import numpy as np
from typing import Union, List, Tuple, Callable
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.special import voigt_profile

from easydynamics.sample import DeltaFunctionComponent
from easydynamics.sample.components import ModelComponent  
from easydynamics.sample import GaussianComponent 
from easydynamics.sample import LorentzianComponent
from easydynamics.sample import SampleModel

from easyscience.variable import Parameter


class ResolutionHandler:

    def convolve(
        self,
        x: np.ndarray,
        sample_model: Union[SampleModel, ModelComponent],
        resolution_model: Union[SampleModel, ModelComponent],
        offset: Union[Parameter, float, None] = None,
        method: str = 'analytical',
        upsample_factor: int = 0,
        selected_component_name: Union[str, None] = None
    ) -> np.ndarray:
        """
        Convolve a sample model with a resolution model using analytical expressions or numerical FFT.
        Accepts SampleModel or ModelComponent for both sample and resolution.
        """

        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or not np.all(np.isfinite(x)):
            raise ValueError("`x` must be a 1D finite array.")
        
        if isinstance(sample_model,SampleModel):
            if not sample_model.components:
                raise ValueError("SampleModel must have at least one component.")
            
        if isinstance(resolution_model,SampleModel):
            if not resolution_model.components:
                raise ValueError("ResolutionModel must have at least one component.")

        if method == 'analytical':
            if isinstance(sample_model,SampleModel) and sample_model._use_detailed_balance:
                raise ValueError("Analytical convolution is not supported with detailed balance.")
            return self._analytical_convolve(x, sample_model, resolution_model, offset, upsample_factor,selected_component_name)
        
        if method == 'numerical':
            return self._numerical_convolve(x, sample_model, resolution_model, offset, upsample_factor,selected_component_name)
        
        if method not in ['analytical', 'numerical']:
            raise ValueError(f"Unknown convolution method: {method}. Choose from 'analytical', or 'numerical'.")


    def _numerical_convolve(
        self,
        x: np.ndarray,
        sample_model: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]],
        resolution_model: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]],
        offset: Union[Parameter, np.ndarray, None] = None,
        upsample_factor: int = 5,
        selected_component_name: Union[str, None] = None
    ) -> np.ndarray:
        """
        Numerical convolution using FFT with optional upsampling + extended range.

        sample_model / resolution_model may be:
          - SampleModel
          - ModelComponent
          - Callable: f(x: np.ndarray) -> np.ndarray
        """
        
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or not np.all(np.isfinite(x)):
            raise ValueError("`x` must be a 1D finite array.")


        #TODO: Add support for more span for the dense grid
        def is_uniform(xarr, rtol=1e-5):
            dx = np.diff(xarr)
            return np.allclose(dx, dx[0], rtol=rtol)

        # Build dense grid
        if upsample_factor == 0:
            if not is_uniform(x):
                raise ValueError("Input array `x` must be uniformly spaced if upsample_factor = 0.")
            x_dense = x
        else:
            x_min, x_max = x.min(), x.max()
            span = (x_max - x_min)
            extra = 0.2 * span
            extended_min = x_min - extra
            extended_max = x_max + extra
            num_points = len(x) * upsample_factor
            x_dense = np.linspace(extended_min, extended_max, num_points)
        if offset is None:
            off=0.0
        elif isinstance(offset, Parameter):
            off=offset.value
        elif isinstance(offset, float):
            off=offset
        else:
            raise TypeError(f"Expected offset to be Parameter, float, or None, got {type(offset)}")

        # Handle offset for even length of x in convolution
        if len(x_dense) %2  == 0:
            off2 = -0.5 * (x_dense[1] - x_dense[0])
        else:
            off2 = 0.0

        # Handle the case when x is not symmetric around zero. The resolution is still centered around zero (or close to it), so it needs to be evaluated there.
        if not np.isclose(x_dense.mean(), 0.0):
            span = x_dense.max() - x_dense.min()
            x_dense_resolution = np.linspace(-0.5 * span, 0.5 * span, len(x_dense))
        else:
            x_dense_resolution = x_dense
        
        # Evaluate on dense grid
        sample_vals = self._evaluate_any(sample_model, x_dense - off - off2, selected_component_name)
        resolution_vals = self._evaluate_any(resolution_model, x_dense_resolution)

        # Convolution
        convolved = fftconvolve(sample_vals, resolution_vals, mode='same')
        convolved *= (x_dense[1] - x_dense[0])  # normalize

        # Add delta contributions
        if isinstance(sample_model, SampleModel):
            for comp in sample_model.components.values():
                if isinstance(comp, DeltaFunctionComponent):
                    if selected_component_name is None or comp.name == selected_component_name:
                        convolved += comp.area.value * self._evaluate_any(resolution_model, x_dense - off - comp.center.value)
        elif isinstance(sample_model, DeltaFunctionComponent):
            convolved += sample_model.area.value * self._evaluate_any(resolution_model, x_dense - off - sample_model.center.value)

        if isinstance(resolution_model, SampleModel):
            for comp in resolution_model.components.values():
                if isinstance(comp, DeltaFunctionComponent):
                    convolved += comp.area.value * self._evaluate_any(sample_model, x_dense - off - comp.center.value)
        elif isinstance(resolution_model, DeltaFunctionComponent):
            convolved += resolution_model.area.value * self._evaluate_any(sample_model, x_dense - off - resolution_model.center.value)

        #TODO: if both resolution and sample are delta functions, we should let the user know that they are wrong.

        if upsample_factor > 0:
            return interp1d(x_dense, convolved, kind='linear', bounds_error=False, fill_value=0.0)(x)
        else:
            return convolved

    def _analytical_convolve(
        self,
        x: np.ndarray,
        sample_model: Union[SampleModel, ModelComponent],
        resolution_model: Union[SampleModel, ModelComponent],
        offset: Union[Parameter, float, None] = None,
        upsample_factor: int = 5,
        selected_component_name: Union[str, None] = None
    ) -> np.ndarray:
        """
        Convolve sample with resolution. Accepts SampleModel or single ModelComponent for each.
        - Uses analytic registry for supported pairs.
        - For non-analytic pairs, falls back to a single FFT per sample component
          against the sum of its leftover resolution components using numerical_convolve
          (passing a callable for the summed resolution).
        - Handles delta functions analytically.
        """

        
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or not np.all(np.isfinite(x)):
            raise ValueError("`x` must be a 1D finite array.")
        
        if offset is None:
            off=0.0
        elif isinstance(offset, Parameter):
            off=offset.value
        elif isinstance(offset, float):
            off=offset
        else:
            raise TypeError(f"Expected offset to be Parameter, float, or None, got {type(offset)}")

        # make into lists of components
        if selected_component_name is not None:
            sample_components=self._flatten_to_components(sample_model[selected_component_name])
        else:
            sample_components = self._flatten_to_components(sample_model)
        resolution_components = self._flatten_to_components(resolution_model)

        total = np.zeros_like(x, dtype=float)

        for s in sample_components:
            not_analytical_components: List[ModelComponent] = []

            for r in resolution_components:
                handled, contrib = self._try_analytic_pair(x, s, r, off)
                if handled:
                    total += contrib
                else:
                    not_analytical_components.append(r)

            if not_analytical_components:
                # Sum of non-analytic components
                def rsum(xx: np.ndarray) -> np.ndarray:
                    out = np.zeros_like(xx, dtype=float)
                    for rr in not_analytical_components:
                        out += rr.evaluate(xx)
                    return out

                total += self._numerical_convolve(
                    x=x,
                    sample_model=s,                 # single component
                    resolution_model=rsum,          # sum of components that cannot be handled analytically
                    offset=offset,
                    upsample_factor=upsample_factor
                )

        return total

    def _try_analytic_pair(
        self,
        x: np.ndarray,
        s: ModelComponent,
        r: ModelComponent,
        off: float
    ) -> Tuple[bool, np.ndarray]:
        """
        Attempt an analytic convolution for component pair (s, r).
        Returns (True, contribution) if handled, else (False, zeros).
        """
        # Delta functions
        if isinstance(s, DeltaFunctionComponent):
            return True, s.area.value * r.evaluate(x - s.center.value - off)

        if isinstance(r, DeltaFunctionComponent):
            return True, r.area.value * s.evaluate(x - r.center.value - off)

        # Gaussian + Gaussian --> Gaussian
        if isinstance(s, GaussianComponent) and isinstance(r, GaussianComponent):
            width = np.sqrt(s.width.value**2 + r.width.value**2)
            area  = s.area.value * r.area.value
            center = (s.center.value + r.center.value) + off
            return True, self.gaussian_eval(x, center, width, area)

        # Lorentzian + Lorentzian --> Lorentzian
        if isinstance(s, LorentzianComponent) and isinstance(r, LorentzianComponent):
            width = s.width.value + r.width.value
            area  = s.area.value * r.area.value
            center = (s.center.value + r.center.value) + off
            return True, self.lorentzian_eval(x, center, width, area)

        # Gaussian + Lorentzian --> Voigt 
        if (isinstance(s, GaussianComponent) and isinstance(r, LorentzianComponent)) or \
           (isinstance(s, LorentzianComponent) and isinstance(r, GaussianComponent)):
            if isinstance(s, GaussianComponent):
                G, L = s, r
            else:
                G, L = r, s
            center = (G.center.value + L.center.value) + off
            area   = G.area.value * L.area.value
            return True, self.voigt_eval(x, center, G.width.value, L.width.value, area)

        return False, np.zeros_like(x, dtype=float)

    # ---------------------- helpers & evals -----------------------

    @staticmethod
    def gaussian_eval(x, center, width, area):
        return area * 1/(np.sqrt(2 * np.pi) * width) * np.exp(-0.5 * ((x - center) / width) ** 2)

    @staticmethod
    def lorentzian_eval(x, center, width, area):
        return area * width/np.pi / ((x - center)**2 + width**2)

    @staticmethod
    def voigt_eval(x, center, g_width, l_width, area):
        return area * voigt_profile(x - center, g_width, l_width)

    @staticmethod
    def _flatten_to_components(m: Union[SampleModel, ModelComponent]) -> List[ModelComponent]:
        if isinstance(m, SampleModel):
            return list(m.components.values())
        elif isinstance(m, ModelComponent):
            return [m]
        else:
            raise TypeError(f"Expected SampleModel or ModelComponent, got {type(m)}")

    @staticmethod
    def _evaluate_any(m: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]], x: np.ndarray, selected_component_name: Union[str, None] = None) -> np.ndarray:
        if callable(m):
            return m(x)
        if isinstance(m, (SampleModel, ModelComponent)):
            if selected_component_name is not None:
                return m.evaluate_component(x, name=selected_component_name)
            return m.evaluate(x)
        raise TypeError(f"Expected SampleModel, ModelComponent, or callable, got {type(m)}")


