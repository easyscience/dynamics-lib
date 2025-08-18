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
        offset: Union[Parameter, None] = None,
        method: str = 'analytical',
        upsample_factor: int = 0
    ) -> np.ndarray:
        """
        Convolve a sample model with a resolution model using analytical expressions or numerical FFT.
        Accepts SampleModel or single ModelComponent for both sample and resolution.
        """

        if method == 'analytical':
            return self.analytical_convolve(x, sample_model, resolution_model, offset, upsample_factor)
        if method == 'numerical':
            return self.numerical_convolve(x, sample_model, resolution_model, offset, upsample_factor)
        if method not in ['analytical', 'numerical']:
            raise ValueError(f"Unknown method: {method}. Choose from 'analytical', or 'numerical'.")


    def numerical_convolve(
        self,
        x: np.ndarray,
        sample_model: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]],
        resolution_model: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]],
        offset: Union[Parameter, None] = None,
        upsample_factor: int = 5
    ) -> np.ndarray:
        """
        Numerical convolution using FFT with optional upsampling + extended range.

        sample_model / resolution_model may be:
          - SampleModel
          - ModelComponent
          - Callable: f(x: np.ndarray) -> np.ndarray
        """
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

        off = offset.value if offset is not None else 0.0

        # Evaluate on dense grid
        sample_vals = self._evaluate_any(sample_model, x_dense - off)
        resolution_vals = self._evaluate_any(resolution_model, x_dense)

        # Convolution
        convolved = fftconvolve(sample_vals, resolution_vals, mode='same')
        convolved *= (x_dense[1] - x_dense[0])  # normalize

        # Add delta contributions
        if isinstance(sample_model, SampleModel):
            for comp in sample_model.components.values():
                if isinstance(comp, DeltaFunctionComponent):
                    convolved += comp.area.value * resolution_model.evaluate(x_dense - off)
        elif isinstance(sample_model, DeltaFunctionComponent):
            convolved += sample_model.area.value * resolution_model.evaluate(x_dense - off)

        if upsample_factor > 0:
            return interp1d(x_dense, convolved, kind='linear', bounds_error=False, fill_value=0.0)(x)
        else:
            return convolved

    def analytical_convolve(
        self,
        x: np.ndarray,
        sample_model: Union[SampleModel, ModelComponent],
        resolution_model: Union[SampleModel, ModelComponent],
        offset: Union[Parameter, None] = None,
        upsample_factor: int = 5
    ) -> np.ndarray:
        """
        Convolve sample with resolution. Accepts SampleModel or single ModelComponent for each.
        - Uses analytic registry for supported pairs.
        - For non-analytic pairs, falls back to a single FFT per sample component
          against the sum of its leftover resolution components using numerical_convolve
          (passing a callable for the summed resolution).
        - Handles delta functions analytically.
        """
        off = offset.value if offset is not None else 0.0

        # Normalize to lists of components
        sample_components = self._flatten_to_components(sample_model)
        resolution_components = self._flatten_to_components(resolution_model)

        total = np.zeros_like(x, dtype=float)

        for s in sample_components:
            hard_R: List[ModelComponent] = []

            for r in resolution_components:
                handled, contrib = self._try_analytic_pair(x, s, r, off)
                if handled:
                    total += contrib
                else:
                    hard_R.append(r)

            if hard_R:
                # Sum of hard resolution parts as a callable; avoids _SumComponent
                def rsum(xx: np.ndarray) -> np.ndarray:
                    out = np.zeros_like(xx, dtype=float)
                    for rr in hard_R:
                        out += rr.evaluate(xx)
                    return out

                total += self.numerical_convolve(
                    x=x,
                    sample_model=s,                 # single component
                    resolution_model=rsum,          # callable sum
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
    def _evaluate_any(m: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]], x: np.ndarray) -> np.ndarray:
        if callable(m):
            return m(x)
        if isinstance(m, (SampleModel, ModelComponent)):
            return m.evaluate(x)
        raise TypeError(f"Expected SampleModel, ModelComponent, or callable, got {type(m)}")




# import numpy as np
# from easydynamics.sample import GaussianComponent, LorentzianComponent, VoigtComponent, DeltaFunctionComponent
# from easydynamics.sample import SampleModel

# from scipy.signal import fftconvolve
# from scipy.special import voigt_profile

# from easyscience.variable import Parameter

# from typing import Union
# from easydynamics.sample.components import ModelComponent

# class ResolutionHandler:


#     def convolve(self,
#                  x: np.ndarray,
#                     sample_model: Union[SampleModel, ModelComponent],
#                     resolution_model: SampleModel,
#                     offset: Union[Parameter, None] = None,
#                     method: str = 'auto',
#                     upsample_factor: int = 0) -> np.ndarray:
#         """        Convolve a sample model with a resolution model using analytical expressions or numerical FFT.
#         Args:
#             x (np.ndarray): Evaluation points.
#             sample_model (SampleModel or ModelComponent): Signal model.
#             resolution_model (SampleModel): Resolution model.
#             offset (Parameter): Offset parameter for alignment.
#             method (str): Convolution method ('auto', 'analytical', 'numerical').
#             upsample_factor (int): Factor by which to upsample (0 = no upsampling).
#         Returns:
#             np.ndarray: Convolved model evaluated on x.
#         """

#         if method == 'auto':
#             self.auto_decide_method(sample_model, resolution_model)
#         elif method == 'analytical':
#             return self.analytical_convolve(x, sample_model, resolution_model, offset)
#         elif method == 'numerical':
#             return self.numerical_convolve(x, sample_model, resolution_model, offset, upsample_factor)
#         else:
#             raise ValueError(f"Unknown method: {method}. Choose from 'auto', 'analytical', or 'numerical'.")
        
#     def auto_decide_method(self, sample_model: Union[SampleModel, ModelComponent], resolution_model: SampleModel):
#         """
#         Automatically decide the convolution method based. Use analytical if sample_model._use_detailed_balance is False or if the widths are large enough,
#         otherwise use numerical convolution.
#         Use analytical if the sample_model is a ModelComponent.
#         """

#         if isinstance(sample_model, ModelComponent):
#             return 'analytical'
        
#         if not sample_model._use_detailed_balance:
#             return 'analytical'
        
        
#     def numerical_convolve(self, 
#                         x: np.ndarray,
#                         sample_model: Union[SampleModel, ModelComponent],
#                         resolution_model: Union[SampleModel, ModelComponent],
#                         offset: Union[Parameter,None] = None,
#                         upsample_factor: int = 5) -> np.ndarray: #TODO: remove standard value
#         """
#         Perform numerical convolution using FFT, with optional upsampling and extended evaluation range.

#         Args:
#             x (np.ndarray): Evaluation points.
#             sample_model (SampleModel or ModelComponent): Signal model.
#             resolution_model (SampleModel): Resolution model.
#             offset (Parameter): Offset parameter for alignment.
#             upsample_factor (int): Factor by which to upsample (0 = no upsampling).

#         Returns:
#             np.ndarray: Convolved model evaluated on x.
#         """

#         def is_uniform(x, rtol=1e-5):
#             dx = np.diff(x)
#             return np.allclose(dx, dx[0], rtol=rtol)

#         if upsample_factor == 0:
#             if not is_uniform(x):
#                 raise ValueError("Input array `x` must be uniformly spaced if upsample_factor = 0.")
#             x_dense = x
#         else:
#             # Extend range by 20% of the total width to improve accuracy at the edges
#             x_min, x_max = x.min(), x.max()
#             dx = (x_max - x_min)
#             extra = 0.2 * dx
#             extended_min = x_min - extra
#             extended_max = x_max + extra

#             # Use more points on the dense grid
#             num_points = len(x) * upsample_factor
#             x_dense = np.linspace(extended_min, extended_max, num_points)

#         offset_value = offset.value if offset is not None else 0.0

#         # Evaluate on dense grid
#         sample_vals = sample_model.evaluate(x_dense - offset_value)
#         resolution_vals = resolution_model.evaluate(x_dense)

#         # Convolution
#         convolved = fftconvolve(sample_vals, resolution_vals, mode='same')
#         convolved *= (x_dense[1] - x_dense[0])  # Normalize

#         # Add delta contributions
#         if isinstance(sample_model, SampleModel):
#             for comp in sample_model.components.values():
#                 if isinstance(comp, DeltaFunctionComponent):
#                     convolved += comp.area.value * resolution_model.evaluate(x_dense - offset_value)
#         elif isinstance(sample_model, DeltaFunctionComponent):
#             convolved += sample_model.area.value * resolution_model.evaluate(x_dense - offset_value)

#         # Interpolate back if upsampled
#         if upsample_factor > 0:
#             from scipy.interpolate import interp1d
#             interp_func = interp1d(x_dense, convolved, kind='linear', bounds_error=False, fill_value=0.0)
#             return interp_func(x)
#         else:
#             return convolved


# # TODO: add support for convolution with components instead of only SampleModels
# # TODO: add support for delta function
#     def analytical_convolve(self, x: np.ndarray, sample_model: Union[SampleModel, ModelComponent], resolution_model: Union[SampleModel, ModelComponent]) -> np.ndarray:
#         """
#         Convolve a sample model with a resolution model.

#         Args:
#             x (np.ndarray): Evaluation points.
#             sample_model (SampleModel): Signal model.
#             resolution_model (SampleModel): Resolution model.

#         Returns:
#             np.ndarray: Convolved model evaluated on x.
#         """
#         total = np.zeros_like(x, dtype=float)

#         for s_name, s_comp in sample_model.components.items():
#             matched = False
#             for r_name, r_comp in resolution_model.components.items():

#                 # === Gaussian + Gaussian → Gaussian ===
#                 if isinstance(s_comp, GaussianComponent) and isinstance(r_comp, GaussianComponent):
#                     width = np.sqrt(s_comp.width.value**2 + r_comp.width.value**2)
#                     area = s_comp.area.value * r_comp.area.value
#                     center = s_comp.center.value + r_comp.center.value 
#                     total += self.gaussian_eval(x, center, width, area)
#                     matched = True
#                     break

#                 # === Lorentzian + Lorentzian → Lorentzian ===
#                 elif isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, LorentzianComponent):
#                     width = s_comp.width.value + r_comp.width.value
#                     area = s_comp.area.value * r_comp.area.value
#                     center = s_comp.center.value + r_comp.center.value  
#                     total += self.lorentzian_eval(x, center, width, area)
#                     matched = True
#                     break

#                 # === Gaussian + Lorentzian → Voigt ===
#                 elif (
#                     isinstance(s_comp, GaussianComponent) and isinstance(r_comp, LorentzianComponent)
#                 ) or (
#                     isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, GaussianComponent)
#                 ):
#                     G = s_comp if isinstance(s_comp, GaussianComponent) else r_comp
#                     L = r_comp if isinstance(r_comp, LorentzianComponent) else s_comp
#                     center = G.center.value + L.center.value 
#                     area = G.area.value * L.area.value
#                     total += self.voigt_eval(x, center, G.width.value, L.width.value, area)
#                     matched = True
#                     break

#             if not matched:
#                 raise NotImplementedError(
#                     f"Convolution not implemented for: {type(s_comp).__name__} + {type(r_comp).__name__}"
#                 )

#         return total

    
#     @staticmethod
#     def gaussian_eval(x, center, width, area):
#         norm = area / (width * np.sqrt(2 * np.pi))
#         return norm * np.exp(-0.5 * ((x - center) / width) ** 2)

#     @staticmethod
#     def lorentzian_eval(x, center, width, area):
#         norm = area / (np.pi * width)
#         return norm / (1 + ((x - center) / width) ** 2)

#     @staticmethod
#     def voigt_eval(x, center, g_width, l_width, area):
#         return area * voigt_profile(x - center, g_width, l_width)