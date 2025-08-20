import pytest
import numpy as np
from scipy.integrate import simpson

from scipy.special import voigt_profile

from easyscience.variable import Parameter
from easydynamics.sample import SampleModel, GaussianComponent, LorentzianComponent, DeltaFunctionComponent
from easydynamics.sample.components import ModelComponent


from easydynamics.resolution import ResolutionHandler

 
class TestConvolution:
    @pytest.fixture
    def sample_model(self):
        test_sample_model = SampleModel(name="TestSampleModel")
        test_sample_model.add_component(LorentzianComponent(center=0.1, width=0.2, area=2.0))
        return test_sample_model

    @pytest.fixture
    def resolution_model(self):
        test_resolution_model = SampleModel(name="TestResolutionModel")
        test_resolution_model.add_component(GaussianComponent(center=0.2, width=0.3, area=3.0))
        return test_resolution_model
       
    @pytest.fixture
    def x(self):
        return np.linspace(-50, 50, 50001)


# Test analytical convolution of components
    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_gauss_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss = GaussianComponent(center=0.1, width=0.3, area=2)
        resolution_gauss = GaussianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample_gauss, resolution_model=resolution_gauss, offset=offset_obj, method=method)

        #EXPECT
        expected_width = np.sqrt(sample_gauss.width.value**2 + resolution_gauss.width.value**2)
        expected_area = sample_gauss.area.value * resolution_gauss.area.value
        expected_center = sample_gauss.center.value + resolution_gauss.center.value + expected_shift
        expected_result = expected_area * np.exp(-0.5 * ((x - expected_center) / expected_width)**2) / (np.sqrt(2 * np.pi) * expected_width)

        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_lorentzian_lorentzian(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_lorentzian = LorentzianComponent(center=0.1, width=0.3, area=2)
        resolution_lorentzian = LorentzianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_lorentzian, resolution_model=resolution_lorentzian,offset=offset_obj,method=method, upsample_factor=5)

        #EXPECT
        expected_width = sample_lorentzian.width.value + resolution_lorentzian.width.value
        expected_area = sample_lorentzian.area.value * resolution_lorentzian.area.value
        expected_center = sample_lorentzian.center.value + resolution_lorentzian.center.value + expected_shift
        expected_result = expected_area * expected_width / np.pi / ((x - expected_center)**2 + expected_width**2)

        np.testing.assert_allclose(convolution, expected_result, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_gauss_lorentzian(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss = GaussianComponent(center=0.1, width=0.3, area=2)
        resolution_lorentzian = LorentzianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample_gauss, resolution_model=resolution_lorentzian, offset=offset_obj, method=method, upsample_factor=5)

        #EXPECT
        expected_center = sample_gauss.center.value + resolution_lorentzian.center.value + expected_shift
        expected_area = sample_gauss.area.value * resolution_lorentzian.area.value
        expected_result = expected_area * voigt_profile(
            x - expected_center,
            sample_gauss.width.value,
            resolution_lorentzian.width.value
        )

        np.testing.assert_allclose(convolution, expected_result, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_lorentzian_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        resolution_gauss = GaussianComponent(center=0.1, width=0.3, area=2)
        sample_lorentzian = LorentzianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_lorentzian, resolution_model=resolution_gauss,offset=offset_obj,method=method,upsample_factor=5)

        #EXPECT
        expected_center = sample_lorentzian.center.value + resolution_gauss.center.value + expected_shift
        expected_area = sample_lorentzian.area.value * resolution_gauss.area.value
        expected_result = expected_area * voigt_profile(
            x - expected_center,
            resolution_gauss.width.value,
            sample_lorentzian.width.value
        )

        np.testing.assert_allclose(convolution, expected_result, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_delta_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_delta = DeltaFunctionComponent(name="Delta", center=0.1, area=2)
        resolution_gauss = GaussianComponent(center=0.2, width=0.3, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_delta, resolution_model=resolution_gauss,offset = offset_obj, method=method)

        #EXPECT
        expected_center = sample_delta.center.value + resolution_gauss.center.value + expected_shift
        expected_area = sample_delta.area.value * resolution_gauss.area.value
        expected_result = expected_area * np.exp(-0.5 * ((x - expected_center) / resolution_gauss.width.value)**2) / (np.sqrt(2 * np.pi) * resolution_gauss.width.value)

        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_analytical_components_gauss_delta(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss = GaussianComponent(center=0.1, width=0.2, area=2)
        resolution_delta = DeltaFunctionComponent(name="Delta", center=0.2, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_gauss, resolution_model=resolution_delta,offset = offset_obj, method=method)

        #EXPECT
        expected_center = sample_gauss.center.value + resolution_delta.center.value + expected_shift
        expected_area = sample_gauss.area.value * resolution_delta.area.value
        expected_result = expected_area * np.exp(-0.5 * ((x - expected_center) / sample_gauss.width.value)**2) / (np.sqrt(2 * np.pi) * sample_gauss.width.value)

        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_models_gauss_gauss_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss1 = GaussianComponent(center=0.1, width=0.3, area=2,name="SampleGauss1")
        sample_gauss2 = GaussianComponent(center=0.2, width=0.4, area=3,name="SampleGauss2")
        resolution_gauss = GaussianComponent(center=0.3, width=0.5, area=4)

        sample= SampleModel(name="SampleModel")
        sample.add_component(sample_gauss1)
        sample.add_component(sample_gauss2)

        resolution = SampleModel(name="ResolutionModel")
        resolution.add_component(resolution_gauss)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample, resolution_model=resolution, offset=offset_obj, method=method)

        #EXPECT
        expected_width1 = np.sqrt(sample_gauss1.width.value**2 + resolution_gauss.width.value**2)
        expected_width2 = np.sqrt(sample_gauss2.width.value**2 + resolution_gauss.width.value**2)
        expected_area1 = sample_gauss1.area.value * resolution_gauss.area.value
        expected_area2 = sample_gauss2.area.value * resolution_gauss.area.value
        expected_center1 = sample_gauss1.center.value + resolution_gauss.center.value + expected_shift
        expected_center2 = sample_gauss2.center.value + resolution_gauss.center.value + expected_shift

        expected_result = (
            expected_area1 * np.exp(-0.5 * ((x - expected_center1) / expected_width1)**2) / (np.sqrt(2 * np.pi) * expected_width1) +
            expected_area2 * np.exp(-0.5 * ((x - expected_center2) / expected_width2)**2) / (np.sqrt(2 * np.pi) * expected_width2)
        )
        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

#  import numpy as np
# from typing import Union, List, Tuple, Callable
# from scipy.signal import fftconvolve
# from scipy.interpolate import interp1d
# from scipy.special import voigt_profile

# from easydynamics.sample import DeltaFunctionComponent
# from easydynamics.sample.components import ModelComponent  
# from easydynamics.sample import GaussianComponent 
# from easydynamics.sample import LorentzianComponent
# from easydynamics.sample import SampleModel

# from easyscience.variable import Parameter


# class ResolutionHandler:

#     def convolve(
#         self,
#         x: np.ndarray,
#         sample_model: Union[SampleModel, ModelComponent],
#         resolution_model: Union[SampleModel, ModelComponent],
#         offset: Union[Parameter, None] = None,
#         method: str = 'analytical',
#         upsample_factor: int = 0
#     ) -> np.ndarray:
#         """
#         Convolve a sample model with a resolution model using analytical expressions or numerical FFT.
#         Accepts SampleModel or ModelComponent for both sample and resolution.
#         """

#         x = np.asarray(x, dtype=float)
#         if x.ndim != 1 or not np.all(np.isfinite(x)):
#             raise ValueError("`x` must be a 1D finite array.")

#         if method == 'analytical':
#             if isinstance(sample_model,SampleModel) and sample_model._use_detailed_balance:
#                 raise ValueError("Analytical convolution is not supported with detailed balance.")
#             return self.analytical_convolve(x, sample_model, resolution_model, offset, upsample_factor)
#         if method == 'numerical':
#             return self.numerical_convolve(x, sample_model, resolution_model, offset, upsample_factor)
#         if method not in ['analytical', 'numerical']:
#             raise ValueError(f"Unknown convolution method: {method}. Choose from 'analytical', or 'numerical'.")


#     def numerical_convolve(
#         self,
#         x: np.ndarray,
#         sample_model: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]],
#         resolution_model: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]],
#         offset: Union[Parameter, None] = None,
#         upsample_factor: int = 5
#     ) -> np.ndarray:
#         """
#         Numerical convolution using FFT with optional upsampling + extended range.

#         sample_model / resolution_model may be:
#           - SampleModel
#           - ModelComponent
#           - Callable: f(x: np.ndarray) -> np.ndarray
#         """

#         #TODO: Add support for more span for the dense grid
#         def is_uniform(xarr, rtol=1e-5):
#             dx = np.diff(xarr)
#             return np.allclose(dx, dx[0], rtol=rtol)

#         # Build dense grid
#         if upsample_factor == 0:
#             if not is_uniform(x):
#                 raise ValueError("Input array `x` must be uniformly spaced if upsample_factor = 0.")
#             x_dense = x
#         else:
#             x_min, x_max = x.min(), x.max()
#             span = (x_max - x_min)
#             extra = 0.2 * span
#             extended_min = x_min - extra
#             extended_max = x_max + extra
#             num_points = len(x) * upsample_factor
#             x_dense = np.linspace(extended_min, extended_max, num_points)

#         off = offset.value if offset is not None else 0.0

#         # Evaluate on dense grid
#         sample_vals = self._evaluate_any(sample_model, x_dense - off)
#         resolution_vals = self._evaluate_any(resolution_model, x_dense)

#         # Convolution
#         convolved = fftconvolve(sample_vals, resolution_vals, mode='same')
#         convolved *= (x_dense[1] - x_dense[0])  # normalize

#         # Add delta contributions
#         if isinstance(sample_model, SampleModel):
#             for comp in sample_model.components.values():
#                 if isinstance(comp, DeltaFunctionComponent):
#                     convolved += comp.area.value * self._evaluate_any(resolution_model, x_dense-off)
#         elif isinstance(sample_model, DeltaFunctionComponent):
#             convolved += sample_model.area.value * self._evaluate_any(resolution_model, x_dense - off)

#         if isinstance(resolution_model, SampleModel):
#             for comp in resolution_model.components.values():
#                 if isinstance(comp, DeltaFunctionComponent):
#                     convolved += comp.area.value * self._evaluate_any(sample_model, x_dense - off)
#         elif isinstance(resolution_model, DeltaFunctionComponent):
#             convolved += resolution_model.area.value * self._evaluate_any(sample_model, x_dense - off)

#         #TODO: if both resolution and sample are delta functions, we should let the user know that they are wrong.

#         if upsample_factor > 0:
#             return interp1d(x_dense, convolved, kind='linear', bounds_error=False, fill_value=0.0)(x)
#         else:
#             return convolved

#     def analytical_convolve(
#         self,
#         x: np.ndarray,
#         sample_model: Union[SampleModel, ModelComponent],
#         resolution_model: Union[SampleModel, ModelComponent],
#         offset: Union[Parameter, None] = None,
#         upsample_factor: int = 5
#     ) -> np.ndarray:
#         """
#         Convolve sample with resolution. Accepts SampleModel or single ModelComponent for each.
#         - Uses analytic registry for supported pairs.
#         - For non-analytic pairs, falls back to a single FFT per sample component
#           against the sum of its leftover resolution components using numerical_convolve
#           (passing a callable for the summed resolution).
#         - Handles delta functions analytically.
#         """
#         off = offset.value if offset is not None else 0.0

#         # make into lists of components
#         sample_components = self._flatten_to_components(sample_model)
#         resolution_components = self._flatten_to_components(resolution_model)

#         total = np.zeros_like(x, dtype=float)

#         for s in sample_components:
#             not_analytical_components: List[ModelComponent] = []

#             for r in resolution_components:
#                 handled, contrib = self._try_analytic_pair(x, s, r, off)
#                 if handled:
#                     total += contrib
#                 else:
#                     not_analytical_components.append(r)

#             if not_analytical_components:
#                 # Sum of non-analytic components
#                 def rsum(xx: np.ndarray) -> np.ndarray:
#                     out = np.zeros_like(xx, dtype=float)
#                     for rr in not_analytical_components:
#                         out += rr.evaluate(xx)
#                     return out

#                 total += self.numerical_convolve(
#                     x=x,
#                     sample_model=s,                 # single component
#                     resolution_model=rsum,          # sum of components that cannot be handled analytically
#                     offset=offset,
#                     upsample_factor=upsample_factor
#                 )

#         return total

#     def _try_analytic_pair(
#         self,
#         x: np.ndarray,
#         s: ModelComponent,
#         r: ModelComponent,
#         off: float
#     ) -> Tuple[bool, np.ndarray]:
#         """
#         Attempt an analytic convolution for component pair (s, r).
#         Returns (True, contribution) if handled, else (False, zeros).
#         """
#         # Delta functions
#         if isinstance(s, DeltaFunctionComponent):
#             return True, s.area.value * r.evaluate(x - s.center.value - off)

#         if isinstance(r, DeltaFunctionComponent):
#             return True, r.area.value * s.evaluate(x - r.center.value - off)

#         # Gaussian + Gaussian --> Gaussian
#         if isinstance(s, GaussianComponent) and isinstance(r, GaussianComponent):
#             width = np.sqrt(s.width.value**2 + r.width.value**2)
#             area  = s.area.value * r.area.value
#             center = (s.center.value + r.center.value) + off
#             return True, self.gaussian_eval(x, center, width, area)

#         # Lorentzian + Lorentzian --> Lorentzian
#         if isinstance(s, LorentzianComponent) and isinstance(r, LorentzianComponent):
#             width = s.width.value + r.width.value
#             area  = s.area.value * r.area.value
#             center = (s.center.value + r.center.value) + off
#             return True, self.lorentzian_eval(x, center, width, area)

#         # Gaussian + Lorentzian --> Voigt 
#         if (isinstance(s, GaussianComponent) and isinstance(r, LorentzianComponent)) or \
#            (isinstance(s, LorentzianComponent) and isinstance(r, GaussianComponent)):
#             if isinstance(s, GaussianComponent):
#                 G, L = s, r
#             else:
#                 G, L = r, s
#             center = (G.center.value + L.center.value) + off
#             area   = G.area.value * L.area.value
#             return True, self.voigt_eval(x, center, G.width.value, L.width.value, area)

#         return False, np.zeros_like(x, dtype=float)

#     # ---------------------- helpers & evals -----------------------

#     @staticmethod
#     def gaussian_eval(x, center, width, area):
#         return area * 1/(np.sqrt(2 * np.pi) * width) * np.exp(-0.5 * ((x - center) / width) ** 2)

#     @staticmethod
#     def lorentzian_eval(x, center, width, area):
#         return area * width/np.pi / ((x - center)**2 + width**2)

#     @staticmethod
#     def voigt_eval(x, center, g_width, l_width, area):
#         return area * voigt_profile(x - center, g_width, l_width)

#     @staticmethod
#     def _flatten_to_components(m: Union[SampleModel, ModelComponent]) -> List[ModelComponent]:
#         if isinstance(m, SampleModel):
#             return list(m.components.values())
#         elif isinstance(m, ModelComponent):
#             return [m]
#         else:
#             raise TypeError(f"Expected SampleModel or ModelComponent, got {type(m)}")

#     @staticmethod
#     def _evaluate_any(m: Union[SampleModel, ModelComponent, Callable[[np.ndarray], np.ndarray]], x: np.ndarray) -> np.ndarray:
#         if callable(m):
#             return m(x)
#         if isinstance(m, (SampleModel, ModelComponent)):
#             return m.evaluate(x)
#         raise TypeError(f"Expected SampleModel, ModelComponent, or callable, got {type(m)}")


