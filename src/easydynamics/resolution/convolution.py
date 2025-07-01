import numpy as np
from easydynamics.sample import GaussianComponent, LorentzianComponent, VoigtComponent, DeltaFunctionComponent
from easydynamics.sample import SampleModel

from scipy.signal import fftconvolve
from scipy.special import voigt_profile

from easyscience.variable import Parameter

from typing import Union
from easydynamics.sample.components import ModelComponent

class ResolutionHandler:


    def convolve(self,
                 x: np.ndarray,
                    sample_model: Union[SampleModel, ModelComponent],
                    resolution_model: SampleModel,
                    offset: Parameter,
                    method: str = 'auto',
                    upsample_factor: int = 0) -> np.ndarray:
        """        Convolve a sample model with a resolution model using analytical expressions or numerical FFT.
        Args:
            x (np.ndarray): Evaluation points.
            sample_model (SampleModel or ModelComponent): Signal model.
            resolution_model (SampleModel): Resolution model.
            offset (Parameter): Offset parameter for alignment.
            method (str): Convolution method ('auto', 'analytical', 'numerical').
            upsample_factor (int): Factor by which to upsample (0 = no upsampling).
        Returns:
            np.ndarray: Convolved model evaluated on x.
        """

        if method == 'auto':
            self.auto_decide_method(sample_model, resolution_model)
        elif method == 'analytical':
            return self.analytical_convolve(x, sample_model, resolution_model, offset)
        elif method == 'numerical':
            return self.numerical_convolve(x, sample_model, resolution_model, offset, upsample_factor)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'auto', 'analytical', or 'numerical'.")
        
    def auto_decide_method(self, sample_model: Union[SampleModel, ModelComponent], resolution_model: SampleModel):
        """
        Automatically decide the convolution method based. Use analytical if sample_model._use_detailed_balance is False or if the widths are large enough,
        otherwise use numerical convolution.
        Use analaytical if the sample_model is a ModelComponent.
        """

        if isinstance(sample_model, ModelComponent):
            return 'analytical'
        
        if not sample_model._use_detailed_balance:
            return 'analytical'
        
        



#     """
#     Convolution handler that uses analytical expressions where possible:

#     - Gaussian ⊗ Gaussian → Gaussian
#     - Lorentzian ⊗ Lorentzian → Lorentzian
#     - Gaussian ⊗ Lorentzian → Voigt profile
#     - Fallback: Numerical FFT-based convolution
#     """
#     def numerical_convolve(self, 
#                         x: np.ndarray,
#                         sample_model: Union[SampleModel, ModelComponent],
#                         resolution_model: SampleModel,
#                         offset: Parameter,
#                         upsample_factor: int = 0) -> np.ndarray:
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

#         # Evaluate on dense grid
#         sample_vals = sample_model.evaluate(x_dense - offset.value)
#         resolution_vals = resolution_model.evaluate(x_dense)

#         # Convolution
#         convolved = fftconvolve(sample_vals, resolution_vals, mode='same')
#         convolved *= (x_dense[1] - x_dense[0])  # Normalize

#         # Add delta contributions
#         if isinstance(sample_model, SampleModel):
#             for comp in sample_model.components.values():
#                 if isinstance(comp, DeltaFunctionComponent):
#                     convolved += comp.area.value * resolution_model.evaluate(x_dense - offset.value)
#         elif isinstance(sample_model, DeltaFunctionComponent):
#             convolved += sample_model.area.value * resolution_model.evaluate(x_dense - offset.value)

#         # Interpolate back if upsampled
#         if upsample_factor > 0:
#             from scipy.interpolate import interp1d
#             interp_func = interp1d(x_dense, convolved, kind='linear', bounds_error=False, fill_value=0.0)
#             return interp_func(x)
#         else:
#             return convolved



# # TODO: add support for convolution with components instead of only SampleModels
# # TODO: add support for delta function
#     def convolve(self, x: np.ndarray, sample_model: SampleModel, resolution_model: SampleModel) -> np.ndarray:
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