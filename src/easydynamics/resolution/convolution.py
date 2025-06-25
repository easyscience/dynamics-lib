import numpy as np
from easydynamics.sample import GaussianComponent, LorentzianComponent, VoigtComponent, DeltaFunctionComponent
from easydynamics.sample import SampleModel

from scipy.signal import fftconvolve
from scipy.special import voigt_profile

from easyscience.variable import Parameter
class ResolutionHandler:
    """
    Convolution handler that uses analytical expressions where possible:

    - Gaussian ⊗ Gaussian → Gaussian
    - Lorentzian ⊗ Lorentzian → Lorentzian
    - Gaussian ⊗ Lorentzian → Voigt profile
    - Fallback: Numerical FFT-based convolution
    """


    def numerical_convolve(self, x: np.ndarray, sample_model: SampleModel, resolution_model: SampleModel,offset:Parameter) -> np.ndarray:
        """
        Perform numerical convolution using FFT.

        Args:
            x (np.ndarray): Evaluation points.
            sample_model (SampleModel): Signal model.
            resolution_model (SampleModel): Resolution model.

        Returns:
            np.ndarray: Convolved model evaluated on x.
        """

        'TODO: implement upsampling and interpolation to avoid issues with sparse data and non-uniform spacing'

        # Evaluate both models at the same points
        sample_values = sample_model.evaluate(x-offset.value) # TODO: do not evaluate the delta function here. For now, the delta function evaluates to 0 everywhere.
        resolution_values = resolution_model.evaluate(x)

        # Perform convolution
        convolved = fftconvolve(sample_values, resolution_values, mode='same')
        # Normalize the result to maintain the area under the curve
        convolved*= (x[1] - x[0])  # Assuming uniform spacing in x


        # Handle delta functions in the sample model
        for name, comp in sample_model.components.items():
            if isinstance(comp,DeltaFunctionComponent):                
                convolved=convolved+ comp.area.value*resolution_model.evaluate(x-offset.value) #TODO: Normalize the resolution model to have area 1

        return convolved


# TODO: add support for convolution with components instead of only SampleModels
# TODO: add support for delta function
    def convolve(self, x: np.ndarray, sample_model: SampleModel, resolution_model: SampleModel) -> np.ndarray:
        """
        Convolve a sample model with a resolution model.

        Args:
            x (np.ndarray): Evaluation points.
            sample_model (SampleModel): Signal model.
            resolution_model (SampleModel): Resolution model.

        Returns:
            np.ndarray: Convolved model evaluated on x.
        """
        total = np.zeros_like(x, dtype=float)

        for s_name, s_comp in sample_model.components.items():
            matched = False
            for r_name, r_comp in resolution_model.components.items():

                # === Gaussian + Gaussian → Gaussian ===
                if isinstance(s_comp, GaussianComponent) and isinstance(r_comp, GaussianComponent):
                    width = np.sqrt(s_comp.width.value**2 + r_comp.width.value**2)
                    area = s_comp.area.value * r_comp.area.value
                    center = s_comp.center.value + r_comp.center.value 
                    total += self.gaussian_eval(x, center, width, area)
                    matched = True
                    break

                # === Lorentzian + Lorentzian → Lorentzian ===
                elif isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, LorentzianComponent):
                    width = s_comp.width.value + r_comp.width.value
                    area = s_comp.area.value * r_comp.area.value
                    center = s_comp.center.value + r_comp.center.value  
                    total += self.lorentzian_eval(x, center, width, area)
                    matched = True
                    break

                # === Gaussian + Lorentzian → Voigt ===
                elif (
                    isinstance(s_comp, GaussianComponent) and isinstance(r_comp, LorentzianComponent)
                ) or (
                    isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, GaussianComponent)
                ):
                    G = s_comp if isinstance(s_comp, GaussianComponent) else r_comp
                    L = r_comp if isinstance(r_comp, LorentzianComponent) else s_comp
                    center = G.center.value + L.center.value 
                    area = G.area.value * L.area.value
                    total += self.voigt_eval(x, center, G.width.value, L.width.value, area)
                    matched = True
                    break

            if not matched:
                raise NotImplementedError(
                    f"Convolution not implemented for: {type(s_comp).__name__} + {type(r_comp).__name__}"
                )

        return total

    
    @staticmethod
    def gaussian_eval(x, center, width, area):
        norm = area / (width * np.sqrt(2 * np.pi))
        return norm * np.exp(-0.5 * ((x - center) / width) ** 2)

    @staticmethod
    def lorentzian_eval(x, center, width, area):
        norm = area / (np.pi * width)
        return norm / (1 + ((x - center) / width) ** 2)

    @staticmethod
    def voigt_eval(x, center, g_width, l_width, area):
        return area * voigt_profile(x - center, g_width, l_width)