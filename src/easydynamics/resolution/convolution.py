import numpy as np
from easydynamics.sample import GaussianComponent, LorentzianComponent, VoigtComponent, DeltaFunctionComponent
from easydynamics.sample import SampleModel

from scipy.signal import fftconvolve

class ResolutionHandler:
    """
    Convolution handler that uses analytical expressions where possible:

    - Gaussian ⊗ Gaussian → Gaussian
    - Lorentzian ⊗ Lorentzian → Lorentzian
    - Gaussian ⊗ Lorentzian → Voigt profile
    - Fallback: Numerical FFT-based convolution
    """


    def numerical_convolve(self, x: np.ndarray, sample_model: SampleModel, resolution_model: SampleModel) -> np.ndarray:
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
        sample_values = sample_model.evaluate(x)
        resolution_values = resolution_model.evaluate(x)

        # Perform convolution
        convolved = fftconvolve(sample_values, resolution_values, mode='same')
        # Normalize the result to maintain the area under the curve
        convolved*= (x[1] - x[0])  # Assuming uniform spacing in x


        # Handle delta functions in the sample model
        for comp in sample_model.components:
            if isinstance(comp,DeltaFunctionComponent):                
                convolved=convolved+ resolution_model.evaluate(x)

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

        #TODO: Handle units properly
        #TODO: Allow resolution model that has multiple components that are not all centered
        #TODO: Implement numerical convolution for cases not handled analytically
        total = np.zeros_like(x)

        for s_comp in sample_model.components:
            matched = False
            for r_comp in resolution_model.components:
                if isinstance(s_comp, GaussianComponent) and isinstance(r_comp, GaussianComponent):
                    width = np.sqrt(s_comp.width.value**2 + r_comp.width.value**2)
                    area = s_comp.area * r_comp.area
                    # conv = GaussianComponent(center=s_comp.center.value, width=width, area=area).evaluate(x) # I am not allowed to make new components, since it makes new Parameters

                    conv = self.gaussian_eval(x, s_comp.center.value, width, area)
                    total += conv
                    matched = True
                    break

                elif isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, LorentzianComponent):
                    width = s_comp.width.value + r_comp.width.value
                    area = s_comp.area * r_comp.area
                    # conv = LorentzianComponent(center=s_comp.center.value, width=width, area=area).evaluate(x)
                    conv = self.lorentzian_eval(x, s_comp.center.value, width, area)
                    total += conv
                    matched = True
                    break

                elif (isinstance(s_comp, GaussianComponent) and isinstance(r_comp, LorentzianComponent)) or \
                     (isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, GaussianComponent)):
                    G = s_comp if isinstance(s_comp, GaussianComponent) else r_comp
                    L = r_comp if isinstance(r_comp, LorentzianComponent) else s_comp
                    center = G.center.value #TODO: Handle case where centers are different 
                    area = G.area * L.area
                    voigt = VoigtComponent(center=center, Gwidth=G.width.value, Lwidth=L.width.value, area=area).evaluate(x)
                    total += voigt
                    matched = True
                    break

            if not matched:
                raise NotImplementedError("Not yet implemented for this combination of components.")

        return total
    
    @staticmethod
    def gaussian_eval(x, center, width, area):
        norm = area / (width * np.sqrt(2 * np.pi))
        return norm * np.exp(-0.5 * ((x - center) / width) ** 2)

    @staticmethod
    def lorentzian_eval(x, center, width, area):
        norm = area / (np.pi * width)
        return norm / (1 + ((x - center) / width) ** 2)
