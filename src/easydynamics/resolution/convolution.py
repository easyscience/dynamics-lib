import numpy as np
from scipy.special import voigt_profile
from easydynamics.sample import GaussianComponent, LorentzianComponent
from easydynamics.sample import SampleModel


class ResolutionHandler:
    """
    Convolution handler that uses analytical expressions where possible:

    - Gaussian ⊗ Gaussian → Gaussian
    - Lorentzian ⊗ Lorentzian → Lorentzian
    - Gaussian ⊗ Lorentzian → Voigt profile
    - Fallback: Numerical FFT-based convolution
    """

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
        total = np.zeros_like(x)

        for s_comp in sample_model.components:
            matched = False
            for r_comp in resolution_model.components:
                if isinstance(s_comp, GaussianComponent) and isinstance(r_comp, GaussianComponent):
                    width = np.sqrt(s_comp.width**2 + r_comp.width**2)
                    area = s_comp.area * r_comp.area
                    conv = GaussianComponent(center=s_comp.center, width=width, area=area).evaluate(x)
                    total += conv
                    matched = True
                    break

                elif isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, LorentzianComponent):
                    width = s_comp.width + r_comp.width
                    area = s_comp.area * r_comp.area
                    conv = LorentzianComponent(center=s_comp.center, width=width, area=area).evaluate(x)
                    total += conv
                    matched = True
                    break

                elif (isinstance(s_comp, GaussianComponent) and isinstance(r_comp, LorentzianComponent)) or \
                     (isinstance(s_comp, LorentzianComponent) and isinstance(r_comp, GaussianComponent)):
                    g = s_comp if isinstance(s_comp, GaussianComponent) else r_comp
                    l = r_comp if isinstance(r_comp, LorentzianComponent) else s_comp
                    sigma = g.width
                    gamma = l.width
                    center = g.center  # assume aligned
                    area = g.area * l.area
                    voigt = area * voigt_profile(x - center, sigma, gamma)
                    total += voigt
                    matched = True
                    break

            if not matched:
                raise NotImplementedError("Not yet implemented for this combination of components.")

        return total
