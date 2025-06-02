import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from easydynamics.sample import GaussianComponent, LorentzianComponent
from easydynamics.sample import SampleModel

def convolve_numerical_interpolated(x, model_component, resolution_model, *,
                                     upsample_factor=None, num_points=None,
                                     x_min=None, x_max=None):
    """
    Perform numerical convolution via FFT with interpolation.

    Args:
        x (np.ndarray): Points to evaluate the convolution at.
        model_component: Component to convolve.
        resolution_model: Resolution model.
        upsample_factor (int, optional): Upsampling factor (default: 4 if not specified).
        num_points (int, optional): Total number of high-resolution points.
        x_min (float, optional): Minimum of resolution axis.
        x_max (float, optional): Maximum of resolution axis.

    Returns:
        np.ndarray: Convolved result interpolated back to original x.
    """
    if upsample_factor is None and num_points is None:
        upsample_factor = 4

    if num_points is None:
        num_points = len(x) * upsample_factor

    # Determine default interval using max width of resolution or 10% margin on model range
    if x_min is None or x_max is None:
        res_centers = []
        res_widths = []
        for comp in resolution_model.components:
            if hasattr(comp, 'center') and hasattr(comp, 'width'):
                res_centers.append(comp.center)
                res_widths.append(comp.width)

        if res_centers and res_widths:
            avg_center = np.mean(res_centers)
            max_width = max(res_widths)
            half_range_resolution = 10 * max_width
        else:
            avg_center = 0.0
            half_range_resolution = 0.0

        model_min, model_max = np.min(x), np.max(x)
        model_center = 0.5 * (model_min + model_max)
        model_half_range = 0.5 * (model_max - model_min)
        half_range_model_margin = 1.1 * model_half_range  # 10% margin

        half_range = max(half_range_resolution, half_range_model_margin)
        x_min = model_center - half_range
        x_max = model_center + half_range

    x_hr = np.linspace(x_min, x_max, num_points)
    model_hr = model_component.evaluate(x_hr)
    resolution_hr = resolution_model.evaluate(x_hr)

    dx = x_hr[1] - x_hr[0]
    conv_hr = scipy.signal.fftconvolve(model_hr, resolution_hr, mode='same') * dx

    interp_func = interp1d(x_hr, conv_hr, kind='linear', bounds_error=False, fill_value=0)
    return interp_func(x)

class HybridResolutionHandler:
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
                conv = convolve_numerical_interpolated(x, s_comp, resolution_model, upsample_factor=6)
                total += conv

        return total
