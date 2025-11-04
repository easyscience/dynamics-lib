import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.signal import fftconvolve
from scipy.special import voigt_profile

from easydynamics.sample_model import DeltaFunction, Gaussian, Lorentzian, SampleModel
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.detailed_balance import (
    _detailed_balance_factor as detailed_balance_factor,
)

Numerical = Union[float, int]


def convolution(
    energy: np.ndarray,
    sample_model: Union[SampleModel, ModelComponent],
    resolution_model: Union[SampleModel, ModelComponent],
    offset: Optional[Union[Parameter, float, None]] = None,
    method: Optional[str] = "auto",
    upsample_factor: Optional[int] = 0,
    extension_factor: Optional[float] = 0.2,
    temperature: Optional[Union[Parameter, float, None]] = None,
    temperature_unit: Union[str, sc.Unit] = "K",
    energy_unit: Optional[Union[str, sc.Unit]] = "meV",
    normalize_detailed_balance: Optional[bool] = True,
) -> np.ndarray:
    """
    Calculate the convolution of a sample model with a resolution model using analytical expressions or numerical FFT.
    Accepts SampleModel or ModelComponent for both sample and resolution.
    If method is 'auto', analytical convolution is preferred when possible, otherwise numerical convolution is used.
    Detailed balancing is included if temperature is provided. This requires numerical convolution and that the units
    of energy and temperature are provided. An error will be raised if the units are not compatible.
    The calculated model is shifted by the specified offset.

    Examples:
    energy = np.linspace(-10, 10, 100)
    sample = SampleModel()
    sample.add_component(Gaussian(name="SampleGaussian", area=1.0, center=0.1, width=1.0))
    resolution = Gaussian(name="ResolutionGaussian", area=1.0, center=0.0, width=0.5)
    result = convolution(energy, sample, resolution, offset=0.2)

    energy = np.linspace(-10, 10, 100)
    sample = SampleModel()
    sample.add_component(Gaussian(name="Gaussian", area=1.0, center=0.1, width=1.0))
    sample.add_component(DampedHarmonicOscillator(name="DHO", area=2.0, center=1.5, width=0.2))
    sample.add_component(DeltaFunction(name="Delta", area=0.5, center=0.0))

    resolution = SampleModel()
    resolution.add_component(Gaussian(name="ResolutionGaussian", area=0.8, center=0.0, width=0.5))
    resolution.add_component(Lorentzian(name="ResolutionLorentzian", area=0.2, center=0.1, width=0.3))

    result_auto = convolution(energy, sample, resolution, offset=0.2, method='auto', upsample_factor=5, extension_factor=0.2)
    result_numerical = convolution(energy, sample, resolution, offset=0.2, method='numerical', upsample_factor=5, extension_factor=0.2)


    Args:
        energy : np.ndarray
            1D array of energy transfer where the convolution is evaluated.
        sample_model : SampleModel or ModelComponent
            The sample model to be convolved.
        resolution_model : SampleModel or ModelComponent
            The resolution model to convolve with.
        offset : Parameter, float, or None, optional
            The offset to apply to the x values before convolution.
        method : str, optional
            The convolution method to use: 'auto', 'analytical' or 'numerical'. Default is 'auto'.
        upsample_factor : int, optional
            The factor by which to upsample the input data before numerical convolution. Default is 0 (no upsampling).
        extension_factor : float, optional
            The factor by which to extend the input data range before numerical convolution. Default is 0.2.
        temperature : Parameter, float, or None, optional
            The temperature to use for detailed balance calculations. Default is None.
        temperature_unit : str or sc.Unit, optional
            The unit of the temperature parameter. Default is 'K'.
        energy_unit : str or sc.Unit, optional
            The unit of the energy. Default is 'meV'.
        normalize_detailed_balance : bool, optional
            Whether to normalize the detailed balance factor. Default is True.
    """

    # Input validation
    if not isinstance(energy, np.ndarray):
        raise TypeError(
            f"`x` is an instance of {type(energy).__name__}, but must be a numpy array."
        )

    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 1 or not np.all(np.isfinite(energy)):
        raise ValueError("`x` must be a 1D finite array.")

    if not isinstance(sample_model, (SampleModel, ModelComponent)):
        raise TypeError(
            f"`sample_model` is an instance of {type(sample_model).__name__}, but must be SampleModel or ModelComponent."
        )

    if not isinstance(resolution_model, (SampleModel, ModelComponent)):
        raise TypeError(
            f"`resolution_model` is an instance of {type(resolution_model).__name__}, but must be SampleModel or ModelComponent."
        )

    if isinstance(sample_model, SampleModel):
        if not sample_model.components:
            raise ValueError("SampleModel must have at least one component.")

    if isinstance(resolution_model, SampleModel):
        if not resolution_model.components:
            raise ValueError("ResolutionModel must have at least one component.")

    # Handle offset
    if offset is None:
        offset_float = 0.0
    elif isinstance(offset, Parameter):
        offset_float = offset.value
    elif isinstance(offset, Numerical):
        offset_float = float(offset)
    else:
        raise TypeError(
            f"Expected offset to be Parameter, number, or None, got {type(offset)}"
        )

    if not isinstance(upsample_factor, int) or upsample_factor < 0:
        raise ValueError("upsample_factor must be a non-negative integer.")

    if not isinstance(extension_factor, float) or extension_factor < 0.0:
        raise ValueError("extension_factor must be a non-negative float.")

    if temperature is not None:
        if energy_unit is None:
            raise ValueError(
                "energy_unit must be provided when temperature is specified."
            )
        if not isinstance(energy_unit, (str, sc.Unit)):
            raise TypeError(
                f"Expected energy_unit to be str or sc.Unit, got {type(energy_unit)}"
            )

    use_numerical_convolution_as_fallback = False
    if method == "auto":
        if temperature is not None:
            method = "numerical"
        else:
            method = "analytical"
            use_numerical_convolution_as_fallback = True

    if method == "analytical":
        if temperature is not None:
            raise ValueError(
                "Analytical convolution is not supported with detailed balance. Set method to 'numerical' instead or set the temperature to None."
            )
        return _analytical_convolution(
            x=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset_float=offset_float,
            use_numerical_convolution_as_fallback=use_numerical_convolution_as_fallback,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
        )
    elif method == "numerical":
        return _numerical_convolution(
            x=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset_float=offset_float,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
            temperature=temperature,
            temperature_unit=temperature_unit,
            x_unit=energy_unit,
            normalize_detailed_balance=normalize_detailed_balance,
        )
    else:
        raise ValueError(
            f"Unknown convolution method: {method}. Choose from 'analytical', or 'numerical'."
        )


def _numerical_convolution(
    x: np.ndarray,
    sample_model: Union[SampleModel, ModelComponent],
    resolution_model: Union[SampleModel, ModelComponent],
    offset_float: Optional[float] = 0.0,
    upsample_factor: Optional[int] = 5,
    extension_factor: Optional[float] = 0.2,
    temperature: Optional[Union[Parameter, float]] = None,
    temperature_unit: Optional[Union[str, sc.Unit]] = "K",
    x_unit: Optional[Union[str, sc.Unit]] = "meV",
    normalize_detailed_balance: Optional[bool] = True,
) -> np.ndarray:
    """
    Numerical convolution using FFT with optional upsampling + extended range.
    Includes detailed balance correction if temperature is provided.

    Args:
        x : np.ndarray
            1D array of x values where the convolution is evaluated.
        sample_model : SampleModel or ModelComponent
            The sample model to be convolved.
        resolution_model : SampleModel or ModelComponent
            The resolution model to convolve with.
        offset_float : float, or None, optional
            The offset to apply to the input array.
        upsample_factor : int, optional
            The factor by which to upsample the input data before convolution. Default is 5.
        extension_factor : float, optional
            The factor by which to extend the input data range before convolution. Default is 0.2.
        temperature : Parameter, float, or None, optional
            The temperature to use for detailed balance correction. Default is None.
        temperature_unit : str or sc.Unit, optional
            The unit of the temperature parameter. Default is 'K'.
        x_unit : str or sc.Unit, optional
            The unit of the x parameter. Default is 'meV'.
        normalize_detailed_balance : bool, optional
            Whether to normalize the detailed balance factor. Default is True.
    Returns:
        np.ndarray
            The convolved values evaluated at x.
    """

    # Build dense grid
    x_dense = _create_dense_grid(
        x, upsample_factor=upsample_factor, extension_factor=extension_factor
    )

    dx = x_dense[1] - x_dense[0]
    span = x_dense.max() - x_dense.min()
    # Handle offset for even length of x in convolution.
    # The convolution of two arrays of length N is of length 2N-1. When using 'same' mode, only the central N points are kept,
    # so the output has the same length as the input.
    # However, if N is even, the center falls between two points, leading to a half-bin offset.
    # For example, if N=4, the convolution has length 7, and when we select the 4 central points we either get
    # indices [2,3,4,5] or [1,2,3,4], both of which are offset by 0.5*dx from the true center at index 3.5.
    if len(x_dense) % 2 == 0:
        x_even_length_offset = -0.5 * dx
    else:
        x_even_length_offset = 0.0

    # Handle the case when x is not symmetric around zero. The resolution is still centered around zero (or close to it), so it needs to be evaluated there.
    if not np.isclose(x_dense.mean(), 0.0):
        x_dense_centered = np.linspace(-0.5 * span, 0.5 * span, len(x_dense))
    else:
        x_dense_centered = x_dense

    # Give warnings if peaks are very wide or very narrow
    _check_width_thresholds(sample_model, span, dx, "sample model")
    _check_width_thresholds(resolution_model, span, dx, "resolution model")

    # Evaluate on dense grid and interpolate at the end
    if isinstance(sample_model, SampleModel):
        sample_vals = sample_model.evaluate_without_delta(
            x_dense - offset_float - x_even_length_offset
        )
    elif isinstance(sample_model, DeltaFunction):
        sample_vals = np.zeros_like(x_dense)
    else:
        sample_vals = sample_model.evaluate(
            x_dense - offset_float - x_even_length_offset
        )

    # Detailed balance correction
    if temperature is not None:
        detailed_balance_factor_correction = detailed_balance_factor(
            energy=x_dense,
            temperature=temperature,
            energy_unit=x_unit,
            temperature_unit=temperature_unit,
            divide_by_temperature=normalize_detailed_balance,
        )
        sample_vals *= detailed_balance_factor_correction

    # Delta functions are handled separately for accuracy
    if isinstance(resolution_model, SampleModel):
        resolution_vals = resolution_model.evaluate_without_delta(x_dense_centered)
    elif isinstance(resolution_model, DeltaFunction):
        resolution_vals = np.zeros_like(x_dense_centered)
    else:
        resolution_vals = resolution_model.evaluate(x_dense_centered)

    # Convolution
    convolved = fftconvolve(sample_vals, resolution_vals, mode="same")
    convolved *= dx  # normalize

    if upsample_factor > 0:
        # interpolate back to original x grid
        convolved = np.interp(x, x_dense, convolved, left=0.0, right=0.0)

    # Add delta contributions on original grid
    # collect deltas
    sample_deltas = _find_delta_components(sample_model)
    resolution_deltas = _find_delta_components(resolution_model)

    # error if both contain delta(s)
    if sample_deltas and resolution_deltas:
        raise ValueError(
            "Both sample_model and resolution_model contain delta functions. "
            "Their convolution is not defined."
        )

    # if sample has deltas, convolve each delta with the resolution_model
    for delta in sample_deltas:
        convolved += delta.area.value * resolution_model.evaluate(
            x - offset_float - delta.center.value
        )

    # if resolution has deltas, convolve each delta with the sample_model
    for delta in resolution_deltas:
        convolved += delta.area.value * sample_model.evaluate(
            x - offset_float - delta.center.value
        )

    return convolved


def _analytical_convolution(
    x: np.ndarray,
    sample_model: Union[SampleModel, ModelComponent],
    resolution_model: Union[SampleModel, ModelComponent],
    offset_float: float = 0.0,
    use_numerical_convolution_as_fallback: bool = False,
    upsample_factor: int = 5,
    extension_factor: float = 0.2,
) -> np.ndarray:
    """
    Convolve sample with resolution analytically if possible. Accepts SampleModel or single ModelComponent for each.
    Possible analytical convolutions are any combination of delta functions, Gaussians, and Lorentzians.
    Falls back to numerical convolution for other pairs of functions

    Most validation happens in the main `convolution` function.

    Args:
        x : np.ndarray
            1D array of x values where the convolution is evaluated.
        sample_model : SampleModel or ModelComponent
            The sample model to be convolved.
        resolution_model : SampleModel or ModelComponent
            The resolution model to convolve with.
        offset_float : float
            The offset to apply to the convolution.
        upsample_factor : int, optional
            The factor by which to upsample the input data before numerical convolution. Improves accuracy at the cost of speed. Default is 5
        extension_factor : float, optional
            The factor by which to extend the input data range before numerical convolution. Improves accuracy at the edges of the data. Default is 0.2
    Returns:
        np.ndarray
            The convolved values evaluated at x.

    Raises:
        ValueError
            If both sample_model and resolution_model contain delta functions.

    """

    # prepare list of components
    if isinstance(sample_model, SampleModel):
        sample_components = sample_model.components
    else:
        sample_components = [sample_model]

    if isinstance(resolution_model, SampleModel):
        resolution_components = resolution_model.components
    else:
        resolution_components = [resolution_model]

    total = np.zeros_like(x, dtype=float)

    # loop over sample components, making a list of components that cannot be handled analytically
    for s in sample_components:
        not_analytical_components = SampleModel(name="not_analytical")

        # Go through resolution components, adding analytical contributions where possible, making a list of those that cannot be handled analytically
        for r in resolution_components:
            handled, contrib = _try_analytic_pair(x, s, r, offset_float)
            if handled:
                total += contrib
            else:
                not_analytical_components.add_component(r)

        if not_analytical_components:
            if use_numerical_convolution_as_fallback:
                total += _numerical_convolution(
                    x=x,
                    sample_model=s,  # single component
                    resolution_model=not_analytical_components,
                    offset_float=offset_float,
                    upsample_factor=upsample_factor,
                    extension_factor=extension_factor,
                )
            else:
                raise ValueError(
                    f"Could not find analytical convolution for sample component '{s.name}' with resolution model '{not_analytical_components.name}'. "
                    "Set method to 'auto' or 'numerical'."
                )

    return total


# ---------------------- helpers & evals -----------------------
def _try_analytic_pair(
    x: np.ndarray,
    sample_component: ModelComponent,
    resolution_component: ModelComponent,
    off: float,
) -> Tuple[bool, np.ndarray]:
    """
    Attempt an analytic convolution for component pair (sample_component, resolution_component).
    Returns (True, contribution) if handled, else (False, zeros).
    The convolution of two gaussian components results in another gaussian component with width sqrt(w1^2 + w2^2).
    The convolution of two lorentzian components results in another lorentzian component with width w1 + w2.
    The convolution of a gaussian and a lorentzian results in a voigt profile.
    The convolution of a delta function with any component results in the same component shifted by the delta center.
    All areas are multiplied.

    Args:
        x : np.ndarray
            1D array of x values where the convolution is evaluated.
        sample_component : ModelComponent
            The sample component to be convolved.
        resolution_component : ModelComponent
            The resolution component to convolve with.
        off : float
            The offset to apply to the convolution.

    Returns:
        Tuple[bool, np.ndarray]:
            - bool: True if analytical convolution was computed, False otherwise
            - np.ndarray: The convolution result if computed, or zeros if not handled
    """
    # Delta functions
    if isinstance(sample_component, DeltaFunction) and isinstance(
        resolution_component, DeltaFunction
    ):
        raise ValueError("Convolution of two delta functions is not defined.")

    if isinstance(sample_component, DeltaFunction):
        return True, sample_component.area.value * resolution_component.evaluate(
            x - sample_component.center.value - off
        )

    if isinstance(resolution_component, DeltaFunction):
        return True, resolution_component.area.value * sample_component.evaluate(
            x - resolution_component.center.value - off
        )

    # Gaussian + Gaussian --> Gaussian with width sqrt(w1^2 + w2^2)
    if isinstance(sample_component, Gaussian) and isinstance(
        resolution_component, Gaussian
    ):
        width = np.sqrt(
            sample_component.width.value**2 + resolution_component.width.value**2
        )
        area = sample_component.area.value * resolution_component.area.value
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + off
        return True, _gaussian_eval(x, center, width, area)

    # Lorentzian + Lorentzian --> Lorentzian with width w1 + w2
    if isinstance(sample_component, Lorentzian) and isinstance(
        resolution_component, Lorentzian
    ):
        width = sample_component.width.value + resolution_component.width.value
        area = sample_component.area.value * resolution_component.area.value
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + off
        return True, _lorentzian_eval(x, center, width, area)

    # Gaussian + Lorentzian --> Voigt
    if (
        isinstance(sample_component, Gaussian)
        and isinstance(resolution_component, Lorentzian)
    ) or (
        isinstance(sample_component, Lorentzian)
        and isinstance(resolution_component, Gaussian)
    ):
        if isinstance(sample_component, Gaussian):
            gaussian, lorentzian = sample_component, resolution_component
        else:
            gaussian, lorentzian = resolution_component, sample_component
        center = (gaussian.center.value + lorentzian.center.value) + off
        area = gaussian.area.value * lorentzian.area.value
        return True, _voigt_eval(
            x, center, gaussian.width.value, lorentzian.width.value, area
        )

    return False, np.zeros_like(x, dtype=float)


def _gaussian_eval(
    x: np.ndarray, center: float, width: float, area: float
) -> np.ndarray:
    """
    Evaluate a Gaussian function. y = (area / (sqrt(2pi) * width)) * exp(-0.5 * ((x - center) / width)^2)
    All checks are handled in the calling function.

    Args:
        x : np.ndarray
            1D array of x values where the Gaussian is evaluated.
        center : float
            The center of the Gaussian.
        width : float
            The width (sigma) of the Gaussian.
        area : float
            The area under the Gaussian curve.
    Returns:
        np.ndarray
            The evaluated Gaussian values at x.
    """
    return (
        area
        * 1
        / (np.sqrt(2 * np.pi) * width)
        * np.exp(-0.5 * ((x - center) / width) ** 2)
    )


def _lorentzian_eval(
    x: np.ndarray, center: float, width: float, area: float
) -> np.ndarray:
    """
    Evaluate a Lorentzian function. y = (area * width / pi) / ((x - center)^2 + width^2).
    All checks are handled in the calling function.

    Args:
        x : np.ndarray
            1D array of x values where the Lorentzian is evaluated.
        center : float
            The center of the Lorentzian.
        width : float
            The width (HWHM) of the Lorentzian.
        area : float
            The area under the Lorentzian.
    Returns:
        np.ndarray
            The evaluated Lorentzian values at x.
    """
    return area * width / np.pi / ((x - center) ** 2 + width**2)


def _voigt_eval(
    x: np.ndarray, center: float, g_width: float, l_width: float, area: float
) -> np.ndarray:
    """
    Evaluate a Voigt profile function using scipy's voigt_profile.
    Args:
        x : np.ndarray
            1D array of x values where the Voigt profile is evaluated.
        center : float
            The center of the Voigt profile.
        g_width : float
            The Gaussian width (sigma) of the Voigt profile.
        l_width : float
            The Lorentzian width (HWHM) of the Voigt profile.
        area : float
            The area under the Voigt profile.
    Returns:
        np.ndarray
            The evaluated Voigt profile values at x.
    """

    return area * voigt_profile(x - center, g_width, l_width)


def _check_width_thresholds(
    model: Union[SampleModel, ModelComponent], span: float, dx: float, model_type: str
) -> None:
    """
    Helper function to check and warn if components are wide compared to the span of the data, or narrow compared to the spacing.
    In both cases, the convolution accuracy may be compromised.
    Args:
        model : SampleModel or ModelComponent
            The model to check.
        dx : float
            The bin spacing of the input x array.
        span : float
            The total span of the input x array.
        model_type : str
            A string indicating whether the model is a 'sample model' or 'resolution model' for warning messages.
    returns:
        None
    warns:
        UserWarning
            If the component widths are not appropriate for the data span or bin spacing.

    """

    # The thresholds are illustrated in performance_tests/convolution/convolution_width_thresholds.ipynb
    LARGE_WIDTH_THRESHOLD = (
        0.1  # Threshold for large widths compared to span - warn if width > 10% of span
    )
    SMALL_WIDTH_THRESHOLD = (
        1.0  # Threshold for small widths compared to bin spacing - warn if width < dx
    )

    # Handle SampleModel or ModelComponent
    if isinstance(model, SampleModel):
        components = model.components
    else:
        components = [model]  # Treat single ModelComponent as a list

    for comp in components:
        if hasattr(comp, "width"):
            if comp.width.value > LARGE_WIDTH_THRESHOLD * span:
                warnings.warn(
                    f"The width of the {model_type} component '{comp.name}' ({comp.width.value}) is large compared to the span of the input "
                    f"array ({span}). This may lead to inaccuracies in the convolution. Increase extension_factor to improve accuracy.",
                    UserWarning,
                )
            if comp.width.value < SMALL_WIDTH_THRESHOLD * dx:
                warnings.warn(
                    f"The width of the {model_type} component '{comp.name}' ({comp.width.value}) is small compared to the spacing of the input "
                    f"array ({dx}). This may lead to inaccuracies in the convolution. Increase upsample_factor to improve accuracy.",
                    UserWarning,
                )


def _find_delta_components(
    model: Union[SampleModel, ModelComponent],
) -> List[DeltaFunction]:
    """Return a list of DeltaFunction instances contained in `model`.

    Args:
        model : SampleModel or ModelComponent
            The model to search for DeltaFunction components.
    Returns:
        List[DeltaFunction]
            A list of DeltaFunction components found in the model.
    """
    if isinstance(model, DeltaFunction):
        return [model]
    if isinstance(model, SampleModel):
        return [c for c in model.components if isinstance(c, DeltaFunction)]
    return []


def _create_dense_grid(
    x: np.ndarray, upsample_factor: int = 5, extension_factor: float = 0.2
) -> np.ndarray:
    """
    Create a dense grid by upsampling and extending the input x array.

    Args:
        x : np.ndarray
            1D array of x values.
        upsample_factor : int, optional
            The factor by which to upsample the input data. Default is 5.
        extension_factor : float, optional
            The factor by which to extend the input data range. Default is 0.2.
    Returns:
        np.ndarray
            The dense grid created by upsampling and extending x.
    """
    if upsample_factor == 0:
        # Check if the array is uniformly spaced.
        x_diff = np.diff(x)
        is_uniform = np.allclose(x_diff, x_diff[0])
        if not is_uniform:
            raise ValueError(
                "Input array `x` must be uniformly spaced if upsample_factor = 0."
            )
        x_dense = x
    else:
        # Create an extended and upsampled x grid
        x_min, x_max = x.min(), x.max()
        span = x_max - x_min
        extra = extension_factor * span
        extended_min = x_min - extra
        extended_max = x_max + extra
        num_points = len(x) * upsample_factor
        x_dense = np.linspace(extended_min, extended_max, num_points)

    return x_dense
