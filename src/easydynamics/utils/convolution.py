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
    x: np.ndarray,
    sample_model: Union[SampleModel, ModelComponent],
    resolution_model: Union[SampleModel, ModelComponent],
    offset: Optional[Union[Parameter, float, None]] = None,
    method: Optional[str] = "analytical",
    upsample_factor: Optional[int] = 0,
    extension_factor: Optional[float] = 0.2,
    temperature: Optional[Union[Parameter, float, None]] = None,
    temperature_unit: Union[str, sc.Unit] = "K",
    x_unit: Optional[Union[str, sc.Unit]] = "meV",
    normalize_detailed_balance: Optional[bool] = True,
) -> np.ndarray:
    """
    Calculate the convolution of a sample model with a resolution model using analytical expressions or numerical FFT.
    Accepts SampleModel or ModelComponent for both sample and resolution.
    The analytical method silently falls back to numerical convolution if no analytical expression is found.
    Detailed balancing is included if temperature is provided. This requires numerical convolution.

    Args:
        x : np.ndarray
            1D array of x values where the convolution is evaluated.
        sample_model : SampleModel or ModelComponent
            The sample model to be convolved.
        resolution_model : SampleModel or ModelComponent
            The resolution model to convolve with.
        offset : Parameter, float, or None, optional
            The offset to apply to the x values before convolution.
        method : str, optional
            The convolution method to use: 'analytical' or 'numerical'. Default is 'analytical'.
        upsample_factor : int, optional
            The factor by which to upsample the input data before numerical convolution. Default is 0 (no upsampling).
        extension_factor : float, optional
            The factor by which to extend the input data range before numerical convolution. Default is 0.2.
        temperature : Parameter, float, or None, optional
            The temperature to use for detailed balance calculations. Default is None.
        temperature_unit : str or sc.Unit, optional
            The unit of the temperature parameter. Default is 'K'.
        x_unit : str or sc.Unit, optional
            The unit of the x parameter. Default is 'meV'.
        normalize_detailed_balance : bool, optional
            Whether to normalize the detailed balance factor. Default is True.
    """

    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError(
            f"`x` is an instance of {type(x).__name__}, but must be a numpy array."
        )

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
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

    if offset is None:
        off = 0.0
    elif isinstance(offset, Parameter):
        off = offset.value
    elif isinstance(offset, Numerical):
        off = float(offset)
    else:
        raise TypeError(
            f"Expected offset to be Parameter, number, or None, got {type(offset)}"
        )

    if method == "analytical":
        if temperature is not None:
            raise ValueError(
                "Analytical convolution is not supported with detailed balance. Set method to 'numerical' instead or set the temperature to None."
            )
        return _analytical_convolution(
            x=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset=off,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
        )
    elif method == "numerical":
        return _numerical_convolution(
            x=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset=off,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
            temperature=temperature,
            temperature_unit=temperature_unit,
            x_unit=x_unit,
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
    offset: Optional[Union[Parameter, np.ndarray]] = None,
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
        offset : Parameter, float, or None, optional
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

    dx = x_dense[1] - x_dense[0]
    span = x_dense.max() - x_dense.min()
    # Handle offset for even length of x in convolution
    if len(x_dense) % 2 == 0:
        x_even_length_offset = -0.5 * dx
    else:
        x_even_length_offset = 0.0

    # Handle the case when x is not symmetric around zero. The resolution is still centered around zero (or close to it), so it needs to be evaluated there.
    if not np.isclose(x_dense.mean(), 0.0):
        x_dense_resolution = np.linspace(-0.5 * span, 0.5 * span, len(x_dense))
    else:
        x_dense_resolution = x_dense

    # Give warnings if peaks are very wide or very narrow
    _check_width_thresholds(sample_model, span, dx, "sample model")
    _check_width_thresholds(resolution_model, span, dx, "resolution model")

    # Evaluate on dense grid and interpolate at the end
    if isinstance(sample_model, SampleModel):
        sample_vals = sample_model.evaluate_without_delta(
            x_dense - offset - x_even_length_offset
        )
    elif isinstance(sample_model, DeltaFunction):
        sample_vals = np.zeros_like(x_dense)
    else:
        sample_vals = sample_model.evaluate(x_dense - offset - x_even_length_offset)

    # Detailed balance correction
    if temperature is not None:
        if isinstance(temperature, Parameter):
            T = temperature.value
            temperature_unit = temperature.unit
        elif isinstance(temperature, float):
            T = temperature
        else:
            raise TypeError(
                f"Expected temperature to be Parameter, float, or None, got {type(temperature)}"
            )

        if x_unit is None:
            raise ValueError("x_unit must be provided when temperature is specified.")
        if not isinstance(x_unit, (str, sc.Unit)):
            raise TypeError(f"Expected x_unit to be str or sc.Unit, got {type(x_unit)}")

        detailed_balance_factor_correction = detailed_balance_factor(
            energy=x_dense,
            temperature=T,
            energy_unit=x_unit,
            temperature_unit=temperature_unit,
            divide_by_temperature=normalize_detailed_balance,
        )
        sample_vals *= detailed_balance_factor_correction

    # Delta functions are handled separately for accuracy
    if isinstance(resolution_model, SampleModel):
        resolution_vals = resolution_model.evaluate_without_delta(x_dense_resolution)
    elif isinstance(resolution_model, DeltaFunction):
        resolution_vals = np.zeros_like(x_dense_resolution)
    else:
        resolution_vals = resolution_model.evaluate(x_dense_resolution)

    # Convolution
    convolved = fftconvolve(sample_vals, resolution_vals, mode="same")
    convolved *= dx  # normalize

    if upsample_factor > 0:
        # interpolate back to original x grid
        convolved = np.interp(x, x_dense, convolved, left=0.0, right=0.0)

    # Add delta contributions on original grid
    # collect deltas
    sample_deltas = _delta_components(sample_model)
    resolution_deltas = _delta_components(resolution_model)

    # error if both contain delta(s)
    if sample_deltas and resolution_deltas:
        raise ValueError(
            "Both sample_model and resolution_model contain delta functions. "
            "Their convolution is not defined."
        )

    # if sample has deltas, convolve each delta with the resolution_model
    for delta in sample_deltas:
        convolved += delta.area.value * resolution_model.evaluate(
            x - offset - delta.center.value
        )

    # if resolution has deltas, convolve each delta with the sample_model
    for delta in resolution_deltas:
        convolved += delta.area.value * sample_model.evaluate(
            x - offset - delta.center.value
        )

    return convolved


def _analytical_convolution(
    x: np.ndarray,
    sample_model: Union[SampleModel, ModelComponent],
    resolution_model: Union[SampleModel, ModelComponent],
    offset: Union[Parameter, float, None] = None,
    upsample_factor: int = 5,
    extension_factor: float = 0.2,
) -> np.ndarray:
    """
    Convolve sample with resolution. Accepts SampleModel or single ModelComponent for each.
    - Uses analytic registry for supported pairs.
    - For non-analytic pairs, falls back to a single FFT per sample component
        against the sum of its leftover resolution components using numerical_convolve
        (passing a callable for the summed resolution).
    - Handles delta functions analytically.
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
            handled, contrib = _try_analytic_pair(x, s, r, offset)
            if handled:
                total += contrib
            else:
                not_analytical_components.add_component(r)

        if not_analytical_components:
            total += _numerical_convolution(
                x=x,
                sample_model=s,  # single component
                resolution_model=not_analytical_components,  # SampleModel with components that cannot be handled analytically
                offset=offset,
                upsample_factor=upsample_factor,
                extension_factor=extension_factor,
            )

    return total


def _try_analytic_pair(
    x: np.ndarray,
    sample_component: ModelComponent,
    resolution_component: ModelComponent,
    off: float,
) -> Tuple[bool, np.ndarray]:
    """
    Attempt an analytic convolution for component pair (sample_component, resolution_component).
    Returns (True, contribution) if handled, else (False, zeros).
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

    # Gaussian + Gaussian --> Gaussian
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
        return True, gaussian_eval(x, center, width, area)

    # Lorentzian + Lorentzian --> Lorentzian
    if isinstance(sample_component, Lorentzian) and isinstance(
        resolution_component, Lorentzian
    ):
        width = sample_component.width.value + resolution_component.width.value
        area = sample_component.area.value * resolution_component.area.value
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + off
        return True, lorentzian_eval(x, center, width, area)

    # Gaussian + Lorentzian --> Voigt
    if (
        isinstance(sample_component, Gaussian)
        and isinstance(resolution_component, Lorentzian)
    ) or (
        isinstance(sample_component, Lorentzian)
        and isinstance(resolution_component, Gaussian)
    ):
        if isinstance(sample_component, Gaussian):
            G, L = sample_component, resolution_component
        else:
            G, L = resolution_component, sample_component
        center = (G.center.value + L.center.value) + off
        area = G.area.value * L.area.value
        return True, voigt_eval(x, center, G.width.value, L.width.value, area)

    return False, np.zeros_like(x, dtype=float)


# ---------------------- helpers & evals -----------------------


@staticmethod
def gaussian_eval(x, center, width, area):
    return (
        area
        * 1
        / (np.sqrt(2 * np.pi) * width)
        * np.exp(-0.5 * ((x - center) / width) ** 2)
    )


@staticmethod
def lorentzian_eval(x, center, width, area):
    return area * width / np.pi / ((x - center) ** 2 + width**2)


@staticmethod
def voigt_eval(x, center, g_width, l_width, area):
    return area * voigt_profile(x - center, g_width, l_width)


@staticmethod
def _check_width_thresholds(model, span, dx, model_type):
    """
    Helper function to check and warn about width thresholds for a given model or component.
    Parameters:
    - model: ModelComponent or SampleModel
    - span: Range of the input data
    - dx: Bin spacing of the input data
    - model_type: 'sample model' or 'resolution model' for proper warning messages
    """
    LARGE_WIDTH_THRESHOLD = 0.1  # Threshold for large widths compared to span
    SMALL_WIDTH_THRESHOLD = 0.5  # Threshold for small widths compared to bin spacing

    # Handle SampleModel or ModelComponent
    if isinstance(model, SampleModel):
        components = model.components
    else:
        components = [model]  # Treat single ModelComponent as a list of one

    for comp in components:
        if hasattr(comp, "width"):
            if comp.width.value > LARGE_WIDTH_THRESHOLD * span:
                warnings.warn(
                    f"The width of the {model_type} component '{comp.name}' ({comp.width.value}) is large compared to the span of the input "
                    f"array ({span}). This may lead to inaccuracies in the convolution.",
                    UserWarning,
                )
            if comp.width.value < SMALL_WIDTH_THRESHOLD * dx:
                warnings.warn(
                    f"The width of the {model_type} component '{comp.name}' ({comp.width.value}) is small compared to the spacing of the input "
                    f"array ({dx}). This may lead to inaccuracies in the convolution.",
                    UserWarning,
                )


def _delta_components(model: Union[SampleModel, ModelComponent]) -> List[DeltaFunction]:
    """Return a list of DeltaFunction instances contained in `model`."""
    if isinstance(model, DeltaFunction):
        return [model]
    if isinstance(model, SampleModel):
        return [c for c in model.components if isinstance(c, DeltaFunction)]
    return []
