import warnings
from typing import Optional, Tuple, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.special import voigt_profile

from easydynamics.sample_model import DeltaFunction, Gaussian, Lorentzian, SampleModel
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.detailed_balance import (
    _detailed_balance_factor as detailed_balance_factor,
)


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
    """
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

    if method == "analytical":
        if isinstance(sample_model, SampleModel) and temperature is not None:
            raise ValueError(
                "Analytical convolution is not supported with detailed balance. Set method to 'numerical' instead or set the temperature to None."
            )
        return _analytical_convolution(
            x=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset=offset,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
        )

    if method == "numerical":
        return _numerical_convolution(
            x=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset=offset,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
            temperature=temperature,
            temperature_unit=temperature_unit,
            x_unit=x_unit,
            normalize_detailed_balance=normalize_detailed_balance,
        )

    if method not in ["analytical", "numerical"]:
        raise ValueError(
            f"Unknown convolution method: {method}. Choose from 'analytical', or 'numerical'."
        )


def _numerical_convolution(
    x: np.ndarray,
    sample_model: Union[SampleModel, ModelComponent, np.ndarray],
    resolution_model: Union[SampleModel, ModelComponent, np.ndarray],
    offset: Union[Parameter, np.ndarray, None] = None,
    upsample_factor: int = 5,
    extension_factor: float = 0.2,
    temperature: Union[Parameter, float, None] = None,
    temperature_unit: Union[str, sc.Unit] = "K",
    x_unit: Optional[Union[str, sc.Unit]] = "meV",
    normalize_detailed_balance: bool = True,
) -> np.ndarray:
    """
    Numerical convolution using FFT with optional upsampling + extended range.

    sample_model / resolution_model may be:
        - SampleModel
        - ModelComponent
        - Callable: f(x: np.ndarray) -> np.ndarray
    offset: Union[Parameter, np.ndarray, None]: The offset on the x axis
    upsample_factor: int: The factor by which to upsample the input array to improve resolution
    extension_factor: float: The factor by which to extend the range of the input array to improve accuracy at the edges
    selected_component_name: Union[str, None]: If provided, the name of the component to be selected for evaluation
    """

    def is_uniform(xarr, rtol=1e-5) -> bool:
        """Check if the array is uniformly spaced."""
        dx = np.diff(xarr)
        return np.allclose(dx, dx[0], rtol=rtol)

    # Build dense grid
    if upsample_factor == 0:
        if not is_uniform(x):
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
    if offset is None:
        off = 0.0
    elif isinstance(offset, Parameter):
        off = offset.value
    elif isinstance(offset, float):
        off = offset
    else:
        raise TypeError(
            f"Expected offset to be Parameter, float, or None, got {type(offset)}"
        )

    dx = x_dense[1] - x_dense[0]
    span = x_dense.max() - x_dense.min()
    # Handle offset for even length of x in convolution
    if len(x_dense) % 2 == 0:
        off2 = -0.5 * dx
    else:
        off2 = 0.0

    # Handle the case when x is not symmetric around zero. The resolution is still centered around zero (or close to it), so it needs to be evaluated there.
    if not np.isclose(x_dense.mean(), 0.0):
        x_dense_resolution = np.linspace(-0.5 * span, 0.5 * span, len(x_dense))
    else:
        x_dense_resolution = x_dense

    # Give warnings if peaks are very wide or very narrow
    _check_width_thresholds(sample_model, span, dx, "sample model")
    _check_width_thresholds(resolution_model, span, dx, "resolution model")

    # Evaluate on dense grid
    if isinstance(sample_model, SampleModel):
        sample_vals = sample_model.evaluate_without_delta(x_dense - off - off2)
    elif isinstance(sample_model, DeltaFunction):
        sample_vals = np.zeros_like(x_dense)
    else:
        sample_vals = sample_model.evaluate(x_dense - off - off2)

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

    if isinstance(resolution_model, SampleModel):
        resolution_vals = resolution_model.evaluate_without_delta(x_dense_resolution)
    elif isinstance(resolution_model, DeltaFunction):
        resolution_vals = np.zeros_like(x_dense_resolution)
    else:
        resolution_vals = resolution_model.evaluate(x_dense_resolution)

    # Convolution
    convolved = fftconvolve(sample_vals, resolution_vals, mode="same")
    convolved *= dx  # normalize

    # Add delta contributions
    if isinstance(sample_model, SampleModel):
        for comp in sample_model.components:
            if isinstance(comp, DeltaFunction):
                convolved += comp.area.value * resolution_model.evaluate(
                    x_dense - off - comp.center.value
                )
    elif isinstance(sample_model, DeltaFunction):
        convolved += sample_model.area.value * resolution_model.evaluate(
            x_dense - off - sample_model.center.value
        )

    if isinstance(resolution_model, SampleModel):
        for comp in resolution_model.components:
            if isinstance(comp, DeltaFunction):
                convolved += comp.area.value * sample_model.evaluate(
                    x_dense - off - comp.center.value
                )
    elif isinstance(resolution_model, DeltaFunction):
        convolved += resolution_model.area.value * sample_model.evaluate(
            x_dense - off - resolution_model.center.value
        )

    # TODO: if both resolution and sample are delta functions, we should let the user know that they are wrong.

    if upsample_factor > 0:
        # interpolate back to original x grid
        return interp1d(
            x_dense, convolved, kind="linear", bounds_error=False, fill_value=0.0
        )(x)
    else:
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

    if offset is None:
        off = 0.0
    elif isinstance(offset, Parameter):
        off = offset.value
    elif isinstance(offset, float):
        off = offset
    else:
        raise TypeError(
            f"Expected offset to be Parameter, float, or None, got {type(offset)}"
        )

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
            handled, contrib = _try_analytic_pair(x, s, r, off)
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
    x: np.ndarray, s: ModelComponent, r: ModelComponent, off: float
) -> Tuple[bool, np.ndarray]:
    """
    Attempt an analytic convolution for component pair (s, r).
    Returns (True, contribution) if handled, else (False, zeros).
    """
    # Delta functions
    if isinstance(s, DeltaFunction):
        return True, s.area.value * r.evaluate(x - s.center.value - off)

    if isinstance(r, DeltaFunction):
        return True, r.area.value * s.evaluate(x - r.center.value - off)

    # Gaussian + Gaussian --> Gaussian
    if isinstance(s, Gaussian) and isinstance(r, Gaussian):
        width = np.sqrt(s.width.value**2 + r.width.value**2)
        area = s.area.value * r.area.value
        center = (s.center.value + r.center.value) + off
        return True, gaussian_eval(x, center, width, area)

    # Lorentzian + Lorentzian --> Lorentzian
    if isinstance(s, Lorentzian) and isinstance(r, Lorentzian):
        width = s.width.value + r.width.value
        area = s.area.value * r.area.value
        center = (s.center.value + r.center.value) + off
        return True, lorentzian_eval(x, center, width, area)

    # Gaussian + Lorentzian --> Voigt
    if (isinstance(s, Gaussian) and isinstance(r, Lorentzian)) or (
        isinstance(s, Lorentzian) and isinstance(r, Gaussian)
    ):
        if isinstance(s, Gaussian):
            G, L = s, r
        else:
            G, L = r, s
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
