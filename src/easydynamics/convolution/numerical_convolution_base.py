import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.convolution_base import ConvolutionBase
from easydynamics.sample_model import (
    SampleModel,
)
from easydynamics.sample_model.components.model_component import ModelComponent

Numerical = Union[float, int]


class NumericalConvolutionBase(ConvolutionBase):
    """
    Base class for numerical convolutions of sample and resolution models.
    Provides methods to handle upsampling, extension, and detailed balance correction.
    This base class has no convolution functionality.

    Args:
    energy : np.ndarray or scipp.Variable
        1D array of energy values where the convolution is evaluated.
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
    energy_unit : str or sc.Unit, optional
        The unit of the energy. Default is 'meV'.
    normalize_detailed_balance : bool, optional
        Whether to normalize the detailed balance correction. Default is True.
    """

    def __init__(
        self,
        energy: Union[np.ndarray, sc.Variable],
        sample_model: Union[SampleModel, ModelComponent],
        resolution_model: Union[SampleModel, ModelComponent],
        offset: Optional[Union[Numerical, Parameter]] = 0.0,
        upsample_factor: Optional[Numerical] = 5,
        extension_factor: Optional[float] = 0.2,
        temperature: Optional[Union[Parameter, float]] = None,
        temperature_unit: Optional[Union[str, sc.Unit]] = "K",
        energy_unit: Optional[Union[str, sc.Unit]] = "meV",
        normalize_detailed_balance: Optional[bool] = True,
    ):
        super().__init__(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            energy_unit=energy_unit,
            offset=offset,
        )

        if temperature is not None:
            if isinstance(temperature, Numerical):
                temperature = Parameter(
                    name="temperature",
                    value=float(temperature),
                    unit=temperature_unit,
                    fixed=True,
                )
            elif not isinstance(temperature, Parameter):
                raise TypeError("Temperature must be a float or Parameter.")
        self._temperature = temperature
        self._temperature_unit = temperature_unit

        self._normalize_detailed_balance = normalize_detailed_balance

        self._upsample_factor = upsample_factor
        self._extension_factor = extension_factor

        # Create a dense grid to improve accuracy. When upsample_factor>1, we evaluate on this grid and interpolate back to the original values at the end
        self._energy_grid = self._create_energy_grid()

    @ConvolutionBase.energy.setter
    def energy(self, energy: np.ndarray) -> None:
        super().energy = energy
        # Recreate dense grid when energy is updated
        self._energy_grid = self._create_energy_grid()

    @property
    def upsample_factor(self) -> Numerical:
        """
        Get the upsample factor.
        """

        return self._upsample_factor

    @upsample_factor.setter
    def upsample_factor(self, factor: Numerical) -> None:
        """
        Set the upsample factor and recreate the dense grid."""
        if factor is None:
            self._upsample_factor = factor
            self._energy_grid = self._create_energy_grid()
            return

        if not isinstance(factor, Numerical):
            raise TypeError("Upsample factor must be a numerical value or None.")
        factor = float(factor)
        if factor < 1.0:
            raise ValueError("Upsample factor must be greater than 1.")

        self._upsample_factor = factor

        # Recreate dense grid when upsample factor is updated
        self._energy_grid = self._create_energy_grid()

    @property
    def extension_factor(self) -> float:
        """
        Get the extension factor.
        The extension factor determines how much the energy range is extended on both sides before convolution.
        0.2 means extending by 20% of the original energy span on each side
        """

        return self._extension_factor

    @extension_factor.setter
    def extension_factor(self, factor: Numerical) -> None:
        """
        Set the extension factor and recreate the dense grid.
        The extension factor determines how much the energy range is extended on both sides before convolution.
        0.2 means extending by 20% of the original energy span on each side.

        Args:
            factor : float
                The new extension factor.

        Raises:
            TypeError: If factor is not a number.
        """
        if not isinstance(factor, Numerical):
            raise TypeError("Extension factor must be a number.")
        if factor < 0.0:
            raise ValueError("Extension factor must be non-negative.")

        self._extension_factor = factor
        # Recreate dense grid when extension factor is updated
        self._energy_grid = self._create_energy_grid()

    @property
    def temperature(self) -> Optional[Parameter]:
        """
        Get the temperature.
        """

        return self._temperature

    @temperature.setter
    def temperature(self, temp: Optional[Union[Parameter, float]]) -> None:
        """
        Set the temperature. If None, disables detailed balance correction and removes the temperature parameter.
        Args:
            temp : Parameter, float, or None
                The temperature to set. The unit will be the same as the existing temperature parameter if it exists, otherwise 'K'.
        Raises:
            TypeError: If temp is not a float, Parameter, or None.
        """

        if temp is None:
            self._temperature = None
        elif isinstance(temp, Numerical):
            if self._temperature is not None:
                self._temperature.value = float(temp)
            else:
                self._temperature = Parameter(
                    name="temperature",
                    value=float(temp),
                    unit=self._temperature_unit,
                    fixed=True,
                )
        elif isinstance(temp, Parameter):
            self._temperature = temp
        else:
            raise TypeError("Temperature must be None, a float or a Parameter.")

    @property
    def normalize_detailed_balance(self) -> bool:
        """
        Get whether to normalize the detailed balance factor.
        """

        return self._normalize_detailed_balance

    @normalize_detailed_balance.setter
    def normalize_detailed_balance(self, normalize: bool) -> None:
        """
        Set whether to normalize the detailed balance factor.
        If True, the detailed balance factor is divided by temperature.
        Args:
            normalize : bool
                Whether to normalize the detailed balance factor.
        Raises:
            TypeError: If normalize is not a bool.
        """

        if not isinstance(normalize, bool):
            raise TypeError("normalize_detailed_balance must be True or False.")

        self._normalize_detailed_balance = normalize

    @dataclass(frozen=True)
    class EnergyGrid:
        """Container for the dense energy grid and related metadata.

        Attributes:
            energy_dense: the (possibly extended & upsampled) energy grid (1D).
            span_original: span of the original energy array (max-min).
            span_dense: span of the dense grid (max-min).
            energy_even_length_offset: -0.5*dE if length is even, else 0.0 — used to correct half-bin shift.
            energy_dense_centered: energy_dense recentered around zero (same length as energy_dense).
            energy_step: grid spacing (dE) of energy_dense (positive float).
        """

        energy_dense: np.ndarray
        span_original: float
        span_dense: float
        energy_even_length_offset: float
        energy_dense_centered: np.ndarray
        energy_step: float

    def _create_energy_grid(
        self,
    ) -> EnergyGrid:
        """
        Create a dense grid by upsampling and extending the input energy array.

        Returns:
            DenseGrid
                The dense grid created by upsampling and extending x.
        """
        if self.upsample_factor is None:
            # Check if the array is uniformly spaced.
            energy_diff = np.diff(self.energy.values)
            is_uniform = np.allclose(energy_diff, energy_diff[0])
            if not is_uniform:
                raise ValueError(
                    "Input array `energy` must be uniformly spaced if upsample_factor = 0."
                )
            energy_dense = self.energy.values

            span = self.energy.values.max() - self.energy.values.min()
        else:
            # Create an extended and upsampled energy grid
            energy_min, energy_max = self.energy.values.min(), self.energy.values.max()
            span = energy_max - energy_min
            extra = self.extension_factor * span
            extended_min = energy_min - extra
            extended_max = energy_max + extra
            num_points = round(len(self.energy.values) * self.upsample_factor)
            energy_dense = np.linspace(extended_min, extended_max, num_points)

        energy_step = energy_dense[1] - energy_dense[0]

        # Handle offset for even length of x in convolution.
        # The convolution of two arrays of length N is of length 2N-1. When using 'same' mode, only the central N points are kept,
        # so the output has the same length as the input.
        # However, if N is even, the center falls between two points, leading to a half-bin offset.
        # For example, if N=4, the convolution has length 7, and when we select the 4 central points we either get
        # indices [2,3,4,5] or [1,2,3,4], both of which are offset by 0.5*dx from the true center at index 3.5.
        if len(energy_dense) % 2 == 0:
            energy_even_length_offset = -0.5 * energy_step
        else:
            energy_even_length_offset = 0.0

        # Handle the case when x is not symmetric around zero. The resolution is still centered around zero (or close to it), so it needs to be evaluated there.
        if not np.isclose(energy_dense.mean(), 0.0):
            energy_dense_centered = np.linspace(
                -0.5 * span, 0.5 * span, len(energy_dense)
            )
        else:
            energy_dense_centered = energy_dense

        energy_grid = self.EnergyGrid(
            energy_dense=energy_dense,
            span_original=span,
            span_dense=span,
            energy_even_length_offset=energy_even_length_offset,
            energy_dense_centered=energy_dense_centered,
            energy_step=energy_step,
        )

        return energy_grid

    def _check_width_thresholds(
        self,
        model: Union[SampleModel, ModelComponent],
        model_name: str,
    ) -> None:
        """
        Helper function to check and warn if components are wide compared to the span of the data, or narrow compared to the spacing.
        In both cases, the convolution accuracy may be compromised.
        Args:
            model : SampleModel or ModelComponent
                The model to check.
            energy_step : float
                The bin spacing of the energy array.
            span : float
                The total span of the energy array.
            model_name : str
                A string indicating whether the model is a 'sample model' or 'resolution model' for warning messages.
        returns:
            None
        warns:
            UserWarning
                If the component widths are not appropriate for the data span or bin spacing.

        """

        # The thresholds are illustrated in performance_tests/convolution/convolution_width_thresholds.ipynb
        LARGE_WIDTH_THRESHOLD = 0.1  # Threshold for large widths compared to span - warn if width > 10% of span
        SMALL_WIDTH_THRESHOLD = 1.0  # Threshold for small widths compared to bin spacing - warn if width < dx

        # Handle SampleModel or ModelComponent
        if isinstance(model, SampleModel):
            components = model.components
        else:
            components = [model]  # Treat single ModelComponent as a list

        for comp in components:
            if hasattr(comp, "width"):
                if (
                    comp.width.value
                    > LARGE_WIDTH_THRESHOLD * self._energy_grid.span_dense
                ):
                    warnings.warn(
                        f"The width of the {model_name} component '{comp.name}' ({comp.width.value}) is large compared to the span of the input "
                        f"array ({self._energy_grid.span_dense}). This may lead to inaccuracies in the convolution. Increase extension_factor to improve accuracy.",
                        UserWarning,
                    )
                if (
                    comp.width.value
                    < SMALL_WIDTH_THRESHOLD * self._energy_grid.energy_step
                ):
                    warnings.warn(
                        f"The width of the {model_name} component '{comp.name}' ({comp.width.value}) is small compared to the spacing of the input "
                        f"array ({self._energy_grid.energy_step}). This may lead to inaccuracies in the convolution. Increase upsample_factor to improve accuracy.",
                        UserWarning,
                    )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"energy=array of shape {self.energy.values.shape}, "
            f"sample_model={self.sample_model}, "
            f"resolution_model={self.resolution_model}, "
            f"energy_unit={self._energy_unit}, "
            f"offset={self.offset}, "
            f"upsample_factor={self.upsample_factor}, "
            f"extension_factor={self.extension_factor}, "
            f"temperature={self.temperature}, "
            f"normalize_detailed_balance={self.normalize_detailed_balance})"
        )
