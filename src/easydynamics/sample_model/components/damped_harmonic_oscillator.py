from __future__ import annotations

from typing import Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin

from .model_component import ModelComponent

Numeric = Union[float, int]


class DampedHarmonicOscillator(CreateParametersMixin, ModelComponent):
    """
    Damped Harmonic Oscillator (DHO). 2*area*center^2*width/pi / ( (x^2 - center^2)^2 + (2*width*x)^2 )

    Args:
        name (str): Name of the component.
        center (Int or float): Resonance frequency, approximately the peak position.
        width (Int or float): Damping constant, approximately the half width at half max (HWHM) of the peaks.
        area (Int or float): Area under the curve.
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
    """

    def __init__(
        self,
        display_name: str = "DampedHarmonicOscillator",
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter = 1.0,
        width: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = "meV",
    ):
        # Validate inputs and create Parameters if not given
        self.validate_unit(unit)
        self._unit = unit

        # These methods live in ValidationMixin
        area = self._create_area_parameter(
            area=area, name=display_name, unit=self._unit
        )
        center = self._create_center_parameter(
            center=center, name=display_name, fix_if_none=False, unit=self._unit
        )
        center.min = 0.0  # Enforce center >= 0 for DHO

        width = self._create_width_parameter(
            width=width, name=display_name, unit=self._unit
        )
        super().__init__(
            display_name=display_name,
            unit=unit,
        )

        self._area = area
        self._center = center
        self._width = width

    @property
    def area(self) -> Parameter:
        """Get the area parameter."""
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """Set the area parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError("area must be a number")
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """Get the center parameter."""
        return self._center

    @center.setter
    def center(self, value: Numeric) -> None:
        """Set the center parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number")
        self._center.value = value

    @property
    def width(self) -> Parameter:
        """Get the width parameter."""
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """Set the width parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError("width must be a number")
        self._width.value = value

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Damped Harmonic Oscillator at the given x values.
        If x is a scipp Variable, the unit of the DHO will be converted to
        match x. The DHO evaluates to 2*area*center^2*width/pi / ( (x^2 - center^2)^2 + (2*width*x)^2 )"""

        x = self._prepare_x_for_evaluate(x)

        normalization = 2 * self.center.value**2 * self.width.value / np.pi
        # No division by zero here, width>0 enforced in setter
        denominator = (x**2 - self.center.value**2) ** 2 + (
            2 * self.width.value * x
        ) ** 2

        return self.area.value * normalization / (denominator)

    def __repr__(self):
        return f"DampedHarmonicOscillator(display_name = {self.display_name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
