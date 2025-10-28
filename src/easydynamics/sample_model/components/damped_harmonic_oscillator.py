from __future__ import annotations

from typing import Optional, Union

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
        name: Optional[str] = "DampedHarmonicOscillator",
        area: Optional[Union[Numeric, Parameter]] = 1.0,
        center: Optional[Union[Numeric, Parameter]] = 1.0,
        width: Optional[Union[Numeric, Parameter]] = 1.0,
        unit: Optional[Union[str, sc.Unit]] = "meV",
    ):
        # Validate inputs and create Parameters if not given
        self.validate_unit(unit)
        self._unit = unit

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=name, unit=self._unit)
        center = self._create_center_parameter(
            center=center, name=name, fix_if_none=False, unit=self._unit
        )
        width = self._create_width_parameter(width=width, name=name, unit=self._unit)

        super().__init__(
            name=name,
            unit=unit,
            area=area,
            center=center,
            width=width,
        )

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Damped Harmonic Oscillator at the given x values.
        If x is a scipp Variable, the unit of the DHO will be converted to
        match x. The DHO evaluates to 2*area*center^2*width/pi / ( (x^2 - center^2)^2 + (2*width*x)^2 )"""

        x = self._prepare_x_for_evaluate(x)

        normalization = 2 * self.center.value**2 * self.width.value / np.pi
        denominator = (x**2 - self.center.value**2) ** 2 + (
            2
            * self.width.value
            * x  # No division by zero here, width>0 enforced in setter
        ) ** 2

        return self.area.value * normalization / (denominator)

    def __repr__(self):
        return f"DampedHarmonicOscillator(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
