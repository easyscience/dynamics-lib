from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from .model_component import ModelComponent

Numeric = Union[float, int]

MINIMUMwidth = 1e-10  # To avoid division by zero


class DampedHarmonicOscillator(ModelComponent):
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
        # this method lives in ModelComponent since it's the same for all components
        self.validate_unit(unit)

        # Area
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")
        if isinstance(area, Numeric):
            area = Parameter(name=name + " area", value=float(area), unit=unit)

        if area.value < 0:
            warnings.warn(
                "The area of the Damped Harmonic Oscillator with name {} is negative, which may not be physically meaningful.".format(
                    name
                )
            )
        else:
            area.min = 0.0

        # Center
        if not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be a number, or a Parameter.")

        if isinstance(center, Numeric):
            center = Parameter(name=name + " center", value=float(center), unit=unit)

        # Width
        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if isinstance(width, Numeric):
            if float(width) < MINIMUMwidth:
                raise ValueError(
                    "The width of a Damped Harmonic Oscillator must be greater than zero."
                )
            width = Parameter(
                name=name + " width", value=float(width), unit=unit, min=MINIMUMwidth
            )
        else:
            if width.value <= 0:
                raise ValueError(
                    "The width of a Damped Harmonic Oscillator must be greater than zero."
                )
            width.min = MINIMUMwidth

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

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str or sc.Unit): The new unit to convert to.
        """
        old_unit = self._unit
        try:
            self.area.convert_unit(unit)
            self.center.convert_unit(unit)
            self.width.convert_unit(unit)
            self._unit = unit
        except Exception as e:
            # Attempt to rollback on failure
            try:
                if hasattr(self.area, "convert_unit"):
                    self.area.convert_unit(old_unit)
                if hasattr(self.center, "convert_unit"):
                    self.center.convert_unit(old_unit)
                if hasattr(self.width, "convert_unit"):
                    self.width.convert_unit(old_unit)
            except Exception:
                pass  # Best effort rollback
            raise e

    def __repr__(self):
        return f"DampedHarmonicOscillator(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
