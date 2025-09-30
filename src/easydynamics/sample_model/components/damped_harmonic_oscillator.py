from __future__ import annotations

from typing import Union

import numpy as np

from easyscience.variable import Parameter

from .model_component import ModelComponent

import scipp as sc

import warnings

Numeric = Union[float, int]


class DampedHarmonicOscillator(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (Int or float): Resonance frequency, approximately the peak position.
        width (Int or float): Damping constant, approximately the half width at half max (HWHM) of the peaks.
        area (Int or float): Area under the curve.
    """

    def __init__(
        self,
        name: str = "DHO",
        center: Numeric = 1.0,
        width: Numeric = 1.0,
        area: Numeric = 1.0,
        unit: Union[str, sc.Unit] = "meV",
    ):
        # Validate inputs
        if not isinstance(area, Numeric):
            raise TypeError("area must be a number.")
        area = float(area)
        if area < 0:
            warnings.warn(
                "The area of the Damped Harmonic Oscillator with name {} is negative, which may not be physically meaningful.".format(
                    name
                )
            )

        if not isinstance(center, Numeric):
            raise TypeError("center must be a number.")

        if isinstance(center, Numeric):
            center = float(center)

        if not isinstance(width, Numeric):
            raise TypeError("width must be a number.")

        width = float(width)
        if width <= 0:
            raise ValueError(
                "The width of a DampedHarmonicOscillator must be greater than zero."
            )

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")

        super().__init__(name=name)
        self._unit = unit  # Set the unit for the component

        # Create Parameters from floats
        self._area = Parameter(name=name + " area", value=area, unit=unit)

        self._center = Parameter(name=name + " center", value=center, unit=unit)

        self._width = Parameter(name=name + " width", value=width, unit=unit, min=0.0)

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self._width.value <= 0:
            raise ValueError(
                "The width of a DampedHarmonicOscillator must be greater than zero."
            )
        if self._area.value < 0:
            warnings.warn(
                "The area of the DampedHarmonicOscillator with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self._unit is not None and x.unit != self._unit:
                try:
                    self.convert_unit(x.unit.name)
                except Exception as e:
                    raise ValueError(
                        f"Input x has unit {x.unit}, but DHO component has unit {self._unit}. Failed to convert DHO to {x.unit}."
                    ) from e
                warnings.warn(
                    f"Input x has unit {x.unit}, but DHO component has unit {self._unit}. Converting DHO to {x.unit}."
                )
        else:
            x_in = x
        return (
            2
            * self._area.value
            * self._center.value**2
            * self._width.value
            / np.pi
            / (
                (x_in**2 - self._center.value**2) ** 2
                + (2 * self._width.value * x_in) ** 2
            )
        )

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self._area, self._center, self._width]

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """

        self._area.convert_unit(unit)
        self._center.convert_unit(unit)
        self._width.convert_unit(unit)
        self._unit = unit

    def copy(self) -> DampedHarmonicOscillator:
        """
        Return a deep copy of this component with independent parameters.
        """

        model_copy = DampedHarmonicOscillator(
            name=self.name,
            area=self._area.value,
            center=self._center.value,
            width=self._width.value,
            unit=self._unit,
        )
        model_copy.area.fixed = self._area.fixed
        model_copy.center.fixed = self._center.fixed
        model_copy.width.fixed = self._width.fixed
        return model_copy

    def __repr__(self):
        return f"DampedHarmonicOscillator(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center},\n width = {self._width})"
