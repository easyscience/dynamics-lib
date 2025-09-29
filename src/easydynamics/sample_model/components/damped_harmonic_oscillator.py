from __future__ import annotations

from typing import Union

import numpy as np

from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent

import scipp as sc

import warnings

Numeric = Union[float, int]


class DampedHarmonicOscillator(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (Numeric or Parameter): Resonance frequency, approximately the peak position.
        width (Numeric or Parameter): Damping constant, approximately the half width at half max (HWHM) of the peaks.
        area (Numeric or Parameter): Area under the curve.
    """

    def __init__(
        self,
        name: str = "DHO",
        center: Union[Numeric, Parameter] = 1.0,
        width: Union[Numeric, Parameter] = 1.0,
        area: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be a number or a Parameter.")

        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(width, Numeric):
            width = float(width)
            if width <= 0:
                raise ValueError(
                    "The width of a DampedHarmonicOscillator must be greater than zero."
                )

        if isinstance(area, Numeric):
            area = float(area)
            if area < 0:
                warnings.warn(
                    "The area of the Damped Harmonic Oscillator with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )

        if isinstance(center, Numeric):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(width, Numeric):
            self.width = Parameter(
                name=name + " width", value=width, unit=unit, min=0.0
            )
        else:
            self.width = width

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.width.value <= 0:
            raise ValueError(
                "The width of a DampedHarmonicOscillator must be greater than zero."
            )
        if self.area.value < 0:
            warnings.warn(
                "The area of the DampedHarmonicOscillator with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(
                    f"Input x has unit {x.unit}, but DHO component has unit {self.unit}. Converting DHO to {x.unit}."
                )
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return (
            2
            * self.area.value
            * self.center.value**2
            * self.width.value
            / np.pi
            / (
                (x_in**2 - self.center.value**2) ** 2
                + (2 * self.width.value * x_in) ** 2
            )
        )

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center, self.width]

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """

        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.width.convert_unit(unit)
        self.unit = unit

    def copy(self) -> DampedHarmonicOscillator:
        """
        Return a deep copy of this component with independent parameters.
        """

        model_copy = DampedHarmonicOscillator(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.width.fixed = self.width.fixed
        return model_copy

    def __repr__(self):
        return f"DampedHarmonicOscillator(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
