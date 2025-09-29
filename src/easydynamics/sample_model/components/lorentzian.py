from __future__ import annotations

from typing import Union

import numpy as np

from easyscience.variable import Parameter

from .model_component import ModelComponent

import scipp as sc

import warnings

Numeric = Union[float, int]


class Lorentzian(ModelComponent):
    """
    Lorentzian function. Creates new EasyScience Parameters if floats are provided, otherwise uses the provided Parameters.

    Args:
        area (Numeric or Parameter): Area of the Lorentzian.
        center (Numeric or Parameter or None): Peak center. If None, defaults to 0 and is fixed.
        width (Numeric or Parameter): Half Width at Half Maximum (HWHM)
    """

    def __init__(
        self,
        name: str = "Lorentzian",
        area: Union[Numeric, Parameter] = 1.0,
        center: Union[Numeric, Parameter, None] = None,
        width: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if isinstance(width, Numeric):
            if width <= 0:
                raise ValueError("The width of a Lorentzian must be greater than zero.")
            width = float(width)
        elif isinstance(width, Parameter):
            if width.value <= 0:
                raise ValueError("The width of a Lorentzian must be greater than zero.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )
            area = float(area)
        elif isinstance(area, Parameter):
            if area.value < 0:
                warnings.warn(
                    "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )

        if isinstance(center, Numeric):
            center = float(center)

        super().__init__(name=name)
        self._unit = unit  # Set the unit for the component

        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        elif isinstance(center, Numeric):
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
            raise ValueError("The width of a Lorentzian must be greater than zero.")
        if self.area.value < 0:
            warnings.warn(
                "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
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
                        f"Input x has unit {x.unit}, but Lorentzian component has unit {self._unit}. Failed to convert Lorentzian to {x.unit}."
                    ) from e
                warnings.warn(
                    f"Input x has unit {x.unit}, but Lorentzian component has unit {self._unit}. Converting Lorentzian to {x.unit}."
                )
        else:
            x_in = x
        return self.area.value * (
            self.width.value
            / np.pi
            / ((x_in - self.center.value) ** 2 + self.width.value**2)
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
        self._unit = unit

    def copy(self) -> Lorentzian:
        model_copy = Lorentzian(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self._unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.width.fixed = self.width.fixed
        return model_copy

    def __repr__(self):
        return f"Lorentzian(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
