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
    Lorentzian function: area*width / (pi * ( (x - center)^2 + width^2 ) )

    Args:
        area (Int or float): Area of the Lorentzian.
        center (Int or float or None): Peak center. If None, defaults to 0 and is fixed.
        width (Int or float): Half Width at Half Maximum (HWHM)
    """

    def __init__(
        self,
        name: str = "Lorentzian",
        area: Numeric = 1.0,
        center: Union[Numeric, None] = None,
        width: Numeric = 1.0,
        unit: Union[str, sc.Unit] = "meV",
    ):
        # Validate inputs
        if not isinstance(area, Numeric):
            raise TypeError("area must be a number.")

        area = float(area)
        if area < 0:
            warnings.warn(
                "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
                    name
                )
            )

        if center is not None and not isinstance(center, Numeric):
            raise TypeError("center must be None, a number.")

        if isinstance(center, Numeric):
            center = float(center)

        if not isinstance(width, Numeric):
            raise TypeError("width must be a number.")

        width = float(width)
        if width <= 0:
            raise ValueError("The width of a Lorentzian must be greater than zero.")

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")

        super().__init__(name=name)
        self._unit = unit  # Set the unit for the component

        # Create Parameters from floats
        self._area = Parameter(name=name + " area", value=area, unit=unit)

        if center is None:
            self._center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        else:
            self._center = Parameter(name=name + " center", value=center, unit=unit)

        self._width = Parameter(name=name + " width", value=width, unit=unit, min=0.0)

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self._width.value <= 0:
            raise ValueError("The width of a Lorentzian must be greater than zero.")
        if self._area.value < 0:
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
        return self._area.value * (
            self._width.value
            / np.pi
            / ((x_in - self._center.value) ** 2 + self._width.value**2)
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

    def copy(self) -> Lorentzian:
        model_copy = Lorentzian(
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
        return f"Lorentzian(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center},\n width = {self._width})"
