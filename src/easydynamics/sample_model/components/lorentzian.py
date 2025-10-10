from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from .model_component import ModelComponent

Numeric = Union[float, int]

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class Lorentzian(ModelComponent):
    """
    Lorentzian function: area*width / (pi * ( (x - center)^2 + width^2 ) )

    Args:
        name (str): Name of the component.
        area (Int or float): Area of the Lorentzian.
        center (Int or float or None): Peak center. If None, defaults to 0 and is fixed.
        width (Int or float): Half Width at Half Maximum (HWHM)
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
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
            raise TypeError("center must be None or a number.")

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
        if area > 0:
            self._area.min = 0.0

        if center is None:
            self._center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        else:
            self._center = Parameter(name=name + " center", value=center, unit=unit)

        self._width = Parameter(
            name=name + " width", value=width, unit=unit, min=MINIMUM_WIDTH
        )

    @property
    def area(self) -> Parameter:
        """Return the area parameter."""
        return self._area

    @area.setter
    def area(self, value: Numeric):
        """Set the area parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("area must be a number.")
        value = float(value)
        if value < 0:
            warnings.warn(
                "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """Return the center parameter."""
        return self._center

    @center.setter
    def center(self, value: Numeric):
        """Set the center parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number.")
        self._center.value = float(value)

    @property
    def width(self) -> Parameter:
        """Return the width parameter."""
        return self._width

    @width.setter
    def width(self, value: Numeric):
        """Set the width parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("width must be a number.")
        value = float(value)
        if value <= 0:
            raise ValueError("The width of a Lorentzian must be greater than zero.")
        self._width.value = value

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Lorentzian at the given x values.
        If x is a scipp Variable, the unit of the Lorentzian will be converted to match x.
        The Lorentzian evaluates to area*width / (pi * ( (x - center)^2 + width^2 ) )"""

        x = self._prepare_x_for_evaluate(x)

        normalization = self._width.value / np.pi
        denominator = (x - self._center.value) ** 2 + self._width.value**2

        return self._area.value * normalization / denominator

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self._area, self._center, self._width]

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str or sc.Unit): The new unit to convert to.
        """

        self._area.convert_unit(unit)
        self._center.convert_unit(unit)
        self._width.convert_unit(unit)
        self._unit = unit

    def copy(self, name: Optional[str] = None) -> Lorentzian:
        """ "
        Return a deep copy of this component with independent parameters.
        """
        if name is None:
            name = "copy of " + self.name

        model_copy = Lorentzian(
            name=name,
            area=self._area.value,
            center=self._center.value,
            width=self._width.value,
            unit=self._unit,
        )
        model_copy._area.fixed = self._area.fixed
        model_copy._center.fixed = self._center.fixed
        model_copy._width.fixed = self._width.fixed
        return model_copy

    def __repr__(self):
        return f"Lorentzian(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center},\n width = {self._width})"
