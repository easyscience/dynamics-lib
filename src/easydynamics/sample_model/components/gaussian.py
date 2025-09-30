from __future__ import annotations

from typing import Union, List, Optional

import numpy as np

from easyscience.variable import Parameter

from .model_component import ModelComponent

import scipp as sc
from scipp import UnitError

import warnings

Numeric = Union[float, int]

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class Gaussian(ModelComponent):
    """
    Gaussian function: area/(width*sqrt(2pi)) * exp(-0.5*((x - center)/width)^2)

    Args:
        name (str): Name of the component.
        area (Int or float): Area of the Gaussian.
        center (Int or float or None): Center of the Gaussian. If None, defaults to 0 and is fixed
        width (Int or float): Standard deviation.
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
    """

    def __init__(
        self,
        name: str = "Gaussian",
        area: Numeric = 1.0,
        center: Union[Numeric, None] = None,
        width: Numeric = 1.0,
        unit: Union[str, sc.Unit] = "meV",
    ):
        # Validate inputs - throw errors before any Parameters are created
        if not isinstance(area, Numeric):
            raise TypeError("area must be a number.")

        area = float(area)
        if area < 0:
            warnings.warn(
                "The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(
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
            raise ValueError("The width of a Gaussian must be greater than zero.")

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
    def area(self, value: Numeric, unit: Optional[Union[str, sc.Unit]] = None):
        """Set the area parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("area must be a number.")
        self._area.value = float(value)
        if unit is not None:
            self._area._unit = unit

    @property
    def center(self) -> Parameter:
        """Return the center parameter."""
        return self._center

    @center.setter
    def center(self, value: Numeric, unit: Optional[Union[str, sc.Unit]] = None):
        """Set the center parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number.")
        self._center.value = float(value)
        if unit is not None:
            self._center._unit = unit

    @property
    def width(self) -> Parameter:
        """Return the width parameter."""
        return self._width

    @width.setter
    def width(self, value: Numeric, unit: Optional[Union[str, sc.Unit]] = None):
        """Set the width parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("width must be a number.")
        self._width.value = float(value)
        if unit is not None:
            self._width._unit = unit

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        """Evaluate the Gaussian at the given x values.
        If x is a scipp Variable, the unit of the Gaussian will be converted to match x.
        The Gaussian evaluates to area/(width*sqrt(2pi)) * exp(-0.5*((x - center)/width)^2)"""

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self._unit is not None and x.unit != self._unit:
                try:
                    self.convert_unit(x.unit.name)
                except Exception as e:
                    raise UnitError(
                        f"Input x has unit {x.unit}, but Gaussian component has unit {self._unit}. Failed to convert Gaussian to {x.unit}."
                    ) from e

                warnings.warn(
                    f"Input x has unit {x.unit}, but Gaussian component has unit {self._unit}. Converting Gaussian to {x.unit}."
                )
        else:
            x_in = x

        if any(np.isnan(x_in)):
            raise ValueError("Input x contains NaN values.")

        if any(np.isinf(x_in)):
            raise ValueError("Input x contains infinite values.")

        normalization = 1 / (np.sqrt(2 * np.pi) * self._width.value)
        exponent = -0.5 * ((x_in - self._center.value) / self._width.value) ** 2

        return self._area.value * normalization * np.exp(exponent)

    def get_parameters(self) -> List[Parameter]:
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

    def copy(self, name: Optional[str] = None) -> Gaussian:
        """
        Return a deep copy of this component with independent parameters.
        """
        if name is None:
            name = "copy of " + self.name

        model_copy = Gaussian(
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
        return f"Gaussian(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center},\n width = {self._width})"
