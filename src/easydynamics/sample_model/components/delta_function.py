from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from .model_component import ModelComponent

Numeric = Union[float, int]


class DeltaFunction(ModelComponent):
    """
    Delta function. Evaluates to zero everywhere, except in convolutions, where it acts as an identity. This is handled in the ResolutionHandler.

    Args:
        name (str): Name of the component.
        center (Int or float or None): Center of the delta function. If None, defaults to 0 and is fixed.
        area (Int or float): Total area under the curve.
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
    """

    def __init__(
        self,
        name: str = "DeltaFunction",
        center: Union[None, Numeric] = None,
        area: Numeric = 1.0,
        unit: Union[str, sc.Unit] = "meV",
    ):
        # Validate inputs
        if not isinstance(area, Numeric):
            raise TypeError("area must be a number.")

        if area < 0:
            warnings.warn(
                "The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(
                    name
                )
            )
        area = float(area)

        if center is not None and not isinstance(center, Numeric):
            raise TypeError("center must be None or a number.")

        if isinstance(center, Numeric):
            center = float(center)

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")

        super().__init__(name=name)
        self._unit = unit

        # Create Parameters from floats, or set Parameters if already provided
        self._area = Parameter(name=name + " area", value=area, unit=unit)
        if area > 0:
            self._area.min = 0.0

        if center is None:
            self._center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        else:
            self._center = Parameter(name=name + " center", value=center, unit=unit)

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
                "The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(
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
    def unit(self) -> Union[str, sc.Unit]:
        """Return the unit of the component."""
        return self._unit

    @unit.setter
    def unit(self, value: Union[str, sc.Unit]):
        """Set the unit of the component."""
        if not isinstance(value, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")
        self._unit = value

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Delta function at the given x values.
        The Delta function evaluates to zero everywhere, except at the center. Its numerical integral is equal to the area.
        It acts as an identity in convolutions."""
        # TODO: Consider adding support for evaluation without resolution convolution

        x = self._prepare_x_for_evaluate(x)
        model = np.zeros_like(x, dtype=float)

        if min(x) <= self._center.value <= max(x):
            # if center within x-range, delta is non-zero in this interval
            # otherwise do nothing
            idx = np.argmin(np.abs(x - self._center.value))
            if len(x) > 1:
                dx = (max(x) - min(x)) / (len(x) - 1)  # domain spacing
            else:
                dx = 1.0
            model[idx] = self._area.value / dx

        return model

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self._area, self._center]

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str or sc.Unit): The new unit to convert to.
        """
        self._area.convert_unit(unit)
        self._center.convert_unit(unit)
        self._unit = unit

    def copy(self, name: Optional[str] = None) -> DeltaFunction:
        """
        Return a deep copy of this component with independent parameters.
        """
        if name is None:
            name = "copy of " + self.name

        model_copy = DeltaFunction(
            name=name,
            area=self._area.value,
            center=self._center.value,
            unit=self._unit,
        )
        model_copy._area.fixed = self._area.fixed
        model_copy._center.fixed = self._center.fixed
        return model_copy

    def __repr__(self):
        return f"DeltaFunction(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center}"
