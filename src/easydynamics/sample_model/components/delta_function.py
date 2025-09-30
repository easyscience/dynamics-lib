from __future__ import annotations

from typing import Union, Optional

from easyscience.variable import Parameter

from .model_component import ModelComponent

import scipp as sc
from scipp import UnitError

import numpy as np

import warnings

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
        center: Union[None, Numeric, Parameter] = None,
        area: Numeric = 1.0,
        unit="meV",
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
        self._area.value = float(value)

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

    def evaluate(self, x):
        """ "Evaluate the Delta function at the given x values.
        The Delta function evaluates to zero everywhere, except in convolutions, where it acts as an identity. This is handled in the ResolutionHandler."""
        # TODO: Consider adding support for evaluation without resolution convolution

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self._unit is not None and x.unit != self._unit:
                try:
                    self.convert_unit(x.unit.name)
                except Exception as e:
                    raise UnitError(
                        f"Input x has unit {x.unit}, but DeltaFunction component has unit {self._unit}. Failed to convert DeltaFunction to {x.unit}."
                    ) from e

                warnings.warn(
                    f"Input x has unit {x.unit}, but DeltaFunction component has unit {self._unit}. Converting DeltaFunction to {x.unit}."
                )
        else:
            x_in = x

        if any(np.isnan(x_in)):
            raise ValueError("Input x contains NaN values.")

        if any(np.isinf(x_in)):
            raise ValueError("Input x contains infinite values.")

        return 0 * x_in

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self._area, self._center]

    def convert_unit(self, unit):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
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
