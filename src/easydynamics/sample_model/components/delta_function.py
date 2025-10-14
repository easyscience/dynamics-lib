from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from .model_component import ModelComponent

Numeric = Union[float, int]

EPSILON = 1e-8  # small number to avoid floating point issues


class DeltaFunction(ModelComponent):
    """
    Delta function. Evaluates to zero everywhere, except in convolutions, where it acts as an identity. This is handled in the ResolutionHandler.
    If the center is not provided, it will be centered at 0 and fixed, which is typically what you want in QENS.

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

        super().__init__(name=name, unit=unit)
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

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Delta function at the given x values.
        The Delta function evaluates to zero everywhere, except at the center. Its numerical integral is equal to the area.
        It acts as an identity in convolutions."""

        # x assumed sorted, 1D numpy array
        x = self._prepare_x_for_evaluate(x)
        model = np.zeros_like(x, dtype=float)
        center = self._center.value
        area = self._area.value

        if x.min() - EPSILON <= center <= x.max() + EPSILON:
            # nearest index
            i = np.argmin(np.abs(x - center))

            # left half-width
            if i == 0:
                left = x[1] - x[0] if x.size > 1 else 0.0
            else:
                left = x[i] - x[i - 1]

            # right half-width
            if i == x.size - 1:
                right = x[-1] - x[-2] if x.size > 1 else 0.0
            else:
                right = x[i + 1] - x[i]

            # effective bin width: half left + half right
            bin_width = 0.5 * (left + right)

            model[i] = area / bin_width

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
