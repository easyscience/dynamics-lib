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
        name: Optional[str] = "DeltaFunction",
        center: Optional[Union[None, Numeric, Parameter]] = None,
        area: Optional[Union[Numeric, Parameter]] = 1.0,
        unit: Union[str, sc.Unit] = "meV",
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
                "The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(
                    name
                )
            )
        else:
            area.min = 0.0

        # Center
        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number, or a Parameter.")

        if center is None:
            center = Parameter(name=name + " center", value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Numeric):
            center = Parameter(name=name + " center", value=float(center), unit=unit)

        super().__init__(
            name=name,
            unit=unit,
            area=area,
            center=center,
        )

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Delta function at the given x values.
        The Delta function evaluates to zero everywhere, except at the center. Its numerical integral is equal to the area.
        It acts as an identity in convolutions."""

        # x assumed sorted, 1D numpy array
        x = self._prepare_x_for_evaluate(x)
        model = np.zeros_like(x, dtype=float)
        center = self.center.value
        area = self.area.value

        if x.min() - EPSILON <= center <= x.max() + EPSILON:
            # nearest index
            i = np.argmin(np.abs(x - center))

            # left half-width
            if i == 0:
                left = x[1] - x[0] if x.size > 1 else 0.5
            else:
                left = x[i] - x[i - 1]

            # right half-width
            if i == x.size - 1:
                right = x[-1] - x[-2] if x.size > 1 else 0.5
            else:
                right = x[i + 1] - x[i]

            # effective bin width: half left + half right
            bin_width = 0.5 * (left + right)

            model[i] = area / bin_width

        return model

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str or sc.Unit): The new unit to convert to.
        """
        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self._unit = unit

    def __repr__(self):
        return f"DeltaFunction(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center}"
