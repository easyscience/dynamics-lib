from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin

from .model_component import ModelComponent

Numeric = Union[float, int]

EPSILON = 1e-8  # small number to avoid floating point issues


class DeltaFunction(CreateParametersMixin, ModelComponent):
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
        self.validate_unit(unit)
        self._unit = unit

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=name, unit=self._unit)
        center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, unit=self._unit
        )

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

    def __repr__(self):
        return f"DeltaFunction(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center}"
