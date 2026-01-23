# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.utils.utils import Numeric

from .model_component import ModelComponent


class Lorentzian(CreateParametersMixin, ModelComponent):
    """
    Lorentzian function:
    area*width / (pi * ( (x - center)^2 + width^2 ) )
    If the center is not provided, it will be centered at 0 and fixed,
    which is typically what you want in QENS.

    Args:
        area (Int, float or Parameter): Area of the Lorentzian.
        center (Int, float, None or Parameter): Peak center.
        If None, defaults to 0 and is fixed.
        width (Int, float or Parameter):
        Half Width at Half Maximum (HWHM)
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV"
        display_name (str): Display name of the component.
        unique_name (str or None): Unique name of the component.
        If None, a unique_name is automatically generated.
    """

    def __init__(
        self,
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter | None = None,
        width: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = 'meV',
        display_name: str | None = 'Lorentzian',
        unique_name: str | None = None,
    ):
        super().__init__(
            display_name=display_name,
            unit=unit,
            unique_name=unique_name,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=display_name, unit=self._unit)
        center = self._create_center_parameter(
            center=center, name=display_name, fix_if_none=True, unit=self._unit
        )
        width = self._create_width_parameter(width=width, name=display_name, unit=self._unit)

        self._area = area
        self._center = center
        self._width = width

    @property
    def area(self) -> Parameter:
        """Get the area parameter."""
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """Set the area parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError('area must be a number')
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """Get the center parameter."""
        return self._center

    @center.setter
    def center(self, value: Numeric | None) -> None:
        """Set the center parameter value."""
        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    @property
    def width(self) -> Parameter:
        """Get the width parameter."""
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """Set the width parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError('width must be a number')
        self._width.value = value

    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        """Evaluate the Lorentzian at the given x values.

        If x is a scipp Variable, the unit of the Lorentzian will be
        converted to match x.
        The Lorentzian evaluates to
        area*width / (pi * ( (x - center)^2 + width^2 ) )
        """

        x = self._prepare_x_for_evaluate(x)

        normalization = self.width.value / np.pi
        denominator = (x - self.center.value) ** 2 + self.width.value**2

        return self.area.value * normalization / denominator

    def __repr__(self):
        return f'Lorentzian(unique_name = {self.unique_name}, unit = {self._unit},\n \
            area = {self.area},\n center = {self.center},\n width = {self.width})'
