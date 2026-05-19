# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

if TYPE_CHECKING:
    import scipp as sc
    from easyscience.variable import Parameter


class Lorentzian(CreateParametersMixin, ModelComponent):
    r"""
    Model of a Lorentzian function.

    The intensity is given by $$ I(x) = \frac{A}{\pi} \frac{\Gamma}{(x - x_0)^2 + \Gamma^2}, $$
    where $A$ is the area, $x_0$ is the center, and $\Gamma$ is the half width at half maximum
    (HWHM).

    If the center is not provided, it will be centered at 0 and fixed, which is typically what you
    want in QENS.
    """

    def __init__(
        self,
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter | None = None,
        width: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = 'meV',
        name: str = 'Lorentzian',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Lorentzian component.

        Parameters
        ----------
        area : Numeric | Parameter, default=1.0
            Area of the Lorentzian.
        center : Numeric | Parameter | None, default=None
            Center of the Lorentzian. If None, defaults to 0 and is fixed.
        width : Numeric | Parameter, default=1.0
            Half width at half maximum (HWHM).
        unit : str | sc.Unit, default='meV'
            Unit of the parameters.
        name : str, default='Lorentzian'
            Name of the component for indexing.
        display_name : str | None, default=None
            Display name for the component.
        unique_name : str | None, default=None
            Unique name of the component. If None, a unique_name is automatically generated. By
            default, None.
        """

        super().__init__(
            unit=unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=name, unit=self._unit)
        center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, unit=self._unit
        )
        width = self._create_width_parameter(width=width, name=name, unit=self._unit)

        self._area = area
        self._center = center
        self._width = width

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns
        -------
        Parameter
            The area parameter.
        """
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Set the value of the area parameter.

        Parameters
        ----------
        value : Numeric
            The new value for the area parameter.

        Raises
        ------
        TypeError
            If the value is not a number.
        """
        if not isinstance(value, Numeric):
            raise TypeError('area must be a number')
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter.

        Returns
        -------
        Parameter
            The center parameter.
        """
        return self._center

    @center.setter
    def center(self, value: Numeric | None) -> None:
        """
        Set the value of the center parameter.

        Parameters
        ----------
        value : Numeric | None
            The new value for the center parameter. If None, defaults to 0 and is fixed.

        Raises
        ------
        TypeError
            If the value is not a number or None.
        """

        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    @property
    def width(self) -> Parameter:
        """
        Get the width parameter (HWHM).

        Returns
        -------
        Parameter
            The width parameter.
        """
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """
        Set the width parameter value (HWHM).

        Parameters
        ----------
        value : Numeric
            The new value for the width parameter.

        Raises
        ------
        TypeError
            If the value is not a number.
        ValueError
            If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('width must be a number')

        if float(value) <= 0:
            raise ValueError('width must be positive')
        self._width.value = value

    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        r"""
        Evaluate the Lorentzian at the given x values.

        If x is a scipp Variable, the unit of the Lorentzian will be converted to match x. The
        intensity is given by

        $$ I(x) = \frac{A}{\pi} \frac{\Gamma}{(x - x_0)^2 + \Gamma^2}, $$

        where $A$ is the area, $x_0$ is the center, and $\Gamma$ is the half width at half maximum
        (HWHM).

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the Lorentzian.

        Returns
        -------
        np.ndarray
            The intensity of the Lorentzian at the given x values.
        """

        x = self._prepare_x_for_evaluate(x)

        normalization = self.width.value / np.pi
        denominator = (x - self.center.value) ** 2 + self.width.value**2

        return self.area.value * normalization / denominator

    def __repr__(self) -> str:
        """
        Return a string representation of the Lorentzian.

        Returns
        -------
        str
            A string representation of the Lorentzian.
        """
        return (
            f'Lorentzian(name = {self.name}, display_name = {self.display_name}, '
            f'unit = {self._unit},\n'
            f'    area = {self.area},\n'
            f'    center = {self.center},\n'
            f'    width = {self.width})'
        )
