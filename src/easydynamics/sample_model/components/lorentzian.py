# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipp as sc

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

if TYPE_CHECKING:
    from easyscience.variable import Parameter


class Lorentzian(CreateParametersMixin, ModelComponent):
    r"""
    Model of a Lorentzian function.

    $$ I(x) = \frac{A}{\pi} \frac{\Gamma}{(x - x_0)^2 + \Gamma^2} $$

    where $A$ is the area, $x_0$ is the center, and $\Gamma$ is the HWHM. area has unit = x_unit *
    y_unit; center and width have unit = x_unit.

    If the center is not provided, it will be centered at 0 and fixed, which is typically what you
    want in QENS.

    Examples
    --------
    **Creating a Lorentzian with a fixed center (typical QENS use)**

    By default the center is fixed at 0, which is the typical setup for a QENS quasi-elastic line:
    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    l = sm.Lorentzian(area=1.0, width=0.3)
    x = np.linspace(-2, 2, 100)
    values = l.evaluate(x)
    ```

    **Creating a Lorentzian with a free center and modifying parameters**

    Pass a numeric value for ``center`` to leave it free during fitting:
    ```python
    import easydynamics.sample_model as sm

    l = sm.Lorentzian(area=2.0, center=0.5, width=0.3, name='QE peak')
    l.area = 3.0
    l.width = 0.2
    ```
    """

    def __init__(
        self,
        area: Numeric = 1.0,
        center: Numeric | None = None,
        width: Numeric = 1.0,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'Lorentzian',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Lorentzian component.

        Parameters
        ----------
        area : Numeric, default=1.0
            Integrated area under the Lorentzian.  Unit is ``x_unit * y_unit``.
        center : Numeric | None, default=None
            Peak position in x_unit.  If None, defaults to 0 and the center parameter is fixed.
        width : Numeric, default=1.0
            Half-width at half-maximum (HWHM, gamma) in x_unit.  Must be strictly positive.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  center and width are stored in this unit. area_unit = x_unit *
            y_unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='Lorentzian'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier.  Auto-generated if None.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        self._area = self._create_area_parameter(
            area=area, name=name, x_unit=self._x_unit, y_unit=self._y_unit
        )
        self._center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, x_unit=self._x_unit
        )
        self._width = self._create_width_parameter(width=width, name=name, x_unit=self._x_unit)

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns
        -------
        Parameter
            The area Parameter with unit ``x_unit * y_unit``.
        """
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New area value (in current area unit = x_unit * y_unit).

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
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
            The center Parameter with unit ``x_unit``.
        """
        return self._center

    @center.setter
    def center(self, value: Numeric | None) -> None:
        """
        Parameters
        ----------
        value : Numeric | None
            New center value in x_unit.  If None, the center is set to 0 and the parameter is
            fixed.

        Raises
        ------
        TypeError
            If *value* is not None and not a numeric type.
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
            The HWHM (gamma) Parameter with unit ``x_unit``.
        """
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New HWHM value in x_unit.  Must be strictly positive.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('width must be a number')
        if float(value) <= 0:
            raise ValueError('width must be positive')
        self._width.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        r"""
        Evaluate the Lorentzian at x.

        Parameters in the model's own units are temporarily converted to x's unit for the
        computation — the model is never mutated.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
        output : str, default='numpy'
            'numpy' returns np.ndarray; 'scipp' returns sc.Variable with y_unit.

        Returns
        -------
        np.ndarray | sc.Variable
            Evaluated Lorentzian values at x.
        """
        x_vals, detected_unit, dim = self._prepare_x_for_evaluate(x)
        eval_unit = detected_unit or self._x_unit
        eval_area_unit = str(sc.Unit(eval_unit) * sc.Unit(self._y_unit))

        center = self._resolve_param_value(self._center, eval_unit)
        width = self._resolve_param_value(self._width, eval_unit)
        area = self._resolve_param_value(self._area, eval_area_unit)

        normalization = width / np.pi
        denominator = (x_vals - center) ** 2 + width**2
        result = area * normalization / denominator

        if output == 'scipp':
            return sc.array(dims=[dim], values=result, unit=self._y_unit)
        return result

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert x-axis parameters (center, width) and area to new_x_unit.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.
        """
        self._convert_x_unit_area_based(
            new_x_unit=new_x_unit,
            x_params=[self._center, self._width],
            area_param=self._area,
        )

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis (output) unit by rescaling the area parameter.

        The area is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(new_y_unit=new_y_unit, area_param=self._area)

    def __repr__(self) -> str:
        """
        Return a string representation of the Lorentzian.

        Returns
        -------
        str
            A string representation of the Lorentzian.
        """
        return (
            f'{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, '
            f'x_unit = {self._x_unit}, y_unit = {self._y_unit},\n'
            f'    area = {self.area},\n'
            f'    center = {self.center},\n'
            f'    width = {self.width})'
        )
