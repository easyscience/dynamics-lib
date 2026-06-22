# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import scipp as sc
from scipy.special import voigt_profile

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

if TYPE_CHECKING:
    import numpy as np
    from easyscience.variable import Parameter


class Voigt(CreateParametersMixin, ModelComponent):
    r"""
    Voigt profile — convolution of Gaussian and Lorentzian.

    area has unit = x_unit * y_unit; center, gaussian_width, lorentzian_width have unit = x_unit.
    """

    def __init__(
        self,
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter | None = None,
        gaussian_width: Numeric | Parameter = 1.0,
        lorentzian_width: Numeric | Parameter = 1.0,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'Voigt',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        area : Numeric | Parameter, default=1.0
            Integrated area under the Voigt profile.  Unit is ``x_unit * y_unit``.
        center : Numeric | Parameter | None, default=None
            Peak position in x_unit.  If None, defaults to 0 and the center parameter is fixed.
        gaussian_width : Numeric | Parameter, default=1.0
            Gaussian component standard deviation (sigma) in x_unit.  Must be strictly positive.
        lorentzian_width : Numeric | Parameter, default=1.0
            Lorentzian component HWHM (gamma) in x_unit.  Must be strictly positive.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  center, gaussian_width, and lorentzian_width are stored in this
            unit.  area_unit = x_unit * y_unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='Voigt'
            Internal name used for parameter labelling.
        display_name : str | None, default=None
            Human-readable name.  Falls back to *name* if None.
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
        self._gaussian_width = self._create_width_parameter(
            width=gaussian_width, name=name, param_name='gaussian_width', x_unit=self._x_unit
        )
        self._lorentzian_width = self._create_width_parameter(
            width=lorentzian_width, name=name, param_name='lorentzian_width', x_unit=self._x_unit
        )

    @property
    def area(self) -> Parameter:
        """
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
    def gaussian_width(self) -> Parameter:
        """
        Returns
        -------
        Parameter
            The Gaussian component width (sigma) Parameter with unit ``x_unit``.
        """
        return self._gaussian_width

    @gaussian_width.setter
    def gaussian_width(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New Gaussian width (sigma) in x_unit.  Must be strictly positive.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('gaussian_width must be a number')
        if float(value) <= 0:
            raise ValueError('gaussian_width must be positive')
        self._gaussian_width.value = value

    @property
    def lorentzian_width(self) -> Parameter:
        """
        Returns
        -------
        Parameter
            The Lorentzian component HWHM (gamma) Parameter with unit ``x_unit``.
        """
        return self._lorentzian_width

    @lorentzian_width.setter
    def lorentzian_width(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New Lorentzian HWHM (gamma) in x_unit.  Must be strictly positive.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('lorentzian_width must be a number')
        if float(value) <= 0:
            raise ValueError('lorentzian_width must be positive')
        self._lorentzian_width.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        """
        Evaluate the Voigt at x. Model is never mutated for unit differences.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Input x values.
        output : str, default='numpy'
            'numpy' returns np.ndarray; 'scipp' returns sc.Variable with y_unit.

        Returns
        -------
        np.ndarray | sc.Variable
            Evaluated Voigt profile values at x.
        """
        x_vals, detected_unit, dim = self._prepare_x_for_evaluate(x)
        eval_unit = detected_unit or self._x_unit
        eval_area_unit = str(sc.Unit(eval_unit) * sc.Unit(self._y_unit))

        center = self._resolve_param_value(self._center, eval_unit)
        gw = self._resolve_param_value(self._gaussian_width, eval_unit)
        lw = self._resolve_param_value(self._lorentzian_width, eval_unit)
        area = self._resolve_param_value(self._area, eval_area_unit)

        result = area * voigt_profile(x_vals - center, gw, lw)

        if output == 'scipp':
            return sc.array(dims=[dim], values=result, unit=self._y_unit)
        return result

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert x-axis parameters (center, widths) and area to new_x_unit.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.

        Raises
        ------
        TypeError
            If *new_x_unit* is not a ``str`` or ``sc.Unit``.
        Exception
            If the unit conversion fails.  On failure the component is rolled back to its original
            units.
        """
        if not isinstance(new_x_unit, (str, sc.Unit)):
            raise TypeError(f'x_unit must be a string or sc.Unit, got {type(new_x_unit).__name__}')
        old_x_unit = self._x_unit
        new_area_unit = str(sc.Unit(new_x_unit) * sc.Unit(self._y_unit))
        try:
            self._center.convert_unit(new_x_unit)
            self._gaussian_width.convert_unit(new_x_unit)
            self._lorentzian_width.convert_unit(new_x_unit)
            self._area.convert_unit(new_area_unit)
            self._x_unit = str(new_x_unit) if isinstance(new_x_unit, sc.Unit) else new_x_unit
        except Exception as e:
            try:
                old_area_unit = str(sc.Unit(old_x_unit) * sc.Unit(self._y_unit))
                self._center.convert_unit(old_x_unit)
                self._gaussian_width.convert_unit(old_x_unit)
                self._lorentzian_width.convert_unit(old_x_unit)
                self._area.convert_unit(old_area_unit)
            except Exception:  # noqa: S110
                pass
            raise e

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit by rescaling the area parameter.

        The area is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.

        Raises
        ------
        TypeError
            If *new_y_unit* is not a ``str`` or ``sc.Unit``.
        Exception
            If the unit conversion fails.  On failure the component is rolled back to its original
            units.
        """
        if not isinstance(new_y_unit, (str, sc.Unit)):
            raise TypeError(f'y_unit must be a string or sc.Unit, got {type(new_y_unit).__name__}')
        old_y_unit = self._y_unit
        new_area_unit = str(sc.Unit(self._x_unit) * sc.Unit(new_y_unit))
        try:
            self._area.convert_unit(new_area_unit)
            self._y_unit = str(new_y_unit) if isinstance(new_y_unit, sc.Unit) else new_y_unit
        except Exception as e:
            try:
                old_area_unit = str(sc.Unit(self._x_unit) * sc.Unit(old_y_unit))
                self._area.convert_unit(old_area_unit)
            except Exception:  # noqa: S110
                pass
            raise e

    def __repr__(self) -> str:
        return (
            f'Voigt(name = {self.name}, display_name = {self.display_name}, '
            f'x_unit = {self._x_unit}, y_unit = {self._y_unit},\n'
            f'    area = {self.area},\n'
            f'    center = {self.center},\n'
            f'    gaussian_width = {self.gaussian_width},\n'
            f'    lorentzian_width = {self.lorentzian_width})'
        )
