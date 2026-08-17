# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.utils.utils import Numeric

MINIMUM_WIDTH = 1e-10  # To avoid division by zero
MINIMUM_AREA = 0.0  # To avoid negative areas
DHO_MINIMUM_CENTER = 1e-10  # To avoid zero center in DHO


class CreateParametersMixin:
    """
    Provides parameter creation and validation methods for model components.

    area_unit = x_unit * y_unit, so when y_unit='dimensionless', area_unit = x_unit.
    """

    @staticmethod
    def _set_bounded_parameter_value(param: Parameter, value: Numeric, label: str) -> None:
        """
        Assign a value to a bounded parameter, raising instead of silently clamping.

        easyscience's ``Parameter.value`` setter silently clamps out-of-bounds values to the
        nearest bound, which corrupts the parameter (e.g. assigning -1.0 to an area with
        ``min=0`` stores 0.0). Component setters route assignments through this helper so a
        bounds violation raises a clear error instead.

        Parameters
        ----------
        param : Parameter
            The parameter to assign to.
        value : Numeric
            The new value.
        label : str
            Name of the parameter used in error messages (e.g. ``'area'``, ``'width'``).

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* violates the parameter's bounds.
        """
        if not isinstance(value, Numeric):
            raise TypeError(f'{label} must be a number')
        value = float(value)
        if value < param.min or value > param.max:
            raise ValueError(
                f'Cannot set {label} to {value}: it violates the parameter bounds '
                f'[{param.min}, {param.max}]. Adjust the bounds first if this value is intended.'
            )
        param.value = value

    def _create_area_parameter(
        self,
        area: Numeric,
        name: str,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        minimum_area: float = MINIMUM_AREA,
    ) -> Parameter:
        """
        Create a Parameter for the area with unit = x_unit * y_unit.

        Parameters
        ----------
        area : Numeric
            Initial area value.
        name : str
            Base name used to label the Parameter (``name + ' area'``).
        x_unit : str | sc.Unit, default='meV'
            X-axis unit.  The resulting area unit is ``x_unit * y_unit``.
        y_unit : str | sc.Unit, default='dimensionless'
            Y-axis unit.  The resulting area unit is ``x_unit * y_unit``.
        minimum_area : float, default=MINIMUM_AREA
            Lower bound applied to the Parameter when the area is non-negative. When *area* is
            negative no lower bound is set and a :class:`UserWarning` is issued.

        Returns
        -------
        Parameter
            Configured area Parameter with ``unit = x_unit * y_unit``.

        Raises
        ------
        TypeError
            If *area* is not a numeric type.
        ValueError
            If *area* is not finite.
        """
        if not isinstance(area, Numeric):
            raise TypeError('area must be a number.')

        if not np.isfinite(area):
            raise ValueError('area must be a finite number.')

        area_unit = str(sc.Unit(x_unit) * sc.Unit(y_unit))
        area_param = Parameter(name=name + ' area', value=float(area), unit=area_unit)

        if area_param.value < 0:
            warnings.warn(
                f'The area of {name} is negative, which may not be physically meaningful.',
                UserWarning,
                stacklevel=3,
            )
        else:
            area_param.min = minimum_area

        return area_param

    def _create_center_parameter(
        self,
        center: Numeric | None,
        name: str,
        fix_if_none: bool,
        x_unit: str | sc.Unit = 'meV',
        enforce_minimum_center: bool = False,
    ) -> Parameter:
        """
        Create a Parameter for the center with unit = x_unit.

        Parameters
        ----------
        center : Numeric | None
            Initial center value.  If None, the center is set to 0.0 and ``fixed`` is controlled by
            *fix_if_none*.
        name : str
            Base name used to label the Parameter (``name + ' center'``).
        fix_if_none : bool
            Whether to fix the Parameter when *center* is None.
        x_unit : str | sc.Unit, default='meV'
            X-axis unit, applied to the center Parameter.
        enforce_minimum_center : bool, default=False
            If True, the Parameter's lower bound is raised to ``DHO_MINIMUM_CENTER`` (1e-10) to
            prevent a zero center.

        Returns
        -------
        Parameter
            Configured center Parameter with ``unit = x_unit``.

        Raises
        ------
        TypeError
            If *center* is not None and not a numeric type.
        ValueError
            If *center* is not None and not finite.
        """
        if center is not None and not isinstance(center, Numeric):
            raise TypeError('center must be None or a number.')

        if center is None:
            center_param = Parameter(
                name=name + ' center',
                value=0.0,
                unit=x_unit,
                fixed=fix_if_none,
            )
        else:
            if not np.isfinite(center):
                raise ValueError('center must be None or a finite number.')
            center_param = Parameter(name=name + ' center', value=float(center), unit=x_unit)

        if enforce_minimum_center:
            # Clamp the value before raising the bound: Parameter.min rejects a new minimum
            # that is larger than the current value.
            if center_param.value < DHO_MINIMUM_CENTER:
                center_param.value = DHO_MINIMUM_CENTER
            if center_param.min < DHO_MINIMUM_CENTER:
                center_param.min = DHO_MINIMUM_CENTER
        return center_param

    def _create_width_parameter(
        self,
        width: Numeric,
        name: str,
        param_name: str = 'width',
        x_unit: str | sc.Unit = 'meV',
        minimum_width: float = MINIMUM_WIDTH,
    ) -> Parameter:
        """
        Create a Parameter for the width with unit = x_unit.

        Parameters
        ----------
        width : Numeric
            Initial width value.  Must be strictly positive (>= *minimum_width*).
        name : str
            Base name used to label the Parameter (``name + ' ' + param_name``).
        param_name : str, default='width'
            Logical name of the parameter used in the label and error messages (e.g.
            ``'gaussian_width'``, ``'lorentzian_width'``).
        x_unit : str | sc.Unit, default='meV'
            X-axis unit, applied to the width Parameter.
        minimum_width : float, default=MINIMUM_WIDTH
            Absolute lower bound for the width to prevent division-by-zero.

        Returns
        -------
        Parameter
            Configured width Parameter with ``unit = x_unit`` and ``min = minimum_width``.

        Raises
        ------
        TypeError
            If *width* is not a numeric type.
        ValueError
            If *width* is not finite or is smaller than *minimum_width*.
        """
        if not isinstance(width, Numeric):
            raise TypeError(f'{param_name} must be a number.')

        if not np.isfinite(width):
            raise ValueError(f'{param_name} must be a finite number')

        if float(width) < minimum_width:
            raise ValueError(
                f'The {param_name} of a {self.__class__.__name__} must be greater than zero.'
            )
        return Parameter(
            name=name + ' ' + param_name,
            value=float(width),
            unit=x_unit,
            min=minimum_width,
        )
