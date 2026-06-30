# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipp as sc

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

EPSILON = 1e-8  # tolerance for bin-edge comparisons

if TYPE_CHECKING:
    from easyscience.variable import Parameter


class DeltaFunction(CreateParametersMixin, ModelComponent):
    """
    Delta function.

    When called directly, returns zero everywhere except at the bin nearest to ``center``, where it
    returns ``area / bin_width``. In convolutions it acts as an identity element (handled by the
    ``Convolution`` class). area has unit = x_unit * y_unit; center has unit = x_unit.

    If the center is not provided, it will be centered at 0 and fixed, which is typically what you
    want in QENS.

    Examples
    --------
    **Creating a DeltaFunction (elastic line)**

    The DeltaFunction evaluates to zero everywhere when called directly. It acts as an identity in
    convolutions, making it useful for modelling the elastic line in QENS:
    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    delta = sm.DeltaFunction(area=1.0)
    x = np.linspace(-2, 2, 100)
    values = delta.evaluate(x)  # all zeros except at the bin nearest to center
    ```

    **Creating a DeltaFunction with a free center**

    Pass a numeric value for ``center`` to place the elastic line at a specific energy transfer:
    ```python
    import easydynamics.sample_model as sm

    delta = sm.DeltaFunction(area=0.7, center=0.5)
    delta.area = 0.5
    ```
    """

    def __init__(
        self,
        center: Numeric | None = None,
        area: Numeric = 1.0,
        x_unit: str | sc.Unit = "meV",
        y_unit: str | sc.Unit = "dimensionless",
        name: str = "DeltaFunction",
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Delta function.

        Parameters
        ----------
        center : Numeric | None, default=None
            Position of the delta function in x_unit.  If None, defaults to 0 and the center
            parameter is fixed.
        area : Numeric, default=1.0
            Integrated area (weight) of the delta function.  Unit is ``x_unit * y_unit``.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  center is stored in this unit. area_unit = x_unit * y_unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='DeltaFunction'
            Name of the component.
        display_name : str | None, default=None
            Display name of the component, shown when plotting.  Falls back to *name* if None.
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
            raise TypeError("area must be a number")
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
            raise TypeError("center must be a number")
        self._center.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = "numpy",
    ) -> np.ndarray | sc.Variable:
        """
        Evaluate the Delta function at x.

        Parameters in the model's own units are temporarily converted to x's unit for the
        computation.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Input x values.
        output : str, default='numpy'
            'numpy' returns np.ndarray; 'scipp' returns sc.Variable with y_unit.

        Returns
        -------
        np.ndarray | sc.Variable
            Zero everywhere, with a single non-zero bin nearest the center when center falls within
            the x range.

        Notes
        -----
        When ``center`` falls within the x range, the bin nearest to ``center`` receives ``area /
        bin_width`` rather than zero.  In convolutions, the DeltaFunction acts as an identity
        element (handled by the Convolution class).
        """
        x_vals, detected_unit, dim = self._prepare_x_for_evaluate(x)
        eval_unit = detected_unit or self._x_unit
        eval_area_unit = str(sc.Unit(eval_unit) * sc.Unit(self._y_unit))

        center = self._resolve_param_value(self._center, eval_unit)
        area = self._resolve_param_value(self._area, eval_area_unit)

        model = np.zeros_like(x_vals, dtype=float)

        if x_vals.min() - EPSILON <= center <= x_vals.max() + EPSILON:
            # nearest index
            i = np.argmin(np.abs(x_vals - center))

            # left half-width
            if i == 0:
                left = x_vals[1] - x_vals[0] if x_vals.size > 1 else 0.5
            else:
                left = x_vals[i] - x_vals[i - 1]

            # right half-width
            if i == x_vals.size - 1:
                right = x_vals[-1] - x_vals[-2] if x_vals.size > 1 else 0.5
            else:
                right = x_vals[i + 1] - x_vals[i]

            # effective bin width: half left + half right
            bin_width = 0.5 * (left + right)
            model[i] = area / bin_width

        if output == "scipp":
            return sc.array(dims=[dim], values=model, unit=self._y_unit)
        return model

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert x-axis parameters (center) and area to new_x_unit.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.
        """
        self._convert_x_unit_area_based(
            new_x_unit=new_x_unit,
            x_params=[self._center],
            area_param=self._area,
        )

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit by rescaling the area parameter.

        The area is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(new_y_unit=new_y_unit, area_param=self._area)

    def __repr__(self) -> str:
        """
        Return a string representation of the Delta function.

        Returns
        -------
        str
            A string representation of the Delta function.
        """
        return (
            f"{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, "
            f"x_unit = {self.x_unit}, y_unit = {self.y_unit},\n"
            f"    area = {self.area},\n"
            f"    center = {self.center})"
        )
