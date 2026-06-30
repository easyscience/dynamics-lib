# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import _assert_valid_unit


class Exponential(CreateParametersMixin, ModelComponent):
    r"""
    Model of an exponential function.

    $$ I(x) = A e^{B (x-x_0)} $$

    where $A$ is the amplitude, $x_0$ is the center, and $B$ is the rate. amplitude has unit =
    x_unit * y_unit; center has unit = x_unit; rate has unit = 1/x_unit.

    Examples
    --------
    **Creating an Exponential with a fixed center**

    By default the center is fixed at 0. A negative ``rate`` gives a decaying exponential:
    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    exp = sm.Exponential(amplitude=1.0, rate=-0.5)
    x = np.linspace(0, 5, 100)
    values = exp.evaluate(x)
    ```

    **Creating an Exponential with a free center and modifying parameters**

    Pass a numeric value for ``center`` to leave it free during fitting:
    ```python
    import easydynamics.sample_model as sm

    exp = sm.Exponential(amplitude=2.0, center=1.0, rate=-1.0, name='Background')
    exp.amplitude = 3.0
    exp.rate = -0.5
    ```
    """

    def __init__(
        self,
        amplitude: Numeric = 1.0,
        center: Numeric | None = None,
        rate: Numeric = 1.0,
        x_unit: str | sc.Unit = "meV",
        y_unit: str | sc.Unit = "dimensionless",
        name: str = "Exponential",
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Exponential component.

        Parameters
        ----------
        amplitude : Numeric, default=1.0
            Pre-exponential factor A.  Unit is ``x_unit * y_unit``.
        center : Numeric | None, default=None
            Reference point x_0 in x_unit.  If None, defaults to 0 and the center parameter is
            fixed.
        rate : Numeric, default=1.0
            Exponential rate B in units of ``1/x_unit``.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  center is stored in this unit; rate is stored in ``1/x_unit``.
            amplitude_unit = x_unit * y_unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='Exponential'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier.  Auto-generated if None.

        Raises
        ------
        TypeError
            If *amplitude* or *rate* is not numeric.
        ValueError
            If *amplitude* or *rate* is not finite.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        x_unit_str = str(x_unit) if isinstance(x_unit, sc.Unit) else x_unit
        amplitude_unit = str(sc.Unit(x_unit_str) * sc.Unit(self._y_unit))

        if not isinstance(amplitude, Numeric):
            raise TypeError("amplitude must be a number.")
        if not np.isfinite(amplitude):
            raise ValueError("amplitude must be finite.")
        self._amplitude = Parameter(
            name=name + " amplitude", value=float(amplitude), unit=amplitude_unit
        )

        self._center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, x_unit=self._x_unit
        )

        if not isinstance(rate, Numeric):
            raise TypeError("rate must be a number.")
        if not np.isfinite(rate):
            raise ValueError("rate must be finite.")
        self._rate = Parameter(
            name=name + " rate", value=float(rate), unit="1/" + x_unit_str
        )

    @property
    def amplitude(self) -> Parameter:
        """
        Get the amplitude parameter.

        Returns
        -------
        Parameter
            The amplitude Parameter with unit ``x_unit * y_unit``.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New amplitude value (in current amplitude unit = x_unit * y_unit).

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        """
        if not isinstance(value, Numeric):
            raise TypeError("amplitude must be a number")
        self._amplitude.value = value

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter.

        Returns
        -------
        Parameter
            The center (x_0) Parameter with unit ``x_unit``.
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

    @property
    def rate(self) -> Parameter:
        """
        Get the rate parameter.

        Returns
        -------
        Parameter
            The exponential rate (B) Parameter with unit ``1/x_unit``.
        """
        return self._rate

    @rate.setter
    def rate(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New exponential rate in ``1/x_unit``.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        """
        if not isinstance(value, Numeric):
            raise TypeError("rate must be a number")
        self._rate.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = "numpy",
    ) -> np.ndarray | sc.Variable:
        r"""
        Evaluate the Exponential at x.

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
            Evaluated exponential values at x.
        """
        x_vals, detected_unit, dim = self._prepare_x_for_evaluate(x)
        eval_unit = detected_unit or self._x_unit
        eval_area_unit = str(sc.Unit(eval_unit) * sc.Unit(self._y_unit))
        eval_rate_unit = "1/" + str(eval_unit)

        center = self._resolve_param_value(self._center, eval_unit)
        rate = self._resolve_param_value(self._rate, eval_rate_unit)
        amplitude = self._resolve_param_value(self._amplitude, eval_area_unit)

        exponent = rate * (x_vals - center)
        result = amplitude * np.exp(exponent)

        if output == "scipp":
            return sc.array(dims=[dim], values=result, unit=self._y_unit)
        return result

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert center and amplitude to new_x_unit, rate to 1/new_x_unit.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.  The
            rate unit is set to ``1/new_x_unit``.

        Raises
        ------
        Exception
            If the unit conversion fails.  On failure the component is rolled back to its original
            units.
        """
        _assert_valid_unit(new_x_unit)
        old_x_unit = self._x_unit
        new_x_str = str(new_x_unit) if isinstance(new_x_unit, sc.Unit) else new_x_unit
        new_area_unit = str(sc.Unit(new_x_str) * sc.Unit(self._y_unit))
        try:
            self._center.convert_unit(new_x_unit)
            self._amplitude.convert_unit(new_area_unit)
            self._rate.convert_unit("1/" + new_x_str)
            self._x_unit = new_x_str
        except Exception as e:
            try:
                old_area_unit = str(sc.Unit(old_x_unit) * sc.Unit(self._y_unit))
                self._center.convert_unit(old_x_unit)
                self._amplitude.convert_unit(old_area_unit)
                self._rate.convert_unit("1/" + str(old_x_unit))
            except Exception:  # noqa: S110
                pass
            raise e

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit by rescaling the amplitude parameter.

        The amplitude is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(
            new_y_unit=new_y_unit, area_param=self._amplitude
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the Exponential.

        Returns
        -------
        str
            A string representation of the Exponential.
        """
        return (
            f"{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, "
            f"x_unit = {self._x_unit}, y_unit = {self._y_unit},\n "
            f"    amplitude = {self.amplitude},\n "
            f"    center = {self.center},\n "
            f"    rate = {self.rate})"
        )
