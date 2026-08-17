# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric


class Exponential(CreateParametersMixin, ModelComponent):
    r"""
    Model of an exponential function.

    $$ I(x) = A e^{B (x-x_0)} $$

    where $A$ is the amplitude, $x_0$ is the center, and $B$ is the rate. amplitude has unit =
    y_unit; center has unit = x_unit; rate has unit = 1/x_unit.

    Examples
    --------
    **Creating an Exponential with a fixed center**

    By default the center is fixed at 0. A negative ``rate`` gives a decaying exponential:
    ```python
    import numpy as np
    import easydynamics as edyn

    exp = edyn.Exponential(amplitude=1.0, rate=-0.5)
    x = np.linspace(0, 5, 100)
    values = exp.evaluate(x)
    ```

    **Creating an Exponential with a free center and modifying parameters**

    Pass a numeric value for ``center`` to leave it free during fitting:
    ```python
    import easydynamics as edyn

    exp = edyn.Exponential(amplitude=2.0, center=1.0, rate=-1.0, name='Background')
    exp.amplitude = 3.0
    exp.rate = -0.5
    ```
    """

    def __init__(
        self,
        amplitude: Numeric = 1.0,
        center: Numeric | None = None,
        rate: Numeric = 1.0,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'Exponential',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Exponential component.

        Parameters
        ----------
        amplitude : Numeric, default=1.0
            Pre-exponential factor A.  Unit is ``y_unit``.
        center : Numeric | None, default=None
            Reference point x_0 in x_unit.  If None, defaults to 0 and the center parameter is
            fixed.
        rate : Numeric, default=1.0
            Exponential rate B in units of ``1/x_unit``.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  center is stored in this unit; rate is stored in ``1/x_unit``.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).  amplitude is stored in this unit.
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

        if not isinstance(amplitude, Numeric):
            raise TypeError('amplitude must be a number.')
        if not np.isfinite(amplitude):
            raise ValueError('amplitude must be finite.')
        self._amplitude = Parameter(
            name=name + ' amplitude', value=float(amplitude), unit=self.y_unit
        )

        self._center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, x_unit=self.x_unit
        )

        if not isinstance(rate, Numeric):
            raise TypeError('rate must be a number.')
        if not np.isfinite(rate):
            raise ValueError('rate must be finite.')
        self._rate = Parameter(name=name + ' rate', value=float(rate), unit='1/' + x_unit_str)

    @property
    def amplitude(self) -> Parameter:
        """
        Get the amplitude parameter.

        Returns
        -------
        Parameter
            The amplitude Parameter with unit ``y_unit``.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New amplitude value (in current amplitude unit = y_unit).

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        """
        if not isinstance(value, Numeric):
            raise TypeError('amplitude must be a number')
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
            raise TypeError('center must be a number')
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
            raise TypeError('rate must be a number')
        self._rate.value = value

    def _evaluate_values(self, x_vals: np.ndarray, eval_unit: str | None) -> np.ndarray:
        r"""
        Evaluate the Exponential at x_vals.

        Parameters in the model's own units are temporarily converted to eval_unit for the
        computation.

        Parameters
        ----------
        x_vals : np.ndarray
            Raw x values expressed in eval_unit.
        eval_unit : str | None
            The unit of x_vals.

        Returns
        -------
        np.ndarray
            Evaluated exponential values at x_vals.
        """
        eval_rate_unit = None if eval_unit is None else '1/' + str(eval_unit)

        center = self._resolve_param_value(self._center, eval_unit)
        rate = self._resolve_param_value(self._rate, eval_rate_unit)
        # The amplitude carries y_unit only, so it is unaffected by the x evaluation unit.
        amplitude = self._amplitude.value

        exponent = rate * (x_vals - center)
        return amplitude * np.exp(exponent)

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert center to new_x_unit and rate to 1/new_x_unit.

        The amplitude carries ``y_unit`` only and is unaffected.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.  The
            rate unit is set to ``1/new_x_unit``.
        """
        self._convert_x_unit_area_based(
            new_x_unit=new_x_unit,
            x_params=[self._center],
            inverse_params=[self._rate],
        )

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit by rescaling the amplitude parameter.

        The amplitude is rescaled from ``old_y_unit`` to ``new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(new_y_unit=new_y_unit, y_params=[self._amplitude])

    def __repr__(self) -> str:
        """
        Return a string representation of the Exponential.

        Returns
        -------
        str
            A string representation of the Exponential.
        """
        return (
            f'{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, '
            f'x_unit = {self.x_unit}, y_unit = {self.y_unit},\n '
            f'    amplitude = {self.amplitude},\n '
            f'    center = {self.center},\n '
            f'    rate = {self.rate})'
        )
