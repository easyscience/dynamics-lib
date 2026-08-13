# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorBase
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

_CoefficientsInput = Sequence[Numeric | Parameter] | dict[int, Numeric]


class Polynomial(ModelComponent):
    r"""
    Polynomial function component.

    $$ I(x) = c_0 + c_1 x + c_2 x^2 + ... + c_N x^N $$

    Coefficients are stored as dimensionless Parameters. When x_unit changes, the coefficient
    values are rescaled so the evaluated result stays the same. The output unit is y_unit.

    Examples
    --------
    **Creating a constant background (degree 0)**

    ```python
    import numpy as np
    import easydynamics as edyn

    poly = edyn.Polynomial(coefficients=[1.5])
    x = np.linspace(-5, 5, 100)
    values = poly.evaluate(x)
    ```

    **Creating a linear background (degree 1)**

    Coefficients are ordered as ``[c0, c1, ...]``, where ``c0`` is the constant term:
    ```python
    import easydynamics as edyn

    poly = edyn.Polynomial(coefficients=[2.0, 0.1], name='Background')
    poly.coefficients = [1.5, 0.05]
    ```

    **Creating a sparse polynomial from a dict**

    Powers that are not listed are filled with coefficients fixed to zero:
    ```python
    import easydynamics as edyn

    poly = edyn.Polynomial(coefficients={2: 1.5})  # 1.5*x^2, with c0 and c1 fixed at 0
    ```

    **Changing the degree after construction**

    ```python
    import easydynamics as edyn

    poly = edyn.Polynomial(coefficients=[2.0, 0.1])
    poly.add_coefficient(0.05)  # now 2.0 + 0.1*x + 0.05*x^2
    removed = poly.remove_coefficient()  # returns 0.05, back to 2.0 + 0.1*x
    ```
    """

    def __init__(
        self,
        coefficients: _CoefficientsInput = (0.0,),
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'Polynomial',
        display_name: str | None = None,
        unique_name: str | None = None,
        suppress_warnings: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        coefficients : _CoefficientsInput, default=(0.0,)
            Either an ordered sequence of polynomial coefficients ``[c0, c1, ..., cN]`` where the
            polynomial is ``c0 + c1*x + c2*x^2 + ... + cN*x^N``, or a sparse ``dict`` mapping
            integer powers to numeric values (e.g. ``{2: 1.5}`` for ``1.5*x^2``).

            For a sequence, each element may be a plain numeric value (wrapped into a dimensionless
            :class:`Parameter`) or an existing :class:`Parameter` instance.  For a dict, powers not
            present are filled with fixed-to-zero Parameters, and the degree is taken from the
            largest key.  Must contain at least one element.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  When the x_unit is changed via :meth:`convert_x_unit`, coefficient
            values are rescaled by power-law factors so the evaluated output remains unchanged.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='Polynomial'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier.  Auto-generated if None.
        suppress_warnings : bool, default=False
            Whether to suppress warnings

        Raises
        ------
        TypeError
            If *coefficients* is not a list, tuple, ndarray, or dict, if any sequence element is
            neither numeric nor a :class:`Parameter`, or if any dict key is not an integer or dict
            value is not numeric.
        ValueError
            If *coefficients* is empty, or if any dict key is negative.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        if not isinstance(coefficients, (list, tuple, np.ndarray, dict)):
            raise TypeError(
                'coefficients must be a sequence (list/tuple/ndarray) '
                'of numbers or Parameter objects, or a dict mapping powers to numbers.'
            )

        if len(coefficients) == 0:
            raise ValueError('At least one coefficient must be provided.')

        self._coefficients: list[Parameter] = []

        if isinstance(coefficients, dict):
            for key in coefficients:
                # bool is a subclass of int, so reject it explicitly
                if not isinstance(key, int) or isinstance(key, bool):
                    raise TypeError('Dict keys must be integers representing polynomial powers.')
                if key < 0:
                    raise ValueError('Dict keys (powers) must be non-negative integers.')
            for i in range(max(coefficients) + 1):
                if i not in coefficients:
                    param = Parameter(name=f'{name}_c{i}', value=0.0, fixed=True)
                elif isinstance(coefficients[i], Numeric):
                    param = Parameter(name=f'{name}_c{i}', value=float(coefficients[i]))
                else:
                    raise TypeError('Each coefficient value must be a number.')
                self._coefficients.append(param)
        else:
            for i, coef in enumerate(coefficients):
                if isinstance(coef, Parameter):
                    param = coef
                elif isinstance(coef, Numeric):
                    param = Parameter(name=f'{name}_c{i}', value=float(coef))
                else:
                    raise TypeError(
                        'Each coefficient must be either a numeric value or a Parameter.'
                    )
                self._coefficients.append(param)

        # Tracks the current x_unit scale for convert_x_unit power-law rescaling
        self._x_unit_helper = sc.scalar(value=1.0, unit=x_unit)

        self.suppress_warnings = suppress_warnings

    @property
    def coefficients(self) -> list[Parameter]:
        """
        Get the coefficients of the polynomial as a list of Parameters.
        Returns
        -------
        list[Parameter]
            A shallow copy of the internal coefficient list ``[c0, c1, ..., cN]``.  Modifying the
            returned list does not affect the model; use the setter to replace values.
        """
        return list(self._coefficients)

    @coefficients.setter
    def coefficients(self, coeffs: Sequence[Numeric | Parameter]) -> None:
        """
        Set the coefficients of the polynomial.
        Parameters
        ----------
        coeffs : Sequence[Numeric | Parameter]
            New coefficient values.  Must be a list, tuple, or ndarray and must have the same
            length as the current number of coefficients. Numeric values update the existing
            Parameter's ``.value``; a Parameter instance replaces the stored Parameter entirely.

        Raises
        ------
        TypeError
            If *coeffs* is not a list, tuple, or ndarray, or if any element is neither numeric nor
            a Parameter.
        ValueError
            If the length of *coeffs* does not match the current number of coefficients.
        """
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise TypeError(
                'coefficients must be a sequence (list/tuple/ndarray) of numbers or Parameter.'
            )
        if len(coeffs) != len(self._coefficients):
            raise ValueError(
                'Number of coefficients must match the existing number of coefficients.'
            )
        for i, coef in enumerate(coeffs):
            if isinstance(coef, Parameter):
                self._coefficients[i] = coef
            elif isinstance(coef, Numeric):
                self._coefficients[i].value = float(coef)
            else:
                raise TypeError('Each coefficient must be either a numeric value or a Parameter.')

    def coefficient_values(self) -> list[float]:
        """
        Get the coefficients of the polynomial as a list.
        Returns
        -------
        list[float]
            Current numeric values of all coefficients ``[c0.value, c1.value, ..., cN.value]``.
        """
        return [param.value for param in self._coefficients]

    @property
    def degree(self) -> int:
        """
        Returns
        -------
        int
            Polynomial degree, equal to ``len(coefficients) - 1``.
        """
        return len(self._coefficients) - 1

    @degree.setter
    def degree(self, _value: int) -> None:
        """
        The degree is determined by the number of coefficients and cannot be set directly.

        Parameters
        ----------
        _value : int
            Ignored; this setter always raises :exc:`AttributeError`.

        Raises
        ------
        AttributeError
            Always raised when this setter is called.
        """
        raise AttributeError(
            'The degree of the polynomial is determined by the number of coefficients '
            'and cannot be set directly.'
        )

    def add_coefficient(self, value: Numeric = 0.0, fixed: bool = False) -> None:
        """
        Add a new coefficient at the next highest power, increasing the degree by one.

        Parameters
        ----------
        value : Numeric, default=0.0
            The numeric value of the new coefficient.
        fixed : bool, default=False
            If True, the new coefficient is fixed (not free for fitting).

        Raises
        ------
        TypeError
            If *value* is not a numeric value.
        """
        if not isinstance(value, Numeric):
            raise TypeError('value must be a numeric value.')
        new_power = len(self._coefficients)
        self._coefficients.append(
            Parameter(name=f'{self.name}_c{new_power}', value=float(value), fixed=fixed)
        )

    def remove_coefficient(self) -> float:
        """
        Remove the highest-power coefficient, decreasing the degree by one.

        Returns
        -------
        float
            The value of the removed coefficient.

        Raises
        ------
        ValueError
            If only one coefficient remains; a Polynomial must always keep at least one.
        """
        if len(self._coefficients) == 1:
            raise ValueError(
                'Cannot remove the last coefficient. The Polynomial must have at least one '
                'coefficient.'
            )
        return self._coefficients.pop().value

    @property
    def suppress_warnings(self) -> bool:
        """
        Get whether or not to suppress warnings.
        """
        return self._suppress_warnings

    @suppress_warnings.setter
    def suppress_warnings(self, value: bool) -> None:
        """
        Choose whether or not to suppress warnings.

        Parameters
        ----------
        value : bool
            Whether or not to suppress warnings

        Raises
        ------
        TypeError
            If suppress_warnings is not True or False
        """
        if not isinstance(value, bool):
            raise TypeError('Suppress_warnings must be True or False')
        self._suppress_warnings = value

    def _evaluate_values(self, x_vals: np.ndarray, eval_unit: str | None) -> np.ndarray:
        r"""
        Evaluate the Polynomial at x_vals.

        When x_vals is expressed in a different unit than the stored x_unit, coefficient values are
        temporarily rescaled (same power-law logic as convert_x_unit) without mutation.

        Parameters
        ----------
        x_vals : np.ndarray
            Raw x values expressed in eval_unit.
        eval_unit : str | None
            The unit of x_vals.

        Returns
        -------
        np.ndarray
            Evaluated polynomial values.
        """
        if (
            eval_unit is not None
            and self.x_unit is not None
            and sc.Unit(eval_unit) != sc.Unit(self.x_unit)
        ):
            # Temporary coefficient rescaling — no mutation
            helper = sc.scalar(1.0, unit=self.x_unit)
            helper_in_x = sc.to_unit(helper, eval_unit)
            scale = helper.value / helper_in_x.value
            coeff_vals = [p.value * scale**i for i, p in enumerate(self._coefficients)]
        else:
            coeff_vals = [p.value for p in self._coefficients]

        result = np.zeros_like(x_vals, dtype=float)
        for i, cv in enumerate(coeff_vals):
            result += cv * np.power(x_vals, i)

        if not self._suppress_warnings and any(result < 0):
            warnings.warn(
                f'The Polynomial with unique_name {self.unique_name} has negative values, '
                'which may not be physically meaningful.',
                UserWarning,
                stacklevel=3,
            )

        return result

    def get_all_variables(self) -> list[DescriptorBase]:
        """
        Returns
        -------
        list[DescriptorBase]
            The coefficient Parameters that constitute the fittable variables of this polynomial
            component.
        """
        return list(self._coefficients)

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert the x-axis unit by rescaling coefficients with power-law factors.

        Each coefficient ``c_i`` is rescaled by ``(old_scale / new_scale) ** i`` so the evaluated
        polynomial output is unchanged after the conversion.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.

        Raises
        ------
        UnitError
            If *new_x_unit* is not a valid unit string or ``sc.Unit``, or if the conversion between
            the current unit and *new_x_unit* fails.
        """
        if not isinstance(new_x_unit, (str, sc.Unit)):
            raise UnitError('new_x_unit must be a string or a scipp unit.')

        conversion_value_before = self._x_unit_helper.value
        self._x_unit_helper = sc.to_unit(self._x_unit_helper, unit=new_x_unit)
        conversion_value_after = self._x_unit_helper.value
        for i, param in enumerate(self._coefficients):
            param.value *= (conversion_value_before / conversion_value_after) ** i

        self._x_unit = str(new_x_unit) if isinstance(new_x_unit, sc.Unit) else new_x_unit

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Rescale all coefficients so the evaluated output remains the same physical value.

        All coefficients are multiplied by the conversion factor from ``old_y_unit`` to
        ``new_y_unit`` so that ``I(x) [new_y_unit]`` represents the same physical quantity as
        ``I(x) [old_y_unit]``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.  Must be dimensionally compatible with the current y_unit.

        Raises
        ------
        UnitError
            If *new_y_unit* is not a valid unit string or ``sc.Unit``, or if the conversion between
            the current y_unit and *new_y_unit* fails.
        """
        if not isinstance(new_y_unit, (str, sc.Unit)):
            raise UnitError('new_y_unit must be a string or a scipp unit.')

        old_y_unit = self.y_unit or 'dimensionless'
        new_y_str = str(new_y_unit) if isinstance(new_y_unit, sc.Unit) else new_y_unit

        # Compute conversion factor: 1 old_y_unit expressed in new_y_unit
        y_helper = sc.scalar(1.0, unit=old_y_unit)
        y_helper_new = sc.to_unit(y_helper, new_y_str)
        scale = y_helper_new.value / y_helper.value

        for param in self._coefficients:
            param.value *= scale
        self._y_unit = new_y_str

    def __repr__(self) -> str:
        coeffs_str = ', '.join(f'{param.name}={param.value}' for param in self._coefficients)
        return (
            f'{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, '
            f'x_unit = {self.x_unit}, y_unit = {self.y_unit},\n'
            f'    coefficients = [{coeffs_str}])'
        )
