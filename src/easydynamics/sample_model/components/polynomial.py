# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorBase
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.utils.utils import Numeric

from .model_component import ModelComponent

if TYPE_CHECKING:
    from collections.abc import Sequence


class Polynomial(ModelComponent):
    r"""
    Polynomial function component.

    The intensity is given by $$ I(x) = c_0 + c_1 x + c_2 x^2 + ... + c_N x^N, $$ where $C_i$ are
    the coefficients.
    """

    def __init__(
        self,
        coefficients: Sequence[Numeric | Parameter] = (0.0,),
        unit: str | sc.Unit = 'meV',
        display_name: str | None = 'Polynomial',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Polynomial component.

        Parameters
        ----------
        coefficients : Sequence[Numeric | Parameter], default=(0.0,)
            Coefficients c0, c1, ..., cN.
        unit : str | sc.Unit, default='meV'
            Unit of the Polynomial component.
        display_name : str | None, default='Polynomial'
            Display name of the Polynomial component.
        unique_name : str | None, default=None
            Unique name of the component. If None, a unique_name is automatically generated. By
            default, None.

        Raises
        ------
        TypeError
            If coefficients is not a sequence of numbers or Parameters or if any item in
            coefficients is not a number or Parameter.
        ValueError
            If coefficients is an empty sequence.
        """

        super().__init__(display_name=display_name, unit=unit, unique_name=unique_name)

        if not isinstance(coefficients, (list, tuple, np.ndarray)):
            raise TypeError(
                'coefficients must be a sequence (list/tuple/ndarray) \
                    of numbers or Parameter objects.'
            )

        if len(coefficients) == 0:
            raise ValueError('At least one coefficient must be provided.')

        # Internal storage of Parameter objects
        self._coefficients: list[Parameter] = []

        # Coefficients are treated as dimensionless Parameters
        for i, coef in enumerate(coefficients):
            if isinstance(coef, Parameter):
                param = coef
            elif isinstance(coef, Numeric):
                param = Parameter(name=f'{display_name}_c{i}', value=float(coef))
            else:
                raise TypeError('Each coefficient must be either a numeric value or a Parameter.')
            self._coefficients.append(param)

        # Helper scipp scalar to track unit conversions
        # (value initialized to 1 with provided unit)
        self._unit_conversion_helper = sc.scalar(value=1.0, unit=unit)

    @property
    def coefficients(self) -> list[Parameter]:
        """
        Get the coefficients of the polynomial as a list of Parameters.

        Returns
        -------
        list[Parameter]
            The coefficients of the polynomial.
        """
        return list(self._coefficients)

    @coefficients.setter
    def coefficients(self, coeffs: Sequence[Numeric | Parameter]) -> None:
        """
        Set the coefficients of the polynomial.

        Length must match current number of coefficients.

        Parameters
        ----------
        coeffs : Sequence[Numeric | Parameter]
            New coefficients as a sequence of numbers or Parameters.

        Raises
        ------
        TypeError
            If coeffs is not a sequence of numbers or Parameters or if any item in coeffs is not a
            number or Parameter.
        ValueError
            If the length of coeffs does not match the existing number of coefficients.
        """
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise TypeError(
                'coefficients must be a sequence (list/tuple/ndarray) of numbers or Parameter .'
            )
        if len(coeffs) != len(self._coefficients):
            raise ValueError(
                'Number of coefficients must match the existing number of coefficients.'
            )
        for i, coef in enumerate(coeffs):
            if isinstance(coef, Parameter):
                # replace parameter
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
            The coefficient values of the polynomial.
        """
        return [param.value for param in self._coefficients]

    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        r"""
        Evaluate the Polynomial at the given x values.

        The intensity is given by $$ I(x) = c_0 + c_1 x + c_2 x^2 + ...
        + c_N x^N, $$ where $C_i$ are the coefficients.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the Polynomial.

        Returns
        -------
        np.ndarray
            The evaluated Polynomial at the given x values.
        """

        x = self._prepare_x_for_evaluate(x)

        result = np.zeros_like(x, dtype=float)
        for i, param in enumerate(self._coefficients):
            result += param.value * np.power(x, i)

        if any(result < 0):
            warnings.warn(
                f'The Polynomial with unique_name {self.unique_name} has negative values, '
                'which may not be physically meaningful.',
                UserWarning,
                stacklevel=2,
            )
        return result

    @property
    def degree(self) -> int:
        """
        Get the degree of the polynomial.

        Returns
        -------
        int
            The degree of the polynomial.
        """
        return len(self._coefficients) - 1

    @degree.setter
    def degree(self, _value: int) -> None:
        """
        The degree is determined by the number of coefficients and cannot be set directly.

        Parameters
        ----------
        _value : int
            The new degree of the polynomial.

        Raises
        ------
        AttributeError
            Always raised since degree cannot be set directly.
        """
        raise AttributeError(
            'The degree of the polynomial is determined by the number of coefficients '
            'and cannot be set directly.'
        )

    def get_all_variables(self) -> list[DescriptorBase]:
        """
        Get all variables from the model component.

        Returns
        -------
        list[DescriptorBase]
            List of variables in the component.
        """
        return list(self._coefficients)

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the polynomial.

        Parameters
        ----------
        unit : str | sc.Unit
            The target unit to convert to.

        Raises
        ------
        UnitError
            If the provided unit is not a string or sc.Unit.
        """

        if not isinstance(unit, (str, sc.Unit)):
            raise UnitError('unit must be a string or a scipp unit.')

        # Find out how much the unit changes
        # by converting a helper variable
        conversion_value_before = self._unit_conversion_helper.value
        self._unit_conversion_helper = sc.to_unit(self._unit_conversion_helper, unit=unit)
        conversion_value_after = self._unit_conversion_helper.value
        for i, param in enumerate(self._coefficients):
            param.value *= (
                conversion_value_before / conversion_value_after
            ) ** i  # set the values directly to the appropriate power

        self._unit = unit

    def __repr__(self) -> str:
        """
        Return a string representation of the Polynomial.

        Returns
        -------
        str
            A string representation of the Polynomial.
        """

        coeffs_str = ', '.join(f'{param.name}={param.value}' for param in self._coefficients)
        return (
            f'Polynomial(unique_name = {self.unique_name}, '
            f'unit = {self._unit},\n coefficients = [{coeffs_str}])'
        )


# from typing import Callable, Dict
# class UserDefinedComponent(ModelComponent):
#     """
#     User-defined model component, defined via a custom function.

#     Args:
#         func (Callable): Function accepting (x, params) and returning
# np.ndarray.
#         params (dict): Parameters passed to the function.
#     """

#     def __init__(
#         self, name, func: Callable[[np.ndarray, Dict], np.ndarray],
# params: Dict
#     ):
#         super().__init__(name=name)
#         self.func = func
#         self.params = params

#     def evaluate(self, x):
#         return self.func(x, self.params)
