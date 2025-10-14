from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError

from .model_component import ModelComponent

Numeric = Union[float, int]


class Polynomial(ModelComponent):
    """
    Polynomial function component. c0 + c1*x + c2*x^2 + ... + cN*x^N

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(
        self,
        name: str = "Polynomial",
        coefficients: Union[list[float], np.ndarray] = [0.0],
        unit: Union[str, sc.Unit] = "meV",
    ):
        if not isinstance(coefficients, (list, np.ndarray)):
            raise TypeError("coefficients must be a list or ndarray of floats.")

        if not all(isinstance(c, Numeric) for c in coefficients):
            raise TypeError("All coefficients must be numbers.")

        super().__init__(name=name, unit=unit)
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        self._coefficients = []
        # Coefficients are dimensionless, since powers of units are difficult to handle otherwise
        for i, coef in enumerate(coefficients):
            self._coefficients.append(Parameter(name=f"{name}_c{i}", value=coef))
        self._unit_conversion_helper = sc.scalar(value=1.0, unit=unit)

    @property
    def coefficients(self) -> list[Parameter]:
        """Get the coefficients of the polynomial as a list of Parameters."""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coeffs: list[float]) -> None:
        """Set the coefficients of the polynomial from a list of floats."""
        if not isinstance(coeffs, list):
            raise TypeError("coefficients must be a list of floats.")
        if not all(isinstance(c, Numeric) for c in coeffs):
            raise TypeError("All coefficients must be numbers.")
        if len(coeffs) != len(self._coefficients):
            raise ValueError(
                "Number of coefficients must match the existing number of coefficients."
            )
        for i, coef in enumerate(coeffs):
            self._coefficients[i].value = coef

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Polynomial at the given x values.
        The Polynomial evaluates to c0 + c1*x + c2*x^2 + ... + cN*x^N
        """

        x = self._prepare_x_for_evaluate(x)

        result = np.zeros_like(x, dtype=float)
        for i, param in enumerate(self._coefficients):
            result += param.value * np.power(x, i)

        if any(result < 0):
            warnings.warn(
                "The Polynomial with name {} has negative values, which may not be physically meaningful.".format(
                    self.name
                )
            )
        return result

    def degree(self) -> int:
        """Return the degree of the polynomial."""
        return len(self._coefficients) - 1

    def get_parameters(self) -> list[Parameter]:
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return self._coefficients

    def copy(self, name: Optional[str] = None) -> Polynomial:
        """
        Return a deep copy of this component with independent parameters.
        """
        if name is None:
            name = "copy of " + self.name

        model_copy = Polynomial(
            name=self.name, coefficients=[param.value for param in self._coefficients]
        )
        for i, param in enumerate(model_copy.coefficients):
            param.fixed = self._coefficients[i].fixed
        return model_copy

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """Convert the unit of the polynomial.
        Args:
            unit (str or sc.Unit): The target unit to convert to.
        """

        if not isinstance(unit, (str, sc.Unit)):
            raise UnitError("unit must be a string or a scipp unit.")

        # Find out how much the unit changes by converting in a helper variable
        conversion_value_before = self._unit_conversion_helper.value
        self._unit_conversion_helper = sc.to_unit(
            self._unit_conversion_helper, unit=unit
        )
        conversion_value_after = self._unit_conversion_helper.value
        for i, param in enumerate(self._coefficients):
            param.value *= (
                conversion_value_before / conversion_value_after
            ) ** i  # set the values directly to the appropriate power

        self._unit = unit

    def __repr__(self) -> str:
        coeffs_str = ", ".join(
            f"{param.name}={param.value}" for param in self._coefficients
        )
        return f"Polynomial(name = {self.name}, unit = {self._unit},\n coefficients = [{coeffs_str}])"


# from typing import Callable, Dict
# class UserDefinedComponent(ModelComponent):
#     """
#     User-defined model component, defined via a custom function.

#     Args:
#         func (Callable): Function accepting (x, params) and returning np.ndarray.
#         params (dict): Parameters passed to the function.
#     """

#     def __init__(
#         self, name, func: Callable[[np.ndarray, Dict], np.ndarray], params: Dict
#     ):
#         super().__init__(name=name)
#         self.func = func
#         self.params = params

#     def evaluate(self, x):
#         return self.func(x, self.params)
