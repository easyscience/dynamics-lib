from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorBase, Parameter
from scipp import UnitError

from .model_component import ModelComponent

Numeric = float | int


class Polynomial(ModelComponent):
    """
    Polynomial function component. c0 + c1*x + c2*x^2 + ... + cN*x^N

    Args:
        display_name (str): Display name of the Polynomial component.
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
        unit (str or sc.Unit): Unit of the Polynomial component.
    """

    def __init__(
        self,
        coefficients: Sequence[Numeric | Parameter] = (0.0,),
        unit: str | sc.Unit = "meV",
        display_name: str | None = "Polynomial",
        unique_name: str | None = None,
    ):
        super().__init__(display_name=display_name, unit=unit, unique_name=unique_name)

        if not isinstance(coefficients, (list, tuple, np.ndarray)):
            raise TypeError(
                "coefficients must be a sequence (list/tuple/ndarray) of numbers or Parameter objects."
            )

        if len(coefficients) == 0:
            raise ValueError("At least one coefficient must be provided.")

        # Internal storage of Parameter objects
        self._coefficients: list[Parameter] = []

        # Coefficients are treated as dimensionless Parameters
        for i, coef in enumerate(coefficients):
            if isinstance(coef, Parameter):
                param = coef
            elif isinstance(coef, Numeric):
                param = Parameter(name=f"{display_name}_c{i}", value=float(coef))
            else:
                raise TypeError(
                    "Each coefficient must be either a numeric value or a Parameter."
                )
            self._coefficients.append(param)

        # Helper scipp scalar to track unit conversions (value initialized to 1 with provided unit)
        self._unit_conversion_helper = sc.scalar(value=1.0, unit=unit)

    @property
    def coefficients(self) -> list[Parameter]:
        """Get the coefficients of the polynomial as a list of Parameters."""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coeffs: Sequence[Numeric | Parameter]) -> None:
        """Replace the coefficients. Length must match current number of coefficients."""
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise TypeError(
                "coefficients must be a sequence (list/tuple/ndarray) of numbers or Parameter objects."
            )
        if len(coeffs) != len(self._coefficients):
            raise ValueError(
                "Number of coefficients must match the existing number of coefficients."
            )
        for i, coef in enumerate(coeffs):
            if isinstance(coef, Parameter):
                # replace parameter
                self._coefficients[i] = coef
            elif isinstance(coef, Numeric):
                self._coefficients[i].value = float(coef)
            else:
                raise TypeError(
                    "Each coefficient must be either a numeric value or a Parameter."
                )

    @property
    def coefficient_values(self) -> list[float]:
        """Get the coefficients of the polynomial as a list."""
        coefficient_list = [param.value for param in self._coefficients]
        return coefficient_list

    def evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
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
                f"The Polynomial with name {self.display_name} has negative values, "
                "which may not be physically meaningful.",
                UserWarning,
            )
        return result

    @property
    def degree(self) -> int:
        """Return the degree of the polynomial."""
        return len(self._coefficients) - 1

    @degree.setter
    def degree(self, value: int) -> None:
        raise AttributeError(
            "The degree of the polynomial is determined by the number of coefficients and cannot be set directly."
        )

    def get_all_variables(self) -> list[DescriptorBase]:
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return self._coefficients

    def convert_unit(self, unit: str | sc.Unit):
        """Convert the unit of the polynomial.
        Args:
            unit (str or sc.Unit): The target unit to convert to.
        """

        if not isinstance(unit, (str, sc.Unit)):
            raise UnitError("unit must be a string or a scipp unit.")

        # Find out how much the unit changes by converting a helper variable
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
        return f"Polynomial(display_name = {self.display_name}, unit = {self._unit},\n coefficients = [{coeffs_str}])"


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
