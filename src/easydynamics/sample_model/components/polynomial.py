from __future__ import annotations

from typing import Union

import numpy as np

from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent

import scipp as sc

import warnings

Numeric = Union[float, int]


class Polynomial(ModelComponent):
    """
    Polynomial function component. Supports units, but not conversion between units.

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(
        self,
        name: str = "Polynomial",
        coefficients: Union[list[float], np.ndarray] = [0.0],
        unit: str = "meV",
    ):
        if not isinstance(coefficients, (list, tuple, np.ndarray)):
            raise TypeError("coefficients must be a list, tuple or ndarray of floats.")

        if not all(isinstance(c, Numeric) for c in coefficients):
            raise TypeError("All coefficients must be numbers.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        super().__init__(name=name)
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        self.coefficients = [
            Parameter(
                name=f"{name}_c{i}",
                value=coef,
            )
            for i, coef in enumerate(coefficients)
        ]
        self.unit = unit

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> np.ndarray:
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                raise ValueError(
                    f"Input x has unit {x.unit}, but Polynomial component has unit {self.unit}. Change the unit of the Polynomial and try again. "
                )
        else:
            x_in = x
        result = np.zeros_like(x_in, dtype=float)
        for i, param in enumerate(self.coefficients):
            result += param.value * np.power(x_in, i)

        if any(result < 0):
            warnings.warn(
                "The Polynomial with name {} has negative values, which may not be physically meaningful.".format(
                    self.name
                )
            )
        return result

    def degree(self):
        return len(self.coefficients) - 1

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return self.coefficients

    def copy(self) -> Polynomial:
        """
        Return a deep copy of this component with independent parameters.
        """

        model_copy = Polynomial(
            name=self.name, coefficients=[param.value for param in self.coefficients]
        )
        for i, param in enumerate(model_copy.coefficients):
            param.fixed = self.coefficients[i].fixed
        return model_copy

    def __repr__(self):
        coeffs_str = ", ".join(
            f"{param.name}={param.value}" for param in self.coefficients
        )
        return f"Polynomial(name = {self.name}, unit = {self.unit},\n coefficients = [{coeffs_str}])"

    def convert_unit(self, unit):
        raise NotImplementedError(
            "Unit conversion is not implemented for Polynomial components. The automatic unit converter does not like powers of units. "
        )


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
