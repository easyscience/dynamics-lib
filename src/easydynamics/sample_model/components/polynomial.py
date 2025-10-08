from __future__ import annotations

import warnings
from typing import Union

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

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")

        super().__init__(name=name)
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        dimless = sc.units.dimensionless

        # Build Parameters with appropriate units
        self.coefficients = []
        for i, coef in enumerate(coefficients):
            coef_unit = dimless if i == 0 else f"1 / ({unit}**{i})"
            self.coefficients.append(
                Parameter(name=f"{name}_c{i}", value=coef, unit=coef_unit)
            )
        # scipp converts units like "1 / (meV**2)" to SI units (3.89e+43 1/J**2)
        # EasyScience then converts this to 1/J**2 when setting the Parameter, and the value is scaled accordingly.
        # We therefore convert back to 1 / (meV**2), which becomes (3.89e+43 1/J**2), and set the value again
        for i, coef in enumerate(coefficients):
            coef_unit = f"1 / ({unit}**{i})"
            if i == 0:
                continue  # dimensionless, no scaling
            self.coefficients[i].convert_unit(coef_unit)
            self.coefficients[i].value = coef

        self._unit = unit

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Polynomial at the given x values.
        The Polynomial evaluates to c0 + c1*x + c2*x^2 + ... + cN*x^N
        """

        x = self._prepare_x_for_evaluate(x)

        result = np.zeros_like(x, dtype=float)
        for i, param in enumerate(self.coefficients):
            result += param.value * np.power(x, i)

        if any(result < 0):
            warnings.warn(
                "The Polynomial with name {} has negative values, which may not be physically meaningful.".format(
                    self.name
                )
            )
        return result

    def degree(self):
        """Return the degree of the polynomial."""
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
        return f"Polynomial(name = {self.name}, unit = {self._unit},\n coefficients = [{coeffs_str}])"

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """Convert the unit of the polynomial.
        Args:
            unit (str or sc.Unit): The target unit to convert to.
        """

        if not isinstance(unit, (str, sc.Unit)):
            raise UnitError("unit must be a string or a scipp unit.")

        for i, param in enumerate(self.coefficients):
            if i == 0:
                continue  # dimensionless, no scaling
            print(
                f"Converting coefficient {param.name} from {param.unit} to 1 / ({unit}**{i})"
            )
            param.convert_unit(f"1 / ({unit}**{i})")

        self._unit = unit


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
