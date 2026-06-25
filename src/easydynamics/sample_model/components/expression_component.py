# SPDX-FileCopyrightText: 2026 EasyScience contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import ClassVar

import scipp as sc
import sympy as sp
from easyscience.variable import Parameter
from scipy.special import erf

if TYPE_CHECKING:
    import numpy as np

from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric


class ExpressionComponent(ModelComponent):
    """
    Model component defined by a symbolic expression.

    The expression must contain ``x`` as the independent variable. All other symbols are treated as
    free parameters, which can be accessed and set as attributes after construction. Supported
    functions include ``exp``, ``sin``, ``cos``, ``sqrt``, ``erf``, and others — see the
    ``_ALLOWED_FUNCS`` class variable for the full list.

    Examples
    --------
    **Defining a custom Gaussian expression**

    Parameters are given as a dictionary of initial values and can be accessed as attributes after
    construction:
    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    expr = sm.ExpressionComponent(
        'A * exp(-(x - x0)**2 / (2*sigma**2))',
        parameters={'A': 10, 'x0': 0, 'sigma': 1},
        x_unit='meV',
        display_name='Gaussian Peak',
    )
    x = np.linspace(-3, 3, 100)
    values = expr.evaluate(x)
    ```

    **Modifying parameter values after construction**

    Parameters can be set directly as attributes:
    ```python
    expr.A = 5
    expr.sigma = 0.5
    ```
    """

    _ALLOWED_FUNCS: ClassVar[dict[str, object]] = {
        'exp': sp.exp,
        'log': sp.log,
        'ln': sp.log,
        'sqrt': sp.sqrt,
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'sinc': sp.sinc,
        'cot': sp.cot,
        'sec': sp.sec,
        'csc': sp.csc,
        'asin': sp.asin,
        'acos': sp.acos,
        'atan': sp.atan,
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        'abs': sp.Abs,
        'sign': sp.sign,
        'floor': sp.floor,
        'ceil': sp.ceiling,
        'erf': sp.erf,
    }

    _ALLOWED_CONSTANTS: ClassVar[dict[str, object]] = {
        'pi': sp.pi,
        'E': sp.E,
    }

    _RESERVED_NAMES: ClassVar[dict[str, object]] = {'x'}

    def __init__(
        self,
        expression: str,
        parameters: dict[str, Numeric] | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'Expression',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the ExpressionComponent.

        Parameters
        ----------
        expression : str
            The symbolic expression as a string. Must contain 'x' as the independent variable.
        parameters : dict[str, Numeric] | None, default=None
            Dictionary of parameter names and their initial values.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='Expression'
            Name used for parameter labelling and serialization.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Unique name for the component.

        Raises
        ------
        ValueError
            If the expression is invalid or does not contain 'x'.
        TypeError
            If any parameter value is not numeric.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        if 'np.' in expression:
            raise ValueError(
                'NumPy syntax (np.*) is not supported. '
                "Use functions like 'exp', 'sin', etc. directly."
            )

        self._expression_str = expression

        locals_dict = {}
        locals_dict.update(self._ALLOWED_FUNCS)
        locals_dict.update(self._ALLOWED_CONSTANTS)

        try:
            self._expr = sp.sympify(expression, locals=locals_dict)
        except Exception as e:
            raise ValueError(f'Invalid expression: {expression}') from e

        symbols = self._expr.free_symbols
        symbol_names = sorted(str(s) for s in symbols)

        if 'x' not in symbol_names:
            raise ValueError("Expression must contain 'x' as independent variable")

        allowed_function_names = set(self._ALLOWED_FUNCS) | {
            func.__name__ for func in self._ALLOWED_FUNCS.values()
        }

        unknown_function_names: set[str] = set()
        function_atoms = self._expr.atoms(sp.Function)
        for function_atom in function_atoms:
            function_name = function_atom.func.__name__
            if function_name not in allowed_function_names:
                unknown_function_names.add(function_name)

        unknown_functions = sorted(unknown_function_names)
        if unknown_functions:
            raise ValueError(
                f'Unsupported function(s) in expression: {", ".join(unknown_functions)}'
            )

        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                f'Parameters must be None or a dictionary, got {type(parameters).__name__}'
            )

        if parameters is not None:
            for name, value in parameters.items():
                if not isinstance(value, (Numeric, Parameter, dict)):
                    raise TypeError(
                        f"Parameter '{name}' must be numeric, "
                        f'a Parameter instance, or a dictionary, got {type(value).__name__}'
                    )
        parameters = parameters or {}
        self._parameters: dict[str, Parameter] = {}

        self._symbol_names = symbol_names
        for name in self._symbol_names:
            if name in self._RESERVED_NAMES:
                continue

            value = parameters.get(name, 1.0)
            if isinstance(value, Parameter):
                self._parameters[name] = value
            elif isinstance(value, dict) and value.get('@class') == 'Parameter':
                self._parameters[name] = Parameter.from_dict(value)
            else:
                self._parameters[name] = Parameter(
                    name=name,
                    value=value,
                    unit=self._x_unit,
                )

        ordered_symbols = [sp.Symbol(name) for name in self._symbol_names]
        self._func = sp.lambdify(
            ordered_symbols,
            self._expr,
            modules=[{'erf': erf}, 'numpy'],
        )

    @property
    def expression(self) -> str:
        return self._expression_str

    @expression.setter
    def expression(self, _new_expr: str) -> None:
        raise AttributeError('Expression cannot be changed after initialization')

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        """
        Evaluate the expression for given x values.

        Unit conversion of parameters is not supported for ExpressionComponent. If x has a
        different unit than x_unit, a warning is issued and x values are used as-is.
        """
        x_vals, detected_unit, dim = self._prepare_x_for_evaluate(x)

        if detected_unit is not None and detected_unit != str(self._x_unit):
            warnings.warn(
                f'Input x has unit {detected_unit} but {self.__class__.__name__} has '
                f'x_unit {self._x_unit}. ExpressionComponent cannot auto-convert parameters. '
                'x values are used as-is.',
                UserWarning,
                stacklevel=2,
            )

        args = []
        for name in self._symbol_names:
            if name == 'x':
                args.append(x_vals)
            else:
                args.append(self._parameters[name].value)

        result = self._func(*args)

        if output == 'scipp':
            return sc.array(dims=[dim], values=result, unit=self._y_unit)
        return result

    def get_all_variables(self) -> list[Parameter]:
        return list(self._parameters.values())

    def convert_x_unit(self, _new_unit: str | sc.Unit) -> None:
        raise NotImplementedError('Unit conversion is not implemented for ExpressionComponent')

    def convert_y_unit(self, _new_unit: str | sc.Unit) -> None:
        raise NotImplementedError('Unit conversion is not implemented for ExpressionComponent')

    def __getattr__(self, name: str) -> Parameter:
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __setattr__(self, name: str, value: Numeric) -> None:
        if '_parameters' in self.__dict__ and name in self._parameters:
            param = self._parameters[name]
            if not isinstance(value, Numeric):
                raise TypeError(f'{name} must be numeric')
            param.value = value
        else:
            super().__setattr__(name, value)

    def __dir__(self) -> list[str]:
        return super().__dir__() + list(self._parameters.keys())

    def __repr__(self) -> str:
        param_str = ', '.join(f'{k}={v.value}' for k, v in self._parameters.items())
        return (
            f'ExpressionComponent(name={self.name}, display_name={self.display_name}, '
            f'x_unit={self._x_unit}, y_unit={self._y_unit},\n'
            f"    expr='{self._expression_str}',\n"
            f'    parameters={{ {param_str} }} )'
        )
