# SPDX-FileCopyrightText: 2026 EasyScience contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import scipy
import sympy as sp
from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

if TYPE_CHECKING:
    import numpy as np
    import scipp as sc


class ExpressionComponent(ModelComponent):
    """
    Model component defined by a symbolic expression.

    Example: expr = ExpressionComponent( "A * exp(-(x - x0)**2 / (2*sigma**2))", parameters={"A":
    10, "x0": 0, "sigma": 1}, )

        expr.A = 5 y = expr.evaluate(x)
    """

    # -------------------------
    # Allowed symbolic functions
    # -------------------------
    _ALLOWED_FUNCS: ClassVar[dict[str, object]] = {
        # Exponentials & logs
        'exp': sp.exp,
        'log': sp.log,
        'ln': sp.log,
        'sqrt': sp.sqrt,
        # Trigonometric
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
        # Hyperbolic
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        # Misc
        'abs': sp.Abs,
        'sign': sp.sign,
        'floor': sp.floor,
        'ceil': sp.ceiling,
        # Special functions
        'erf': sp.erf,
    }

    # -------------------------
    # Allowed constants
    # -------------------------
    _ALLOWED_CONSTANTS: ClassVar[dict[str, object]] = {
        'pi': sp.pi,
        'E': sp.E,
    }

    _RESERVED_NAMES: ClassVar[dict[str, object]] = {'x'}

    def __init__(
        self,
        expression: str,
        parameters: dict[str, Numeric] | None = None,
        unit: str | sc.Unit = 'meV',
        display_name: str | None = 'Expression',
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
        unit : str | sc.Unit, default='meV'
            Unit of the output.
        display_name : str | None, default='Expression'
            Display name for the component.
        unique_name : str | None, default=None
            Unique name for the component.

        Raises
        ------
        ValueError
            If the expression is invalid or does not contain 'x'.
        TypeError
            If any parameter value is not numeric.
        """
        super().__init__(unit=unit, display_name=display_name, unique_name=unique_name)

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

        # Extract symbols from the expression
        symbols = self._expr.free_symbols
        symbol_names = sorted(str(s) for s in symbols)

        if 'x' not in symbol_names:
            raise ValueError("Expression must contain 'x' as independent variable")

        # Reject unknown functions early so invalid expressions fail at init,
        # not later during numerical evaluation.
        allowed_function_names = set(self._ALLOWED_FUNCS) | {
            func.__name__ for func in self._ALLOWED_FUNCS.values()
        }

        # Walk all function-call nodes in the parsed expression (e.g. sin(x), foo(x)).
        # Keep only function names that are not in our allowlist.
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

        # Create parameters
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                f'Parameters must be None or a dictionary, got {type(parameters).__name__}'
            )

        if parameters is not None:
            for name, value in parameters.items():
                if not isinstance(value, Numeric):
                    raise TypeError(f"Parameter '{name}' must be numeric")
        parameters = parameters or {}
        self._parameters: dict[str, Parameter] = {}

        self._symbol_names = symbol_names
        for name in self._symbol_names:
            if name in self._RESERVED_NAMES:
                continue

            value = parameters.get(name, 1.0)

            self._parameters[name] = Parameter(
                name=name,
                value=value,
                unit=self._unit,
            )

        # Create numerical function
        ordered_symbols = [sp.Symbol(name) for name in self._symbol_names]

        self._func = sp.lambdify(
            ordered_symbols,
            self._expr,
            modules=[{'erf': scipy.special.erf}, 'numpy'],
        )

        # -------------------------
        # Properties
        # -------------------------

    @property
    def expression(self) -> str:
        """Return the original expression string."""
        return self._expression_str

    @expression.setter
    def expression(self, _new_expr: str) -> None:
        """
        Prevent changing the expression after initialization.

        Parameters
        ----------
        _new_expr : str
            New expression string (ignored).

        Raises
        ------
        AttributeError
            Always raised to prevent changing the expression.
        """
        raise AttributeError('Expression cannot be changed after initialization')

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
    ) -> np.ndarray:
        """
        Evaluate the expression for given x values.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Input values for the independent variable.

        Returns
        -------
        np.ndarray
            Evaluated results.
        """
        x = self._prepare_x_for_evaluate(x)

        args = []
        for name in self._symbol_names:
            if name == 'x':
                args.append(x)
            else:
                args.append(self._parameters[name].value)

        return self._func(*args)

    def get_all_variables(self) -> list[Parameter]:
        """
        Return all parameters.

        Returns
        -------
        list[Parameter]
            List of all parameters in the expression.
        """
        return list(self._parameters.values())

    def convert_unit(self, _new_unit: str | sc.Unit) -> None:
        """
        Convert the unit of the expression.

        Unit conversion is not implemented for ExpressionComponent.

        Parameters
        ----------
        _new_unit : str | sc.Unit
            The new unit to convert to (ignored).

        Raises
        ------
        NotImplementedError
            Always raised to indicate unit conversion is not supported.
        """

        raise NotImplementedError('Unit conversion is not implemented for ExpressionComponent')

    # -------------------------
    # dunder methods
    # -------------------------

    def __getattr__(self, name: str) -> Parameter:
        """
        Allow access to parameters as attributes.

        Parameters
        ----------
        name : str
            Name of the parameter to access.

        Raises
        ------
        AttributeError
            If the parameter does not exist.

        Returns
        -------
        Parameter
            The parameter with the given name.
        """
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")

    def __setattr__(self, name: str, value: Numeric) -> None:
        """
        Allow setting parameter values as attributes.

        Parameters
        ----------
        name : str
            Name of the parameter to set.
        value : Numeric
            New value for the parameter.

        Raises
        ------
        TypeError
            If the value is not numeric.
        """
        if '_parameters' in self.__dict__ and name in self._parameters:
            param = self._parameters[name]

            if not isinstance(value, Numeric):
                raise TypeError(f'{name} must be numeric')

            param.value = value
        else:
            # For other attributes, use default behavior
            super().__setattr__(name, value)

    def __dir__(self) -> list[str]:
        """
        Include parameter names in dir() output for better IDE support.

        Returns
        -------
        list[str]
            List of attribute names, including parameters.
        """
        return super().__dir__() + list(self._parameters.keys())

    def __repr__(self) -> str:
        """Repr function."""
        param_str = ', '.join(f'{k}={v.value}' for k, v in self._parameters.items())
        return (
            f'{self.__class__.__name__}(\n'
            f"  expr='{self._expression_str}',\n"
            f'  unit={self._unit},\n'
            f'  parameters={{ {param_str} }}\n'
            f')'
        )
