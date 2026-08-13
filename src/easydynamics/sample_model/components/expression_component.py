# SPDX-FileCopyrightText: 2026 EasyScience contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING
from typing import ClassVar

import scipp as sc
import sympy as sp
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from scipy.special import erf

if TYPE_CHECKING:
    import numpy as np

from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import _assert_valid_unit
from easydynamics.utils.utils import hbar
from easydynamics.utils.utils import kb


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
    import easydynamics as edyn

    expr = edyn.ExpressionComponent(
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

    **Giving parameters units**

    Parameters are dimensionless by default. Units can be given per parameter at construction, or
    relabelled later with ``set_unit`` (the numeric value is kept as-is). When units are in use,
    the unit of the evaluated expression is derived from the parameter units and x_unit (see
    ``output_unit``), and a warning is issued if it does not match y_unit:
    ```python
    expr = edyn.ExpressionComponent(
        'A * exp(-(x - x0)**2 / (2*sigma**2))',
        parameters={'A': 10, 'x0': 0, 'sigma': 1},
        parameter_units={'A': '1/meV', 'x0': 'meV', 'sigma': 'meV'},
        y_unit='1/meV',
    )
    expr.set_unit('A', '1/meV')
    ```

    **Physical constants**

    The symbols ``hbar`` (in meV*s) and ``kb`` (in meV/K) are provided automatically as read-only
    constants (DescriptorNumbers) when they appear in the expression:
    ```python
    boltzmann = edyn.ExpressionComponent(
        'exp(-x / (kb * T))',
        parameters={'T': 300.0},
        parameter_units={'T': 'K'},
    )
    ```
    Use e.g. ``boltzmann.kb.convert_unit('eV/K')`` to work in another unit (this rescales the
    value, unlike ``set_unit``).
    """

    _ALLOWED_FUNCS: ClassVar[dict[str, object]] = {
        # exponential / logarithmic
        'exp': sp.exp,
        'log': sp.log,
        'ln': sp.log,
        'sqrt': sp.sqrt,
        # trigonometric
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'sinc': sp.sinc,
        'cot': sp.cot,
        'sec': sp.sec,
        'csc': sp.csc,
        # inverse trigonometric
        'asin': sp.asin,
        'acos': sp.acos,
        'atan': sp.atan,
        # hyperbolic
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        # rounding / sign
        'abs': sp.Abs,
        'sign': sp.sign,
        'floor': sp.floor,
        'ceil': sp.ceiling,
        # special functions
        'erf': sp.erf,
    }

    _ALLOWED_CONSTANTS: ClassVar[dict[str, object]] = {
        'pi': sp.pi,
        'E': sp.E,
    }

    _RESERVED_NAMES: ClassVar[dict[str, object]] = {'x'}

    # Physical constants provided automatically as read-only DescriptorNumbers when their
    # symbol appears in the expression: name -> (source constant from utils, default unit).
    _PHYSICAL_CONSTANTS: ClassVar[dict[str, tuple[DescriptorNumber, str]]] = {
        'hbar': (hbar, 'meV*s'),
        'kb': (kb, 'meV/K'),
    }

    # Sympy functions that preserve the unit of their argument; all other allowed functions
    # require a dimensionless argument and return a dimensionless result.
    _UNIT_PRESERVING_FUNCS: ClassVar[tuple] = (sp.Abs, sp.floor, sp.ceiling)
    # Functions that accept any unit but return a dimensionless result.
    _DIMENSIONLESS_RESULT_FUNCS: ClassVar[tuple] = (sp.sign,)

    # parameter_units is applied at construction only; the resulting units are captured in
    # the serialized Parameter dicts, so the argument is skipped during serialization.
    _REDIRECT: ClassVar[dict[str, object]] = {'parameter_units': None}

    def __init__(
        self,
        expression: str,
        parameters: dict[str, Numeric] | None = None,
        parameter_units: dict[str, str | sc.Unit] | None = None,
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
            The symbolic expression as a string. Must contain 'x' as the independent variable. The
            symbols ``hbar`` and ``kb`` are provided automatically as read-only physical constants
            (in meV*s and meV/K respectively) unless overridden via *parameters*.
        parameters : dict[str, Numeric] | None, default=None
            Dictionary of parameter names and their initial values. Parameters that are not given a
            unit are dimensionless.
        parameter_units : dict[str, str | sc.Unit] | None, default=None
            Optional units per parameter name. Each entry sets the unit of the named parameter
            without rescaling its value (see :meth:`set_unit`), and takes precedence over the unit
            of a Parameter instance given in *parameters*. When units are in use, a warning is
            issued if the expression's output unit does not match y_unit.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='Expression'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Unique name for the component.

        Raises
        ------
        ValueError
            If the expression is invalid or does not contain 'x', or if parameter_units names a
            parameter that is not in the expression.
        TypeError
            If any parameter value is not numeric, or if parameter_units is not a dictionary.
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
                if not isinstance(value, (Numeric, Parameter, dict)):
                    raise TypeError(
                        f"Parameter '{name}' must be numeric, "
                        f'a Parameter instance, or a dictionary, got {type(value).__name__}'
                    )
        parameters = parameters or {}
        self._parameters: dict[str, Parameter] = {}
        self._constants: dict[str, DescriptorNumber] = {}

        self._symbol_names = symbol_names
        for name in self._symbol_names:
            if name in self._RESERVED_NAMES:
                continue

            # Physical constants are provided automatically, unless the user explicitly
            # supplies a parameter with the same name.
            if name in self._PHYSICAL_CONSTANTS and name not in parameters:
                self._constants[name] = self._create_physical_constant(name)
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
                    unit='dimensionless',
                )

        if parameter_units is not None:
            if not isinstance(parameter_units, dict):
                raise TypeError(
                    f'parameter_units must be None or a dictionary, '
                    f'got {type(parameter_units).__name__}'
                )
            constant_names = sorted(set(parameter_units) & set(self._constants))
            if constant_names:
                raise ValueError(
                    f'parameter_units given for physical constant(s): '
                    f'{", ".join(constant_names)}. Use e.g. '
                    f'expr.{constant_names[0]}.convert_unit(...) to change the unit of a '
                    f'constant (this rescales its value).'
                )
            unknown_names = sorted(set(parameter_units) - set(self._parameters))
            if unknown_names:
                raise ValueError(
                    f'parameter_units given for unknown parameter(s): {", ".join(unknown_names)}'
                )
            for parameter_name, parameter_unit in parameter_units.items():
                # Relabel without the per-call output-unit warning; the consistency check
                # runs once below, after all units are applied.
                self._relabel_parameter_unit(parameter_name, parameter_unit)

        # Create numerical function
        ordered_symbols = [sp.Symbol(name) for name in self._symbol_names]
        self._func = sp.lambdify(
            ordered_symbols,
            self._expr,
            modules=[{'erf': erf}, 'numpy'],
        )

        self._warn_if_output_unit_mismatch()

        # -------------------------
        # Properties
        # -------------------------

    @property
    def expression(self) -> str:
        """
        Return the original expression string.

        Returns
        -------
        str
             The original expression string provided at initialization.
        """
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

    def _evaluate_values(self, x_vals: np.ndarray, eval_unit: str | None) -> np.ndarray:
        """
        Evaluate the expression for the given x values.

        Unit conversion of parameters is not supported for ExpressionComponent. If x_vals is
        expressed in a different unit than x_unit, a UnitError is raised — silently evaluating the
        expression with wrongly-scaled x would corrupt the result.

        Parameters
        ----------
        x_vals : np.ndarray
            Raw x values expressed in eval_unit.
        eval_unit : str | None
            The unit of x_vals.

        Raises
        ------
        sc.UnitError
            If eval_unit differs from the component's x_unit.

        Returns
        -------
        np.ndarray
            Evaluated results.
        """
        if (
            eval_unit is not None
            and self.x_unit is not None
            and sc.Unit(eval_unit) != sc.Unit(self.x_unit)
        ):
            raise sc.UnitError(
                f'Input x has unit {eval_unit} but {self.__class__.__name__} has '
                f'x_unit {self.x_unit}. ExpressionComponent cannot auto-convert its parameters; '
                f'convert x to {self.x_unit} before evaluating.'
            )

        args = []
        for name in self._symbol_names:
            if name == 'x':
                args.append(x_vals)
            elif name in self._constants:
                args.append(self._constants[name].value)
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

    def set_unit(self, name: str, unit: str | sc.Unit) -> None:
        """
        Set the unit of a parameter without rescaling its value.

        This relabels the unit: the numeric value, bounds, and variance are kept as-is. Use
        ``Parameter.convert_unit`` instead to rescale a value into a compatible unit. Issues a
        warning if the resulting output unit of the expression no longer matches y_unit. Raises the
        same exceptions as :meth:`_relabel_parameter_unit` on invalid input.

        Parameters
        ----------
        name : str
            Name of the parameter whose unit to set.
        unit : str | sc.Unit
            The new unit.
        """
        self._relabel_parameter_unit(name, unit)
        self._warn_if_output_unit_mismatch()

    def _relabel_parameter_unit(self, name: str, unit: str | sc.Unit) -> None:
        """
        Validate and apply a unit relabel to a parameter (see set_unit).

        Unit validation is delegated to ``_assert_valid_unit``, which raises ``ValueError`` for a
        string that is not a valid scipp unit.

        Parameters
        ----------
        name : str
            Name of the parameter whose unit to set.
        unit : str | sc.Unit
            The new unit.

        Raises
        ------
        TypeError
            If name is not a string or unit is not a string or sc.Unit.
        KeyError
            If no parameter with the given name exists.
        AttributeError
            If the parameter is a physical constant or a dependent parameter.
        """
        if not isinstance(name, str):
            raise TypeError(f'name must be a string, got {type(name).__name__}')
        if '_constants' in self.__dict__ and name in self._constants:
            raise AttributeError(
                f"'{name}' is a physical constant. Use expr.{name}.convert_unit(...) to change "
                f'its unit (this rescales its value).'
            )
        if '_parameters' not in self.__dict__ or name not in self._parameters:
            raise KeyError(f"No parameter named '{name}' in this {self.__class__.__name__}.")
        _assert_valid_unit(unit)
        new_unit = sc.Unit(str(unit))

        param = self._parameters[name]
        if not param.independent:
            raise AttributeError(
                f"Cannot set the unit of dependent parameter '{name}'; its unit is controlled "
                f'by the dependency expression.'
            )

        # Parameter.unit is read-only and convert_unit rescales the value, so a pure relabel
        # has to swap the underlying scipp scalars (value and bounds) directly.
        param._scalar = sc.scalar(  # ruff: ignore[private-member-access]
            param.value,
            unit=new_unit,
            variance=param._scalar.variance,  # ruff: ignore[private-member-access]
        )
        param._min = sc.scalar(param._min.value, unit=new_unit)  # ruff: ignore[private-member-access]
        param._max = sc.scalar(param._max.value, unit=new_unit)  # ruff: ignore[private-member-access]

    @classmethod
    def _create_physical_constant(cls, name: str) -> DescriptorNumber:
        """
        Create a DescriptorNumber for a supported physical constant.

        The constant is a per-instance copy of the shared constant in ``easydynamics.utils``,
        converted to its default unit, so converting its unit on one component does not affect
        others.

        Parameters
        ----------
        name : str
            The constant's symbol name (a key of ``_PHYSICAL_CONSTANTS``).

        Returns
        -------
        DescriptorNumber
            The constant's value in its default unit.
        """
        source, default_unit = cls._PHYSICAL_CONSTANTS[name]
        constant = copy(source)
        constant.convert_unit(default_unit)
        return constant

    @property
    def constants(self) -> dict[str, DescriptorNumber]:
        """
        Get the physical constants used by the expression.

        Returns
        -------
        dict[str, DescriptorNumber]
            The automatically provided constants (e.g. ``hbar``, ``kb``) keyed by symbol name.
        """
        return dict(self._constants)

    @property
    def output_unit(self) -> str:
        """
        Get the unit of the evaluated expression, derived from x_unit and the parameter units.

        The unit is propagated through the expression tree: addition requires compatible units,
        multiplication and powers combine units, and functions like ``exp`` or ``sin`` require a
        dimensionless argument. Propagation raises ``sc.UnitError`` if the expression is not
        unit-consistent (e.g. adding meV to a dimensionless quantity, or taking ``exp`` of a
        quantity with a unit).

        Returns
        -------
        str
            The unit of the evaluated expression.
        """
        return str(self._canonical_unit(self._propagate_unit(self._expr)))

    def _canonical_unit(self, unit: sc.Unit) -> sc.Unit:
        """
        Replace a unit by an equal, cleaner-printing candidate where possible.

        Unit algebra accumulates tiny floating-point scale factors (e.g. ``meV * 1/meV`` prints as
        ``0.999999999999999889`` rather than ``dimensionless``); scipp's unit equality still
        matches, so map such units onto the common candidates for readable output.

        Parameters
        ----------
        unit : sc.Unit
            The unit to canonicalize.

        Returns
        -------
        sc.Unit
            An equal unit with a cleaner string representation, or the input unit unchanged.
        """
        candidate_strings = ['dimensionless', self.y_unit, self.x_unit]
        for candidate_string in candidate_strings:
            if candidate_string is None:
                continue
            candidate = sc.Unit(candidate_string)
            if unit == candidate:
                return candidate
        return unit

    def _propagate_unit(self, node: sp.Basic) -> sc.Unit:
        """
        Recursively derive the unit of a sympy expression node.

        Parameters
        ----------
        node : sp.Basic
            The sympy expression node to derive the unit of.

        Raises
        ------
        sc.UnitError
            If the node mixes incompatible units, applies a function that requires a dimensionless
            argument to a quantity with a unit, or raises a quantity with a unit to a symbolic or
            unsupported power.

        Returns
        -------
        sc.Unit
            The unit of the node.
        """
        dimensionless = sc.Unit('dimensionless')

        if isinstance(node, sp.Symbol):
            name = str(node)
            if name == 'x':
                return sc.Unit(self.x_unit) if self.x_unit is not None else dimensionless
            if name in self._constants:
                return sc.Unit(str(self._constants[name].unit))
            return sc.Unit(str(self._parameters[name].unit))

        if node.is_number:
            return dimensionless

        if isinstance(node, sp.Add):
            units = [self._propagate_unit(argument) for argument in node.args]
            for unit in units[1:]:
                try:
                    sc.to_unit(sc.scalar(1.0, unit=unit), units[0])
                except sc.UnitError as e:
                    raise sc.UnitError(
                        f'The expression adds quantities with incompatible units '
                        f'{units[0]} and {unit}.'
                    ) from e
            return units[0]

        if isinstance(node, sp.Mul):
            result = dimensionless
            for argument in node.args:
                result = result * self._propagate_unit(argument)
            return result

        if isinstance(node, sp.Pow):
            base_unit = self._propagate_unit(node.base)
            exponent_unit = self._propagate_unit(node.exp)
            if exponent_unit != dimensionless:
                raise sc.UnitError(f'An exponent must be dimensionless, got {exponent_unit}.')
            if base_unit == dimensionless:
                return dimensionless
            if not node.exp.is_number:
                raise sc.UnitError(
                    f'Cannot determine units: quantity with unit {base_unit} is raised to a '
                    f'symbolic power.'
                )
            exponent = float(node.exp)
            if exponent == int(exponent):
                return base_unit ** int(exponent)
            if 2 * exponent == int(2 * exponent):
                # Half-integer power: the square root of an integer power.
                return sc.sqrt(sc.scalar(1.0, unit=base_unit ** int(2 * exponent))).unit
            raise sc.UnitError(
                f'Cannot determine units: quantity with unit {base_unit} is raised to the '
                f'power {exponent}.'
            )

        if isinstance(node, self._UNIT_PRESERVING_FUNCS):
            return self._propagate_unit(node.args[0])

        if isinstance(node, self._DIMENSIONLESS_RESULT_FUNCS):
            self._propagate_unit(node.args[0])
            return dimensionless

        if isinstance(node, sp.Function):
            for argument in node.args:
                argument_unit = self._propagate_unit(argument)
                try:
                    sc.to_unit(sc.scalar(1.0, unit=argument_unit), dimensionless)
                except sc.UnitError as e:
                    raise sc.UnitError(
                        f'{node.func.__name__} requires a dimensionless argument, '
                        f'got {argument_unit}.'
                    ) from e
            return dimensionless

        raise sc.UnitError(
            f'Cannot determine units for expression node {node} of type {type(node).__name__}.'
        )

    def _warn_if_output_unit_mismatch(self) -> None:
        """
        Warn if the expression's output unit does not match y_unit.

        The check only runs when units are in use, i.e. when the expression uses physical constants
        or any parameter has a unit other than dimensionless. Unit-agnostic expressions (all
        parameters dimensionless) stay silent.
        """
        units_in_use = bool(self._constants) or any(
            str(parameter.unit) != 'dimensionless' for parameter in self._parameters.values()
        )
        if not units_in_use:
            return

        try:
            output_unit = sc.Unit(self.output_unit)
        except sc.UnitError as e:
            warnings.warn(
                f'The expression is not unit-consistent: {e}',
                UserWarning,
                stacklevel=3,
            )
            return

        y_unit = sc.Unit(self.y_unit) if self.y_unit is not None else sc.Unit('dimensionless')
        if output_unit != y_unit:
            warnings.warn(
                f'The expression evaluates to unit {output_unit}, which does not match '
                f'y_unit {y_unit}. The evaluated values are labelled with y_unit; adjust the '
                f'parameter units or y_unit to make them consistent.',
                UserWarning,
                stacklevel=3,
            )

    def convert_x_unit(self, _new_unit: str | sc.Unit) -> None:
        """
        Convert the x-axis unit of the expression.

        Unit conversion is not implemented for ExpressionComponent. Should it ever be needed, the
        viable path is dimensional analysis on the parameter units: for each parameter, determine
        the power n of the x-dimension in its unit and rescale its value by the x-unit conversion
        factor to the power n (the generalization of Polynomial's power-law rescaling). This only
        works when x_unit has a single unambiguous dimension.

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

    def convert_y_unit(self, _new_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit of the expression.

        Unit conversion is not implemented for ExpressionComponent. See convert_x_unit for the
        approach that would make it possible.

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

    def __getattr__(self, name: str) -> Parameter | DescriptorNumber:
        """
        Allow access to parameters and physical constants as attributes.

        Parameters
        ----------
        name : str
            Name of the parameter or constant to access.

        Raises
        ------
        AttributeError
            If the parameter does not exist.

        Returns
        -------
        Parameter | DescriptorNumber
            The parameter or constant with the given name.
        """
        if '_parameters' in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if '_constants' in self.__dict__ and name in self._constants:
            return self._constants[name]
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
        AttributeError
            If the name refers to a physical constant.
        TypeError
            If the value is not numeric.
        """
        if '_constants' in self.__dict__ and name in self._constants:
            raise AttributeError(f"'{name}' is a physical constant and cannot be set.")
        if '_parameters' in self.__dict__ and name in self._parameters:
            param = self._parameters[name]
            if not isinstance(value, Numeric):
                raise TypeError(f'{name} must be numeric')
            param.value = value
        else:
            super().__setattr__(name, value)

    def __dir__(self) -> list[str]:
        """
        Include parameter and constant names in dir() output for better IDE support.

        Returns
        -------
        list[str]
            List of attribute names, including parameters and constants.
        """
        return super().__dir__() + list(self._parameters.keys()) + list(self._constants.keys())

    def __repr__(self) -> str:
        """
        Return a string representation of the ExpressionComponent.

        Returns
        -------
        str
            String representation of the ExpressionComponent.
        """
        param_str = ', '.join(f'{k}={v.value}' for k, v in self._parameters.items())
        constants_str = ''
        if self._constants:
            constants_str = ',\n    constants={ ' + ', '.join(
                f'{k}={v.value} {v.unit}' for k, v in self._constants.items()
            )
            constants_str += ' }'
        return (
            f'{self.__class__.__name__}(name={self.name}, display_name={self.display_name}, '
            f'x_unit={self.x_unit}, y_unit={self.y_unit},\n'
            f"    expr='{self._expression_str}',\n"
            f'    parameters={{ {param_str} }}{constants_str} )'
        )
