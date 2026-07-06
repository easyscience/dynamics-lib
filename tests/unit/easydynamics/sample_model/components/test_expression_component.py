# SPDX-FileCopyrightText: 2026 EasyScience contributors
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model import ExpressionComponent
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian


class TestExpressionComponent:
    @pytest.fixture
    def expr(self):
        return ExpressionComponent(
            'A * exp(-(x - x0)**2 / (2*sigma**2))',
            parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
            x_unit='meV',
            display_name='TestExpression',
        )

    def test_init_valid(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        assert expr.display_name == 'TestExpression'
        assert expr.x_unit == 'meV'

        assert expr.A.value == pytest.approx(2.0)
        assert expr.x0.value == pytest.approx(0.5)
        assert expr.sigma.value == pytest.approx(0.6)

    def test_init_without_parameters(self):
        # WHEN THEN
        expr = ExpressionComponent('A * x', parameters=None)

        # EXPECT
        assert expr.A.value == pytest.approx(1.0)  # default

    def test_init_with_parameter(self):
        # WHEN THEN
        A = Parameter('A', 3.0)
        expr = ExpressionComponent('A * x', parameters={'A': A})

        # EXPECT
        assert expr.A.value == pytest.approx(3.0)

    def test_parameters_are_dimensionless_by_default(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        assert str(expr.A.unit) == 'dimensionless'
        assert str(expr.x0.unit) == 'dimensionless'
        assert str(expr.sigma.unit) == 'dimensionless'

    def test_init_with_parameter_units(self):
        # WHEN THEN
        expr = ExpressionComponent(
            'A * exp(-(x - x0)**2 / (2*sigma**2))',
            parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
            parameter_units={'x0': 'meV', 'sigma': 'meV'},
        )

        # EXPECT: the named parameters get their units, values unchanged; others stay
        # dimensionless
        assert str(expr.x0.unit) == 'meV'
        assert str(expr.sigma.unit) == 'meV'
        assert str(expr.A.unit) == 'dimensionless'
        assert expr.x0.value == pytest.approx(0.5)
        assert expr.sigma.value == pytest.approx(0.6)

    def test_init_parameter_units_overrides_parameter_instance_unit(self):
        # WHEN: a Parameter instance in eV, but parameter_units requests meV
        A = Parameter('A', 3.0, unit='eV')

        # THEN: A*x evaluates to meV**2, which mismatches the default dimensionless y_unit
        with pytest.warns(UserWarning, match='does not match'):
            expr = ExpressionComponent('A * x', parameters={'A': A}, parameter_units={'A': 'meV'})

        # EXPECT: the unit is relabelled without rescaling the value
        assert str(expr.A.unit) == 'meV'
        assert expr.A.value == pytest.approx(3.0)

    def test_init_parameter_units_unknown_name_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='unknown parameter'):
            ExpressionComponent('A * x', parameter_units={'B': 'meV'})

    def test_init_parameter_units_not_a_dict_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='parameter_units must be None or a dictionary'):
            ExpressionComponent('A * x', parameter_units='meV')

    def test_invalid_expression_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='Invalid expression'):
            ExpressionComponent('invalid syntax $$')

    def test_expression_without_x_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match="must contain 'x'"):
            ExpressionComponent('A + 1')

    def test_numpy_syntax_not_allowed(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='NumPy syntax'):
            ExpressionComponent('np.exp(x)')

    def test_invalid_function_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='Unsupported function'):
            ExpressionComponent('A * unknown_func(x)')

    @pytest.mark.parametrize(
        'parameters',
        [
            'not a dict',
            123,
            1.23,
        ],
    )
    def test_parameters_type_validation(self, parameters):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Parameters must be None or a dictionary'):
            ExpressionComponent('A * x', parameters=parameters)

    def test_parameter_value_type_validation(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='must be numeric'):
            ExpressionComponent('A * x', parameters={'A': 'invalid'})

    def test_evaluate(self, expr: ExpressionComponent):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = expr.evaluate(x)

        # EXPECT
        expected = 2.0 * np.exp(-((x - 0.5) ** 2) / (2 * 0.6**2))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_parameter_setter(self, expr: ExpressionComponent):
        # WHEN THEN
        expr.A = 3.0

        # EXPECT
        assert expr.A.value == pytest.approx(3.0)
        assert isinstance(expr.A, Parameter)

    def test_parameter_getter_invalid_name(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match="has no attribute 'invalid_param'"):
            _invalid_param = expr.invalid_param

    def test_parameter_setter_invalid(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='must be numeric'):
            expr.A = 'invalid'

    def test_get_all_variables(self, expr: ExpressionComponent):
        # WHEN
        params = expr.get_all_variables()

        # THEN
        assert all(isinstance(p, Parameter) for p in params)

        # EXPECT
        names = {p.name for p in params}
        assert names == {'A', 'x0', 'sigma'}

    def test_expression_property(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        assert expr.expression == 'A * exp(-(x - x0)**2 / (2*sigma**2))'

    def test_expression_is_read_only(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match='cannot be changed'):
            expr.expression = 'x'

    def test_set_unit_relabels_without_rescaling(self, expr: ExpressionComponent):
        # WHEN
        expr.sigma.min = 0.1
        expr.sigma.max = 10.0

        # THEN: relabelling only sigma leaves the expression unit-inconsistent, which warns
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr.set_unit('sigma', 'meV')

        # EXPECT: unit relabelled; value and bounds keep their numeric values
        assert str(expr.sigma.unit) == 'meV'
        assert expr.sigma.value == pytest.approx(0.6)
        assert expr.sigma.min == pytest.approx(0.1)
        assert expr.sigma.max == pytest.approx(10.0)

    def test_set_unit_accepts_scipp_unit(self, expr: ExpressionComponent):
        # WHEN THEN: relabelling only x0 leaves the expression unit-inconsistent, which warns
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr.set_unit('x0', sc.Unit('meV'))

        # EXPECT
        assert str(expr.x0.unit) == 'meV'

    def test_set_unit_leaves_other_parameters_untouched(self, expr: ExpressionComponent):
        # WHEN THEN
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr.set_unit('x0', 'meV')

        # EXPECT
        assert str(expr.A.unit) == 'dimensionless'
        assert str(expr.sigma.unit) == 'dimensionless'

    def test_set_unit_unknown_parameter_raises(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(KeyError, match="No parameter named 'B'"):
            expr.set_unit('B', 'meV')

    def test_set_unit_invalid_unit_raises(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='not a valid scipp unit'):
            expr.set_unit('A', 'not_a_unit')

    @pytest.mark.parametrize(
        'name, unit, expected_message',
        [
            (123, 'meV', 'name must be a string'),
            ('A', 123, 'unit must be a string or sc.Unit'),
        ],
        ids=['non_string_name', 'non_unit_unit'],
    )
    def test_set_unit_type_validation(
        self, expr: ExpressionComponent, name, unit, expected_message
    ):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=expected_message):
            expr.set_unit(name, unit)

    def test_set_unit_dependent_parameter_raises(self, expr: ExpressionComponent):
        # WHEN: make A dependent on sigma
        expr.A.make_dependent_on(
            dependency_expression='2 * sigma',
            dependency_map={'sigma': expr.sigma},
        )

        # THEN EXPECT
        with pytest.raises(AttributeError, match='dependent parameter'):
            expr.set_unit('A', 'meV')

    def test_parameter_units_survive_copy(self):
        # WHEN
        expr = ExpressionComponent(
            'A * x + B',
            parameters={'A': 2.0, 'B': 3.0},
            parameter_units={'A': '1/meV'},
        )

        # THEN
        expr_copy = copy(expr)

        # EXPECT: per-parameter units round-trip through to_dict/from_dict
        assert str(expr_copy.A.unit) == '1/meV'
        assert str(expr_copy.B.unit) == 'dimensionless'
        assert expr_copy.A.value == pytest.approx(2.0)

    def test_convert_x_unit_not_implemented(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(NotImplementedError, match='not implemented'):
            expr.convert_x_unit('microeV')

    def test_convert_y_unit_not_implemented(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(NotImplementedError, match='not implemented'):
            expr.convert_y_unit('1/meV')

    def test_evaluate_scipp_output(self, expr: ExpressionComponent):
        # WHEN
        x = np.linspace(-2, 2, 30)
        # THEN
        result = expr.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 30
        np.testing.assert_allclose(result.values, expr.evaluate(x, output='numpy'))

    def test_missing_parameter_defaults(self):
        # WHEN THEN
        expr = ExpressionComponent('A * x + B', parameters={'A': 2.0})

        # EXPECT
        assert expr.A.value == pytest.approx(2.0)
        assert expr.B.value == pytest.approx(1.0)  # default

    def test_dir_includes_parameters(self, expr: ExpressionComponent):
        # WHEN THEN
        attributes = dir(expr)

        # EXPECT
        assert 'A' in attributes
        assert 'x0' in attributes
        assert 'sigma' in attributes

    def test_repr(self, expr: ExpressionComponent):
        # WHEN THEN
        repr_str = repr(expr)

        # EXPECT
        assert 'ExpressionComponent' in repr_str
        assert 'expr=' in repr_str
        assert 'parameters=' in repr_str
        assert 'A=2.0' in repr_str

    def test_evaluate_scalar_input(self, expr: ExpressionComponent):
        # WHEN
        x = 0.5
        # THEN
        result = expr.evaluate(x)
        # EXPECT
        expected = 2.0 * np.exp(-((x - 0.5) ** 2) / (2 * 0.6**2))
        assert np.isclose(result, expected)

    def test_reserved_name_not_parameter(self):
        # WHEN
        expr = ExpressionComponent('x + A', parameters={'A': 2.0})
        # THEN
        params = expr.get_all_variables()
        names = {p.name for p in params}
        # EXPECT
        assert 'A' in names
        assert 'x' not in names  # x is reserved

    def test_copy(self, expr: ExpressionComponent):
        # WHEN THEN
        expr_copy = copy(expr)

        # EXPECT the copy is a new instance with the same properties
        assert expr_copy is not expr
        assert isinstance(expr_copy, ExpressionComponent)
        assert expr_copy.expression == expr.expression
        assert expr_copy.x_unit == expr.x_unit
        assert expr_copy.y_unit == expr.y_unit
        assert expr_copy.display_name == expr.display_name

        assert expr_copy.A.value == pytest.approx(expr.A.value)
        assert expr_copy.x0.value == pytest.approx(expr.x0.value)
        assert expr_copy.sigma.value == pytest.approx(expr.sigma.value)

    def test_erf(self):
        # WHEN
        expr = ExpressionComponent('erf(x)')
        x = np.array([-1.0, 0.0, 1.0])

        # THEN
        result = expr.evaluate(x)

        # EXPECT
        expected = np.array([-0.84270079, 0.0, 0.84270079])  # erf(-1), erf(0), erf(1)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_evaluate_raises_when_input_unit_differs_from_x_unit():
    # GIVEN an ExpressionComponent with x_unit meV
    expr = ExpressionComponent('A * x', parameters={'A': 2.0}, x_unit='meV')
    x = sc.array(dims=['x'], values=[1.0, 2.0], unit='ueV')
    # WHEN evaluating with x in a different unit THEN EXPECT a UnitError
    with pytest.raises(sc.UnitError, match=r'cannot auto-convert its parameters'):
        expr.evaluate(x)


GAUSSIAN_EXPRESSION = 'A / (sigma*sqrt(2*pi)) * exp(-(x - x0)**2 / (2*sigma**2))'
GAUSSIAN_UNITS = {'A': 'meV', 'x0': 'meV', 'sigma': 'meV'}


class TestExpressionComponentUnitCorrectness:
    """Compare unit-aware expressions against the built-in components."""

    @pytest.fixture
    def gaussian_expr(self):
        return ExpressionComponent(
            GAUSSIAN_EXPRESSION,
            parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
            parameter_units=GAUSSIAN_UNITS,
            x_unit='meV',
        )

    def test_matches_gaussian_component(self, gaussian_expr):
        # WHEN
        gaussian = Gaussian(area=2.0, center=0.5, width=0.6, x_unit='meV')
        x = np.linspace(-3, 3, 101)

        # THEN EXPECT: same values on a plain numpy grid
        np.testing.assert_allclose(gaussian_expr.evaluate(x), gaussian.evaluate(x))

    def test_matches_gaussian_component_scipp_input_and_output(self, gaussian_expr):
        # WHEN
        gaussian = Gaussian(area=2.0, center=0.5, width=0.6, x_unit='meV')
        x = sc.linspace('energy', -3.0, 3.0, 101, unit='meV')

        # THEN
        expr_result = gaussian_expr.evaluate(x, output='scipp')
        gaussian_result = gaussian.evaluate(x, output='scipp')

        # EXPECT: same values and the same output unit
        np.testing.assert_allclose(expr_result.values, gaussian_result.values)
        assert expr_result.unit == gaussian_result.unit

    def test_matches_gaussian_component_in_other_unit(self):
        # WHEN: the built-in Gaussian auto-converts its meV parameters to a ueV grid; the
        # expression is given the physically identical parameters expressed in ueV directly.
        gaussian = Gaussian(area=2.0, center=0.5, width=0.6, x_unit='meV')
        expr = ExpressionComponent(
            GAUSSIAN_EXPRESSION,
            parameters={'A': 2000.0, 'x0': 500.0, 'sigma': 600.0},
            parameter_units={'A': 'ueV', 'x0': 'ueV', 'sigma': 'ueV'},
            x_unit='ueV',
        )
        x = sc.linspace('energy', -3000.0, 3000.0, 101, unit='ueV')

        # THEN EXPECT: identical physical results (the Gaussian output is dimensionless, so no
        # y-scale factor is involved)
        np.testing.assert_allclose(expr.evaluate(x), gaussian.evaluate(x))

    def test_matches_lorentzian_component(self):
        # WHEN
        lorentzian = Lorentzian(area=2.0, center=0.5, width=0.3, x_unit='meV')
        # Note: 'gamma' would collide with sympy's gamma function, so use 'hwhm'
        expr = ExpressionComponent(
            'A / pi * hwhm / ((x - x0)**2 + hwhm**2)',
            parameters={'A': 2.0, 'x0': 0.5, 'hwhm': 0.3},
            parameter_units={'A': 'meV', 'x0': 'meV', 'hwhm': 'meV'},
            x_unit='meV',
        )
        x = np.linspace(-3, 3, 101)

        # THEN EXPECT
        np.testing.assert_allclose(expr.evaluate(x), lorentzian.evaluate(x))

    def test_consistent_units_do_not_warn(self):
        # WHEN THEN EXPECT: a fully unit-consistent expression constructs without warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ExpressionComponent(
                GAUSSIAN_EXPRESSION,
                parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
                parameter_units=GAUSSIAN_UNITS,
                x_unit='meV',
            )

    def test_unit_agnostic_expression_does_not_warn(self):
        # WHEN THEN EXPECT: without any parameter units the consistency check stays silent,
        # even though x_unit is meV and the parameters are dimensionless
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ExpressionComponent(
                GAUSSIAN_EXPRESSION,
                parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
                x_unit='meV',
            )


class TestExpressionComponentOutputUnit:
    def test_output_unit_gaussian_is_dimensionless(self):
        # WHEN: area in meV divided by sigma in meV
        expr = ExpressionComponent(
            GAUSSIAN_EXPRESSION,
            parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
            parameter_units=GAUSSIAN_UNITS,
            x_unit='meV',
        )

        # THEN EXPECT
        assert expr.output_unit == 'dimensionless'

    def test_output_unit_with_dimensionless_area_is_one_over_x(self):
        # WHEN: dimensionless amplitude over a meV width gives 1/meV
        with pytest.warns(UserWarning, match='does not match'):
            expr = ExpressionComponent(
                GAUSSIAN_EXPRESSION,
                parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
                parameter_units={'x0': 'meV', 'sigma': 'meV'},
                x_unit='meV',
            )

        # THEN EXPECT
        assert sc.Unit(expr.output_unit) == sc.Unit('1/meV')

    def test_output_unit_matching_y_unit_does_not_warn(self):
        # WHEN THEN EXPECT: declaring the matching y_unit silences the mismatch warning
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            expr = ExpressionComponent(
                GAUSSIAN_EXPRESSION,
                parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
                parameter_units={'x0': 'meV', 'sigma': 'meV'},
                x_unit='meV',
                y_unit='1/meV',
            )
        assert sc.Unit(expr.output_unit) == sc.Unit(expr.y_unit)

    def test_output_unit_preserved_by_abs(self):
        # WHEN
        with pytest.warns(UserWarning, match='does not match'):
            expr = ExpressionComponent(
                'abs(x - x0)', parameters={'x0': 0.5}, parameter_units={'x0': 'meV'}
            )

        # THEN EXPECT
        assert sc.Unit(expr.output_unit) == sc.Unit('meV')

    def test_output_unit_sign_is_dimensionless(self):
        # WHEN
        expr = ExpressionComponent(
            'A * sign(x - x0)', parameters={'A': 1.0, 'x0': 0.5}, parameter_units={'x0': 'meV'}
        )

        # THEN EXPECT
        assert expr.output_unit == 'dimensionless'

    def test_output_unit_sqrt_of_squared_quantity(self):
        # WHEN: sqrt(sigma**2) is a half-integer power of a quantity with units
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            expr = ExpressionComponent(
                'x / sqrt(2*pi*sigma**2) + offset',
                parameters={'sigma': 0.5, 'offset': 1.0},
                parameter_units={'sigma': 'meV'},
                x_unit='meV',
            )

        # THEN EXPECT: meV / sqrt(meV**2) + dimensionless = dimensionless
        assert expr.output_unit == 'dimensionless'

    def test_output_unit_incompatible_addition_raises(self):
        # WHEN
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr = ExpressionComponent(
                'x + A', parameters={'A': 1.0}, parameter_units={'A': 'K'}, x_unit='meV'
            )

        # THEN EXPECT
        with pytest.raises(sc.UnitError, match='incompatible units'):
            _ = expr.output_unit

    def test_output_unit_exp_of_dimensional_quantity_raises(self):
        # WHEN
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr = ExpressionComponent(
                'exp(x * A)', parameters={'A': 1.0}, parameter_units={'A': 'meV'}, x_unit='meV'
            )

        # THEN EXPECT
        with pytest.raises(sc.UnitError, match='dimensionless argument'):
            _ = expr.output_unit

    def test_output_unit_symbolic_power_of_dimensional_base_raises(self):
        # WHEN
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr = ExpressionComponent(
                'sigma**n * x',
                parameters={'sigma': 0.5, 'n': 2.0},
                parameter_units={'sigma': 'meV'},
            )

        # THEN EXPECT
        with pytest.raises(sc.UnitError, match='symbolic power'):
            _ = expr.output_unit

    def test_output_unit_exponent_with_unit_raises(self):
        # WHEN: the exponent itself carries a unit
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr = ExpressionComponent(
                'x**A', parameters={'A': 1.0}, parameter_units={'A': 'meV'}, x_unit='meV'
            )

        # THEN EXPECT
        with pytest.raises(sc.UnitError, match='exponent must be dimensionless'):
            _ = expr.output_unit

    def test_output_unit_quarter_power_of_dimensional_base_raises(self):
        # WHEN: only integer and half-integer powers of quantities with units are supported
        with pytest.warns(UserWarning, match='not unit-consistent'):
            expr = ExpressionComponent(
                'sigma**0.25 * x', parameters={'sigma': 0.5}, parameter_units={'sigma': 'meV'}
            )

        # THEN EXPECT
        with pytest.raises(sc.UnitError, match='raised to the power'):
            _ = expr.output_unit

    def test_output_unit_with_x_unit_none(self):
        # WHEN: without an x_unit, x is treated as dimensionless in the propagation
        with pytest.warns(UserWarning, match='does not match'):
            expr = ExpressionComponent(
                'A * x', parameters={'A': 1.0}, parameter_units={'A': 'meV'}, x_unit=None
            )

        # THEN EXPECT
        assert sc.Unit(expr.output_unit) == sc.Unit('meV')

    def test_evaluate_raises_for_genuinely_different_x_unit(self):
        # WHEN: x carries a different (but compatible) unit than x_unit
        expr = ExpressionComponent(
            'A * x', parameters={'A': 1.0}, parameter_units={'A': '1/meV'}, x_unit='meV'
        )
        x = sc.linspace('energy', -1.0, 1.0, 11, unit='ueV')

        # THEN EXPECT: ExpressionComponent cannot auto-convert, so evaluating with
        # wrongly-scaled x raises instead of silently producing wrong values
        with pytest.raises(sc.UnitError, match='cannot auto-convert'):
            expr.evaluate(x)

    def test_evaluate_does_not_warn_for_equivalent_unit_spelling(self):
        # WHEN: 'ueV' and scipp's canonical micro-sign spelling are the same unit
        expr = ExpressionComponent(
            'A * x', parameters={'A': 1.0}, parameter_units={'A': '1/ueV'}, x_unit='ueV'
        )
        x = sc.linspace('energy', -1.0, 1.0, 11, unit='\u00b5eV')

        # THEN EXPECT: no spurious warning from the differing spellings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            expr.evaluate(x)

    def test_set_unit_warns_when_breaking_consistency(self):
        # WHEN: a consistent expression
        expr = ExpressionComponent(
            'A * (x - x0)',
            parameters={'A': 1.0, 'x0': 0.5},
            parameter_units={'A': '1/meV', 'x0': 'meV'},
            x_unit='meV',
        )

        # THEN EXPECT: relabelling A breaks the output unit
        with pytest.warns(UserWarning, match='does not match'):
            expr.set_unit('A', '1/ueV')


class TestExpressionComponentPhysicalConstants:
    def test_kb_constant_value_and_unit(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN EXPECT: Boltzmann constant in meV/K, a read-only DescriptorNumber
        assert expr.kb.value == pytest.approx(0.08617333, rel=1e-6)
        assert sc.Unit(str(expr.kb.unit)) == sc.Unit('meV/K')
        assert isinstance(expr.kb, DescriptorNumber)
        assert not isinstance(expr.kb, Parameter)

    def test_hbar_constant_value_and_unit(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-(x * tau / hbar)**2)', parameters={'tau': 1.0}, parameter_units={'tau': 'ps'}
        )

        # THEN EXPECT: hbar in meV*s, a read-only DescriptorNumber
        assert expr.hbar.value == pytest.approx(6.582119569509065e-13, rel=1e-6)
        assert sc.Unit(str(expr.hbar.unit)) == sc.Unit('meV*s')
        assert isinstance(expr.hbar, DescriptorNumber)
        assert not isinstance(expr.hbar, Parameter)

    def test_boltzmann_factor_value(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )
        x = np.array([0.0, 10.0, 25.0])

        # THEN
        result = expr.evaluate(x)

        # EXPECT
        expected = np.exp(-x / (0.08617333262145177 * 300.0))
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_constants_are_unit_consistent(self):
        # WHEN THEN EXPECT: x/(kb*T) is dimensionless, so no warning is raised
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            expr = ExpressionComponent(
                'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
            )
        assert expr.output_unit == 'dimensionless'

    def test_constant_with_dimensionless_temperature_warns(self):
        # WHEN THEN EXPECT: forgetting the unit on T makes x/(kb*T) carry kelvins
        with pytest.warns(UserWarning, match='not unit-consistent'):
            ExpressionComponent('exp(-x / (kb * T))', parameters={'T': 300.0})

    def test_constants_property(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN EXPECT
        assert set(expr.constants) == {'kb'}

    def test_constant_not_in_get_all_variables(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN EXPECT: constants are not fittable variables
        names = {p.name for p in expr.get_all_variables()}
        assert names == {'T'}

    def test_constant_cannot_be_set(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN EXPECT
        with pytest.raises(AttributeError, match='physical constant'):
            expr.kb = 5.0

    def test_set_unit_on_constant_raises(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN EXPECT
        with pytest.raises(AttributeError, match='physical constant'):
            expr.set_unit('kb', 'meV/K')

    def test_parameter_units_for_constant_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='physical constant'):
            ExpressionComponent(
                'exp(-x / (kb * T))',
                parameters={'T': 300.0},
                parameter_units={'T': 'K', 'kb': 'meV/K'},
            )

    def test_constant_convert_unit_rescales(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN
        expr.kb.convert_unit('eV/K')

        # EXPECT: the value is rescaled with the unit
        assert expr.kb.value == pytest.approx(8.617333e-05, rel=1e-6)

    def test_user_parameter_overrides_constant(self):
        # WHEN: the user explicitly provides 'kb' as a parameter
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            expr = ExpressionComponent('exp(-x / (kb * T))', parameters={'kb': 2.0, 'T': 300.0})

        # THEN EXPECT: it is an ordinary (fittable, dimensionless) parameter
        assert not expr.constants
        assert expr.kb.value == pytest.approx(2.0)
        assert expr.kb.fixed is False
        names = {p.name for p in expr.get_all_variables()}
        assert names == {'kb', 'T'}

    def test_constant_in_dir(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN EXPECT
        assert 'kb' in dir(expr)

    def test_constant_in_repr(self):
        # WHEN
        expr = ExpressionComponent(
            'exp(-x / (kb * T))', parameters={'T': 300.0}, parameter_units={'T': 'K'}
        )

        # THEN EXPECT
        assert 'kb' in repr(expr)
