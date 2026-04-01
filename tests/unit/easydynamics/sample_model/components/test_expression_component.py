# SPDX-FileCopyrightText: 2026 EasyScience contributors
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from easyscience.variable import Parameter

from easydynamics.sample_model import ExpressionComponent


class TestExpressionComponent:
    @pytest.fixture
    def expr(self):
        return ExpressionComponent(
            'A * exp(-(x - x0)**2 / (2*sigma**2))',
            parameters={'A': 2.0, 'x0': 0.5, 'sigma': 0.6},
            unit='meV',
            display_name='TestExpression',
        )

    def test_init_valid(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        assert expr.display_name == 'TestExpression'
        assert expr.unit == 'meV'

        assert expr.A.value == 2.0
        assert expr.x0.value == 0.5
        assert expr.sigma.value == 0.6

    def test_init_without_parameters(self):
        # WHEN THEN
        expr = ExpressionComponent('A * x', parameters=None)

        # EXPECT
        assert expr.A.value == 1.0  # default

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
        assert expr.A.value == 3.0
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

    def test_convert_unit_not_implemented(self, expr: ExpressionComponent):
        # WHEN THEN EXPECT
        with pytest.raises(NotImplementedError, match='not implemented'):
            expr.convert_unit('microeV')

    def test_missing_parameter_defaults(self):
        # WHEN THEN
        expr = ExpressionComponent('A * x + B', parameters={'A': 2.0})

        # EXPECT
        assert expr.A.value == 2.0
        assert expr.B.value == 1.0  # default

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
        result = expr.evaluate(x)

        # THEN
        expected = 2.0 * np.exp(-((x - 0.5) ** 2) / (2 * 0.6**2))
        assert np.isclose(result, expected)

    def test_reserved_name_not_parameter(self):
        # WHEN
        expr = ExpressionComponent('x + A', parameters={'A': 2.0})

        # THEN
        params = expr.get_all_variables()
        names = {p.name for p in params}

        assert 'A' in names
        assert 'x' not in names  # x is reserved
