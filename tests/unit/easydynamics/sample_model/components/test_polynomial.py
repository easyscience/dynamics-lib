# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.sample_model import Polynomial


class TestPolynomial:
    @pytest.fixture
    def polynomial(self):
        return Polynomial(
            name='PolynomialName',
            display_name='TestPolynomial',
            coefficients=[1.0, -2.0, 3.0],
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        polynomial = Polynomial()

        # EXPECT
        assert polynomial.display_name == 'Polynomial'
        assert polynomial.coefficients[0].value == pytest.approx(0.0)
        assert polynomial.x_unit == 'meV'
        assert polynomial.y_unit == 'dimensionless'

    def test_initialization(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        assert polynomial.display_name == 'TestPolynomial'
        assert polynomial.coefficients[0].value == pytest.approx(1.0)
        assert polynomial.coefficients[1].value == pytest.approx(-2.0)
        assert polynomial.coefficients[2].value == pytest.approx(3.0)

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'coefficients': 'invalid'},
                'coefficients must be ',
            ),
            (
                {'coefficients': [1.0, 'invalid', 3.0]},
                'Each coefficient must be ',
            ),
            (
                {'coefficients': [1.0, -2.0, 3.0], 'x_unit': 123},
                'unit must be ',
            ),
            (
                {'coefficients': [1.0, -2.0, 3.0], 'x_unit': 'meV', 'y_unit': 123},
                'unit must be ',
            ),
            (
                {'coefficients': None},
                'coefficients must be ',
            ),
            ({'coefficients': {}}, 'coefficients must be '),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=expected_message):
            Polynomial(display_name='TestPolynomial', **kwargs)

    def test_init_no_coefficients_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'At least one coefficient must be provided.'):
            Polynomial(display_name='TestPolynomial', coefficients=[])

    def test_negative_value_warns_in_evaluate(self):
        # WHEN THEN
        test_polynomial = Polynomial(display_name='TestPolynomial', coefficients=[-1.0])
        # EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            test_polynomial.evaluate(np.array([0.0, 1.0, 2.0]))

    def test_evaluate(self, polynomial: Polynomial):
        # WHEN
        x = np.array([0.0, 1.0, 2.0])

        # THEN
        result = polynomial.evaluate(x)

        # EXPECT
        expected_result = 1.0 - 2.0 * x + 3.0 * x**2
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_degree(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        assert polynomial.degree == 2

    def test_degree_setter_raises(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match='cannot be set directly'):
            polynomial.degree = 3

    @pytest.mark.parametrize(
        'values',
        [
            [2.0, 0.0, -1.0],  # all floats
            [
                Parameter('p0', 2.0),
                Parameter('p1', 0.0),
                Parameter('p2', -1.0),
            ],  # all Parameters
            [2.0, Parameter('p1', 0.0), -1.0],  # mixed numbers and Parameters
        ],
    )
    def test_set_coefficients(self, polynomial: Polynomial, values):
        """Test that coefficients can be updated from numeric values
        or Parameters."""
        # WHEN
        polynomial.coefficients = values

        # THEN EXPECT: Parameter values match the new inputs
        for i, val in enumerate(values):
            expected = val.value if isinstance(val, Parameter) else val
            assert np.isclose(polynomial.coefficients[i].value, expected)

    def test_set_coefficients_wrong_length_raises(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='Number of coefficients'):
            polynomial.coefficients = [1.0, 2.0]  # shorter list

    def test_set_coefficients_invalid_type_raises(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            polynomial.coefficients = [1.0, 'invalid', 3.0]

    @pytest.mark.parametrize(
        'invalid_coeffs, expected_message',
        [
            ([None, 2.0, 3.0], 'Each coefficient must be '),
            ([1.0, 2.0, 'invalid'], 'Each coefficient must be '),
            ('not a list', 'coefficients must be '),
        ],
    )
    def test_set_coefficients_raises(self, invalid_coeffs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=expected_message):
            polynomial = Polynomial(display_name='TestPolynomial', coefficients=[1.0, -2.0, 3.0])
            polynomial.coefficients = invalid_coeffs

    def test_coefficient_values(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        coeff_values = polynomial.coefficient_values()
        assert coeff_values == [1.0, -2.0, 3.0]

    def test_get_all_parameters(self, polynomial: Polynomial):
        # WHEN THEN
        params = polynomial.get_all_parameters()

        # EXPECT
        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'PolynomialName_c0',
            'PolynomialName_c1',
            'PolynomialName_c2',
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_convert_x_unit(self, polynomial: Polynomial):
        # WHEN
        polynomial.convert_x_unit('microeV')

        # THEN EXPECT
        assert polynomial._x_unit == 'microeV'
        assert np.isclose(polynomial.coefficients[0].value, 1.0)
        assert np.isclose(polynomial.coefficients[1].value, -2.0 * 1e-3)
        assert np.isclose(polynomial.coefficients[2].value, 3.0 * 1e-6)

    def test_convert_x_unit_raises_invalid_unit(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        with pytest.raises(Exception, match='unit must be '):
            polynomial.convert_x_unit(123)

    def test_copy(self, polynomial: Polynomial):
        # WHEN THEN
        polynomial_copy = copy(polynomial)

        # EXPECT
        assert polynomial_copy is not polynomial
        assert polynomial_copy.display_name == polynomial.display_name
        assert len(polynomial_copy.coefficients) == len(polynomial.coefficients)
        for original_coeff, copied_coeff in zip(
            polynomial.get_all_parameters(),
            polynomial_copy.get_all_parameters(),
            strict=True,
        ):
            assert copied_coeff.value == original_coeff.value
            assert copied_coeff.fixed == original_coeff.fixed

    def test_y_unit_custom(self):
        # WHEN THEN
        p = Polynomial(coefficients=[1.0, 2.0], x_unit='meV', y_unit='1/meV')
        # EXPECT
        assert p.y_unit == '1/meV'

    def test_y_unit_setter_raises(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            polynomial.y_unit = '1/meV'

    def test_convert_y_unit_scales_all_coefficients(self):
        # WHEN: polynomial with two non-zero coefficients and a physical y_unit
        p = Polynomial(coefficients=[3.0, 1.0], x_unit='meV', y_unit='meV^-1')
        x = np.array([2.0])
        val_before = p.evaluate(x)[0]  # 3.0 + 1.0*2.0 = 5.0 [meV^-1]

        # THEN
        p.convert_y_unit('eV^-1')

        # EXPECT: both coefficients rescaled by 1000 (1 meV^-1 = 1000 eV^-1)
        assert p.y_unit == 'eV^-1'
        assert np.isclose(p.coefficients[0].value, 3000.0)
        assert np.isclose(p.coefficients[1].value, 1000.0)
        assert np.isclose(p.evaluate(x)[0], val_before * 1000.0)

    def test_evaluate_scipp_output(self):
        # WHEN
        p = Polynomial(coefficients=[1.0, 2.0], x_unit='meV')
        x = np.linspace(-3, 3, 40)
        # THEN
        result = p.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 40
        np.testing.assert_allclose(result.values, p.evaluate(x, output='numpy'))

    def test_evaluate_scipp_output_with_y_unit(self):
        # WHEN
        p = Polynomial(coefficients=[1.0, 2.0], x_unit='meV', y_unit='1/meV')
        x = np.linspace(-3, 3, 40)
        # THEN
        result = p.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('1/meV')

    def test_repr(self, polynomial: Polynomial):
        # WHEN THEN
        repr_str = repr(polynomial)

        # EXPECT
        assert 'name = PolynomialName' in repr_str
        assert 'coefficients =' in repr_str

    def test_evaluate_with_scipp_x_different_compatible_unit(self):
        # WHEN: polynomial with x_unit='meV', coefficients [1.0, 1.0] → f(x) = 1 + x
        p = Polynomial(coefficients=[1.0, 1.0], x_unit='meV')
        # THEN: evaluate with x in eV (compatible unit) — triggers unit-rescaling branch
        x_eV = sc.array(dims=['x'], values=np.array([0.001, 0.002]), unit='eV')
        result = p.evaluate(x_eV)
        # EXPECT: 0.001 eV = 1 meV → f(1)=2, 0.002 eV = 2 meV → f(2)=3; state not mutated
        np.testing.assert_allclose(result, [2.0, 3.0], rtol=1e-5)
        assert p.x_unit == 'meV'

    def test_convert_y_unit_invalid_type_raises(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        with pytest.raises(UnitError, match='new_y_unit must be a string or a scipp unit'):
            polynomial.convert_y_unit(123)

    def test_convert_y_unit_rollback_on_failure(self):
        # WHEN
        p = Polynomial(coefficients=[1.0, 2.0], x_unit='meV')
        # THEN
        with pytest.raises(UnitError):
            p.convert_y_unit('K')
        # EXPECT: state rolled back
        assert p.y_unit == 'dimensionless'
        assert np.isclose(p.coefficients[0].value, 1.0)
        assert np.isclose(p.coefficients[1].value, 2.0)
