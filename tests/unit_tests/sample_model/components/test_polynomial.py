from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter

from easydynamics.sample_model import Polynomial


class TestPolynomial:
    @pytest.fixture
    def polynomial(self):
        return Polynomial(display_name="TestPolynomial", coefficients=[1.0, -2.0, 3.0])

    def test_init_no_inputs(self):
        # WHEN THEN
        polynomial = Polynomial()

        # EXPECT
        assert polynomial.display_name == "Polynomial"
        assert polynomial.coefficients[0].value == 0.0
        assert polynomial.unit == "meV"

    def test_initialization(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        assert polynomial.display_name == "TestPolynomial"
        assert polynomial.coefficients[0].value == 1.0
        assert polynomial.coefficients[1].value == -2.0
        assert polynomial.coefficients[2].value == 3.0

    @pytest.mark.parametrize(
        "kwargs, expected_message",
        [
            (
                {"coefficients": "invalid"},
                "coefficients must be ",
            ),
            (
                {"coefficients": [1.0, "invalid", 3.0]},
                "Each coefficient must be ",
            ),
            (
                {"coefficients": [1.0, -2.0, 3.0], "unit": 123},
                "unit must be ",
            ),
            (
                {"coefficients": None},
                "coefficients must be ",
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            Polynomial(display_name="TestPolynomial", **kwargs)

    def test_negative_value_warns_in_evaluate(self):
        # WHEN THEN
        test_polynomial = Polynomial(display_name="TestPolynomial", coefficients=[-1.0])
        # EXPECT
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
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

    @pytest.mark.parametrize(
        "values",
        [
            [2.0, 0.0, -1.0],  # all floats
            [
                Parameter("p0", 2.0),
                Parameter("p1", 0.0),
                Parameter("p2", -1.0),
            ],  # all Parameters
            [2.0, Parameter("p1", 0.0), -1.0],  # mixed numbers and Parameters
        ],
    )
    def test_set_coefficients(self, polynomial: Polynomial, values):
        """Test that coefficients can be updated from numeric values or Parameters."""
        # WHEN
        polynomial.coefficients = values

        # THEN EXPECT: Parameter values match the new inputs
        for i, val in enumerate(values):
            if isinstance(val, Parameter):
                expected = val.value
            else:
                expected = val
            assert np.isclose(polynomial.coefficients[i].value, expected)

    def test_set_coefficients_wrong_length_raises(self, polynomial: Polynomial):
        """Ensure that setting coefficients with mismatched length raises an error."""
        with pytest.raises(ValueError, match="Number of coefficients"):
            polynomial.coefficients = [1.0, 2.0]  # shorter list

    def test_set_coefficients_invalid_type_raises(self, polynomial: Polynomial):
        """Ensure that invalid coefficient types raise a TypeError."""
        with pytest.raises(TypeError):
            polynomial.coefficients = [1.0, "invalid", 3.0]

    @pytest.mark.parametrize(
        "invalid_coeffs, expected_message",
        [
            ([None, 2.0, 3.0], "Each coefficient must be "),
            ([1.0, 2.0, "invalid"], "Each coefficient must be "),
            ("not a list", "coefficients must be "),
        ],
    )
    def test_set_coefficients_raises(self, invalid_coeffs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            polynomial = Polynomial(
                display_name="TestPolynomial", coefficients=[1.0, -2.0, 3.0]
            )
            polynomial.coefficients = invalid_coeffs

    def test_coefficient_values(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        coeff_values = polynomial.coefficient_values
        assert coeff_values == [1.0, -2.0, 3.0]

    def test_get_all_parameters(self, polynomial: Polynomial):
        # WHEN THEN
        params = polynomial.get_all_parameters()

        # EXPECT
        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            "TestPolynomial_c0",
            "TestPolynomial_c1",
            "TestPolynomial_c2",
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_convert_unit(self, polynomial: Polynomial):
        # WHEN
        polynomial.convert_unit("microeV")

        # THEN EXPECT
        assert polynomial._unit == "microeV"
        assert np.isclose(polynomial.coefficients[0].value, 1.0)
        assert np.isclose(polynomial.coefficients[1].value, -2.0 * 1e-3)
        assert np.isclose(polynomial.coefficients[2].value, 3.0 * 1e-6)

    def test_copy(self, polynomial: Polynomial):
        # WHEN THEN
        polynomial_copy = copy(polynomial)

        # EXPECT
        assert polynomial_copy is not polynomial
        assert polynomial_copy.display_name == polynomial.display_name
        assert len(polynomial_copy.coefficients) == len(polynomial.coefficients)
        for original_coeff, copied_coeff in zip(
            polynomial.get_all_parameters(), polynomial_copy.get_all_parameters()
        ):
            assert copied_coeff.value == original_coeff.value
            assert copied_coeff.fixed == original_coeff.fixed

    def test_repr(self, polynomial: Polynomial):
        # WHEN THEN
        repr_str = repr(polynomial)

        # EXPECT
        assert "Polynomial" in repr_str
        assert "name = TestPolynomial" in repr_str
        assert "coefficients =" in repr_str
