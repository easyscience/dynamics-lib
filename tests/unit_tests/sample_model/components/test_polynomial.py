import numpy as np
import pytest
from easyscience.variable import Parameter

from easydynamics.sample_model import Polynomial


class TestPolynomial:
    @pytest.fixture
    def polynomial(self):
        return Polynomial(name="TestPolynomial", coefficients=[1.0, -2.0, 3.0])

    def test_initialization(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        assert polynomial.name == "TestPolynomial"
        assert polynomial.coefficients[0].value == 1.0
        assert polynomial.coefficients[1].value == -2.0
        assert polynomial.coefficients[2].value == 3.0

    @pytest.mark.parametrize(
        "kwargs, expected_message",
        [
            (
                {"coefficients": "invalid"},
                "coefficients must be a list or ndarray of floats",
            ),
            (
                {"coefficients": [1.0, "invalid", 3.0]},
                "All coefficients must be numbers.",
            ),
            (
                {"coefficients": [1.0, -2.0, 3.0], "unit": 123},
                "unit must be a string or a scipp unit",
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            Polynomial(name="TestPolynomial", **kwargs)

    def test_no_coefficients_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError, match="At least one coefficient must be provided"
        ):
            Polynomial(name="TestPolynomial", coefficients=[])

    def test_negative_value_warns_in_evaluate(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            test_polynomial = Polynomial(name="TestPolynomial", coefficients=[-1.0])
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
        assert polynomial.degree() == 2

    def test_get_parameters(self, polynomial: Polynomial):
        # WHEN THEN
        params = polynomial.get_parameters()

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
        polynomial_copy = polynomial.copy()

        # EXPECT
        assert polynomial_copy is not polynomial
        assert polynomial_copy.name == polynomial.name
        assert len(polynomial_copy.coefficients) == len(polynomial.coefficients)
        for original_coeff, copied_coeff in zip(
            polynomial.coefficients, polynomial_copy.coefficients
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
