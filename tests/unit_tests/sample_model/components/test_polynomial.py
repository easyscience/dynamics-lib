import pytest

import numpy as np
import scipp as sc
from scipp import UnitError

from easydynamics.sample_model import Polynomial

from easyscience.variable import Parameter


class TestPolynomial:
    @pytest.fixture
    def polynomial(self):
        return Polynomial(name="TestPolynomial", coefficients=[1.0, -2.0, 3.0])

    def test_initialization(self, polynomial: Polynomial):
        assert polynomial.name == "TestPolynomial"
        assert polynomial.coefficients[0].value == 1.0
        assert polynomial.coefficients[1].value == -2.0
        assert polynomial.coefficients[2].value == 3.0

    def test_input_type_validation_raises(self):
        with pytest.raises(
            TypeError, match="coefficients must be a list, tuple or ndarray of floats."
        ):
            Polynomial(name="TestPolynomial", coefficients="invalid")

        with pytest.raises(TypeError, match="All coefficients must be numbers."):
            Polynomial(name="TestPolynomial", coefficients=[1.0, "invalid", 3.0])

        with pytest.raises(TypeError, match="unit must be a string or a scipp unit"):
            Polynomial(name="TestPolynomial", coefficients=[1.0, -2.0, 3.0], unit=123)

        with pytest.raises(
            ValueError, match="At least one coefficient must be provided"
        ):
            Polynomial(name="TestPolynomial", coefficients=[])

    def test_negative_value_warns_in_evaluate(self):
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

    def test_evaluate_scipp_array(self, polynomial: Polynomial):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 1.0, 2.0], unit="meV")

        # THEN
        result = polynomial.evaluate(x)

        # EXPECT
        expected_result = 1.0 - 2.0 * x.values + 3.0 * x.values**2
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit_error(self, polynomial: Polynomial):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 1.0, 2.0], unit="microeV")

        # THEN EXPECT
        with pytest.raises(
            UnitError,
            match="Change the unit of the Polynomial and try again",
        ):
            polynomial.evaluate(x)

    def test_degree(self, polynomial: Polynomial):
        # WHEN THEN EXPECT
        assert polynomial.degree() == 2

    def test_get_parameters(self, polynomial: Polynomial):
        # WHEN THEN
        params = polynomial.get_parameters()

        # EXPECT
        assert len(params) == 3
        assert params[0].name == "TestPolynomial_c0"
        assert params[1].name == "TestPolynomial_c1"
        assert params[2].name == "TestPolynomial_c2"
        assert all(isinstance(param, Parameter) for param in params)

    def test_convert_unit_raises_for_polynomial(self, polynomial):
        # WHEN THEN EXPECT
        with pytest.raises(
            NotImplementedError,
            match="Unit conversion is not implemented for Polynomial components. The automatic unit converter does not like powers of units.",
        ):
            polynomial.convert_unit("eV")

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
