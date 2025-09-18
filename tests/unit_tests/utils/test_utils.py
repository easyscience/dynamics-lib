import numpy as np
import pytest

from easydynamics.utils import detailed_balance_factor  # adjust path if needed

class TestDetailedBalanceFactor:

    def test_zero_temperature(self):
        # When
        temperature = 0
        omega = np.array([-1.0, 0.0, 1.0])
        # Then
        result = detailed_balance_factor(omega, temperature)
        # Expect
        expected = np.maximum(omega, 0.0)
        np.testing.assert_array_equal(result, expected)

    def test_negative_temperature_raises(self):
        # When Then Expect
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            detailed_balance_factor(1.0, -10)

    def test_array_input(self):
        # When
        omega = np.linspace(-5, 5, 100)
        T = 300
        # Then
        result = detailed_balance_factor(omega, T)
        # Expect
        assert isinstance(result, np.ndarray)
        assert result.shape == omega.shape

    def test_small_omega_limit(self):
        # When
        T = 300
        omega = np.array([1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
        # Then
        result = detailed_balance_factor(omega, T)
        # Expect
        expected = np.full(5, 8.617333262e-2 * T)  # k_B * T
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_omega_behavior(self):
        # When
        omega = np.linspace(1e2, 1e3, 5)
        T = 1
        # Then
        result = detailed_balance_factor(omega, T)
        # Expect
        np.testing.assert_allclose(result, omega, rtol=1e-2)

    def test_detailed_balance_is_fulfilled(self):
        # When
        T = 10
        omega = np.linspace(0.01,100,101)
        # Then
        detailed_balance_positive= detailed_balance_factor(omega, T)
        detailed_balance_negative = detailed_balance_factor(-omega, T)
        ratio = detailed_balance_positive / detailed_balance_negative 

        # Expect
        expected_ratio = np.exp(omega / (8.617333262e-2 * T))  # k_B in meV/K
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-5)