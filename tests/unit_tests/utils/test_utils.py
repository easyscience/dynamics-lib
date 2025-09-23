import numpy as np
import pytest
import scipp as sc
from easyscience import Parameter

from easydynamics.utils import detailed_balance_factor  

kB_meV_per_K = 8.617333262145e-2  # meV / K

class TestDetailedBalanceFactor:

    def test_zero_temperature(self):
        # When
        temperature = 0
        energy = np.array([-1.0, 0.0, 1.0])
        # Then
        result = detailed_balance_factor(energy, temperature, divide_by_T=False)
        # Expect
        expected = np.maximum(energy, 0.0)
        np.testing.assert_array_equal(result, expected)

    def test_zero_temperature_divide_by_T_error(self):
        # When
        temperature = 0
        energy = np.array([-1.0, 0.0, 1.0])
        # Then Expect
        with pytest.raises(ValueError, match="Cannot divide by T when T=0"):
            detailed_balance_factor(energy, temperature, divide_by_T=True)

    def test_negative_temperature_raises(self):
        # When Then Expect
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            detailed_balance_factor(1.0, -10)

    def test_array_input(self):
        # When
        energy = np.linspace(1, 5, 100)
        T = 300
        # Then
        result = detailed_balance_factor(energy, T)
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * T))) / (kB_meV_per_K * T)  
        assert isinstance(result, np.ndarray)
        assert result.shape == energy.shape
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_input(self):
        # When
        energy = 2.0
        T = 100
        # Then
        result = detailed_balance_factor(energy, T)
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * T))) / (kB_meV_per_K * T) 

        assert isinstance(result, float)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_list_input(self):
        # When
        energy = [1.0, 2.0, 3.0]
        T = 50
        # Then
        result = detailed_balance_factor(energy, T)
        # Expect
        expected = np.array(energy) / (1 - np.exp(-np.array(energy) / (kB_meV_per_K * T))) / (kB_meV_per_K * T)  
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scipp_variable_input(self):
        # When
        energy = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='meV')
        T = sc.scalar(value=100, unit='K')
        # Then
        result = detailed_balance_factor(energy, T)
        # Expect
        expected_values = np.array([1.0, 2.0, 3.0]) / (1 - np.exp(-np.array([1.0, 2.0, 3.0]) / (kB_meV_per_K * 100))) / (kB_meV_per_K * 100)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected_values, rtol=1e-5)

    def test_parameter_temperature(self):
        # When
        energy = np.array([1.0, 2.0, 3.0])
        T_param = Parameter(name='T', value=150, unit='K')
        # Then
        result = detailed_balance_factor(energy, T_param)
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * 150))) / (kB_meV_per_K * 150)

        assert isinstance(result, np.ndarray)
        assert result.shape == energy.shape
        np.testing.assert_allclose(result, expected, rtol=1e-5)        

    def test_small_energy_limit(self):
        # When
        T = 300
        energy = np.array([1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
        # Then
        result = detailed_balance_factor(energy=energy, temperature=T,divide_by_T=False)
        # Expect
        expected = np.full(5, kB_meV_per_K * T)  
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_energy_limit(self):
        # When
        energy = np.linspace(1e2, 1e3, 5)
        T = 1
        # Then
        result = detailed_balance_factor(energy=energy, temperature=T,divide_by_T=False)
        # Expect
        np.testing.assert_allclose(result, energy, rtol=1e-2)

    def test_intermediate_energy(self):
        # When
        energy = np.linspace(1,10,100)
        T = 100
        # Then
        result = detailed_balance_factor(energy=energy, temperature=T,divide_by_T=False)
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * T)))  
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize("divide_by_T", [True, False])
    def test_detailed_balance_is_fulfilled(self, divide_by_T):
        # When
        T = 10
        energy = np.linspace(0.01,100,101)
        # Then
        detailed_balance_positive= detailed_balance_factor(energy=energy, temperature=T, divide_by_T=divide_by_T)
        detailed_balance_negative = detailed_balance_factor(energy=-energy, temperature=T, divide_by_T=divide_by_T)
        ratio = detailed_balance_positive / detailed_balance_negative 

        # Expect
        expected_ratio = np.exp(energy / (kB_meV_per_K * T))
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-5)

    def test_energy_unit(self):
        # When
        energy = np.linspace(1e3,10*1e3,100)
        energy_unit='microeV'
        T = 100/1000  
        # Then
        result = detailed_balance_factor(energy=energy, temperature=T,divide_by_T=False, energy_unit=energy_unit)
        # Expect
        expected = energy / (1 - np.exp(-energy/1000 / (kB_meV_per_K * T)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_energy_unit_warning(self):
        # When
        energy = sc.linspace('energy', 1e3, 100e3, num=100, unit='microeV')
        energy_unit='meV'
        T = 100  

        # Then
        with pytest.warns(UserWarning, match="Input energy has unit µeV, but energy_unit was set to meV. Using µeV."):
            result = detailed_balance_factor(energy=energy, temperature=T,divide_by_T=False, energy_unit=energy_unit)
        # Expect
        expected = energy.values / (1 - np.exp(-energy.values/1000 / (kB_meV_per_K * T)))
        np.testing.assert_allclose(result, expected, rtol=1e-5) 

    def test_temperature_unit(self):
        # When
        energy = np.linspace(1,10,100)
        temperature = 100*1000
        temperature_unit = 'mK'
        # Then
        result = detailed_balance_factor(energy=energy, temperature=temperature, temperature_unit=temperature_unit,divide_by_T=False)
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * temperature/1000)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    
