# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience import Parameter
from scipp import UnitError
from scipp.constants import Boltzmann as kB

from easydynamics.utils import detailed_balance_factor

kB_meV_per_K = sc.to_unit(kB, 'meV/K').value


class TestDetailedBalanceFactor:
    # Input validation tests
    def test_energy_unit_not_string_error(self):
        # When
        energy = 2.0
        T = 100
        energy_unit = 5
        # Then Expect
        with pytest.raises(TypeError, match=r'energy_unit must be a string.'):
            detailed_balance_factor(energy, T, energy_unit=energy_unit)

    @pytest.mark.parametrize('temperature_unit', [5, 5.0, {}, []])
    def test_temperature_unit_not_string_error(self, temperature_unit):
        # When
        energy = 2.0
        T = 100
        # Then Expect
        with pytest.raises(TypeError, match=r'temperature_unit must be a string.'):
            detailed_balance_factor(energy, T, temperature_unit=temperature_unit)

    def test_divide_by_temperature_not_bool_error(self):
        # When
        energy = 2.0
        T = 100
        divide_by_temperature = 'yes'
        # Then Expect
        with pytest.raises(TypeError, match=r'divide_by_temperature must be True or False.'):
            detailed_balance_factor(energy, T, divide_by_temperature=divide_by_temperature)

    @pytest.mark.parametrize(
        'energy',
        [
            2.0,
            [1.0, 2.0, 3.0],
            np.array([2.0]),
            np.linspace(1, 5, 10),
            np.linspace(-5, -1, 10),
        ],
        ids=[
            'single_value',
            'list',
            'np_array_single',
            'np_array_multi',
            'np_array_negative',
        ],
    )
    def test_energy_inputs(self, energy):
        # When
        T = 100
        # Then
        result = detailed_balance_factor(energy, T)
        # Expect
        if isinstance(energy, (np.ndarray)):
            energy_array = energy
        elif isinstance(energy, list):
            energy_array = np.array(energy)
        else:
            energy_array = np.array([energy])
        expected = (
            energy_array / (1 - np.exp(-energy_array / (kB_meV_per_K * T))) / (kB_meV_per_K * T)
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == energy_array.shape
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scipp_variable_input(self):
        # When
        energy = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='meV')
        T = sc.scalar(value=100, unit='K')
        # Then
        result = detailed_balance_factor(energy, T)
        # Expect
        expected_values = (
            np.array([1.0, 2.0, 3.0])
            / (1 - np.exp(-np.array([1.0, 2.0, 3.0]) / (kB_meV_per_K * 100)))
            / (kB_meV_per_K * 100)
        )

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

    # Physical edge cases
    def test_zero_temperature(self):
        # When
        temperature = 0
        energy = np.array([-1.0, 0.0, 1.0])
        # Then
        result = detailed_balance_factor(energy, temperature, divide_by_temperature=False)
        # Expect
        expected = np.maximum(energy, 0.0)
        np.testing.assert_array_equal(result, expected)

    def test_zero_temperature_divide_by_T_error(self):
        # When
        temperature = 0
        energy = np.array([-1.0, 0.0, 1.0])
        # Then Expect
        with pytest.raises(ZeroDivisionError, match='Cannot divide by T when T = 0'):
            detailed_balance_factor(energy, temperature, divide_by_temperature=True)

    def test_zero_temperature_single_value(self):
        # When
        temperature = 0
        energy = 2.0
        # Then
        result = detailed_balance_factor(energy, temperature, divide_by_temperature=False)
        # Expect
        expected = 2.0
        assert result == expected

    def test_negative_temperature_raises(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Temperature must be non-negative'):
            detailed_balance_factor(1.0, -10)

    # Numerical tests
    def test_small_energy_limit(self):
        # When
        T = 300
        energy = np.array([1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
        # Then
        result = detailed_balance_factor(energy=energy, temperature=T, divide_by_temperature=False)
        # Expect
        x = energy / (kB_meV_per_K * T)
        expected = (1 + x / 2 + x**2 / 12) * (kB_meV_per_K * T)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_energy_limit(self):
        # When
        energy = np.linspace(1e2, 1e3, 5)
        T = 1
        # Then
        result = detailed_balance_factor(energy=energy, temperature=T, divide_by_temperature=False)
        # Expect
        np.testing.assert_allclose(result, energy, atol=1e-10)

    def test_intermediate_energy(self):
        # When
        energy = np.linspace(1, 10, 100)
        T = 100
        # Then
        result = detailed_balance_factor(energy=energy, temperature=T, divide_by_temperature=False)
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * T)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize('divide_by_T', [True, False])
    def test_detailed_balance_is_fulfilled(self, divide_by_T):
        # Detailed balance means DBF(E)/DBF(-E) = exp(E/(kB*T))
        # When
        T = 10
        energy = np.linspace(0.01, 100, 101)
        # Then
        detailed_balance_positive = detailed_balance_factor(
            energy=energy, temperature=T, divide_by_temperature=divide_by_T
        )
        detailed_balance_negative = detailed_balance_factor(
            energy=-energy, temperature=T, divide_by_temperature=divide_by_T
        )
        ratio = detailed_balance_positive / detailed_balance_negative

        # Expect
        expected_ratio = np.exp(energy / (kB_meV_per_K * T))
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-5)

    @pytest.mark.parametrize(
        'energy_unit', ['microeV', sc.Unit('microeV')], ids=['str', 'scipp.Unit']
    )
    def test_energy_unit(self, energy_unit):
        # When
        energy = np.linspace(1e3, 10 * 1e3, 100)
        T = 100
        # Then
        result = detailed_balance_factor(
            energy=energy,
            temperature=T,
            divide_by_temperature=False,
            energy_unit=energy_unit,
        )
        # Expect
        expected = energy / (1 - np.exp(-energy / 1000 / (kB_meV_per_K * T)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_energy_unit_warning(self):
        # When
        energy = sc.linspace('energy', 1e3, 10 * 1e3, num=100, unit='microeV')
        energy_unit = 'meV'
        T = 100

        # Then
        with pytest.warns(
            UserWarning,
            match='Input energy has unit [µμ]eV, but energy_unit was set to meV. Using [µμ]eV.',
        ):
            result = detailed_balance_factor(
                energy=energy,
                temperature=T,
                divide_by_temperature=False,
                energy_unit=energy_unit,
            )
        # Expect
        expected = energy.values / (1 - np.exp(-energy.values / 1000 / (kB_meV_per_K * T)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.parametrize('temperature_unit', ['mK', sc.Unit('mK')], ids=['str', 'scipp.Unit'])
    def test_temperature_unit(self, temperature_unit):
        # When
        energy = np.linspace(1, 10, 100)
        temperature = 100 * 1000
        temperature_unit = 'mK'
        # Then
        result = detailed_balance_factor(
            energy=energy,
            temperature=temperature,
            temperature_unit=temperature_unit,
            divide_by_temperature=False,
        )
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * temperature / 1000)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_temperature_unit_warning(self):
        # When
        energy = np.linspace(1, 10, 100)
        temperature = sc.scalar(value=100, unit='mK')
        temperature_unit = 'K'
        # Then
        with pytest.warns(
            UserWarning,
            match='Input temperature has unit mK, but temperature_unit was set to K. Using mK.',
        ):
            result = detailed_balance_factor(
                energy=energy,
                temperature=temperature,
                temperature_unit=temperature_unit,
                divide_by_temperature=False,
            )
        # Expect
        expected = energy / (1 - np.exp(-energy / (kB_meV_per_K * 0.1)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_incompatible_energy_unit_raises(self):
        # When
        energy = 2.0
        T = 100
        energy_unit = 'm'
        temperature_unit = 'K'

        # Then Expect
        with pytest.raises(
            UnitError,
            match='The unit of energy is wrong',
        ):
            detailed_balance_factor(
                energy,
                T,
                energy_unit=energy_unit,
                temperature_unit=temperature_unit,
            )

    def test_incompatible_temperature_unit_raises(self):
        # When
        energy = 2.0
        T = 100
        energy_unit = 'meV'
        temperature_unit = 's'

        # Then Expect
        with pytest.raises(
            UnitError,
            match='The unit of temperature is wrong',
        ):
            detailed_balance_factor(
                energy,
                T,
                energy_unit=energy_unit,
                temperature_unit=temperature_unit,
            )


class TestConvertToScippVariable:
    """Tests for _convert_to_scipp_variable internal helper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from easydynamics.utils.detailed_balance import _convert_to_scipp_variable

        self._fn = _convert_to_scipp_variable

    @pytest.mark.parametrize(
        'name, expected_match',
        [
            ('energy', 'energy must be a number'),
            ('temperature', 'temperature must be a number'),
        ],
        ids=['energy_name', 'other_name'],
    )
    def test_invalid_type_raises_type_error(self, name, expected_match):
        with pytest.raises(TypeError, match=expected_match):
            self._fn({'invalid': 'type'}, name=name, unit='meV')

    def test_invalid_unit_scalar_raises_unit_error(self):
        from scipp import UnitError as ScippUnitError

        with pytest.raises(ScippUnitError, match='Invalid unit string'):
            self._fn(1.0, name='energy', unit='not_a_real_unit_xyz')

    def test_invalid_unit_array_raises_unit_error(self):
        from scipp import UnitError as ScippUnitError

        with pytest.raises(ScippUnitError, match='Invalid unit string'):
            self._fn([1.0, 2.0], name='energy', unit='not_a_real_unit_xyz')
