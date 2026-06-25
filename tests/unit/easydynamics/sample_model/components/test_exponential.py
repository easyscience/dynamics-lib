# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.sample_model import Exponential


class TestExponential:
    @pytest.fixture
    def exponential(self):
        return Exponential(
            name='ExponentialName',
            display_name='TestExponential',
            amplitude=2.0,
            center=0.5,
            rate=1.2,
            x_unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN
        exponential = Exponential()

        # THEN EXPECT
        assert exponential.display_name == 'Exponential'
        assert exponential.amplitude.value == pytest.approx(1.0)
        assert exponential.center.value == pytest.approx(0.0)
        assert exponential.rate.value == pytest.approx(1.0)
        assert exponential.x_unit == 'meV'

    def test_initialization(self, exponential: Exponential):
        # WHEN THEN EXPECT
        assert exponential.display_name == 'TestExponential'
        assert exponential.amplitude.value == pytest.approx(2.0)
        assert exponential.center.value == pytest.approx(0.5)
        assert exponential.rate.value == pytest.approx(1.2)
        assert exponential.x_unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'amplitude': 'invalid', 'center': 0.5, 'rate': 1.0, 'x_unit': 'meV'},
                'amplitude must be a number',
            ),
            (
                {'amplitude': 2.0, 'center': 'invalid', 'rate': 1.0, 'x_unit': 'meV'},
                'center must be None or a number',
            ),
            (
                {'amplitude': 2.0, 'center': 0.5, 'rate': 'invalid', 'x_unit': 'meV'},
                'rate must be a number',
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            Exponential(display_name='TestExponential', **kwargs)

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'amplitude': np.nan, 'center': 0.5, 'rate': 1.0, 'x_unit': 'meV'},
                'amplitude must be finite',
            ),
            (
                {'amplitude': 2.0, 'center': 0.5, 'rate': np.nan, 'x_unit': 'meV'},
                'rate must be finite',
            ),
        ],
    )
    def test_input_value_validation_raises(self, kwargs, expected_message):
        with pytest.raises(ValueError, match=expected_message):
            Exponential(display_name='TestExponential', **kwargs)

    @pytest.mark.parametrize(
        'prop, valid_value, invalid_value, invalid_message',
        [
            ('amplitude', 3.0, 'invalid', r'must be a number'),
            ('center', 0.7, 'invalid', r'must be a number'),
            ('rate', 1.5, 'invalid', r'must be a number'),
        ],
    )
    def test_property_setters(
        self,
        exponential: Exponential,
        prop,
        valid_value,
        invalid_value,
        invalid_message,
    ):
        # set valid
        setattr(exponential, prop, valid_value)
        assert getattr(exponential, prop).value == valid_value

        # invalid
        with pytest.raises(TypeError, match=invalid_message):
            setattr(exponential, prop, invalid_value)

    def test_center_is_fixed_if_set_to_None(self, exponential: Exponential):
        # WHEN
        assert exponential.center.fixed is False

        # THEN
        exponential.center = None

        # EXPECT
        assert exponential.center.value == pytest.approx(0.0)
        assert exponential.center.fixed is True

    def test_evaluate(self, exponential: Exponential):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = exponential.evaluate(x)

        # EXPECT
        expected = 2.0 * np.exp(1.2 * (x - 0.5))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_get_all_parameters(self, exponential: Exponential):
        # WHEN
        params = exponential.get_all_parameters()

        # THEN EXPECT
        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)

        expected_names = {
            'ExponentialName amplitude',
            'ExponentialName center',
            'ExponentialName rate',
        }

        actual_names = {param.name for param in params}

        assert actual_names == expected_names

    def test_convert_unit(self, exponential: Exponential):
        # WHEN
        exponential.convert_x_unit('microeV')

        # THEN EXPECT
        assert exponential.x_unit == 'microeV'

        assert exponential.amplitude.value == pytest.approx(2.0 * 1e3)
        assert exponential.center.value == pytest.approx(0.5 * 1e3)

        # rate should scale inversely
        assert exponential.rate.value == pytest.approx(1.2 / 1e3)
        assert str(exponential.rate.unit) == '1/ueV'

    def test_convert_unit_incorrect_unit_raises(self, exponential: Exponential):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'unit must be a string or sc.Unit'):
            exponential.convert_x_unit(123)

    def test_convert_unit_rollback(self, exponential: Exponential):
        # WHEN
        with pytest.raises(
            UnitError,
            match=r'Failed to convert unit: Conversion from `meV` to `m` is not valid.',
        ):
            exponential.convert_x_unit('m')

        # THEN EXPECT - values should be unchanged
        assert exponential.x_unit == 'meV'
        assert exponential.amplitude.value == pytest.approx(2.0)
        assert exponential.amplitude.unit == 'meV'
        assert exponential.center.value == pytest.approx(0.5)
        assert exponential.center.unit == 'meV'
        assert exponential.rate.value == pytest.approx(1.2)
        assert exponential.rate.unit == '1/meV'

    def test_copy(self, exponential: Exponential):
        # WHEN
        exponential_copy = copy(exponential)

        # THEN EXPECT
        assert exponential_copy is not exponential
        assert exponential_copy.display_name == exponential.display_name

        assert exponential_copy.amplitude.value == exponential.amplitude.value
        assert exponential_copy.amplitude.fixed == exponential.amplitude.fixed

        assert exponential_copy.center.value == exponential.center.value
        assert exponential_copy.center.fixed == exponential.center.fixed

        assert exponential_copy.rate.value == exponential.rate.value
        assert exponential_copy.rate.fixed == exponential.rate.fixed

        assert exponential_copy.x_unit == exponential.x_unit

    def test_repr(self, exponential: Exponential):
        # WHEN
        repr_str = repr(exponential)

        # THEN EXPECT
        assert 'Exponential' in repr_str
        assert 'name = ExponentialName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'amplitude =' in repr_str
        assert 'center =' in repr_str
        assert 'rate =' in repr_str

    def test_y_unit_default(self, exponential: Exponential):
        assert exponential.y_unit == 'dimensionless'

    def test_convert_y_unit(self):
        # GIVEN: x_unit='meV', y_unit='1/meV' → amplitude_unit='dimensionless'
        exp = Exponential(amplitude=1.0, center=0.0, rate=1.0, x_unit='meV', y_unit='1/meV')
        # WHEN: convert y_unit to '1/eV' (same dimension, different scale)
        exp.convert_y_unit('1/eV')
        # EXPECT: y_unit updated and amplitude value rescaled (1e3 factor)
        assert exp.y_unit == '1/eV'
        assert exp.amplitude.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, exponential: Exponential):
        with pytest.raises(TypeError):
            exponential.convert_y_unit(123)

    def test_evaluate_scipp_output(self, exponential: Exponential):
        import scipp as sc

        x = np.linspace(-5, 5, 50)
        result = exponential.evaluate(x, output='scipp')
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 50

    def test_init_rejects_parameter_amplitude(self):
        amplitude_param = Parameter(name='amp', value=3.0, unit='meV')
        with pytest.raises(TypeError, match='amplitude must be a number'):
            Exponential(amplitude=amplitude_param, x_unit='meV')

    def test_init_rejects_parameter_rate(self):
        rate_param = Parameter(name='rate', value=0.5, unit='1/meV')
        with pytest.raises(TypeError, match='rate must be a number'):
            Exponential(rate=rate_param, x_unit='meV')

    def test_convert_y_unit_rollback_on_failure(self):
        exp = Exponential(amplitude=1.0, center=0.0, rate=1.0, x_unit='meV')
        with pytest.raises(UnitError):
            exp.convert_y_unit('K')
        assert exp.y_unit == 'dimensionless'
        assert exp.amplitude.value == pytest.approx(1.0)
