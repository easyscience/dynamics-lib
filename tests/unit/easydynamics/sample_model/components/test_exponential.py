# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
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
        assert exponential.y_unit == 'dimensionless'

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
            (
                {
                    'amplitude': 2.0,
                    'center': 0.5,
                    'rate': 1.0,
                    'x_unit': 'meV',
                    'y_unit': 123,
                },
                'unit must be None, a string',
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
        # WHEN: set a valid value
        setattr(exponential, prop, valid_value)
        # THEN EXPECT
        assert getattr(exponential, prop).value == valid_value

        # WHEN: set an invalid value — THEN EXPECT
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

    def test_convert_x_unit(self, exponential: Exponential):
        # WHEN

        # THEN
        exponential.convert_x_unit('microeV')

        # EXPECT
        assert exponential.x_unit == 'microeV'

        # amplitude carries y_unit only and is unaffected by x-unit conversion
        assert exponential.amplitude.value == pytest.approx(2.0)
        assert exponential.center.value == pytest.approx(0.5 * 1e3)

        # rate should scale inversely
        assert exponential.rate.value == pytest.approx(1.2 / 1e3)
        assert str(exponential.rate.unit) == '1/ueV'

    def test_convert_x_unit_incorrect_unit_raises(self, exponential: Exponential):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'unit must be a string or sc.Unit'):
            exponential.convert_x_unit(123)

    def test_convert_x_unit_rollback(self, exponential: Exponential):
        # WHEN THEN
        with pytest.raises(
            UnitError,
            match=r'Failed to convert unit: Conversion from `meV` to `m` is not valid.',
        ):
            exponential.convert_x_unit('m')

        # EXPECT - values should be unchanged
        assert exponential.x_unit == 'meV'
        assert exponential.amplitude.value == pytest.approx(2.0)
        assert exponential.amplitude.unit == 'dimensionless'
        assert exponential.center.value == pytest.approx(0.5)
        assert exponential.center.unit == 'meV'
        assert exponential.rate.value == pytest.approx(1.2)
        assert exponential.rate.unit == '1/meV'

    def test_copy(self, exponential: Exponential):
        # WHEN

        # THEN
        exponential_copy = copy(exponential)

        # EXPECT
        assert exponential_copy is not exponential
        assert exponential_copy.display_name == exponential.display_name

        assert exponential_copy.amplitude.value == exponential.amplitude.value
        assert exponential_copy.amplitude.fixed == exponential.amplitude.fixed

        assert exponential_copy.center.value == exponential.center.value
        assert exponential_copy.center.fixed == exponential.center.fixed

        assert exponential_copy.rate.value == exponential.rate.value
        assert exponential_copy.rate.fixed == exponential.rate.fixed

        assert exponential_copy.x_unit == exponential.x_unit
        assert exponential_copy.y_unit == exponential.y_unit

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

    def test_y_unit_custom(self):
        # WHEN THEN
        exp = Exponential(amplitude=1.0, center=0.0, rate=1.0, x_unit='meV', y_unit='1/meV')
        # EXPECT
        assert exp.y_unit == '1/meV'

    def test_y_unit_setter_raises(self, exponential: Exponential):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            exponential.y_unit = '1/meV'

    def test_convert_y_unit(self):
        # WHEN: x_unit='meV', y_unit='1/meV' → amplitude_unit='dimensionless'
        exp = Exponential(amplitude=1.0, center=0.0, rate=1.0, x_unit='meV', y_unit='1/meV')
        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        exp.convert_y_unit('1/eV')
        # EXPECT: y_unit updated and amplitude value rescaled (1e3 factor)
        assert exp.y_unit == '1/eV'
        assert exp.amplitude.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, exponential: Exponential):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            exponential.convert_y_unit(123)

    def test_evaluate_scipp_output(self, exponential: Exponential):
        # WHEN
        x = np.linspace(-5, 5, 50)
        # THEN
        result = exponential.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 50
        np.testing.assert_allclose(result.values, exponential.evaluate(x, output='numpy'))

    def test_evaluate_scipp_output_with_y_unit(self):
        # WHEN
        exp = Exponential(amplitude=1.0, center=0.0, rate=1.0, x_unit='meV', y_unit='1/meV')
        x = np.linspace(-5, 5, 50)
        # THEN
        result = exp.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('1/meV')

    def test_init_rejects_parameter_amplitude(self):
        # WHEN THEN EXPECT
        amplitude_param = Parameter(name='amp', value=3.0, unit='meV')
        with pytest.raises(TypeError, match='amplitude must be a number'):
            Exponential(amplitude=amplitude_param, x_unit='meV')

    def test_init_rejects_parameter_rate(self):
        # WHEN THEN EXPECT
        rate_param = Parameter(name='rate', value=0.5, unit='1/meV')
        with pytest.raises(TypeError, match='rate must be a number'):
            Exponential(rate=rate_param, x_unit='meV')

    def test_convert_y_unit_rollback_on_failure(self):
        # WHEN
        exp = Exponential(amplitude=1.0, center=0.0, rate=1.0, x_unit='meV')
        # THEN
        with pytest.raises(UnitError):
            exp.convert_y_unit('K')
        # EXPECT: state rolled back
        assert exp.y_unit == 'dimensionless'
        assert exp.amplitude.value == pytest.approx(1.0)

    def test_evaluate_unchanged_by_convert_x_unit(self):
        # WHEN: regression — the amplitude used to carry x_unit * y_unit, so converting the
        # x unit rescaled the whole curve by the conversion factor
        exp = Exponential(amplitude=2.0, center=0.0, rate=1.0, x_unit='meV')
        before = exp.evaluate(np.array([0.0]))

        # THEN: convert the x-axis unit and evaluate at the same physical point
        exp.convert_x_unit('ueV')
        after = exp.evaluate(np.array([0.0]))

        # EXPECT: the curve is unchanged
        np.testing.assert_allclose(after, before)
