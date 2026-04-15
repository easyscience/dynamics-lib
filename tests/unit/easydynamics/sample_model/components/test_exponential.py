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
            display_name='TestExponential',
            amplitude=2.0,
            center=0.5,
            rate=1.2,
            unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN
        exponential = Exponential()

        # THEN EXPECT
        assert exponential.display_name == 'Exponential'
        assert exponential.amplitude.value == pytest.approx(1.0)
        assert exponential.center.value == pytest.approx(0.0)
        assert exponential.rate.value == pytest.approx(1.0)
        assert exponential.unit == 'meV'

    def test_initialization(self, exponential: Exponential):
        # WHEN THEN EXPECT
        assert exponential.display_name == 'TestExponential'
        assert exponential.amplitude.value == pytest.approx(2.0)
        assert exponential.center.value == pytest.approx(0.5)
        assert exponential.rate.value == pytest.approx(1.2)
        assert exponential.unit == 'meV'

    def test_init_with_parameters(self):
        # WHEN
        amplitude_param = Parameter(name='amp_param', value=3.0, unit='meV')
        center_param = Parameter(name='center_param', value=1.0, unit='meV')
        rate_param = Parameter(name='rate_param', value=0.5, unit='1/meV')

        # THEN
        exponential = Exponential(
            display_name='ParamExponential',
            amplitude=amplitude_param,
            center=center_param,
            rate=rate_param,
            unit='meV',
        )

        # EXPECT
        assert exponential.display_name == 'ParamExponential'
        assert exponential.amplitude is amplitude_param
        assert exponential.center is center_param
        assert exponential.rate is rate_param
        assert exponential.unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'amplitude': 'invalid', 'center': 0.5, 'rate': 1.0, 'unit': 'meV'},
                'amplitude must be a number',
            ),
            (
                {'amplitude': 2.0, 'center': 'invalid', 'rate': 1.0, 'unit': 'meV'},
                'center must be None, a number',
            ),
            (
                {'amplitude': 2.0, 'center': 0.5, 'rate': 'invalid', 'unit': 'meV'},
                'rate must be a number or a Parameter',
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
                {'amplitude': np.nan, 'center': 0.5, 'rate': 1.0, 'unit': 'meV'},
                'amplitude must be a finite number or a Parameter',
            ),
            (
                {'amplitude': 2.0, 'center': 0.5, 'rate': np.nan, 'unit': 'meV'},
                'rate must be a finite number or a Parameter',
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
            'TestExponential amplitude',
            'TestExponential center',
            'TestExponential rate',
        }

        actual_names = {param.name for param in params}

        assert actual_names == expected_names

    def test_convert_unit(self, exponential: Exponential):
        # WHEN
        exponential.convert_unit('microeV')

        # THEN EXPECT
        assert exponential.unit == 'microeV'

        assert exponential.amplitude.value == pytest.approx(2.0 * 1e3)
        assert exponential.center.value == pytest.approx(0.5 * 1e3)

        # rate should scale inversely
        assert exponential.rate.value == pytest.approx(1.2 / 1e3)
        assert str(exponential.rate.unit) == '1/ueV'

    def test_convert_unit_incorrect_unit_raises(self, exponential: Exponential):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'unit must be a string or sc.Unit'):
            exponential.convert_unit(123)

    def test_convert_unit_rollback(self, exponential: Exponential):
        # WHEN
        with pytest.raises(
            UnitError,
            match=r'Failed to convert unit: Conversion from `meV` to `m` is not valid.',
        ):
            exponential.convert_unit('m')

        # THEN EXPECT - values should be unchanged
        assert exponential.unit == 'meV'
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

        assert exponential_copy.unit == exponential.unit

    def test_repr(self, exponential: Exponential):
        # WHEN
        repr_str = repr(exponential)

        # THEN EXPECT
        assert 'Exponential' in repr_str
        assert 'unique_name = Exponential' in repr_str
        assert 'unit = meV' in repr_str
        assert 'amplitude =' in repr_str
        assert 'center =' in repr_str
        assert 'rate =' in repr_str
