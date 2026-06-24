# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson

from easydynamics.sample_model import DampedHarmonicOscillator


class TestDampedHarmonicOscillator:
    @pytest.fixture
    def dho(self):
        return DampedHarmonicOscillator(
            name='TestDHOName',
            display_name='TestDHO',
            area=2.0,
            center=1.5,
            width=0.3,
            x_unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        dho = DampedHarmonicOscillator()

        # EXPECT
        assert dho.display_name == 'DampedHarmonicOscillator'
        assert dho.area.value == pytest.approx(1.0)
        assert dho.center.value == pytest.approx(1.0)
        assert dho.width.value == pytest.approx(1.0)
        assert dho.x_unit == 'meV'

    def test_initialization(self, dho: DampedHarmonicOscillator):
        # WHEN THEN EXPECT
        assert dho.display_name == 'TestDHO'
        assert dho.area.value == pytest.approx(2.0)
        assert dho.center.value == pytest.approx(1.5)
        assert dho.width.value == pytest.approx(0.3)
        assert dho.x_unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'width': 0.6, 'x_unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'width': 0.6, 'x_unit': 'meV'},
                'center must be ',
            ),
            (
                {'area': 2.0, 'center': 0.5, 'width': 'invalid', 'x_unit': 'meV'},
                'width must be a number',
            ),
            (
                {'area': 2.0, 'center': 0.5, 'width': 0.6, 'x_unit': 123},
                'unit must be None',
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            DampedHarmonicOscillator(display_name='DampedHarmonicOscillator', **kwargs)

    def test_negative_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError,
            match=r'The width of a DampedHarmonicOscillator must be greater than zero.',
        ):
            DampedHarmonicOscillator(
                display_name='TestDampedHarmonicOscillator',
                area=2.0,
                center=0.5,
                width=-0.6,
                x_unit='meV',
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            DampedHarmonicOscillator(
                display_name='TestDampedHarmonicOscillator',
                area=-2.0,
                center=0.5,
                width=0.6,
                x_unit='meV',
            )

    @pytest.mark.parametrize(
        'prop, valid_value, invalid_value, invalid_message',
        [
            ('area', 3.0, 'invalid', r'must be a number'),
            ('center', 0.6, 'invalid', r'must be a number'),
            ('width', 0.7, 'invalid', r'must be a number'),
        ],
    )
    def test_property_setters(
        self,
        dho: DampedHarmonicOscillator,
        prop,
        valid_value,
        invalid_value,
        invalid_message,
    ):
        # set valid
        setattr(dho, prop, valid_value)
        assert getattr(dho, prop).value == valid_value

        # invalid
        with pytest.raises(TypeError, match=invalid_message):
            setattr(dho, prop, invalid_value)

    def test_center_setter_negative_raises(self, dho: DampedHarmonicOscillator):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='center must be positive'):
            dho.center = -1.0

    def test_width_must_be_positive(self, dho: DampedHarmonicOscillator):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='width must be positive'):
            dho.width = -0.5

    def test_evaluate(self, dho: DampedHarmonicOscillator):
        # WHEN
        x = np.array([0.0, 1.5, 3.0])

        # THEN
        result = dho.evaluate(x)

        # EXPECT
        expected_result = (
            2 * 2.0 * (1.5**2) * (0.3) / np.pi / ((x**2 - 1.5**2) ** 2 + (2 * 0.3 * x) ** 2)
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_get_all_parameters(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        params = dho.get_all_parameters()

        # EXPECT
        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'TestDHOName area',
            'TestDHOName center',
            'TestDHOName width',
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_area_matches_parameter(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        x = np.linspace(
            -dho.center.value - 20 * dho.width.value,
            dho.center.value + 20 * dho.width.value,
            5000,
        )
        y = dho.evaluate(x)
        numerical_area = simpson(y, x)

        # EXPECT
        assert numerical_area == pytest.approx(dho.area.value, rel=2e-3)

    def test_convert_unit(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        dho.convert_x_unit('microeV')

        # EXPECT
        assert dho.x_unit == 'microeV'
        assert dho.area.value == pytest.approx(2 * 1e3)
        assert dho.center.value == pytest.approx(1.5 * 1e3)
        assert dho.width.value == pytest.approx(0.3 * 1e3)

    def test_copy(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        dho_copy = copy(dho)

        # EXPECT
        assert dho_copy is not dho
        assert dho_copy.display_name == dho.display_name

        assert dho_copy.area.value == dho.area.value
        assert dho_copy.area.fixed == dho.area.fixed

        assert dho_copy.center.value == dho.center.value
        assert dho_copy.center.fixed == dho.center.fixed

        assert dho_copy.width.value == dho.width.value
        assert dho_copy.width.fixed == dho.width.fixed

        assert dho_copy.x_unit == dho.x_unit

    def test_repr(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        repr_str = repr(dho)

        # EXPECT
        assert 'DampedHarmonicOscillator' in repr_str
        assert 'name = TestDHOName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'width =' in repr_str

    def test_y_unit_default(self, dho: DampedHarmonicOscillator):
        assert dho.y_unit == 'dimensionless'

    def test_convert_y_unit(self):
        # GIVEN: x_unit='meV', y_unit='1/meV' → area_unit='dimensionless'
        dho = DampedHarmonicOscillator(
            area=1.0, center=1.0, width=0.3, x_unit='meV', y_unit='1/meV'
        )
        # WHEN: convert y_unit to '1/eV' (same dimension, different scale)
        dho.convert_y_unit('1/eV')
        # EXPECT: y_unit updated and area value rescaled (1e3 factor)
        assert dho.y_unit == '1/eV'
        assert dho.area.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, dho: DampedHarmonicOscillator):
        with pytest.raises(TypeError):
            dho.convert_y_unit(123)

    def test_evaluate_scipp_output(self, dho: DampedHarmonicOscillator):
        import scipp as sc

        x = np.linspace(0.5, 5.0, 50)
        result = dho.evaluate(x, output='scipp')
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 50

    def test_convert_x_unit_invalid_type_raises(self, dho: DampedHarmonicOscillator):
        with pytest.raises(TypeError, match=r'x_unit must be a string or sc\.Unit'):
            dho.convert_x_unit(123)

    def test_convert_x_unit_rollback_on_failure(self, dho: DampedHarmonicOscillator):
        with pytest.raises(UnitError):
            dho.convert_x_unit('m')
        assert dho.x_unit == 'meV'
        assert dho.area.value == pytest.approx(2.0)
        assert dho.center.value == pytest.approx(1.5)
        assert dho.width.value == pytest.approx(0.3)

    def test_convert_y_unit_rollback_on_failure(self):
        dho = DampedHarmonicOscillator(area=1.0, center=1.0, width=0.3, x_unit='meV')
        with pytest.raises(UnitError):
            dho.convert_y_unit('K')
        assert dho.y_unit == 'dimensionless'
        assert dho.area.value == pytest.approx(1.0)
