# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson

from easydynamics.sample_model import Lorentzian


class TestLorentzian:
    @pytest.fixture
    def lorentzian(self):
        return Lorentzian(
            name='LorentzianName',
            display_name='TestLorentzian',
            area=2.0,
            center=0.5,
            width=0.6,
            x_unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        lorentzian = Lorentzian()

        # EXPECT
        assert lorentzian.display_name == 'Lorentzian'
        assert lorentzian.area.value == pytest.approx(1.0)
        assert lorentzian.center.value == pytest.approx(0.0)
        assert lorentzian.width.value == pytest.approx(1.0)
        assert lorentzian.x_unit == 'meV'
        assert lorentzian.center.fixed is True

    def test_initialization(self, lorentzian: Lorentzian):
        # WHEN THEN EXPECT
        assert lorentzian.display_name == 'TestLorentzian'
        assert lorentzian.area.value == pytest.approx(2.0)
        assert lorentzian.center.value == pytest.approx(0.5)
        assert lorentzian.width.value == pytest.approx(0.6)
        assert lorentzian.x_unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'width': 0.6, 'x_unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'width': 0.6, 'x_unit': 'meV'},
                'center must be None',
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
            Lorentzian(display_name='TestLorentzian', **kwargs)

    def test_negative_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError, match=r'The width of a Lorentzian must be greater than zero.'
        ):
            Lorentzian(
                display_name='TestLorentzian',
                area=2.0,
                center=0.5,
                width=-0.6,
                x_unit='meV',
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            Lorentzian(
                display_name='TestLorentzian',
                area=-2.0,
                center=0.5,
                width=0.6,
                x_unit='meV',
            )

    @pytest.mark.parametrize(
        'prop, valid_value, invalid_value, invalid_message',
        [
            ('area', 3.0, 'invalid', r' must be a number'),
            ('center', 0.6, 'invalid', r' must be a number'),
            ('width', 0.7, 'invalid', r' must be a number'),
        ],
    )
    def test_property_setters(
        self, lorentzian: Lorentzian, prop, valid_value, invalid_value, invalid_message
    ):
        # WHEN: set a valid value
        setattr(lorentzian, prop, valid_value)
        # THEN EXPECT
        assert getattr(lorentzian, prop).value == valid_value

        # WHEN: set an invalid value — THEN EXPECT
        with pytest.raises(TypeError, match=invalid_message):
            setattr(lorentzian, prop, invalid_value)

    def test_width_must_be_positive(self, lorentzian: Lorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='width must be positive'):
            lorentzian.width = -0.5

    def test_evaluate(self, lorentzian: Lorentzian):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = lorentzian.evaluate(x)

        # EXPECT
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self, lorentzian: Lorentzian):
        # WHEN
        assert lorentzian.center.fixed is False

        # THEN
        lorentzian.center = None

        # EXPECT
        assert lorentzian.center.value == pytest.approx(0.0)
        assert lorentzian.center.fixed is True

    def test_get_all_parameters(self, lorentzian: Lorentzian):
        # WHEN THEN
        params = lorentzian.get_all_parameters()

        # EXPECT
        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'LorentzianName area',
            'LorentzianName center',
            'LorentzianName width',
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_area_matches_parameter(self, lorentzian: Lorentzian):
        # WHEN THEN
        x = np.linspace(
            lorentzian.center.value - 500 * lorentzian.width.value,
            lorentzian.center.value + 500 * lorentzian.width.value,
            20000,
        )  # Lorentzians have very long tails
        y = lorentzian.evaluate(x)
        numerical_area = simpson(y, x)

        # EXPECT
        assert numerical_area == pytest.approx(lorentzian.area.value, rel=2e-3)

    def test_convert_unit(self, lorentzian: Lorentzian):
        # WHEN THEN
        lorentzian.convert_x_unit('microeV')

        # EXPECT
        assert lorentzian.x_unit == 'microeV'
        assert lorentzian.area.value == pytest.approx(2 * 1e3)
        assert lorentzian.center.value == pytest.approx(0.5 * 1e3)
        assert lorentzian.width.value == pytest.approx(0.6 * 1e3)

    def test_copy(self, lorentzian: Lorentzian):
        # WHEN THEN
        lorentzian_copy = copy(lorentzian)

        # EXPECT
        assert lorentzian_copy is not lorentzian
        assert lorentzian_copy.display_name == lorentzian.display_name

        assert lorentzian_copy.area.value == lorentzian.area.value
        assert lorentzian_copy.area.fixed == lorentzian.area.fixed

        assert lorentzian_copy.center.value == lorentzian.center.value
        assert lorentzian_copy.center.fixed == lorentzian.center.fixed

        assert lorentzian_copy.width.value == lorentzian.width.value
        assert lorentzian_copy.width.fixed == lorentzian.width.fixed

        assert lorentzian_copy.x_unit == lorentzian.x_unit

    def test_repr(self, lorentzian: Lorentzian):
        # WHEN THEN
        repr_str = repr(lorentzian)

        # EXPECT
        assert 'Lorentzian' in repr_str
        assert 'name = LorentzianName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'width =' in repr_str

    def test_y_unit_default(self, lorentzian: Lorentzian):
        # WHEN THEN EXPECT
        assert lorentzian.y_unit == 'dimensionless'

    def test_convert_y_unit(self):
        # WHEN: x_unit='meV', y_unit='1/meV' → area_unit='dimensionless'
        lor = Lorentzian(area=1.0, x_unit='meV', y_unit='1/meV')
        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        lor.convert_y_unit('1/eV')
        # EXPECT: y_unit updated and area value rescaled (1e3 factor)
        assert lor.y_unit == '1/eV'
        assert lor.area.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, lorentzian: Lorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            lorentzian.convert_y_unit(123)

    def test_evaluate_scipp_output(self, lorentzian: Lorentzian):
        # WHEN
        x = np.linspace(-5, 5, 50)
        # THEN
        result = lorentzian.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 50

    def test_convert_x_unit_invalid_type_raises(self, lorentzian: Lorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'x_unit must be a string or sc\.Unit'):
            lorentzian.convert_x_unit(123)

    def test_convert_x_unit_rollback_on_failure(self, lorentzian: Lorentzian):
        # WHEN THEN
        with pytest.raises(UnitError):
            lorentzian.convert_x_unit('m')
        # EXPECT: state rolled back
        assert lorentzian.x_unit == 'meV'
        assert lorentzian.area.value == pytest.approx(2.0)
        assert lorentzian.center.value == pytest.approx(0.5)
        assert lorentzian.width.value == pytest.approx(0.6)

    def test_convert_y_unit_rollback_on_failure(self):
        # WHEN
        lor = Lorentzian(area=1.0, center=0.0, width=0.5, x_unit='meV')
        # THEN
        with pytest.raises(UnitError):
            lor.convert_y_unit('K')
        # EXPECT: state rolled back
        assert lor.y_unit == 'dimensionless'
        assert lor.area.value == pytest.approx(1.0)
