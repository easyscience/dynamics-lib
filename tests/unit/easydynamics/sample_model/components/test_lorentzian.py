# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.integrate import simpson

from easydynamics.sample_model import Lorentzian


class TestLorentzian:
    @pytest.fixture
    def lorentzian(self):
        return Lorentzian(
            display_name='TestLorentzian', area=2.0, center=0.5, width=0.6, unit='meV'
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        lorentzian = Lorentzian()

        # EXPECT
        assert lorentzian.display_name == 'Lorentzian'
        assert lorentzian.area.value == 1.0
        assert lorentzian.center.value == 0.0
        assert lorentzian.width.value == 1.0
        assert lorentzian.unit == 'meV'
        assert lorentzian.center.fixed is True

    def test_initialization(self, lorentzian: Lorentzian):
        # WHEN THEN EXPECT
        assert lorentzian.display_name == 'TestLorentzian'
        assert lorentzian.area.value == 2.0
        assert lorentzian.center.value == 0.5
        assert lorentzian.width.value == 0.6
        assert lorentzian.unit == 'meV'

    def test_init_with_parameters(self):
        # WHEN
        area_param = Parameter(name='area_param', value=3.0, unit='meV')
        center_param = Parameter(name='center_param', value=1.0, unit='meV')
        width_param = Parameter(name='width_param', value=0.8, unit='meV')

        # THEN
        lorentzian = Lorentzian(
            display_name='ParamLorentzian',
            area=area_param,
            center=center_param,
            width=width_param,
            unit='meV',
        )

        # EXPECT
        assert lorentzian.display_name == 'ParamLorentzian'
        assert lorentzian.area is area_param
        assert lorentzian.center is center_param
        assert lorentzian.width is width_param
        assert lorentzian.unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'width': 0.6, 'unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'width': 0.6, 'unit': 'meV'},
                'center must be None',
            ),
            (
                {'area': 2.0, 'center': 0.5, 'width': 'invalid', 'unit': 'meV'},
                'width must be a number',
            ),
            (
                {'area': 2.0, 'center': 0.5, 'width': 0.6, 'unit': 123},
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
            ValueError, match='The width of a Lorentzian must be greater than zero.'
        ):
            Lorentzian(
                display_name='TestLorentzian',
                area=2.0,
                center=0.5,
                width=-0.6,
                unit='meV',
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            Lorentzian(
                display_name='TestLorentzian',
                area=-2.0,
                center=0.5,
                width=0.6,
                unit='meV',
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
        # set valid
        setattr(lorentzian, prop, valid_value)
        assert getattr(lorentzian, prop).value == valid_value

        # invalid
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
        assert lorentzian.center.value == 0.0
        assert lorentzian.center.fixed is True

    def test_get_all_parameters(self, lorentzian: Lorentzian):
        # WHEN THEN
        params = lorentzian.get_all_parameters()

        # EXPECT
        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'TestLorentzian area',
            'TestLorentzian center',
            'TestLorentzian width',
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
        lorentzian.convert_unit('microeV')

        # EXPECT
        assert lorentzian.unit == 'microeV'
        assert lorentzian.area.value == 2 * 1e3
        assert lorentzian.center.value == 0.5 * 1e3
        assert lorentzian.width.value == 0.6 * 1e3

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

        assert lorentzian_copy.unit == lorentzian.unit

    def test_repr(self, lorentzian: Lorentzian):
        # WHEN THEN
        repr_str = repr(lorentzian)

        # EXPECT
        assert 'Lorentzian' in repr_str
        assert 'unique_name = Lorentzian' in repr_str
        assert 'unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'width =' in repr_str
