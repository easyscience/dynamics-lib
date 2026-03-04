# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.integrate import simpson

from easydynamics.sample_model import DampedHarmonicOscillator


class TestDampedHarmonicOscillator:
    @pytest.fixture
    def dho(self):
        return DampedHarmonicOscillator(
            display_name='TestDHO', area=2.0, center=1.5, width=0.3, unit='meV'
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        dho = DampedHarmonicOscillator()

        # EXPECT
        assert dho.display_name == 'DampedHarmonicOscillator'
        assert dho.area.value == 1.0
        assert dho.center.value == 1.0
        assert dho.width.value == 1.0
        assert dho.unit == 'meV'

    def test_initialization(self, dho: DampedHarmonicOscillator):
        # WHEN THEN EXPECT
        assert dho.display_name == 'TestDHO'
        assert dho.area.value == 2.0
        assert dho.center.value == 1.5
        assert dho.width.value == 0.3
        assert dho.unit == 'meV'

    def test_init_with_parameters(self):
        # WHEN
        area_param = Parameter(name='area_param', value=3.0, unit='meV')
        center_param = Parameter(name='center_param', value=1.0, unit='meV')
        width_param = Parameter(name='width_param', value=0.8, unit='meV')

        # THEN
        dho = DampedHarmonicOscillator(
            display_name='Paramdho',
            area=area_param,
            center=center_param,
            width=width_param,
            unit='meV',
        )

        # EXPECT
        assert dho.display_name == 'Paramdho'
        assert dho.area is area_param
        assert dho.center is center_param
        assert dho.width is width_param
        assert dho.unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'width': 0.6, 'unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'width': 0.6, 'unit': 'meV'},
                'center must be ',
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
            DampedHarmonicOscillator(display_name='DampedHarmonicOscillator', **kwargs)

    def test_negative_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError,
            match='The width of a DampedHarmonicOscillator must be greater than zero.',
        ):
            DampedHarmonicOscillator(
                display_name='TestDampedHarmonicOscillator',
                area=2.0,
                center=0.5,
                width=-0.6,
                unit='meV',
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            DampedHarmonicOscillator(
                display_name='TestDampedHarmonicOscillator',
                area=-2.0,
                center=0.5,
                width=0.6,
                unit='meV',
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
            'TestDHO area',
            'TestDHO center',
            'TestDHO width',
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
        dho.convert_unit('microeV')

        # EXPECT
        assert dho.unit == 'microeV'
        assert dho.area.value == 2 * 1e3
        assert dho.center.value == 1.5 * 1e3
        assert dho.width.value == 0.3 * 1e3

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

        assert dho_copy.unit == dho.unit

    def test_repr(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        repr_str = repr(dho)

        # EXPECT
        assert 'DampedHarmonicOscillator' in repr_str
        assert 'name = TestDHO' in repr_str
        assert 'unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'width =' in repr_str
