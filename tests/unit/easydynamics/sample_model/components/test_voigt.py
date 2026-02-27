# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.integrate import simpson
from scipy.special import voigt_profile

from easydynamics.sample_model import Voigt


class TestVoigt:
    @pytest.fixture
    def voigt(self):
        return Voigt(
            display_name='TestVoigt',
            area=2.0,
            center=0.5,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        voigt = Voigt()

        # EXPECT
        assert voigt.display_name == 'Voigt'
        assert voigt.area.value == 1.0
        assert voigt.center.value == 0.0
        assert voigt.gaussian_width.value == 1.0
        assert voigt.lorentzian_width.value == 1.0
        assert voigt.unit == 'meV'
        assert voigt.center.fixed is True

    def test_initialization(self, voigt: Voigt):
        # WHEN THEN EXPECT
        assert voigt.display_name == 'TestVoigt'
        assert voigt.area.value == 2.0
        assert voigt.center.value == 0.5
        assert voigt.gaussian_width.value == 0.6
        assert voigt.lorentzian_width.value == 0.7
        assert voigt.unit == 'meV'

    def test_init_with_parameters(self):
        # WHEN
        area_param = Parameter(name='area_param', value=3.0, unit='meV')
        center_param = Parameter(name='center_param', value=1.0, unit='meV')
        gaussian_width_param = Parameter(name='gaussian_width_param', value=0.8, unit='meV')
        lorentzian_width_param = Parameter(name='lorentzian_width_param', value=0.9, unit='meV')

        # THEN
        voigt = Voigt(
            display_name='ParamVoigt',
            area=area_param,
            center=center_param,
            gaussian_width=gaussian_width_param,
            lorentzian_width=lorentzian_width_param,
            unit='meV',
        )

        # EXPECT
        assert voigt.display_name == 'ParamVoigt'
        assert voigt.area is area_param
        assert voigt.center is center_param
        assert voigt.gaussian_width is gaussian_width_param
        assert voigt.lorentzian_width is lorentzian_width_param
        assert voigt.unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {
                    'area': 'invalid',
                    'center': 0.5,
                    'gaussian_width': 0.6,
                    'lorentzian_width': 0.7,
                    'unit': 'meV',
                },
                'area must be a number',
            ),
            (
                {
                    'area': 2.0,
                    'center': 'invalid',
                    'gaussian_width': 0.6,
                    'lorentzian_width': 0.7,
                    'unit': 'meV',
                },
                'center must be None',
            ),
            (
                {
                    'area': 2.0,
                    'center': 0.5,
                    'gaussian_width': 'invalid',
                    'lorentzian_width': 0.7,
                    'unit': 'meV',
                },
                'gaussian_width must be a number',
            ),
            (
                {
                    'area': 2.0,
                    'center': 0.5,
                    'gaussian_width': 0.6,
                    'lorentzian_width': 'invalid',
                    'unit': 'meV',
                },
                'lorentzian_width must be a number',
            ),
            (
                {
                    'area': 2.0,
                    'center': 0.5,
                    'gaussian_width': 0.6,
                    'lorentzian_width': 0.7,
                    'unit': 123,
                },
                'unit must be None,',
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            Voigt(display_name='TestVoigt', **kwargs)

    def test_negative_gaussian_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError, match='The gaussian_width of a Voigt must be greater than.'
        ):
            Voigt(
                display_name='TestVoigt',
                area=2.0,
                center=0.5,
                gaussian_width=-0.6,
                lorentzian_width=0.7,
                unit='meV',
            )

    def test_negative_lorentzian_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError,
            match='The lorentzian_width of a Voigt must be greater than zero.',
        ):
            Voigt(
                display_name='TestVoigt',
                area=2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=-0.7,
                unit='meV',
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            Voigt(
                display_name='TestVoigt',
                area=-2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit='meV',
            )

    @pytest.mark.parametrize(
        'prop, valid_value, invalid_value, invalid_message',
        [
            ('area', 3.0, 'invalid', r'must be a number'),
            ('center', 0.6, 'invalid', r'must be a number'),
            ('gaussian_width', 0.7, 'invalid', r'must be a number'),
            (
                'lorentzian_width',
                0.7,
                'invalid',
                r'must be a number',
            ),
        ],
    )
    def test_property_setters(
        self, voigt: Voigt, prop, valid_value, invalid_value, invalid_message
    ):
        # set valid
        setattr(voigt, prop, valid_value)
        assert getattr(voigt, prop).value == valid_value

        # invalid
        with pytest.raises(TypeError, match=invalid_message):
            setattr(voigt, prop, invalid_value)

    def test_gaussian_width_must_be_positive(self, voigt: Voigt):
        # WHEN THEN
        with pytest.raises(ValueError, match='gaussian_width must be positive'):
            voigt.gaussian_width = -0.6

    def test_lorentzian_width_must_be_positive(self, voigt: Voigt):
        # WHEN THEN
        with pytest.raises(
            ValueError,
            match='lorentzian_width must be positive',
        ):
            voigt.lorentzian_width = -0.7

    def test_center_is_fixed_if_set_to_None(self, voigt: Voigt):
        # WHEN
        assert voigt.center.fixed is False

        # THEN
        voigt.center = None

        # EXPECT
        assert voigt.center.value == 0.0
        assert voigt.center.fixed is True

    def test_evaluate(self, voigt: Voigt):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = voigt.evaluate(x)

        # EXPECT
        expected_result = 2.0 * voigt_profile(x - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_init_to_None(self):
        # WHEN THEN
        test_voigt = Voigt(
            display_name='TestVoigt',
            area=2.0,
            center=None,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit='meV',
        )

        # EXPECT
        assert test_voigt.center.value == 0.0
        assert test_voigt.center.fixed is True

    def test_convert_unit(self, voigt: Voigt):
        # WHEN THEN
        voigt.convert_unit('microeV')

        # EXPECT
        assert voigt.unit == 'microeV'
        assert voigt.area.value == 2 * 1e3
        assert voigt.center.value == 0.5 * 1e3
        assert voigt.gaussian_width.value == 0.6 * 1e3
        assert voigt.lorentzian_width.value == 0.7 * 1e3

    def test_get_all_parameters(self, voigt: Voigt):
        # WHEN THEN
        params = voigt.get_all_parameters()

        # EXPECT
        assert len(params) == 4
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'TestVoigt area',
            'TestVoigt center',
            'TestVoigt gaussian_width',
            'TestVoigt lorentzian_width',
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_area_matches_parameter(self, voigt: Voigt):
        # WHEN THEN
        x = np.linspace(
            voigt.center.value
            - 100 * voigt.gaussian_width.value
            - 300 * voigt.lorentzian_width.value,
            voigt.center.value
            + 100 * voigt.gaussian_width.value
            + 300 * voigt.lorentzian_width.value,
            20000,
        )  # Voigts have very long tails
        y = voigt.evaluate(x)
        numerical_area = simpson(y, x)

        # EXPECT
        assert numerical_area == pytest.approx(voigt.area.value, rel=2e-3)

    def test_copy(self, voigt: Voigt):
        # WHEN THEN
        voigt_copy = copy(voigt)

        # EXPECT
        assert voigt_copy is not voigt
        assert voigt_copy.display_name == voigt.display_name

        assert voigt_copy.area.value == voigt.area.value
        assert voigt_copy.area.fixed == voigt.area.fixed

        assert voigt_copy.center.value == voigt.center.value
        assert voigt_copy.center.fixed == voigt.center.fixed

        assert voigt_copy.gaussian_width.value == voigt.gaussian_width.value
        assert voigt_copy.gaussian_width.fixed == voigt.gaussian_width.fixed

        assert voigt_copy.lorentzian_width.value == voigt.lorentzian_width.value
        assert voigt_copy.lorentzian_width.fixed == voigt.lorentzian_width.fixed

        assert voigt_copy.unit == voigt.unit

    def test_repr(self, voigt: Voigt):
        # WHEN THEN
        repr_str = repr(voigt)

        # EXPECT
        assert 'Voigt' in repr_str
        assert 'unique_name = Voigt' in repr_str
        assert 'unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'gaussian_width =' in repr_str
        assert 'lorentzian_width =' in repr_str
