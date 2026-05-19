# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.integrate import simpson

from easydynamics.sample_model import Gaussian


class TestGaussian:
    @pytest.fixture
    def gaussian(self):
        return Gaussian(
            name='GaussianName',
            display_name='TestGaussian',
            area=2.0,
            center=0.5,
            width=0.6,
            unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        gaussian = Gaussian()

        # EXPECT
        assert gaussian.display_name == 'Gaussian'
        assert gaussian.area.value == pytest.approx(1.0)
        assert gaussian.center.value == pytest.approx(0.0)
        assert gaussian.width.value == pytest.approx(1.0)
        assert gaussian.unit == 'meV'
        assert gaussian.center.fixed is True

    def test_initialization(self, gaussian: Gaussian):
        # WHEN THEN EXPECT
        assert gaussian.display_name == 'TestGaussian'
        assert gaussian.area.value == pytest.approx(2.0)
        assert gaussian.center.value == pytest.approx(0.5)
        assert gaussian.width.value == pytest.approx(0.6)
        assert gaussian.unit == 'meV'

    def test_init_with_parameters(self):
        # WHEN
        area_param = Parameter(name='area_param', value=3.0, unit='meV')
        center_param = Parameter(name='center_param', value=1.0, unit='meV')
        width_param = Parameter(name='width_param', value=0.8, unit='meV')

        # THEN
        gaussian = Gaussian(
            display_name='ParamGaussian',
            area=area_param,
            center=center_param,
            width=width_param,
            unit='meV',
        )

        # EXPECT
        assert gaussian.display_name == 'ParamGaussian'
        assert gaussian.area is area_param
        assert gaussian.center is center_param
        assert gaussian.width is width_param
        assert gaussian.unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'width': 0.6, 'unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'width': 0.6, 'unit': 'meV'},
                'center must be None, a number',
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
            Gaussian(display_name='TestGaussian', **kwargs)

    def test_negative_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError, match=r'The width of a Gaussian must be greater than zero.'
        ):
            Gaussian(
                display_name='TestGaussian',
                area=2.0,
                center=0.5,
                width=-0.6,
                unit='meV',
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            Gaussian(
                display_name='TestGaussian',
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
        self, gaussian: Gaussian, prop, valid_value, invalid_value, invalid_message
    ):
        # set valid
        setattr(gaussian, prop, valid_value)
        assert getattr(gaussian, prop).value == valid_value

        # invalid
        with pytest.raises(TypeError, match=invalid_message):
            setattr(gaussian, prop, invalid_value)

    def test_width_must_be_positive(self, gaussian: Gaussian):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='width must be positive'):
            gaussian.width = -0.5

    def test_evaluate(self, gaussian: Gaussian):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = gaussian.evaluate(x)

        # EXPECT
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - 0.5) / 0.6) ** 2
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self, gaussian: Gaussian):
        # WHEN
        assert gaussian.center.fixed is False

        # THEN
        gaussian.center = None

        # EXPECT
        assert gaussian.center.value == pytest.approx(0.0)
        assert gaussian.center.fixed is True

    def test_get_all_parameters(self, gaussian: Gaussian):
        # WHEN THEN
        params = gaussian.get_all_parameters()

        # EXPECT
        assert len(params) == 3
        assert all(isinstance(param, Parameter) for param in params)

        expected_names = {
            'GaussianName area',
            'GaussianName center',
            'GaussianName width',
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_area_matches_parameter(self, gaussian: Gaussian):
        # WHEN
        x = np.linspace(
            gaussian.center.value - 10 * gaussian.width.value,
            gaussian.center.value + 10 * gaussian.width.value,
            1000,
        )

        # THEN
        y = gaussian.evaluate(x)

        # EXPECT
        numerical_area = simpson(y, x)
        assert np.isclose(numerical_area, gaussian.area.value, rtol=1e-3)

    def test_convert_unit(self, gaussian: Gaussian):
        # WHEN THEN
        gaussian.convert_unit('microeV')

        # EXPECT
        assert gaussian.unit == 'microeV'
        assert gaussian.area.value == pytest.approx(2 * 1e3)
        assert gaussian.center.value == pytest.approx(0.5 * 1e3)
        assert gaussian.width.value == pytest.approx(0.6 * 1e3)

    def test_copy(self, gaussian: Gaussian):
        # WHEN
        gaussian.area.min = 0.5
        gaussian.width.min = 0.1
        gaussian.area.fixed = True
        gaussian.area.max = 5.0

        # THEN
        gaussian_copy = copy(gaussian)
        # EXPECT
        assert gaussian_copy is not gaussian
        assert gaussian_copy.display_name == gaussian.display_name
        assert gaussian_copy.name == gaussian.name

        assert gaussian_copy.area.value == gaussian.area.value
        assert gaussian_copy.area.fixed == gaussian.area.fixed
        assert gaussian_copy.area.min == gaussian.area.min
        assert gaussian_copy.area.max == gaussian.area.max

        assert gaussian_copy.center.value == gaussian.center.value
        assert gaussian_copy.center.fixed == gaussian.center.fixed
        assert gaussian_copy.center.min == gaussian.center.min
        assert gaussian_copy.center.max == gaussian.center.max

        assert gaussian_copy.width.value == gaussian.width.value
        assert gaussian_copy.width.fixed == gaussian.width.fixed
        assert gaussian_copy.width.min == gaussian.width.min
        assert gaussian_copy.width.max == gaussian.width.max

        assert gaussian_copy.unit == gaussian.unit

    def test_repr(self, gaussian: Gaussian):
        # WHEN THEN
        repr_str = repr(gaussian)
        # EXPECT
        assert 'Gaussian' in repr_str
        assert 'name = GaussianName' in repr_str
        assert 'unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'width =' in repr_str
