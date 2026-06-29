# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
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
            x_unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        gaussian = Gaussian()

        # EXPECT
        assert gaussian.display_name == 'Gaussian'
        assert gaussian.area.value == pytest.approx(1.0)
        assert gaussian.center.value == pytest.approx(0.0)
        assert gaussian.width.value == pytest.approx(1.0)
        assert gaussian.x_unit == 'meV'
        assert gaussian.center.fixed is True

    def test_initialization(self, gaussian: Gaussian):
        # WHEN THEN EXPECT
        assert gaussian.display_name == 'TestGaussian'
        assert gaussian.area.value == pytest.approx(2.0)
        assert gaussian.center.value == pytest.approx(0.5)
        assert gaussian.width.value == pytest.approx(0.6)
        assert gaussian.x_unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'width': 0.6, 'x_unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'width': 0.6, 'x_unit': 'meV'},
                'center must be None or a number',
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
        ids=[
            'invalid area',
            'invalid center',
            'invalid width',
            'invalid unit',
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
                x_unit='meV',
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            Gaussian(
                display_name='TestGaussian',
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
        gaussian.convert_x_unit('microeV')

        # EXPECT
        assert gaussian.x_unit == 'microeV'
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

        assert gaussian_copy.x_unit == gaussian.x_unit

    def test_repr(self, gaussian: Gaussian):
        # WHEN THEN
        repr_str = repr(gaussian)
        # EXPECT
        assert 'Gaussian' in repr_str
        assert 'name = GaussianName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'width =' in repr_str

    def test_y_unit_default(self, gaussian: Gaussian):
        # EXPECT
        assert gaussian.y_unit == 'dimensionless'

    def test_y_unit_custom(self):
        # WHEN
        gaussian = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        # EXPECT
        assert gaussian.y_unit == '1/meV'

    def test_y_unit_setter_raises(self, gaussian: Gaussian):
        # EXPECT
        with pytest.raises(AttributeError):
            gaussian.y_unit = '1/meV'

    def test_convert_y_unit(self):
        # WHEN: x_unit='meV', y_unit='1/meV' → area_unit ≈ dimensionless
        gaussian = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')

        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        gaussian.convert_y_unit('1/eV')

        # EXPECT: unit updated and area value rescaled (1/eV = 1e-3/meV, so value x 1e3)
        assert gaussian.y_unit == '1/eV'
        assert gaussian.area.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, gaussian: Gaussian):
        # EXPECT
        with pytest.raises(TypeError):
            gaussian.convert_y_unit(123)

    def test_evaluate_scipp_output(self, gaussian: Gaussian):
        x = np.linspace(-5, 5, 100)

        # WHEN
        result = gaussian.evaluate(x, output='scipp')

        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 100

    def test_evaluate_scipp_output_with_y_unit(self):
        gaussian = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        x = np.linspace(-5, 5, 100)

        # WHEN
        result = gaussian.evaluate(x, output='scipp')

        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('1/meV')

    def test_convert_x_unit_invalid_type_raises(self, gaussian: Gaussian):
        with pytest.raises(TypeError, match=r'x_unit must be a string or sc\.Unit'):
            gaussian.convert_x_unit(123)

    def test_convert_x_unit_rollback_on_failure(self, gaussian: Gaussian):
        with pytest.raises(UnitError):
            gaussian.convert_x_unit('m')
        assert gaussian.x_unit == 'meV'
        assert gaussian.area.value == pytest.approx(2.0)
        assert gaussian.center.value == pytest.approx(0.5)
        assert gaussian.width.value == pytest.approx(0.6)

    def test_convert_y_unit_rollback_on_failure(self):
        gaussian = Gaussian(area=1.0, center=0.0, width=0.5, x_unit='meV')
        with pytest.raises(UnitError):
            gaussian.convert_y_unit('K')
        assert gaussian.y_unit == 'dimensionless'
        assert gaussian.area.value == pytest.approx(1.0)
