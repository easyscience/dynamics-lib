# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.sample_model import DeltaFunction


class TestDeltaFunction:
    @pytest.fixture
    def delta_function(self):
        return DeltaFunction(
            name='DeltaFunctionName',
            display_name='TestDeltaFunction',
            area=2.0,
            center=0.5,
            x_unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        delta_function = DeltaFunction()

        # EXPECT
        assert delta_function.display_name == 'DeltaFunction'
        assert delta_function.area.value == pytest.approx(1.0)
        assert delta_function.center.value == pytest.approx(0.0)
        assert delta_function.x_unit == 'meV'
        assert delta_function.y_unit == 'dimensionless'
        assert delta_function.center.fixed is True

    def test_initialization(self, delta_function: DeltaFunction):
        # WHEN THEN EXPECT
        assert delta_function.display_name == 'TestDeltaFunction'
        assert delta_function.area.value == pytest.approx(2.0)
        assert delta_function.center.value == pytest.approx(0.5)
        assert delta_function.x_unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'x_unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'x_unit': 'meV'},
                'center must be ',
            ),
            (
                {'area': 2.0, 'center': 0.5, 'x_unit': 123},
                'unit must be ',
            ),
            (
                {'area': 2.0, 'center': 0.5, 'x_unit': 'meV', 'y_unit': 123},
                'unit must be ',
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            DeltaFunction(display_name='TestDeltaFunction', **kwargs)

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            DeltaFunction(display_name='TestDeltaFunction', area=-2.0, center=0.5, x_unit='meV')

    @pytest.mark.parametrize(
        'prop, valid_value, invalid_value, invalid_message',
        [
            ('area', 3.0, 'invalid', r'must be a number'),
            ('center', 0.6, 'invalid', r'must be a number'),
        ],
    )
    def test_property_setters(
        self,
        delta_function: DeltaFunction,
        prop,
        valid_value,
        invalid_value,
        invalid_message,
    ):
        # WHEN: set a valid value
        setattr(delta_function, prop, valid_value)
        # THEN EXPECT
        assert getattr(delta_function, prop).value == valid_value

        # WHEN: set an invalid value — THEN EXPECT
        with pytest.raises(TypeError, match=invalid_message):
            setattr(delta_function, prop, invalid_value)

    def test_evaluate(self, delta_function: DeltaFunction):
        # WHEN
        x = np.linspace(-1, 2, 100)

        # THEN
        result = delta_function.evaluate(x)

        # EXPECT
        expected_result = np.zeros_like(x)

        idx = np.argmin(np.abs(x - delta_function.center.value))
        dx = (max(x) - min(x)) / (len(x) - 1)  # domain spacing
        expected_result[idx] = 2.0 / dx

        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_out_of_bounds(self, delta_function: DeltaFunction):
        # WHEN
        x = np.linspace(1, 2, 100)  # center is at 0.5, so out of bounds

        # THEN
        result = delta_function.evaluate(x)

        # EXPECT
        expected_result = np.zeros_like(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_area_matches_parameter_value(self, delta_function: DeltaFunction):
        # WHEN
        x = np.linspace(-1, 2, 100)

        # THEN
        result = delta_function.evaluate(x)

        # EXPECT
        numerical_area = np.trapezoid(result, x)

        assert np.isclose(numerical_area, delta_function.area.value, rtol=1e-5)

    def test_area_matches_parameter_value_non_uniform_spacing(self, delta_function: DeltaFunction):
        # WHEN
        x = np.array([-1.0, 0.0, 0.4, 0.6, 1.0, 2.0])  # non-uniform spacing

        # THEN
        result = delta_function.evaluate(x)

        # EXPECT
        numerical_area = np.trapezoid(result, x)

        assert np.isclose(numerical_area, delta_function.area.value, rtol=1e-5)

    def test_evaluate_with_incompatible_unit_raises(self, delta_function: DeltaFunction):
        # WHEN
        x = sc.array(dims=['x'], values=[0.0, 0.5, 1.0], unit='nm')
        # THEN EXPECT
        with pytest.raises(
            UnitError,
            match='Input x has unit nm',
        ):
            delta_function.evaluate(x)

    @pytest.mark.parametrize(
        'x, expected_message',
        [
            (np.array([0.0, np.nan, 1.0]), 'Input x contains NaN values.'),
            (np.array([0.0, np.inf, 1.0]), 'Input x contains infinite values.'),
        ],
    )
    def test_evaluate_with_invalid_input_raises(
        self, delta_function: DeltaFunction, x, expected_message
    ):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=expected_message):
            delta_function.evaluate(x)

    def test_center_is_fixed_if_set_to_None(self, delta_function: DeltaFunction):
        # WHEN
        assert delta_function.center.fixed is False

        # THEN
        delta_function.center = None

        # EXPECT
        assert delta_function.center.value == pytest.approx(0.0)
        assert delta_function.center.fixed is True

    def test_get_all_parameters(self, delta_function: DeltaFunction):
        # WHEN THEN
        params = delta_function.get_all_parameters()

        # EXPECT
        assert len(params) == 2
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'DeltaFunctionName area',
            'DeltaFunctionName center',
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_convert_x_unit(self, delta_function: DeltaFunction):
        # WHEN THEN
        delta_function.convert_x_unit('microeV')

        # EXPECT
        assert delta_function.x_unit == 'microeV'
        assert delta_function.area.value == pytest.approx(2 * 1e3)
        assert delta_function.center.value == pytest.approx(0.5 * 1e3)

    def test_copy(self, delta_function: DeltaFunction):
        # WHEN THEN
        delta_copy = copy(delta_function)

        # EXPECT
        assert delta_copy is not delta_function
        assert delta_copy.display_name == delta_function.display_name

        assert delta_copy.area.value == delta_function.area.value
        assert delta_copy.area.fixed == delta_function.area.fixed

        assert delta_copy.center.value == delta_function.center.value
        assert delta_copy.center.fixed == delta_function.center.fixed

        assert delta_copy.x_unit == delta_function.x_unit

    def test_repr(self, delta_function: DeltaFunction):
        # WHEN THEN
        repr_str = repr(delta_function)

        # EXPECT
        assert 'DeltaFunction' in repr_str
        assert 'name = DeltaFunctionName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str

    def test_y_unit_custom(self):
        # WHEN THEN
        delta = DeltaFunction(area=1.0, x_unit='meV', y_unit='1/meV')
        # EXPECT
        assert delta.y_unit == '1/meV'

    def test_y_unit_setter_raises(self, delta_function: DeltaFunction):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            delta_function.y_unit = '1/meV'

    def test_convert_y_unit(self):
        # WHEN: x_unit='meV', y_unit='1/meV' → area_unit='dimensionless'
        delta = DeltaFunction(area=1.0, x_unit='meV', y_unit='1/meV')
        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        delta.convert_y_unit('1/eV')
        # EXPECT: y_unit updated and area value rescaled (1e3 factor)
        assert delta.y_unit == '1/eV'
        assert delta.area.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, delta_function: DeltaFunction):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            delta_function.convert_y_unit(123)

    def test_evaluate_scipp_output(self, delta_function: DeltaFunction):
        # WHEN
        x = np.linspace(-5, 5, 50)
        # THEN
        result = delta_function.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        assert len(result.values) == 50
        np.testing.assert_allclose(result.values, delta_function.evaluate(x, output='numpy'))

    def test_evaluate_scipp_output_with_y_unit(self):
        # WHEN
        delta = DeltaFunction(area=1.0, x_unit='meV', y_unit='1/meV')
        x = np.linspace(-5, 5, 50)
        # THEN
        result = delta.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('1/meV')

    @pytest.mark.parametrize(
        'x, center, expected_idx',
        [
            (np.array([0.5, 1.0, 2.0]), 0.5, 0),  # center at first element (line 202)
            (np.array([0.0, 1.0, 1.5]), 1.5, 2),  # center at last element (line 207)
        ],
        ids=['center_at_first', 'center_at_last'],
    )
    def test_evaluate_center_at_boundary(self, x, center, expected_idx):
        area = 1.0
        delta = DeltaFunction(area=area, center=center, x_unit='meV')
        result = delta.evaluate(x)
        # All elements except the boundary one should be zero
        assert result[expected_idx] > 0.0
        other_indices = [i for i in range(len(x)) if i != expected_idx]
        assert all(result[i] == pytest.approx(0.0) for i in other_indices)
        # Boundary bin width: both left and right are set to the single adjacent spacing
        bin_width = x[1] - x[0] if expected_idx == 0 else x[-1] - x[-2]
        assert np.isclose(result[expected_idx], area / bin_width, rtol=1e-10)

    def test_convert_x_unit_invalid_type_raises(self, delta_function: DeltaFunction):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'x_unit must be a string or sc\.Unit'):
            delta_function.convert_x_unit(123)

    def test_convert_x_unit_rollback_on_failure(self, delta_function: DeltaFunction):
        # WHEN THEN
        with pytest.raises(UnitError):
            delta_function.convert_x_unit('m')
        # EXPECT: state rolled back
        assert delta_function.x_unit == 'meV'
        assert delta_function.area.value == pytest.approx(2.0)
        assert delta_function.center.value == pytest.approx(0.5)

    def test_convert_y_unit_rollback_on_failure(self):
        # WHEN
        delta = DeltaFunction(area=1.0, center=0.0, x_unit='meV', y_unit='dimensionless')
        # THEN
        with pytest.raises(UnitError):
            delta.convert_y_unit('K')
        # EXPECT: state rolled back
        assert delta.y_unit == 'dimensionless'
        assert delta.area.value == pytest.approx(1.0)
