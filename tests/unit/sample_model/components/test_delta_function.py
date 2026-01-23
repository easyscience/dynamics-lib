# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
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
        return DeltaFunction(display_name='TestDeltaFunction', area=2.0, center=0.5, unit='meV')

    def test_init_no_inputs(self):
        # WHEN THEN
        delta_function = DeltaFunction()

        # EXPECT
        assert delta_function.display_name == 'DeltaFunction'
        assert delta_function.area.value == 1.0
        assert delta_function.center.value == 0.0
        assert delta_function.unit == 'meV'
        assert delta_function.center.fixed is True

    def test_initialization(self, delta_function: DeltaFunction):
        # WHEN THEN EXPECT
        assert delta_function.display_name == 'TestDeltaFunction'
        assert delta_function.area.value == 2.0
        assert delta_function.center.value == 0.5
        assert delta_function.unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {'area': 'invalid', 'center': 0.5, 'unit': 'meV'},
                'area must be a number',
            ),
            (
                {'area': 2.0, 'center': 'invalid', 'unit': 'meV'},
                'center must be ',
            ),
            (
                {'area': 2.0, 'center': 0.5, 'unit': 123},
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
            DeltaFunction(display_name='TestDeltaFunction', area=-2.0, center=0.5, unit='meV')

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
        # set valid
        setattr(delta_function, prop, valid_value)
        assert getattr(delta_function, prop).value == valid_value

        # invalid
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
            match='Input x has unit nm, but DeltaFunction component has unit meV. Failed to convert DeltaFunction to nm.',  # noqa: E501
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
        assert delta_function.center.value == 0.0
        assert delta_function.center.fixed is True

    def test_get_all_parameters(self, delta_function: DeltaFunction):
        # WHEN THEN
        params = delta_function.get_all_parameters()

        # EXPECT
        assert len(params) == 2
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'TestDeltaFunction area',
            'TestDeltaFunction center',
        }
        actual_names = {param.name for param in params}
        assert actual_names == expected_names

    def test_convert_unit(self, delta_function: DeltaFunction):
        # WHEN THEN
        delta_function.convert_unit('microeV')

        # EXPECT
        assert delta_function.unit == 'microeV'
        assert delta_function.area.value == 2 * 1e3
        assert delta_function.center.value == 0.5 * 1e3

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

        assert delta_copy.unit == delta_function.unit

    def test_repr(self, delta_function: DeltaFunction):
        # WHEN THEN
        repr_str = repr(delta_function)

        # EXPECT
        assert 'DeltaFunction' in repr_str
        assert 'unique_name = DeltaFunction' in repr_str
        assert 'unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
