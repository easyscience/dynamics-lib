# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin


class TestCreateParametersMixin:
    @pytest.fixture
    def dummy_model(self):
        return CreateParametersMixin()

    # ------------- Area----------------------
    @pytest.mark.parametrize('unit', ['meV', 'eV'])
    @pytest.mark.parametrize('area_input', [2, 2.0])
    def test_create_area_parameter_from_numeric(self, dummy_model, area_input, unit):
        # WHEN THEN
        area_param = dummy_model._create_area_parameter(area_input, 'TestModel', x_unit=unit)

        # EXPECT
        assert isinstance(area_param, Parameter)
        assert area_param.name == 'TestModel area'
        assert area_param.value == pytest.approx(area_input)
        assert area_param.unit == unit
        assert not area_param.fixed
        assert area_param.min == pytest.approx(0.0)

    def test_create_area_parameter_invalid_type_raises(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='area must be a number'):
            dummy_model._create_area_parameter('invalid', 'TestModel')

    @pytest.mark.parametrize(
        'non_finite_area',
        [
            np.nan,
            np.inf,
            -np.inf,
        ],
    )
    def test_create_area_parameter_invalid_numeric_raises(self, dummy_model, non_finite_area):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='area must be a finite number'):
            dummy_model._create_area_parameter(non_finite_area, 'TestModel')

    def test_negative_area_warns(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            area_param = dummy_model._create_area_parameter(-2.0, 'TestModel')

        assert area_param.min == -float('inf')  # No min constraint for negative area

    # ------------- Bounded value assignment ----------------------
    def test_set_bounded_parameter_value_within_bounds(self, dummy_model):
        # WHEN
        param = Parameter(name='p', value=1.0, min=0.0, max=2.0)

        # THEN
        dummy_model._set_bounded_parameter_value(param, 1.5, 'p')

        # EXPECT
        assert param.value == pytest.approx(1.5)

    @pytest.mark.parametrize('out_of_bounds', [-1.0, 3.0], ids=['below_min', 'above_max'])
    def test_set_bounded_parameter_value_out_of_bounds_raises(self, dummy_model, out_of_bounds):
        # WHEN
        param = Parameter(name='p', value=1.0, min=0.0, max=2.0)

        # THEN EXPECT the assignment raises instead of silently clamping, leaving the value
        with pytest.raises(ValueError, match='violates the parameter bounds'):
            dummy_model._set_bounded_parameter_value(param, out_of_bounds, 'p')
        assert param.value == pytest.approx(1.0)

    def test_set_bounded_parameter_value_invalid_type_raises(self, dummy_model):
        # WHEN
        param = Parameter(name='p', value=1.0, min=0.0, max=2.0)

        # THEN EXPECT
        with pytest.raises(TypeError, match='p must be a number'):
            dummy_model._set_bounded_parameter_value(param, 'invalid', 'p')

    # ------------- Center----------------------
    @pytest.mark.parametrize('unit', ['meV', 'eV'])
    @pytest.mark.parametrize('center_input', [0, 0.0])
    def test_create_center_parameter_from_numeric(self, dummy_model, center_input, unit):
        # WHEN THEN
        center_param = dummy_model._create_center_parameter(
            center_input, 'TestModel', fix_if_none=False, x_unit=unit
        )
        # EXPECT
        assert isinstance(center_param, Parameter)
        assert center_param.name == 'TestModel center'
        assert center_param.value == pytest.approx(center_input)
        assert center_param.unit == unit
        assert not center_param.fixed

    @pytest.mark.parametrize('fix_if_none', [True, False])
    def test_create_center_parameter_from_None(self, dummy_model, fix_if_none):
        # WHEN THEN
        center_param = dummy_model._create_center_parameter(
            None, 'TestModel', fix_if_none=fix_if_none
        )
        # EXPECT
        assert isinstance(center_param, Parameter)
        assert center_param.name == 'TestModel center'
        assert center_param.value == pytest.approx(0.0)
        assert center_param.unit == 'meV'
        assert center_param.fixed == fix_if_none

    def test_create_center_parameter_invalid_type_raises(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='center must be None or a number'):
            dummy_model._create_center_parameter('invalid', 'TestModel', fix_if_none=False)

    @pytest.mark.parametrize('center_input', [None, 0.0, 1e-12])
    def test_create_center_parameter_enforce_minimum_clamps(self, dummy_model, center_input):
        # WHEN: a center below the DHO minimum (regression: setting min before clamping the
        # value used to raise ValueError from Parameter.min)
        center_param = dummy_model._create_center_parameter(
            center_input, 'TestModel', fix_if_none=False, enforce_minimum_center=True
        )

        # EXPECT: value clamped up to the minimum, and the bound enforced
        assert center_param.value == pytest.approx(1e-10)
        assert center_param.min == pytest.approx(1e-10)

    @pytest.mark.parametrize(
        'non_finite_center',
        [
            np.nan,
            np.inf,
            -np.inf,
        ],
    )
    def test_create_center_parameter_invalid_numeric_raises(self, dummy_model, non_finite_center):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='center must be None or a finite number'):
            dummy_model._create_center_parameter(non_finite_center, 'TestModel', fix_if_none=False)

    @pytest.mark.parametrize('unit', ['meV', 'eV'])
    @pytest.mark.parametrize('width_input', [2, 2.0])
    def test_create_width_parameter_from_numeric(self, dummy_model, width_input, unit):
        # WHEN THEN
        width_param = dummy_model._create_width_parameter(
            width_input, 'TestModel', param_name='width', x_unit=unit
        )
        # EXPECT
        assert isinstance(width_param, Parameter)
        assert width_param.name == 'TestModel width'
        assert width_param.value == pytest.approx(width_input)
        assert width_param.unit == unit
        assert not width_param.fixed

    def test_create_width_parameter_invalid_type_raises(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='width must be a number'):
            dummy_model._create_width_parameter('invalid', 'TestModel', param_name='width')

    def test_create_width_parameter_negative_raises(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=' must be greater than zero'):
            dummy_model._create_width_parameter(-1, 'TestModel', param_name='width')

    @pytest.mark.parametrize(
        'non_finite_center',
        [
            np.nan,
            np.inf,
            -np.inf,
        ],
    )
    def test_create_width_parameter_invalid_numeric_raises(self, dummy_model, non_finite_center):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='width must be a finite number'):
            dummy_model._create_width_parameter(non_finite_center, 'TestModel')
