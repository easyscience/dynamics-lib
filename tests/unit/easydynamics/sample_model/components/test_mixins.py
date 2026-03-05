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
        area_param = dummy_model._create_area_parameter(area_input, 'TestModel', unit=unit)

        # EXPECT
        assert isinstance(area_param, Parameter)
        assert area_param.name == 'TestModel area'
        assert area_param.value == float(area_input)
        assert area_param.unit == unit
        assert not area_param.fixed
        assert area_param.min == 0.0

    def test_create_area_parameter_from_parameter(self, dummy_model):
        # WHEN
        area_input = Parameter(name='input_area', value=3.0, unit='meV', fixed=True)

        # THEN
        area_param = dummy_model._create_area_parameter(area_input, 'TestModel')

        # EXPECT
        assert area_param is area_input  # Should be the same object
        assert area_param.min == 0.0

    def test_create_area_parameter_invalid_type_raises(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='area must be a number or a Parameter'):
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
        with pytest.raises(ValueError, match='area must be a finite number or a Parameter'):
            dummy_model._create_area_parameter(non_finite_area, 'TestModel')

    def test_negative_area_warns(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            area_param = dummy_model._create_area_parameter(-2.0, 'TestModel')

        assert area_param.min == -float('inf')  # No min constraint for negative area

    # ------------- Center----------------------
    @pytest.mark.parametrize('unit', ['meV', 'eV'])
    @pytest.mark.parametrize('center_input', [0, 0.0])
    def test_create_center_parameter_from_numeric(self, dummy_model, center_input, unit):
        # WHEN THEN
        center_param = dummy_model._create_center_parameter(
            center_input, 'TestModel', fix_if_none=False, unit=unit
        )
        # EXPECT
        assert isinstance(center_param, Parameter)
        assert center_param.name == 'TestModel center'
        assert center_param.value == float(center_input)
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
        assert center_param.value == 0.0
        assert center_param.unit == 'meV'
        assert center_param.fixed == fix_if_none

    def test_create_center_parameter_from_parameter(self, dummy_model):
        # WHEN
        center_input = Parameter(name='input_center', value=1.0, unit='meV', fixed=True)

        # THEN
        center_param = dummy_model._create_center_parameter(
            center_input, 'TestModel', fix_if_none=False
        )

        # EXPECT
        assert center_param is center_input  # Should be the same object

    def test_create_center_parameter_invalid_type_raises(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='center must be None, a number, or a Parameter'):
            dummy_model._create_center_parameter('invalid', 'TestModel', fix_if_none=False)

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
        with pytest.raises(
            ValueError, match='center must be None, a finite number or a Parameter'
        ):
            dummy_model._create_center_parameter(non_finite_center, 'TestModel', fix_if_none=False)

    @pytest.mark.parametrize('unit', ['meV', 'eV'])
    @pytest.mark.parametrize('width_input', [2, 2.0])
    def test_create_width_parameter_from_numeric(self, dummy_model, width_input, unit):
        # WHEN THEN
        width_param = dummy_model._create_width_parameter(
            width_input, 'TestModel', param_name='width', unit=unit
        )
        # EXPECT
        assert isinstance(width_param, Parameter)
        assert width_param.name == 'TestModel width'
        assert width_param.value == float(width_input)
        assert width_param.unit == unit
        assert not width_param.fixed

    def test_create_width_parameter_from_parameter(self, dummy_model):
        # WHEN
        width_input = Parameter(name='input_width', value=3.0, unit='meV', fixed=True)

        # THEN
        width_param = dummy_model._create_width_parameter(
            width_input, 'TestModel', param_name='width'
        )

        # EXPECT
        assert width_param is width_input  # Should be the same object

    def test_create_width_parameter_from_parameter_negative_raises(self, dummy_model):
        # WHEN
        width_input = Parameter(name='input_width', value=-1.0, unit='meV', fixed=True)

        # THEN EXPECT
        with pytest.raises(ValueError, match=' must be greater than zero'):
            dummy_model._create_width_parameter(width_input, 'TestModel', param_name='width')

    def test_create_width_parameter_invalid_type_raises(self, dummy_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='width must be a number or a Parameter'):
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
        with pytest.raises(ValueError, match='width must be a finite number or a Parameter'):
            dummy_model._create_width_parameter(non_finite_center, 'TestModel')
