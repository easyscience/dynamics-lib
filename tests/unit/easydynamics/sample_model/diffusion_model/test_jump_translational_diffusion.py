# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import DescriptorNumber
from scipp import UnitError
from scipp.constants import hbar as scipp_hbar

from easydynamics.sample_model.diffusion_model.jump_translational_diffusion import (
    JumpTranslationalDiffusion,
)

hbar_1 = DescriptorNumber('hbar', 1.0)
hbar = DescriptorNumber.from_scipp('hbar', scipp_hbar)
angstrom = DescriptorNumber('angstrom', 1e-10, unit='m')


class TestJumpTranslationalDiffusion:
    @pytest.fixture
    def jump_diffusion_model(self):
        return JumpTranslationalDiffusion()

    def test_init_default(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        assert jump_diffusion_model.display_name == 'JumpTranslationalDiffusion'
        assert jump_diffusion_model.unit == 'meV'
        assert jump_diffusion_model.scale.value == pytest.approx(1.0)
        assert jump_diffusion_model.diffusion_coefficient.value == pytest.approx(1.0)
        assert jump_diffusion_model.relaxation_time.value == pytest.approx(1.0)

    @pytest.mark.parametrize(
        'kwargs,expected_exception, expected_message',
        [
            (
                {
                    'unit': 123,
                    'scale': 1.0,
                    'diffusion_coefficient': 1.0,
                    'relaxation_time': 1.0,
                },
                UnitError,
                'Invalid unit',
            ),
            (
                {
                    'unit': 'meV',
                    'scale': 'invalid',
                    'diffusion_coefficient': 1.0,
                    'relaxation_time': 1.0,
                },
                TypeError,
                'scale must be a number',
            ),
            (
                {
                    'unit': 'meV',
                    'scale': 1.0,
                    'diffusion_coefficient': 'invalid',
                    'relaxation_time': 1.0,
                },
                TypeError,
                'diffusion_coefficient must be a number',
            ),
            (
                {
                    'unit': 'meV',
                    'scale': 1.0,
                    'diffusion_coefficient': 1.0,
                    'relaxation_time': 'invalid',
                },
                TypeError,
                'relaxation_time must be a number',
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            JumpTranslationalDiffusion(display_name='JumpTranslationalDiffusion', **kwargs)

    def test_diffusion_coefficient_setter(self, jump_diffusion_model):
        # WHEN
        jump_diffusion_model.diffusion_coefficient = 3.0

        # THEN EXPECT
        assert jump_diffusion_model.diffusion_coefficient.value == pytest.approx(3.0)

    def test_diffusion_coefficient_setter_raises(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'diffusion_coefficient must be a number.'):
            jump_diffusion_model.diffusion_coefficient = 'invalid'  # Invalid type

    def test_diffusion_coefficient_setter_negative_raises(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'diffusion_coefficient must be non-negative.'):
            jump_diffusion_model.diffusion_coefficient = -1.0  # Invalid negative value

    def test_relaxation_time_setter(self, jump_diffusion_model):
        # WHEN
        jump_diffusion_model.relaxation_time = 2.5

        # THEN EXPECT
        assert jump_diffusion_model.relaxation_time.value == pytest.approx(2.5)

    def test_relaxation_time_setter_raises(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'relaxation_time must be a number.'):
            jump_diffusion_model.relaxation_time = 'invalid'  # Invalid type

    def test_relaxation_time_setter_negative_raises(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'relaxation_time must be non-negative.'):
            jump_diffusion_model.relaxation_time = -1.0  # Invalid negative value

    def test_calculate_width_type_error(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            jump_diffusion_model.calculate_width(Q='invalid')  # Invalid type

    def test_calculate_width(self, jump_diffusion_model):
        "Test the calculation relying solely on a scipp implementation"
        'instead of our Parameters'
        # WHEN
        Q_values = sc.linspace('Q', 0.5, 1.5, num=6, unit='1/angstrom')
        relaxation_time_sc = jump_diffusion_model.relaxation_time.value * sc.Unit(
            jump_diffusion_model.relaxation_time.unit
        )
        diffusion_coefficient_sc = jump_diffusion_model.diffusion_coefficient.value * sc.Unit(
            jump_diffusion_model.diffusion_coefficient.unit
        )

        # THEN
        widths = jump_diffusion_model.calculate_width(Q_values)

        denominator = diffusion_coefficient_sc * relaxation_time_sc * Q_values**2
        denominator = denominator.to(unit='1')

        # EXPECT
        expected_widths = scipp_hbar * diffusion_coefficient_sc * (Q_values**2) / (1 + denominator)

        expected_widths = expected_widths.to(unit=jump_diffusion_model.unit)

        np.testing.assert_allclose(widths, expected_widths.values, rtol=1e-5)

    def test_calculate_EISF(self, jump_diffusion_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # THEN
        EISF = jump_diffusion_model.calculate_EISF(Q_values)

        # EXPECT
        expected_EISF = np.zeros_like(Q_values)
        np.testing.assert_array_equal(EISF, expected_EISF)

    def test_calculate_EISF_type_error(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            jump_diffusion_model.calculate_EISF(Q='invalid')  # Invalid type

    def test_calculate_QISF(self, jump_diffusion_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # THEN
        QISF = jump_diffusion_model.calculate_QISF(Q_values)

        # EXPECT
        expected_QISF = np.ones_like(Q_values)
        np.testing.assert_array_equal(QISF, expected_QISF)

    def test_calculate_QISF_type_error(self, jump_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            jump_diffusion_model.calculate_QISF(Q='invalid')  # Invalid type

    @pytest.mark.parametrize(
        'Q',
        [
            (0.5),
            ([1.0, 2.0, 3.0]),
            (np.array([1.0, 2.0, 3.0])),
        ],
        ids=[
            'python_scalar',
            'python_list',
            'numpy_array',
        ],
    )
    def test_create_component_collections(self, jump_diffusion_model, Q):
        # WHEN
        jump_diffusion_model.Q = Q

        # THEN
        component_collections = jump_diffusion_model.create_component_collections()

        # EXPECT
        expected_widths = jump_diffusion_model.calculate_width(Q)
        for model_index in range(len(component_collections)):
            model = component_collections[model_index]
            assert len(model) == 1
            component = model[0]
            assert component.width.unit == jump_diffusion_model.unit
            assert np.isclose(component.width.value, expected_widths[model_index])
            assert component.width.independent is False

    def test_write_width_dependency_expression(self, jump_diffusion_model):
        # WHEN THEN
        expression = jump_diffusion_model._write_width_dependency_expression(0.5)

        # EXPECT
        expected_expression = (
            'hbar * D* 0.5 **2/(angstrom**2)/(1 + (D * t* 0.5 **2/(angstrom**2)))'
        )
        assert expression == expected_expression

    def test_write_width_dependency_map_expression(self, jump_diffusion_model):
        # WHEN THEN
        expression_map = jump_diffusion_model._write_width_dependency_map_expression()

        # EXPECT
        expected_map = {
            'D': jump_diffusion_model.diffusion_coefficient,
            't': jump_diffusion_model.relaxation_time,
            'hbar': jump_diffusion_model._hbar,
            'angstrom': jump_diffusion_model._angstrom,
        }

        assert expression_map == expected_map

    def test_write_width_dependency_expression_raises(self, jump_diffusion_model):
        with pytest.raises(TypeError, match='Q must be a float'):
            jump_diffusion_model._write_width_dependency_expression('invalid')

    def test_write_area_dependency_expression_raises(self, jump_diffusion_model):
        with pytest.raises(TypeError, match='QISF must be a float'):
            jump_diffusion_model._write_area_dependency_expression('invalid')

    def test_repr(self, jump_diffusion_model):
        # WHEN THEN
        repr_str = repr(jump_diffusion_model)

        # EXPECT
        assert 'JumpTranslationalDiffusion' in repr_str
        assert 'diffusion_coefficient' in repr_str
        assert 'scale=' in repr_str
