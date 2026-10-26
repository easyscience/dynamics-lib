# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError
from scipp.constants import hbar as scipp_hbar

from easydynamics.sample_model.diffusion_model.hall_ross_jump_diffusion import MINIMUM_WIDTH
from easydynamics.sample_model.diffusion_model.hall_ross_jump_diffusion import (
    HallRossJumpDiffusion,
)


class TestHallRossJumpDiffusion:
    @pytest.fixture
    def hall_ross_model(self):
        return HallRossJumpDiffusion()

    def test_init_default(self, hall_ross_model):
        # WHEN THEN EXPECT
        assert hall_ross_model.display_name == 'HallRossJumpDiffusion'
        assert hall_ross_model.x_unit == 'meV'
        assert hall_ross_model.y_unit == 'dimensionless'
        assert hall_ross_model.scale.value == pytest.approx(1.0)
        assert hall_ross_model.scale.unit == 'meV'
        assert hall_ross_model.residence_time.value == pytest.approx(1.0)
        assert hall_ross_model.jump_length.value == pytest.approx(1.0)

    @pytest.mark.parametrize(
        'kwargs,expected_exception, expected_message',
        [
            (
                {
                    'x_unit': 123,
                    'scale': 1.0,
                    'residence_time': 1.0,
                    'jump_length': 1.0,
                },
                UnitError,
                'Invalid unit',
            ),
            (
                {
                    'y_unit': 123,
                    'scale': 1.0,
                    'residence_time': 1.0,
                    'jump_length': 1.0,
                },
                UnitError,
                'unit must be None, a string',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 'invalid',
                    'residence_time': 1.0,
                    'jump_length': 1.0,
                },
                TypeError,
                'scale must be a number',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 1.0,
                    'residence_time': 'invalid',
                    'jump_length': 1.0,
                },
                TypeError,
                'residence_time must be a number',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 1.0,
                    'residence_time': -1.0,
                    'jump_length': 1.0,
                },
                ValueError,
                'residence_time must be at least',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 1.0,
                    'residence_time': 0.0,
                    'jump_length': 1.0,
                },
                ValueError,
                'residence_time must be at least',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 1.0,
                    'residence_time': 1.0,
                    'jump_length': 'invalid',
                },
                TypeError,
                'jump_length must be a number',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 1.0,
                    'residence_time': 1.0,
                    'jump_length': -1.0,
                },
                ValueError,
                'jump_length must be non-negative',
            ),
        ],
        ids=[
            'invalid_x_unit',
            'invalid_y_unit',
            'invalid_scale_type',
            'invalid_residence_time_type',
            'invalid_residence_time_negative',
            'invalid_residence_time_zero',
            'invalid_jump_length_type',
            'invalid_jump_length_negative',
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            HallRossJumpDiffusion(display_name='HallRossJumpDiffusion', **kwargs)

    def test_residence_time_setter(self, hall_ross_model):
        # WHEN THEN
        hall_ross_model.residence_time = 3.0

        # EXPECT
        assert hall_ross_model.residence_time.value == pytest.approx(3.0)

    def test_residence_time_setter_raises(self, hall_ross_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'residence_time must be a number.'):
            hall_ross_model.residence_time = 'invalid'

    @pytest.mark.parametrize('invalid_value', [-1.0, 0.0, MINIMUM_WIDTH / 10])
    def test_residence_time_setter_below_minimum_raises(self, hall_ross_model, invalid_value):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'residence_time must be at least'):
            hall_ross_model.residence_time = invalid_value

    def test_residence_time_has_minimum_width(self, hall_ross_model):
        # WHEN THEN EXPECT
        assert hall_ross_model.residence_time.min == MINIMUM_WIDTH

    def test_jump_length_setter(self, hall_ross_model):
        # WHEN THEN
        hall_ross_model.jump_length = 2.5

        # EXPECT
        assert hall_ross_model.jump_length.value == pytest.approx(2.5)

    def test_jump_length_setter_raises(self, hall_ross_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'jump_length must be a number.'):
            hall_ross_model.jump_length = 'invalid'

    def test_jump_length_setter_negative_raises(self, hall_ross_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'jump_length must be non-negative.'):
            hall_ross_model.jump_length = -1.0

    def test_calculate_width_type_error(self, hall_ross_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            hall_ross_model.calculate_width(Q='invalid')

    def test_calculate_width(self, hall_ross_model):
        "Test the calculation against a scipp implementation instead of our Parameters"
        # WHEN
        hall_ross_model.residence_time = 2.0
        hall_ross_model.jump_length = 1.5
        Q_values = sc.linspace('Q', 0.5, 1.5, num=6, unit='1/angstrom')
        # sc.values drops the (zero) variance, which scipp cannot broadcast against Q_values
        residence_time_sc = sc.values(hall_ross_model.residence_time.full_value)
        jump_length_sc = sc.values(hall_ross_model.jump_length.full_value)

        # THEN
        widths = hall_ross_model.calculate_width(Q_values)

        # EXPECT
        argument = (Q_values * jump_length_sc).to(unit='dimensionless')
        expected_widths = scipp_hbar / residence_time_sc * (1 - sc.exp(-(argument**2) / 2))
        expected_widths = expected_widths.to(unit=hall_ross_model.x_unit)
        np.testing.assert_allclose(widths, expected_widths.values, rtol=1e-5)

    def test_calculate_width_small_Q_limit_is_brownian(self, hall_ross_model):
        """
        At small Q the width is hbar * D * Q**2 with D = l**2 / (2 tau). Q = 1e-6 also checks
        that the width keeps its precision where 1 - exp(-x) would cancel.
        """
        # WHEN
        hall_ross_model.residence_time = 2.0
        hall_ross_model.jump_length = 1.5
        Q_values = np.array([1e-6, 1e-4])

        # THEN
        widths = hall_ross_model.calculate_width(Q_values)

        # EXPECT
        diffusion_coefficient = sc.scalar(1.5**2 / (2 * 2.0), unit='angstrom**2/ps')
        Q_sc = sc.array(dims=['Q'], values=Q_values, unit='1/angstrom')
        expected_widths = (scipp_hbar * diffusion_coefficient * Q_sc**2).to(unit='meV')
        np.testing.assert_allclose(widths, expected_widths.values, rtol=1e-8)

    def test_calculate_width_large_Q_limit(self, hall_ross_model):
        # WHEN
        hall_ross_model.residence_time = 2.0
        Q_values = np.array([50.0, 100.0])

        # THEN
        widths = hall_ross_model.calculate_width(Q_values)

        # EXPECT width tends to hbar / tau
        expected_width = (scipp_hbar / sc.scalar(2.0, unit='ps')).to(unit='meV')
        np.testing.assert_allclose(widths, expected_width.value, rtol=1e-12)

    def test_calculate_EISF(self, hall_ross_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])

        # THEN
        EISF = hall_ross_model.calculate_EISF(Q_values)

        # EXPECT
        np.testing.assert_array_equal(EISF, np.zeros_like(Q_values))

    def test_calculate_EISF_type_error(self, hall_ross_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            hall_ross_model.calculate_EISF(Q='invalid')

    def test_calculate_QISF(self, hall_ross_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])

        # THEN
        QISF = hall_ross_model.calculate_QISF(Q_values)

        # EXPECT
        np.testing.assert_array_equal(QISF, np.ones_like(Q_values))

    def test_calculate_QISF_type_error(self, hall_ross_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            hall_ross_model.calculate_QISF(Q='invalid')

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
    def test_create_component_collections(self, hall_ross_model, Q):
        # WHEN
        hall_ross_model.Q = Q

        # THEN
        component_collections = hall_ross_model.create_component_collections()

        # EXPECT
        expected_widths = hall_ross_model.calculate_width(Q)
        for model_index in range(len(component_collections)):
            model = component_collections[model_index]
            assert len(model) == 1
            component = model[0]
            assert component.width.unit == hall_ross_model.x_unit
            assert np.isclose(component.width.value, expected_widths[model_index])
            assert component.width.independent is False
            # area.unit = area_unit = x_unit * y_unit
            assert component.area.unit == 'meV'

    def test_component_widths_follow_parameters(self):
        # WHEN
        model = HallRossJumpDiffusion(Q=np.array([0.5, 1.0]))

        # THEN
        model.residence_time = 4.0
        model.jump_length = 2.0

        # EXPECT the dependent widths update with the parameters
        widths = [collection[0].width.value for collection in model.get_component_collections()]
        np.testing.assert_allclose(widths, model.calculate_width(np.array([0.5, 1.0])))

    def test_component_width_keeps_precision_at_small_Q(self):
        """The dependent widths a fit uses must not lose digits where 1 - exp(-x) cancels."""
        # WHEN
        model = HallRossJumpDiffusion(Q=np.array([1e-6]), residence_time=2.0, jump_length=1.5)

        # THEN
        width = model.get_component_collections()[0][0].width.value

        # EXPECT hbar * D * Q**2 with D = l**2 / (2 tau)
        diffusion_coefficient = sc.scalar(1.5**2 / (2 * 2.0), unit='angstrom**2/ps')
        Q_sc = sc.scalar(1e-6, unit='1/angstrom')
        expected_width = (scipp_hbar * diffusion_coefficient * Q_sc**2).to(unit='meV')
        # abs=0: pytest.approx's default absolute tolerance of 1e-12 exceeds this width (~4e-13)
        assert width == pytest.approx(expected_width.value, rel=1e-8, abs=0)

    def test_create_component_collections_installs_collections(self):
        # WHEN
        model = HallRossJumpDiffusion(Q=np.array([1.0, 2.0]))

        # THEN
        collections = model.create_component_collections()

        # EXPECT the returned collections are the installed (live) ones
        assert collections is model.get_component_collections()

    def test_write_width_dependency_expression(self, hall_ross_model):
        # WHEN THEN
        expression = hall_ross_model._write_width_dependency_expression(0.5)

        # EXPECT
        assert expression == '(hbar / tau) * -expm1(-((0.5 * l.value) ** 2) / 2)'

    def test_write_width_dependency_map_expression(self, hall_ross_model):
        # WHEN THEN
        expression_map = hall_ross_model._write_width_dependency_map_expression()

        # EXPECT
        expected_map = {
            'tau': hall_ross_model.residence_time,
            'l': hall_ross_model.jump_length,
            'hbar': hall_ross_model._hbar,
        }
        assert expression_map == expected_map

    def test_write_width_dependency_expression_raises(self, hall_ross_model):
        with pytest.raises(TypeError, match='Q must be a float'):
            hall_ross_model._write_width_dependency_expression('invalid')

    def test_write_area_dependency_expression_raises(self, hall_ross_model):
        with pytest.raises(TypeError, match='QISF must be a float'):
            hall_ross_model._write_area_dependency_expression('invalid')

    def test_y_unit_setter_raises(self, hall_ross_model):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=r'read-only'):
            hall_ross_model.y_unit = '1/meV'

    def test_repr(self, hall_ross_model):
        # WHEN THEN
        repr_str = repr(hall_ross_model)

        # EXPECT
        assert 'HallRossJumpDiffusion' in repr_str
        assert 'residence_time' in repr_str
        assert 'jump_length' in repr_str
        assert 'scale=' in repr_str
        assert 'x_unit=meV, y_unit=dimensionless' in repr_str
