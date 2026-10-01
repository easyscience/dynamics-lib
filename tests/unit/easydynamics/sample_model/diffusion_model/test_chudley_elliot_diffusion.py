# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError
from scipp.constants import hbar as scipp_hbar

from easydynamics.sample_model.diffusion_model.chudley_elliot_jump_diffusion import (
    ChudleyElliotJumpDiffusion,
)


class TestChudleyElliotJumpDiffusion:
    @pytest.fixture
    def chudley_elliot_model(self):
        return ChudleyElliotJumpDiffusion()

    def test_init_default(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        assert chudley_elliot_model.display_name == 'ChudleyElliotJumpDiffusion'
        assert chudley_elliot_model.x_unit == 'meV'
        assert chudley_elliot_model.y_unit == 'dimensionless'
        assert chudley_elliot_model.scale.value == pytest.approx(1.0)
        assert chudley_elliot_model.scale.unit == 'meV'
        assert chudley_elliot_model.residence_time.value == pytest.approx(1.0)
        assert chudley_elliot_model.jump_length.value == pytest.approx(1.0)

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
                # causes a UnitError in scipp. Why?
                # Why does Unit check work for for x_unit, but not for y_unit?
                # UnitError,
                # 'Invalid unit',
                TypeError,
                None,
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
                'residence_time must be non-negative',
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
            'invalid_jump_length_type',
            'invalid_jump_length_negative',
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            ChudleyElliotJumpDiffusion(display_name='ChudleyElliotJumpDiffusion', **kwargs)

    def test_residence_time_setter(self, chudley_elliot_model):
        # WHEN
        chudley_elliot_model.residence_time = 3.0

        # THEN EXPECT
        assert chudley_elliot_model.residence_time.value == pytest.approx(3.0)

    def test_residence_time_setter_raises(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'residence_time must be a number.'):
            chudley_elliot_model.residence_time = 'invalid'  # Invalid type

    def test_residence_time_setter_negative_raises(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'residence_time must be non-negative.'):
            chudley_elliot_model.residence_time = -1.0  # Invalid negative value

    def test_jump_length_setter(self, chudley_elliot_model):
        # WHEN
        chudley_elliot_model.jump_length = 2.5

        # THEN EXPECT
        assert chudley_elliot_model.jump_length.value == pytest.approx(2.5)

    def test_jump_length_setter_raises(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'jump_length must be a number.'):
            chudley_elliot_model.jump_length = 'invalid'  # Invalid type

    def test_jump_length_setter_negative_raises(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'jump_length must be non-negative.'):
            chudley_elliot_model.jump_length = -1.0  # Invalid negative value

    def test_calculate_width_type_error(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            chudley_elliot_model.calculate_width(Q='invalid')  # Invalid type

    def test_calculate_width(self, chudley_elliot_model):
        "Test the calculation relying solely on a scipp implementation"
        'instead of our Parameters'
        # WHEN
        Q_values = sc.linspace('Q', 0.5, 1.5, num=6, unit='1/angstrom')
        residence_time_sc = chudley_elliot_model.residence_time.value * sc.Unit(
            chudley_elliot_model.residence_time.unit
        )
        jump_length_sc = chudley_elliot_model.jump_length.value * sc.Unit(
            chudley_elliot_model.jump_length.unit
        )

        # THEN
        widths = chudley_elliot_model.calculate_width(Q_values)

        # EXPECT
        expected_widths = scipp_hbar / residence_time_sc * (1 - sc.sinc(Q_values * jump_length_sc))

        expected_widths = expected_widths.to(unit=chudley_elliot_model.x_unit)

        np.testing.assert_allclose(widths, expected_widths.values, rtol=1e-5)

    def test_calculate_width_sinc_stability(self, chudley_elliot_model):
        """
        Cross-check the calculate_width result (which uses np.sinc)
        against an explicit sin(x)/x calculation to ensure mathematical
        equivalence and stability.
        """
        # WHEN
        Q_values = sc.linspace('Q', 0.5, 1.5, num=6, unit='1/angstrom')

        residence_time_sc = chudley_elliot_model.residence_time.value * sc.Unit(
            chudley_elliot_model.residence_time.unit
        )
        jump_length_sc = chudley_elliot_model.jump_length.value * sc.Unit(
            chudley_elliot_model.jump_length.unit
        )

        # Model uses np.sinc():
        model_widths = chudley_elliot_model.calculate_width(Q_values)

        # THEN
        # Calculate explicitly using sin(x) / x
        argument = Q_values * jump_length_sc
        prefactor = scipp_hbar / residence_time_sc

        # sc.sin() strictly requires rad or deg unit.
        # Multiply by 1 rad to give the dimensionless argument the correct unit.
        argument_rad = argument * sc.scalar(1.0, unit='rad')

        explicit_sinc = sc.sin(argument_rad) / argument

        expected_widths_sin = prefactor * (1 - explicit_sinc)
        expected_widths_sin = expected_widths_sin.to(unit=chudley_elliot_model.x_unit)

        # EXPECT
        # Both mathematical approaches yield the same result
        np.testing.assert_allclose(model_widths, expected_widths_sin.values, rtol=1e-5)

    def test_calculate_EISF(self, chudley_elliot_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # THEN
        EISF = chudley_elliot_model.calculate_EISF(Q_values)

        # EXPECT
        expected_EISF = np.zeros_like(Q_values)
        np.testing.assert_array_equal(EISF, expected_EISF)

    def test_calculate_EISF_type_error(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            chudley_elliot_model.calculate_EISF(Q='invalid')  # Invalid type

    def test_calculate_QISF(self, chudley_elliot_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # THEN
        QISF = chudley_elliot_model.calculate_QISF(Q_values)

        # EXPECT
        expected_QISF = np.ones_like(Q_values)
        np.testing.assert_array_equal(QISF, expected_QISF)

    def test_calculate_QISF_type_error(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            chudley_elliot_model.calculate_QISF(Q='invalid')  # Invalid type

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
    def test_create_component_collections(self, chudley_elliot_model, Q):
        # WHEN
        chudley_elliot_model.Q = Q

        # THEN
        component_collections = chudley_elliot_model.create_component_collections()

        # EXPECT
        expected_widths = chudley_elliot_model.calculate_width(Q)
        for model_index in range(len(component_collections)):
            model = component_collections[model_index]
            assert len(model) == 1
            component = model[0]
            assert component.width.unit == chudley_elliot_model.x_unit
            assert np.isclose(component.width.value, expected_widths[model_index])
            assert component.width.independent is False
            # area.unit = area_unit = x_unit * y_unit
            assert component.area.unit == 'meV'

    def test_create_component_collections_installs_collections(self):
        # WHEN
        model = ChudleyElliotJumpDiffusion(Q=np.array([1.0, 2.0]))

        # THEN
        collections = model.create_component_collections()

        # EXPECT the returned collections are the installed (live) ones, so callers that
        # follow the docstring get the same objects the model itself uses
        assert collections is model.get_component_collections()

    def test_write_width_dependency_expression(self, chudley_elliot_model):
        # WHEN THEN
        expression = chudley_elliot_model._write_width_dependency_expression(0.5)

        # EXPECT
        expected_expression = '(hbar / tau) * (1 - sin(0.5 * l.value) / (0.5 * l.value) )'
        assert expression == expected_expression

    def test_write_width_dependency_map_expression(self, chudley_elliot_model):
        # WHEN THEN
        expression_map = chudley_elliot_model._write_width_dependency_map_expression()

        # EXPECT
        expected_map = {
            'tau': chudley_elliot_model.residence_time,
            'l': chudley_elliot_model.jump_length,
            'hbar': chudley_elliot_model._hbar,
            'angstrom': chudley_elliot_model._angstrom,
        }

        assert expression_map == expected_map

    def test_write_width_dependency_expression_raises(self, chudley_elliot_model):
        with pytest.raises(TypeError, match='Q must be a float'):
            chudley_elliot_model._write_width_dependency_expression('invalid')

    def test_write_area_dependency_expression_raises(self, chudley_elliot_model):
        with pytest.raises(TypeError, match='QISF must be a float'):
            chudley_elliot_model._write_area_dependency_expression('invalid')

    def test_y_unit_setter_raises(self, chudley_elliot_model):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=r'read-only'):
            chudley_elliot_model.y_unit = '1/meV'

    def test_repr(self, chudley_elliot_model):
        # WHEN THEN
        repr_str = repr(chudley_elliot_model)

        # EXPECT
        assert 'ChudleyElliotJumpDiffusion' in repr_str
        assert 'residence_time' in repr_str
        assert 'jump_length' in repr_str
        assert 'scale=' in repr_str
        # Regression: a stray ')' used to mangle this into 'x_unit=meV), y_unit=...'
        assert 'x_unit=meV, y_unit=dimensionless' in repr_str
