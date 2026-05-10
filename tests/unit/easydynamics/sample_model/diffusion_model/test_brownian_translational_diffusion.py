# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import DescriptorNumber
from scipp import UnitError
from scipp.constants import hbar as scipp_hbar

from easydynamics.sample_model.diffusion_model.brownian_translational_diffusion import (
    BrownianTranslationalDiffusion,
)

hbar_1 = DescriptorNumber('hbar', 1.0)
hbar = DescriptorNumber.from_scipp('hbar', scipp_hbar)
angstrom = DescriptorNumber('angstrom', 1e-10, unit='m')


class TestBrownianTranslationalDiffusion:
    @pytest.fixture
    def brownian_diffusion_model(self):
        return BrownianTranslationalDiffusion()

    def test_init_default(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        assert brownian_diffusion_model.display_name == 'BrownianTranslationalDiffusion'
        assert brownian_diffusion_model.unit == 'meV'
        assert brownian_diffusion_model.scale.value == pytest.approx(1.0)
        assert brownian_diffusion_model.diffusion_coefficient.value == pytest.approx(1.0)

    @pytest.mark.parametrize(
        'kwargs,expected_exception, expected_message',
        [
            (
                {
                    'unit': 123,
                    'scale': 1.0,
                    'diffusion_coefficient': 1.0,
                },
                UnitError,
                'Invalid unit',
            ),
            (
                {
                    'unit': 123,
                    'scale': 'invalid',
                    'diffusion_coefficient': 1.0,
                },
                TypeError,
                'scale must be a number',
            ),
            (
                {
                    'unit': 123,
                    'scale': 1.0,
                    'diffusion_coefficient': 'invalid',
                },
                TypeError,
                'diffusion_coefficient must be a number',
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            BrownianTranslationalDiffusion(display_name='BrownianTranslationalDiffusion', **kwargs)

    def test_diffusion_coefficient_setter(self, brownian_diffusion_model):
        # WHEN
        brownian_diffusion_model.diffusion_coefficient = 3.0

        # THEN EXPECT
        assert brownian_diffusion_model.diffusion_coefficient.value == pytest.approx(3.0)

    def test_diffusion_coefficient_setter_raises(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'diffusion_coefficient must be a number.'):
            brownian_diffusion_model.diffusion_coefficient = 'invalid'  # Invalid type

    def test_diffusion_coefficient_setter_negative_raises(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'diffusion_coefficient must be non-negative.'):
            brownian_diffusion_model.diffusion_coefficient = -1.0  # Invalid negative value

    def test_calculate_width_type_error(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            brownian_diffusion_model.calculate_width(Q='invalid')  # Invalid type

    def test_calculate_width(self, brownian_diffusion_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # WHEN
        widths = brownian_diffusion_model.calculate_width(Q_values)

        # THEN EXPECT
        unit_conversion_factor = sc.to_unit(
            1
            * sc.Unit(brownian_diffusion_model.diffusion_coefficient.unit)
            * scipp_hbar
            / (1 * sc.Unit('Å') ** 2),
            'meV',
        )
        expected_widths = 1.0 * unit_conversion_factor.value * (Q_values**2)
        np.testing.assert_allclose(widths, expected_widths, rtol=1e-5)

    def test_calculate_EISF(self, brownian_diffusion_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # THEN
        EISF = brownian_diffusion_model.calculate_EISF(Q_values)

        # EXPECT
        expected_EISHF = np.zeros_like(Q_values)
        np.testing.assert_array_equal(EISF, expected_EISHF)

    def test_calculate_EISF_type_error(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            brownian_diffusion_model.calculate_EISF(Q='invalid')  # Invalid type

    def test_calculate_QISF(self, brownian_diffusion_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # THEN
        QISF = brownian_diffusion_model.calculate_QISF(Q_values)

        # EXPECT
        expected_QISF = np.ones_like(Q_values)
        np.testing.assert_array_equal(QISF, expected_QISF)

    def test_calculate_QISF_type_error(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be '):
            brownian_diffusion_model.calculate_QISF(Q='invalid')  # Invalid type

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
    def test_create_component_collections(self, brownian_diffusion_model, Q):
        # WHEN

        # THEN
        component_collections = brownian_diffusion_model.create_component_collections(Q=Q)

        # EXPECT
        expected_widths = brownian_diffusion_model.calculate_width(Q)
        for model_index in range(len(component_collections)):
            model = component_collections[model_index]
            assert len(model.components) == 1
            component = model.components[0]
            assert component.width.unit == brownian_diffusion_model.unit
            assert np.isclose(component.width.value, expected_widths[model_index])
            assert component.width.independent is False

    def test_create_component_collections_component_name_must_be_string(
        self, brownian_diffusion_model
    ):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'component_name must be a string.'):
            brownian_diffusion_model.create_component_collections(
                Q=np.array([0.1, 0.2, 0.3]), component_name=123
            )

    def test_create_component_collections_component_display_name_must_be_string(
        self, brownian_diffusion_model
    ):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'component_display_name must be a string.'):
            brownian_diffusion_model.create_component_collections(
                Q=np.array([0.1, 0.2, 0.3]), component_display_name=123
            )

    def test_create_component_collections_Q_type_error(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be a '):
            brownian_diffusion_model.create_component_collections(Q='invalid')  # Invalid type

    def test_create_component_collections_Q_1dimensional_error(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'Q must be a 1-dimensional array.'):
            brownian_diffusion_model.create_component_collections(
                Q=np.array([[0.1, 0.2], [0.3, 0.4]])
            )  # Invalid shape

    def test_write_width_dependency_expression(self, brownian_diffusion_model):
        # WHEN THEN
        expression = brownian_diffusion_model._write_width_dependency_expression(0.5)

        # EXPECT
        expected_expression = 'hbar * D* 0.5 **2*1/(angstrom**2)'
        assert expression == expected_expression

    def test_write_width_dependency_map_expression(self, brownian_diffusion_model):
        # WHEN THEN
        expression_map = brownian_diffusion_model._write_width_dependency_map_expression()

        # EXPECT
        expected_map = {
            'D': brownian_diffusion_model.diffusion_coefficient,
            'hbar': brownian_diffusion_model._hbar,
            'angstrom': brownian_diffusion_model._angstrom,
        }

        assert expression_map == expected_map

    def test_write_width_dependency_expression_raises(self, brownian_diffusion_model):
        with pytest.raises(TypeError, match='Q must be a float'):
            brownian_diffusion_model._write_width_dependency_expression('invalid')

    def test_write_area_dependency_expression_raises(self, brownian_diffusion_model):
        with pytest.raises(TypeError, match='QISF must be a float'):
            brownian_diffusion_model._write_area_dependency_expression('invalid')

    def test_repr(self, brownian_diffusion_model):
        # WHEN THEN
        repr_str = repr(brownian_diffusion_model)

        # EXPECT
        assert 'BrownianTranslationalDiffusion' in repr_str
        assert 'diffusion_coefficient' in repr_str
        assert 'scale=' in repr_str
