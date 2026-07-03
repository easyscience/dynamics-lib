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
        assert brownian_diffusion_model.x_unit == 'meV'
        assert brownian_diffusion_model.y_unit == 'dimensionless'
        assert brownian_diffusion_model.scale.value == pytest.approx(1.0)
        assert brownian_diffusion_model.scale.unit == 'meV'
        assert brownian_diffusion_model.diffusion_coefficient.value == pytest.approx(1.0)

    def test_convert_x_unit_rescales_widths(self):
        # WHEN
        model = BrownianTranslationalDiffusion(diffusion_coefficient=2.4e-9, Q=np.array([1.0]))
        width_mev = model.calculate_width()[0]

        # THEN
        model.convert_x_unit('ueV')

        # EXPECT: calculated width and the component width follow the new unit
        assert model.calculate_width()[0] == pytest.approx(width_mev * 1000)
        collection = model.get_component_collections(0)
        assert sc.Unit(str(collection[0].width.unit)) == sc.Unit('ueV')
        assert collection[0].width.value == pytest.approx(width_mev * 1000)
        assert sc.Unit(str(model.scale.unit)) == sc.Unit('ueV')

    def test_convert_x_unit_converts_collections_in_place(self):
        # WHEN
        model = BrownianTranslationalDiffusion(diffusion_coefficient=2.4e-9, Q=np.array([1.0]))
        collection_before = model.get_component_collections(0)

        # THEN
        model.convert_x_unit('ueV')

        # EXPECT: conversion does not regenerate the collections (regression: rebuilding
        # replaced the component objects, breaking external references)
        assert model.get_component_collections(0) is collection_before

    def test_convert_x_unit_persists_after_dependency_update(self):
        # WHEN
        model = BrownianTranslationalDiffusion(diffusion_coefficient=2.4e-9, Q=np.array([1.0]))
        width = model.get_component_collections(0)[0].width
        model.convert_x_unit('ueV')
        width_uev = width.value

        # THEN: trigger a dependency-graph recompute
        model.diffusion_coefficient = 4.8e-9

        # EXPECT: the dependent width stays in the new unit (regression: a plain convert_unit
        # was reverted to the old desired unit by the next graph update)
        assert sc.Unit(str(width.unit)) == sc.Unit('ueV')
        assert width.value == pytest.approx(2 * width_uev)

    @pytest.mark.parametrize(
        'kwargs,expected_exception, expected_message',
        [
            (
                {
                    'x_unit': 123,
                    'scale': 1.0,
                    'diffusion_coefficient': 1.0,
                },
                UnitError,
                'Invalid unit',
            ),
            (
                {
                    'y_unit': 123,
                    'scale': 1.0,
                    'diffusion_coefficient': 1.0,
                },
                TypeError,
                None,
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 'invalid',
                    'diffusion_coefficient': 1.0,
                },
                TypeError,
                'scale must be a number',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': -123.4,
                    'diffusion_coefficient': 1.0,
                },
                ValueError,
                'scale must be non-negative',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 1.0,
                    'diffusion_coefficient': 'invalid',
                },
                TypeError,
                'diffusion_coefficient must be a number',
            ),
            (
                {
                    'x_unit': 'meV',
                    'scale': 1.0,
                    'diffusion_coefficient': -123.4,
                },
                ValueError,
                'diffusion_coefficient must be non-negative',
            ),
        ],
        ids=[
            'invalid_x_unit',
            'invalid_y_unit',
            'invalid_scale_type',
            'invalid_scale_negative',
            'invalid_diffusion_coefficient_type',
            'invalid_diffusion_coefficient_negative',
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

    def test_calculate_width_scipp_Q_converts_unit(self, brownian_diffusion_model):
        # WHEN: the same Q expressed in 1/angstrom (numpy, assumed) and in 1/nm (scipp)
        Q_values = np.array([0.1, 0.2, 0.3])
        Q_scipp = sc.Variable(dims=['Q'], values=Q_values * 10, unit='1/nm')

        # THEN EXPECT: scipp input is converted to 1/angstrom before the width calculation
        np.testing.assert_allclose(
            brownian_diffusion_model.calculate_width(Q_scipp),
            brownian_diffusion_model.calculate_width(Q_values),
        )

    def test_calculate_EISF(self, brownian_diffusion_model):
        # WHEN
        Q_values = np.array([0.1, 0.2, 0.3])  # Example Q values in Å^-1

        # THEN
        EISF = brownian_diffusion_model.calculate_EISF(Q_values)

        # EXPECT
        expected_EISF = np.zeros_like(Q_values)
        np.testing.assert_array_equal(EISF, expected_EISF)

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
        brownian_diffusion_model.Q = Q
        # THEN
        component_collections = brownian_diffusion_model.create_component_collections()

        # EXPECT
        expected_widths = brownian_diffusion_model.calculate_width(Q)
        for model_index in range(len(component_collections)):
            model = component_collections[model_index]
            assert len(model) == 1
            component = model[0]
            assert component.width.unit == brownian_diffusion_model.x_unit
            assert np.isclose(component.width.value, expected_widths[model_index])
            assert component.width.independent is False
            # area.unit = area_unit = x_unit * y_unit
            assert component.area.unit == 'meV'

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

    def test_y_unit_setter_raises(self, brownian_diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=r'read-only'):
            brownian_diffusion_model.y_unit = '1/meV'

    def test_repr(self, brownian_diffusion_model):
        # WHEN THEN
        repr_str = repr(brownian_diffusion_model)

        # EXPECT
        assert 'BrownianTranslationalDiffusion' in repr_str
        assert 'diffusion_coefficient' in repr_str
        assert 'scale=' in repr_str
        # Regression: a stray ')' used to mangle this into 'x_unit=meV), y_unit=...'
        assert 'x_unit=meV, y_unit=dimensionless' in repr_str
