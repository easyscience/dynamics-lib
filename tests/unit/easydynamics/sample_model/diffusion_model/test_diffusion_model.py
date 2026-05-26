# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from easyscience.variable.parameter import Parameter

from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase


class TestDiffusionModel:
    @pytest.fixture
    def diffusion_model(self):
        return DiffusionModelBase(display_name='TestDiffusionModel', unit='meV')

    def test_init_default(self, diffusion_model):
        # WHEN THEN EXPECT
        assert diffusion_model.display_name == 'TestDiffusionModel'
        assert diffusion_model.unit == 'meV'

    def test_unit_setter_raises(self, diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match=r'Unit is read-only. Use convert_unit to change the unit between allowed types',
        ):
            diffusion_model.unit = 'eV'

    # def test_scale_setter(self, diffusion_model):
    #     # WHEN
    #     diffusion_model.scale = 2.0

    #     # THEN EXPECT
    #     assert diffusion_model.scale.value == pytest.approx(2.0)

    # def test_scale_setter_negative_raises(self, diffusion_model):
    #     # WHEN THEN EXPECT
    #     with pytest.raises(ValueError, match=r'scale must be non-negative.'):
    #         diffusion_model.scale = -1.0  # Invalid negative value

    # def test_scale_setter_raises(self, diffusion_model):
    #     # WHEN THEN EXPECT
    #     with pytest.raises(TypeError, match=r'scale must be a number.'):
    #         diffusion_model.scale = 'invalid'  # Invalid type

    @pytest.mark.parametrize(
        ('attribute', 'value', 'expected'),
        [
            ('scale', 2.0, 2.0),
            ('scale', 0.0, 0.0),
            ('scale', 5, 5.0),
            ('lorentzian_name', 'lorentzian', 'lorentzian'),
            ('lorentzian_name', '', ''),
            ('lorentzian_display_name', 'display', 'display'),
            ('lorentzian_display_name', None, None),
        ],
    )
    def test_setters_valid(
        self,
        diffusion_model,
        attribute,
        value,
        expected,
    ):
        # WHEN

        # THEN
        setattr(diffusion_model, attribute, value)

        # EXPECT
        result = getattr(diffusion_model, attribute)

        # Handle Parameters
        if isinstance(result, Parameter):
            result = result.value

        assert result == expected

    @pytest.mark.parametrize(
        ('attribute', 'value', 'exception', 'message'),
        [
            (
                'scale',
                -1.0,
                ValueError,
                r'scale must be non-negative.',
            ),
            (
                'scale',
                'invalid',
                TypeError,
                r'scale must be a number.',
            ),
            (
                'lorentzian_name',
                1,
                TypeError,
                r'lorentzian_name must be a string.',
            ),
            (
                'lorentzian_name',
                None,
                TypeError,
                r'lorentzian_name must be a string.',
            ),
            (
                'lorentzian_display_name',
                1,
                TypeError,
                r'lorentzian_display_name must be a string or None.',
            ),
            (
                'lorentzian_display_name',
                [],
                TypeError,
                r'lorentzian_display_name must be a string or None.',
            ),
        ],
    )
    def test_setters_invalid(
        self,
        diffusion_model,
        attribute,
        value,
        exception,
        message,
    ):
        # WHEN THEN EXPECT
        with pytest.raises(exception, match=message):
            setattr(diffusion_model, attribute, value)

    def test_repr(self, diffusion_model):
        # WHEN THEN
        repr_str = repr(diffusion_model)

        # EXPECT
        assert 'DiffusionModelBase' in repr_str
        assert 'display_name=TestDiffusionModel' in repr_str
        assert 'unit=meV' in repr_str
