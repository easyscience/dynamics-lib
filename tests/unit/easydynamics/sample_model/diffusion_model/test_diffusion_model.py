# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

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
            match='Unit is read-only. Use convert_unit to change the unit between allowed types',
        ):
            diffusion_model.unit = 'eV'

    def test_scale_setter(self, diffusion_model):
        # WHEN
        diffusion_model.scale = 2.0

        # THEN EXPECT
        assert diffusion_model.scale.value == 2.0

    def test_scale_setter_negative_raises(self, diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='scale must be non-negative.'):
            diffusion_model.scale = -1.0  # Invalid negative value

    def test_scale_setter_raises(self, diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='scale must be a number.'):
            diffusion_model.scale = 'invalid'  # Invalid type

    def test_repr(self, diffusion_model):
        # WHEN THEN
        repr_str = repr(diffusion_model)

        # EXPECT
        assert 'DiffusionModelBase' in repr_str
        assert 'display_name=TestDiffusionModel' in repr_str
        assert 'unit=meV' in repr_str
