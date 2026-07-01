# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from easyscience.variable.parameter import Parameter

from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase


class TestDiffusionModelBase:
    @pytest.fixture
    def diffusion_model(self):
        return DiffusionModelBase()

    def test_init_default(self, diffusion_model):
        # WHEN THEN EXPECT
        assert diffusion_model.display_name == 'DiffusionModel'
        assert diffusion_model.name == 'DiffusionModel'
        assert diffusion_model.lorentzian_name == 'DiffusionModel'
        assert diffusion_model.lorentzian_display_name == 'DiffusionModel'
        assert diffusion_model.x_unit == 'meV'
        assert diffusion_model.y_unit == 'dimensionless'
        # scale.unit = area_unit = x_unit * y_unit = meV * dimensionless = meV
        assert diffusion_model.scale.unit == 'meV'

    def test_init_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'lorentzian_name must be a string'):
            DiffusionModelBase(lorentzian_name=123)

        with pytest.raises(TypeError, match=r'lorentzian_display_name must be a string or None'):
            DiffusionModelBase(lorentzian_display_name=123)

    def test_x_unit_setter_raises(self, diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match=r'read-only',
        ):
            diffusion_model.x_unit = 'eV'

    def test_y_unit_setter_raises(self, diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=r'read-only'):
            diffusion_model.y_unit = '1/meV'

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

    def test_Q_property(self, diffusion_model):
        # WHEN THEN EXPECT
        assert diffusion_model.Q is None

        # THEN
        diffusion_model.Q = [1.0, 2.0, 3.0]

        # EXPECT
        np.testing.assert_allclose(diffusion_model.Q, [1.0, 2.0, 3.0])

        # THEN EXPECT
        with pytest.raises(ValueError, match=r'New Q values are not similar to the old ones'):
            diffusion_model.Q = [10.0, 20.0, 30.0]

        # THEN EXPECT
        with pytest.raises(ValueError, match=r'Clearing Q values requires confirmation'):
            diffusion_model.clear_Q()

        # THEN
        diffusion_model.clear_Q(confirm=True)

        # EXPECT
        assert diffusion_model.Q is None

    def test_repr(self, diffusion_model):
        # WHEN THEN
        repr_str = repr(diffusion_model)

        # EXPECT
        assert 'DiffusionModelBase' in repr_str
        assert 'x_unit=meV' in repr_str

    def test_get_independent_variables(self, diffusion_model):
        # WHEN THEN EXPECT
        independent_vars = diffusion_model.get_independent_variables()

        # EXPECT
        assert independent_vars == []

    @pytest.mark.parametrize(
        'Q_index',
        [
            -1,
            100,
            'string',
        ],
        ids=[
            'negative index',
            'index too large',
            'non-integer index',
        ],
    )
    def test_get_independent_variables_raises(self, diffusion_model, Q_index):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'Q_index must be an integer between 0 and'):
            diffusion_model.get_independent_variables(Q_index=Q_index)

    @pytest.mark.parametrize(
        'Q_index',
        [
            -1,
            100,
            'string',
        ],
        ids=[
            'negative index',
            'index too large',
            'non-integer index',
        ],
    )
    def test_get_all_variables_raises(self, diffusion_model, Q_index):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=r'Q_index must be an integer between 0 and'):
            diffusion_model.get_all_variables(Q_index=Q_index)

    def test_create_component_collections_no_Q(self, diffusion_model):
        # WHEN

        # THEN
        collections = diffusion_model.create_component_collections()

        # EXPECT
        assert collections == []
        assert diffusion_model._component_collections == []

    def test_create_component_collections_with_Q(self, diffusion_model):
        # WHEN
        diffusion_model.Q = [1.0, 2.0, 3.0]

        # THEN
        collections = diffusion_model.create_component_collections()

        # EXPECT
        assert len(collections) == 3
        for collection in collections:
            assert collection.is_empty
        assert diffusion_model._component_collections == collections

    def test_get_component_collections(self, diffusion_model):
        # WHEN
        diffusion_model.Q = [1.0, 2.0]

        # THEN
        collections = diffusion_model.get_component_collections()

        # EXPECT
        assert len(collections) == 2
        assert diffusion_model._component_collections == collections

    def test_get_component_collections_Q_index(self, diffusion_model):
        # WHEN
        diffusion_model.Q = [1.0, 2.0]

        # THEN
        collection = diffusion_model.get_component_collections(Q_index=0)

        # EXPECT
        assert collection == diffusion_model._component_collections[0]

    def test_get_component_collections_Q_index_raises(self, diffusion_model):
        # WHEN
        diffusion_model.Q = [1.0, 2.0]

        # THEN EXPECT
        with pytest.raises(IndexError, match=r'out of bounds'):
            diffusion_model.get_component_collections(Q_index=100)

    def test_ensure_Q_raises(self, diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError,
            match=r'Q must be provided either as an argument or set in the model',
        ):
            diffusion_model._ensure_Q(Q=None)

    def test_ensure_Q_uses_stored_Q(self, diffusion_model):
        # WHEN
        diffusion_model.Q = [1.0, 2.0, 3.0]

        # THEN
        Q = diffusion_model._ensure_Q(Q=None)

        # EXPECT
        np.testing.assert_allclose(Q, [1.0, 2.0, 3.0])

    def test_ensure_Q_uses_argument(self, diffusion_model):
        # WHEN

        # THEN
        Q = diffusion_model._ensure_Q(Q=[1.0, 2.0])

        # EXPECT
        np.testing.assert_allclose(Q, [1.0, 2.0])
