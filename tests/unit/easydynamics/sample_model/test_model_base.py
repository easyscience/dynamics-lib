# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import Mock

import numpy as np
import pytest

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.model_base import ModelBase


class TestModelBase:
    @pytest.fixture
    def model_base(self):
        component1 = Gaussian(
            display_name='TestGaussian1',
            area=1.0,
            center=0.0,
            width=1.0,
            unit='meV',
        )
        component2 = Lorentzian(
            display_name='TestLorentzian1',
            area=2.0,
            center=1.0,
            width=0.5,
            unit='meV',
        )
        component_collection = ComponentCollection()
        component_collection.append_component(component1)
        component_collection.append_component(component2)
        model_base = ModelBase(
            display_name='InitModel',
            components=component_collection,
            unit='meV',
            Q=np.array([1.0, 2.0, 3.0]),
        )

        return model_base

    def test_init(self, model_base):
        # WHEN THEN
        model = model_base

        # EXPECT
        assert model.display_name == 'InitModel'
        assert model.unit == 'meV'
        assert len(model.components) == 2
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))

    def test_init_raises_with_invalid_components(self):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='Components must be ',
        ):
            ModelBase(components='invalid_component')

    def test_evaluate_calls_all_component_collections(self, model_base):
        # WHEN
        x = np.array([0.0, 1.0, 2.0])

        collection1 = Mock()
        collection2 = Mock()

        collection1.evaluate.return_value = np.array([1.0, 2.0, 3.0])
        collection2.evaluate.return_value = np.array([4.0, 5.0, 6.0])

        model_base._component_collections = [collection1, collection2]

        # THEN
        result = model_base.evaluate(x)

        # EXPECT
        collection1.evaluate.assert_called_once_with(x)
        collection2.evaluate.assert_called_once_with(x)

        np.testing.assert_allclose(result[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result[1], np.array([4.0, 5.0, 6.0]))

    def test_evaluate_no_component_collections_raises(self, model_base):
        # WHEN
        x = np.array([0.0, 1.0, 2.0])

        model_base._component_collections = []

        # THEN / EXPECT
        with pytest.raises(ValueError, match='No components'):
            model_base.evaluate(x)

    def test_generate_component_collections_with_Q(self, model_base):
        # WHEN
        model_base._generate_component_collections()

        # THEN
        assert len(model_base._component_collections) == len(model_base.Q)
        for collection in model_base._component_collections:
            assert isinstance(collection, ComponentCollection)
            assert len(collection.components) == 2
            assert isinstance(collection.components[0], Gaussian)
            assert collection.components[0].display_name == 'TestGaussian1'
            assert isinstance(collection.components[1], Lorentzian)
            assert collection.components[1].display_name == 'TestLorentzian1'

    def test_fix_free_all_parameters(self, model_base):
        # WHEN
        model_base.fix_all_parameters()

        # THEN
        for par in model_base.get_all_variables():
            assert par.fixed is True

        # WHEN
        model_base.free_all_parameters()

        # THEN
        for par in model_base.get_all_variables():
            assert par.fixed is False

    def test_get_all_variables(self, model_base):
        # WHEN
        all_vars = model_base.get_all_variables()

        # THEN
        expected_var_display_names = {
            'TestGaussian1 area',
            'TestGaussian1 center',
            'TestGaussian1 width',
            'TestLorentzian1 area',
            'TestLorentzian1 center',
            'TestLorentzian1 width',
        }

        retrieved_var_display_names = {var.display_name for var in all_vars}

        assert expected_var_display_names == retrieved_var_display_names
        assert len(all_vars) == 18

    def test_get_all_variables_with_Q_index(self, model_base):
        # WHEN
        all_vars = model_base.get_all_variables(Q_index=1)

        # THEN
        expected_var_display_names = {
            'TestGaussian1 area',
            'TestGaussian1 center',
            'TestGaussian1 width',
            'TestLorentzian1 area',
            'TestLorentzian1 center',
            'TestLorentzian1 width',
        }

        retrieved_var_display_names = {var.display_name for var in all_vars}

        assert expected_var_display_names == retrieved_var_display_names
        assert len(all_vars) == 6

    def test_get_all_variables_with_invalid_Q_index_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            IndexError,
            match='Q_index 5 is out of bounds for component collections of length 3',
        ):
            model_base.get_all_variables(Q_index=5)

    def test_get_all_variables_with_nonint_Q_index_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='Q_index must be an int or None, got str',
        ):
            model_base.get_all_variables(Q_index='invalid_index')

    def test_get_component_collection(self, model_base):
        # WHEN THEN
        collection = model_base.get_component_collection(Q_index=0)
        # EXPECT
        assert collection is model_base._component_collections[0]

    def test_get_component_collection_invalid_index_type_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='Q_index must be an int, got str',
        ):
            model_base.get_component_collection(Q_index='invalid_index')

    def test_get_component_collection_invalid_index_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            IndexError,
            match='Q_index 5 is out of bounds for ',
        ):
            model_base.get_component_collection(Q_index=5)

    def test_append_and_remove_and_clear_component(self, model_base):
        # WHEN
        new_component = Gaussian(unique_name='NewGaussian')

        # THEN
        model_base.append_component(new_component)

        # EXPECT
        assert len(model_base.components) == 3
        assert model_base.components[-1] is new_component

        # THEN
        model_base.remove_component('NewGaussian')

        # EXPECT
        assert len(model_base.components) == 2

        # THEN
        model_base.clear_components()

        # EXPECT
        assert len(model_base.components) == 0

    def test_append_component_collection(self, model_base):
        # WHEN
        new_collection = ComponentCollection()
        new_component1 = Lorentzian()
        new_component2 = Gaussian()
        new_collection.append_component(new_component1)
        new_collection.append_component(new_component2)

        # THEN
        model_base.append_component(new_collection)

        # EXPECT
        assert len(model_base.components) == 4
        assert model_base.components[-2] is new_component1
        assert model_base.components[-1] is new_component2

    def test_append_component_invalid_type_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(TypeError, match=' must be '):
            model_base.append_component('invalid_component')

    def test_unit_property(self, model_base):
        # WHEN
        unit = model_base.unit

        # THEN / EXPECT
        assert unit == 'meV'

    def test_unit_setter_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(AttributeError, match='Use convert_unit to change '):
            model_base.unit = 'K'

    def test_convert_unit(self, model_base):
        # WHEN
        model_base.convert_unit('eV')

        # THEN / EXPECT
        assert model_base.unit == 'eV'
        for component in model_base.components:
            assert component.unit == 'eV'

    def test_convert_unit_invalid_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(Exception):
            model_base.convert_unit('invalid_unit')

    def test_components_setter(self, model_base):
        # WHEN
        new_component = Lorentzian(unique_name='NewLorentzian')
        model_base.components = new_component

        # THEN / EXPECT
        assert len(model_base.components) == 1
        assert model_base.components[0] is new_component

    def test_components_setter_collection(self, model_base):
        # WHEN
        new_collection = ComponentCollection()
        new_component1 = Gaussian()
        new_component2 = Lorentzian()
        new_collection.append_component(new_component1)
        new_collection.append_component(new_component2)

        model_base.components = new_collection

        # THEN / EXPECT
        assert len(model_base.components) == 2
        assert model_base.components[0] is new_component1
        assert model_base.components[1] is new_component2

    def test_components_setter_invalid_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='Components must be ',
        ):
            model_base.components = 'invalid_component'

    def test_Q_setter(self, model_base):
        # WHEN
        new_Q = [0.5, 1.5, 2.5]
        model_base.Q = new_Q

        # THEN / EXPECT
        np.testing.assert_array_equal(model_base.Q, np.array(new_Q))

    def test_repr(self, model_base):
        # WHEN
        repr_str = repr(model_base)

        # THEN / EXPECT
        assert 'unique_name' in repr_str
        assert 'unit' in repr_str
        assert 'Q = ' in repr_str
        assert 'components = ' in repr_str
