# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import Mock

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.model_base import ModelBase


class TestModelBase:
    @pytest.fixture
    def model_base(self):
        component1 = Gaussian(
            name='TestGaussian1Name',
            display_name='TestGaussian1',
            area=1.0,
            center=0.0,
            width=1.0,
            x_unit='meV',
        )
        component2 = Lorentzian(
            name='TestLorentzian1Name',
            display_name='TestLorentzian1',
            area=2.0,
            center=1.0,
            width=0.5,
            x_unit='meV',
        )
        component_collection = ComponentCollection()
        component_collection.append_component(component1)
        component_collection.append_component(component2)
        return ModelBase(
            display_name='InitModel',
            components=component_collection,
            x_unit='meV',
            Q=np.array([1.0, 2.0, 3.0]),
        )

    def test_init(self, model_base):
        # WHEN THEN
        model = model_base

        # EXPECT
        assert model.display_name == 'InitModel'
        assert model.x_unit == 'meV'
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
        model_base._component_collections_is_dirty = False

        # THEN
        result = model_base.evaluate(x)

        # EXPECT
        collection1.evaluate.assert_called_once_with(x, output='numpy')
        collection2.evaluate.assert_called_once_with(x, output='numpy')

        np.testing.assert_allclose(result[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result[1], np.array([4.0, 5.0, 6.0]))

    def test_evaluate_no_component_collections_raises(self, model_base):
        # WHEN
        x = np.array([0.0, 1.0, 2.0])

        model_base._component_collections = []
        model_base._component_collections_is_dirty = False

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
            assert len(collection) == 2
            assert isinstance(collection[0], Gaussian)
            assert collection[0].display_name == 'TestGaussian1'
            assert isinstance(collection[1], Lorentzian)
            assert collection[1].display_name == 'TestLorentzian1'

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

        # EXPECT
        expected_var_display_names = {
            'TestGaussian1Name area',
            'TestGaussian1Name center',
            'TestGaussian1Name width',
            'TestLorentzian1Name area',
            'TestLorentzian1Name center',
            'TestLorentzian1Name width',
        }

        retrieved_var_display_names = {var.display_name for var in all_vars}

        assert expected_var_display_names == retrieved_var_display_names
        assert len(all_vars) == 18
        assert len(all_vars) == len(set(all_vars))

    def test_get_all_variables_with_Q_index(self, model_base):
        # WHEN
        all_vars = model_base.get_all_variables(Q_index=1)

        # THEN
        expected_var_display_names = {
            'TestGaussian1Name area',
            'TestGaussian1Name center',
            'TestGaussian1Name width',
            'TestLorentzian1Name area',
            'TestLorentzian1Name center',
            'TestLorentzian1Name width',
        }

        retrieved_var_display_names = {var.display_name for var in all_vars}

        assert expected_var_display_names == retrieved_var_display_names
        assert len(all_vars) == 6
        assert len(all_vars) == len(set(all_vars))

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
        new_component = Gaussian(name='NewGaussian')

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

    def test_x_unit_property(self, model_base):
        # WHEN
        unit = model_base.x_unit

        # THEN / EXPECT
        assert unit == 'meV'

    def test_x_unit_setter_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(AttributeError):
            model_base.x_unit = 'K'

    def test_convert_x_unit(self, model_base):
        # WHEN
        model_base.convert_x_unit('eV')

        # THEN / EXPECT
        assert model_base.x_unit == 'eV'
        for component in model_base.components:
            assert component.x_unit == 'eV'

    def test_convert_x_unit_invalid_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(UnitError):
            model_base.convert_x_unit('invalid_unit')

    def test_convert_x_unit_incorrect_unit_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Unit must be a string or sc.Unit'):
            model_base.convert_x_unit(123)

    def test_components_setter_none(self, model_base):
        # WHEN THEN
        model_base.components = None
        # EXPECT
        assert len(model_base.components) == 0

    def test_convert_x_unit_rollback_when_old_unit_none(self):
        # WHEN: model with _x_unit=None (rollback branch is skipped when old_unit is None)
        component = Gaussian(name='G', area=1.0, center=0.0, width=0.5, x_unit='meV')
        model = ModelBase(display_name='M', x_unit=None, components=component)
        model._x_unit = None
        # THEN
        with pytest.raises(UnitError):
            model.convert_x_unit('m')  # incompatible unit triggers failure
        # EXPECT: Gaussian's own atomic rollback keeps it at 'meV' even though
        # ModelBase's outer rollback loop is skipped when old_unit is None
        assert component.x_unit == 'meV'

    def test_convert_x_unit_rollback_on_failure(self, model_base):
        # WHEN THEN
        with pytest.raises(UnitError):
            model_base.convert_x_unit('m')
        # EXPECT: state rolled back
        assert model_base.x_unit == 'meV'
        for component in model_base.components:
            assert component.x_unit == 'meV'

    def test_convert_y_unit_rollback_on_failure(self, model_base):
        # WHEN THEN
        with pytest.raises(UnitError):
            model_base.convert_y_unit('K')
        # EXPECT: state rolled back
        assert model_base.y_unit == 'dimensionless'

    def test_components_setter(self, model_base):
        # WHEN
        new_component = Lorentzian(name='NewLorentzian')
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

    def test_Q_setter_raises_if_Q_is_not_similar(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(ValueError, match='New Q values are not similar to'):
            model_base.Q = [10.0, 20.0, 30.0]

    @pytest.mark.parametrize(
        'new_Q',
        [
            [1.0, 2.0, 3.0],
            np.array([1.0, 2.0, 3.0]),
            sc.Variable(dims=['Q'], values=[1.0, 2.0, 3.0], unit='1/angstrom'),
        ],
        ids=['list', 'numpy_array', 'scipp_variable'],
    )
    def test_Q_setter_with_similar_Q(self, model_base, new_Q):
        # WHEN
        old_Q = model_base.Q

        # THEN
        model_base.Q = new_Q

        # EXPECT
        np.testing.assert_array_equal(model_base.Q, old_Q)

    def test_Q_setter_with_none(self, model_base):
        # WHEN
        old_Q = model_base.Q

        # THEN
        model_base.Q = None

        # THEN / EXPECT
        assert model_base.Q is old_Q

    def test_Q_setter_when_current_Q_is_none(self, model_base):
        # WHEN
        model_base._Q = None
        new_Q = [0.5, 1.5, 2.5]

        # THEN
        model_base.Q = new_Q

        # EXPECT
        np.testing.assert_array_equal(model_base.Q, np.array(new_Q))

    def test_clear_Q(self, model_base):
        # WHEN
        #
        # THEN
        model_base.clear_Q(confirm=True)

        # EXPECT
        assert model_base.Q is None

    def test_clear_Q_raises_without_confirm(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(ValueError, match='Clearing Q values requires confirmation'):
            model_base.clear_Q()

    def test_normalize_area(self, model_base):
        # WHEN

        # THEN
        model_base.normalize_area()

        # EXPECT
        for collection in model_base._component_collections:
            total_area = sum(component.area.value for component in collection)
            assert total_area == pytest.approx(1.0)

    def test_repr(self, model_base):
        # WHEN
        repr_str = repr(model_base)

        # THEN / EXPECT
        assert 'unique_name' in repr_str
        assert 'unit' in repr_str
        assert 'Q=' in repr_str
        assert 'components=' in repr_str

    def test_y_unit_default(self, model_base):
        # WHEN THEN EXPECT
        assert model_base.y_unit == 'dimensionless'

    def test_convert_y_unit(self):
        # WHEN: model with components where y_unit='1/meV' so area_unit ≈ dimensionless
        g = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        lor = Lorentzian(area=1.0, x_unit='meV', y_unit='1/meV')
        cc = ComponentCollection(components=[g, lor])
        model = ModelBase(components=cc, x_unit='meV', Q=np.array([1.0]))

        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        model.convert_y_unit('1/eV')

        # EXPECT: model y_unit and all template components updated
        assert model.y_unit == '1/eV'
        for component in model.components:
            assert component.y_unit == '1/eV'
        # Component collections rebuilt on demand reflect the new unit
        for component in model.get_component_collection(0):
            assert component.y_unit == '1/eV'

    def test_convert_y_unit_invalid_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            model_base.convert_y_unit(123)
