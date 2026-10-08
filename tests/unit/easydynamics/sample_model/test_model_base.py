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

        # EXPECT
        assert model_base.display_name == 'InitModel'
        assert model_base.x_unit == 'meV'
        assert model_base.y_unit == 'dimensionless'
        assert len(model_base.components) == 2
        assert isinstance(model_base.Q, sc.Variable)
        assert model_base.Q.dims == ('Q',)
        assert model_base.Q.unit == sc.Unit('1/angstrom')
        np.testing.assert_array_equal(model_base.Q.values, np.array([1.0, 2.0, 3.0]))

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
            match='Q_index 5 is out of bounds for Q of length 3',
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
            match='Q_index 5 is out of bounds for Q of length 3',
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
        # Build collections before conversion so we can verify in-place update
        _ = model_base.get_component_collection(0)
        assert model_base._component_collections_is_dirty is False
        collection_before = model_base._component_collections[0]

        # WHEN
        model_base.convert_x_unit('eV')

        # THEN / EXPECT: dirty flag NOT set and same collections reused (not rebuilt)
        assert model_base._component_collections_is_dirty is False
        assert model_base._component_collections[0] is collection_before

        assert model_base.x_unit == 'eV'
        for component in model_base.components:
            assert component.x_unit == 'eV'
        for collection in model_base._component_collections:
            for component in collection:
                assert component.x_unit == 'eV'

    def test_init_propagates_units_to_template_collection(self):
        "Regression: the template ComponentCollection must carry the model's units"
        # WHEN
        model = ModelBase(
            display_name='M',
            x_unit='ueV',
            y_unit='counts',
            components=Gaussian(name='G', x_unit='ueV', y_unit='counts'),
            Q=np.array([1.0, 2.0]),
        )

        # THEN EXPECT: template and per-Q collections carry the model units
        assert model._components.x_unit == 'ueV'
        assert model._components.y_unit == 'counts'
        collection = model.get_component_collection(0)
        assert collection.x_unit == 'ueV'
        assert collection.y_unit == 'counts'
        assert collection.get_fit_targets()[0].y_unit == 'counts'

    def test_convert_x_unit_updates_template_collection_unit(self, model_base):
        "Regression: conversion must update the template collection's own unit attribute"
        # WHEN
        model_base.convert_x_unit('eV')

        # THEN EXPECT: the template collection follows, so per-Q collections regenerated
        # later (e.g. after a Q change) carry the new unit
        assert model_base._components.x_unit == 'eV'
        model_base._component_collections_is_dirty = True
        assert model_base.get_component_collection(0).x_unit == 'eV'

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

    def test_convert_x_unit_rollback_restores_collections(self):
        # WHEN: a model with built per-Q collections
        component = Gaussian(name='G', area=1.0, center=0.0, width=0.5, x_unit='meV')
        model = ModelBase(display_name='M', components=component, Q=[1.0, 2.0])
        collection = model.get_component_collection(0)

        # THEN: an incompatible unit fails and triggers the rollback of components and
        # collections
        with pytest.raises(UnitError):
            model.convert_x_unit('m')

        # EXPECT
        assert model.x_unit == 'meV'
        assert component.x_unit == 'meV'
        assert collection[0].x_unit == 'meV'

    def test_component_collections_empty_without_Q(self):
        # WHEN: a model without Q regenerates its collections
        model = ModelBase(display_name='M', components=Gaussian(name='G'))

        # THEN EXPECT: no per-Q collections and therefore no variables
        assert model.get_all_variables() == []
        assert model._component_collections == []

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
            sc.Variable(dims=['Q'], values=[10.0, 20.0, 30.0], unit='1/nm'),
        ],
        ids=['list', 'numpy_array', 'scipp_variable', 'scipp_variable_other_unit'],
    )
    def test_Q_setter_with_similar_Q(self, model_base, new_Q):
        # WHEN
        old_Q = model_base.Q

        # THEN
        model_base.Q = new_Q

        # EXPECT
        np.testing.assert_array_equal(model_base.Q.values, old_Q.values)

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
        np.testing.assert_array_equal(model_base.Q.values, np.array(new_Q))

    def test_Q_stored_as_scipp_in_inverse_angstrom(self, model_base):
        # WHEN: a scipp Q in 1/nm
        model_base._Q = None
        new_Q = sc.Variable(dims=['Q'], values=[5.0, 10.0], unit='1/nm')

        # THEN
        model_base.Q = new_Q

        # EXPECT: stored canonically in 1/angstrom
        assert isinstance(model_base.Q, sc.Variable)
        assert model_base.Q.unit == sc.Unit('1/angstrom')
        np.testing.assert_allclose(model_base.Q.values, [0.5, 1.0])

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

    def test_y_unit_setter_raises(self, model_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(AttributeError):
            model_base.y_unit = '1/meV'

    def test_convert_y_unit(self):
        # WHEN: model with components where y_unit='1/meV' so area_unit ≈ dimensionless
        g = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        lor = Lorentzian(area=1.0, x_unit='meV', y_unit='1/meV')
        cc = ComponentCollection(components=[g, lor])
        model = ModelBase(components=cc, x_unit='meV', Q=np.array([1.0]))

        # Build collections before conversion so we can verify in-place update
        _ = model.get_component_collection(0)
        assert model._component_collections_is_dirty is False
        collection_before = model._component_collections[0]

        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        model.convert_y_unit('1/eV')

        # EXPECT: dirty flag NOT set and same collections reused (not rebuilt)
        assert model._component_collections_is_dirty is False
        assert model._component_collections[0] is collection_before

        # EXPECT: model y_unit and template components updated
        assert model.y_unit == '1/eV'
        for component in model.components:
            assert component.y_unit == '1/eV'
        assert g.area.value == pytest.approx(1e3)
        assert lor.area.value == pytest.approx(1e3)
        # EXPECT: component collections updated in-place (not rebuilt from templates)
        for component in collection_before:
            assert component.y_unit == '1/eV'

    def test_convert_y_unit_invalid_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            model_base.convert_y_unit(123)

    #############
    # State versioning
    #############

    def test_evaluate_without_Q_names_the_cause(self):
        "Regression: the error used to claim 'no components' when Q was the missing piece"
        # WHEN a model with components but no Q
        model = ModelBase(display_name='M', components=Gaussian(name='G'))

        # THEN EXPECT
        with pytest.raises(ValueError, match='Q is not set'):
            model.evaluate(np.array([0.0, 1.0]))

    def test_state_version_reading_does_not_mutate(self, model_base):
        # WHEN
        version = model_base.state_version

        # THEN EXPECT repeated reads return the same value and rebuild nothing
        assert model_base.state_version == version
        assert model_base.component_collections_is_dirty is True
        assert model_base._component_collections == []

    def test_state_version_changes_on_component_and_Q_changes(self, model_base):
        # WHEN
        version = model_base.state_version

        # THEN appending a component through the model
        model_base.append_component(Gaussian(name='SVGaussian'))
        # EXPECT
        assert model_base.state_version > version
        version = model_base.state_version

        # THEN removing a component through the model
        model_base.remove_component('SVGaussian')
        # EXPECT
        assert model_base.state_version > version
        version = model_base.state_version

        # THEN clearing Q
        model_base.clear_Q(confirm=True)
        # EXPECT
        assert model_base.state_version > version

    def test_state_version_changes_on_in_place_template_mutation(self, model_base):
        # WHEN collections are current, so the dirty flag alone would report clean
        _ = model_base.get_component_collection(0)
        assert model_base.component_collections_is_dirty is False
        version = model_base.state_version

        # THEN mutating the live template collection in place, bypassing the model's methods
        model_base.components.append_component(Gaussian(name='LiveGaussian'))

        # EXPECT the mutation is visible without any callback
        assert model_base.state_version > version
        assert model_base.component_collections_is_dirty is True

    def test_evaluate_includes_component_appended_to_live_collection(self, model_base):
        "Regression: components appended via the live template collection were invisible"
        # WHEN a model whose collections were already built and evaluated
        x = np.linspace(-5, 5, 101)
        result_before = model_base.evaluate(x)

        # THEN appending directly to the live template collection and evaluating again
        model_base.components.append_component(
            Gaussian(name='LiveGaussian', area=10.0, center=0.0, width=1.0)
        )
        result_after = model_base.evaluate(x)

        # EXPECT the new component contributes to the output at every Q
        extra = Gaussian(name='Reference', area=10.0, center=0.0, width=1.0).evaluate(x)
        for before, after in zip(result_before, result_after, strict=True):
            np.testing.assert_allclose(after, before + extra, rtol=1e-10)
