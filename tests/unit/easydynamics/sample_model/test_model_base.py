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

    def test_fix_all_parameters_before_Q_applies_to_later_collections(self):
        # WHEN
        model = ModelBase(components=Gaussian(width=0.01))

        # THEN
        model.fix_all_parameters()
        model.Q = np.array([1.0, 2.0])

        # EXPECT the template and the per-Q copies built from it are fixed
        assert all(par.fixed for par in model.components.get_fittable_parameters())
        all_vars = model.get_all_variables()
        assert len(all_vars) == 6
        assert all(par.fixed for par in all_vars)

    def test_free_all_parameters_before_Q_applies_to_later_collections(self):
        # WHEN
        gaussian = Gaussian(width=0.01)
        gaussian.fix_all_parameters()
        model = ModelBase(components=gaussian)

        # THEN
        model.free_all_parameters()
        model.Q = np.array([1.0, 2.0])

        # EXPECT the template and the per-Q copies built from it are free
        assert not any(par.fixed for par in model.components.get_fittable_parameters())
        all_vars = model.get_all_variables()
        assert len(all_vars) == 6
        assert not any(par.fixed for par in all_vars)

    def test_fix_all_parameters_with_Q_keeps_per_Q_collections(self, model_base):
        # WHEN
        collection = model_base.get_component_collection(Q_index=0)
        collection[0].area.value = 5.0

        # THEN
        model_base.fix_all_parameters()

        # EXPECT the per-Q collections are fixed in place, not rebuilt, and the
        # template is fixed so a later rebuild also comes out fixed
        assert model_base.get_component_collection(Q_index=0) is collection
        assert collection[0].area.value == pytest.approx(5.0)
        assert all(par.fixed for par in model_base.components.get_fittable_parameters())

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

    #############
    # Incremental per-Q updates (issue #220)
    #############

    def test_append_component_keeps_per_Q_state(self, model_base):
        # WHEN per-Q collections are built and one per-Q parameter is fitted and fixed
        width = model_base.get_component_collection(1)['TestGaussian1Name'].width
        width.value = 0.123
        width.fixed = True
        version = model_base.state_version

        # THEN
        model_base.append_component(Gaussian(name='NewGaussian', area=3.0))

        # EXPECT the per-Q parameter is untouched and every Q gets its own copy of the new one
        assert model_base.get_component_collection(1)['TestGaussian1Name'].width is width
        assert width.value == pytest.approx(0.123)
        assert width.fixed is True
        new_components = [
            model_base.get_component_collection(i)['NewGaussian'] for i in range(len(model_base.Q))
        ]
        assert all(c is not model_base.components['NewGaussian'] for c in new_components)
        assert len({id(c.area) for c in new_components}) == len(model_base.Q)
        assert all(c.area.value == pytest.approx(3.0) for c in new_components)
        assert model_base.state_version > version
        assert model_base.component_collections_is_dirty is False

    def test_append_component_collection_inserts_in_template_order(self, model_base):
        # WHEN per-Q collections are built
        _ = model_base.get_component_collection(0)

        # THEN
        model_base.append_component(
            ComponentCollection(components=[Lorentzian(name='NewL'), Gaussian(name='NewG')])
        )

        # EXPECT every per-Q collection lists the components in template order
        template_names = model_base.components.list_component_names()
        for i in range(len(model_base.Q)):
            names = model_base.get_component_collection(i).list_component_names()
            assert names == template_names

    def test_remove_component_keeps_per_Q_state(self, model_base):
        # WHEN per-Q collections are built and one per-Q parameter is fitted and fixed
        width = model_base.get_component_collection(1)['TestGaussian1Name'].width
        width.value = 0.123
        width.fixed = True
        version = model_base.state_version

        # THEN removing an unrelated component
        model_base.remove_component('TestLorentzian1Name')

        # EXPECT it is gone at every Q, and the remaining per-Q parameter is untouched
        for i in range(len(model_base.Q)):
            names = model_base.get_component_collection(i).list_component_names()
            assert names == ['TestGaussian1Name']
        assert model_base.get_component_collection(1)['TestGaussian1Name'].width is width
        assert width.value == pytest.approx(0.123)
        assert width.fixed is True
        assert model_base.state_version > version
        assert model_base.component_collections_is_dirty is False

    def test_append_and_remove_component_rebuild_when_collections_stale(self, model_base):
        # WHEN per-Q collections have not been built yet
        assert model_base.component_collections_is_dirty is True

        # THEN
        model_base.append_component(Gaussian(name='NewGaussian'))
        model_base.remove_component('TestLorentzian1Name')

        # EXPECT nothing is patched; the next access rebuilds from the template
        assert model_base.component_collections_is_dirty is True
        names = model_base.get_component_collection(0).list_component_names()
        assert names == ['TestGaussian1Name', 'NewGaussian']

    def test_remove_component_rebuilds_when_per_Q_collection_was_edited(self, model_base):
        "Regression: a template index applied to an edited per-Q collection removed the wrong one"
        # WHEN one per-Q collection was edited directly, so it no longer lines up with the template
        model_base.get_component_collection(0).pop('TestGaussian1Name')

        # THEN
        model_base.remove_component('TestLorentzian1Name')

        # EXPECT a full rebuild instead of removing by position
        assert model_base.component_collections_is_dirty is True
        for i in range(len(model_base.Q)):
            names = model_base.get_component_collection(i).list_component_names()
            assert names == ['TestGaussian1Name']

    def test_append_component_rebuilds_when_per_Q_collection_was_edited(self, model_base):
        # WHEN one per-Q collection was edited directly, so it no longer lines up with the template
        model_base.get_component_collection(0).pop('TestGaussian1Name')

        # THEN
        model_base.append_component(Gaussian(name='NewGaussian'))

        # EXPECT a full rebuild instead of inserting at a misaligned position
        assert model_base.component_collections_is_dirty is True
        template_names = model_base.components.list_component_names()
        for i in range(len(model_base.Q)):
            names = model_base.get_component_collection(i).list_component_names()
            assert names == template_names

    def test_append_component_rebuilds_when_collection_count_does_not_match_Q(self, model_base):
        # WHEN the collections are flagged current but there is not one per Q
        model_base._component_collections = []
        model_base._component_collections_is_dirty = False

        # THEN
        model_base.append_component(Gaussian(name='NewGaussian'))

        # EXPECT a full rebuild instead of a no-op patch that leaves no collections
        assert model_base.component_collections_is_dirty is True
        assert len(model_base.get_component_collection(0)) == 3

    #############
    # Single-parameter access (issue #70)
    #############

    def test_get_parameters(self, model_base):
        # WHEN THEN
        widths = model_base.get_parameters('TestGaussian1Name', 'width')

        # EXPECT one Parameter per Q, in Q order
        assert widths == [
            model_base.get_component_collection(i)['TestGaussian1Name'].width for i in range(3)
        ]

    def test_get_parameters_with_Q_index(self, model_base):
        # WHEN THEN
        widths = model_base.get_parameters('TestGaussian1Name', 'width', Q_index=1)

        # EXPECT
        assert widths == [model_base.get_component_collection(1)['TestGaussian1Name'].width]

    def test_get_parameters_without_Q_is_empty(self):
        # WHEN
        model = ModelBase(components=Gaussian(name='G'))

        # THEN EXPECT
        assert model.get_parameters('G', 'width') == []

    def test_get_parameters_unknown_component_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(KeyError, match='No item with name "Missing" found'):
            model_base.get_parameters('Missing', 'width')

    @pytest.mark.parametrize('parameter_name', ['missing', 'name'])
    def test_get_parameters_unknown_parameter_raises(self, model_base, parameter_name):
        # WHEN THEN EXPECT an attribute that is missing or is not a Parameter is rejected
        with pytest.raises(ValueError, match=f"has no Parameter '{parameter_name}'"):
            model_base.get_parameters('TestGaussian1Name', parameter_name)

    def test_fix_parameter(self, model_base):
        # WHEN THEN
        model_base.fix_parameter('TestGaussian1Name', 'width')

        # EXPECT only that Parameter is fixed, at every Q and in the template
        assert all(p.fixed for p in model_base.get_parameters('TestGaussian1Name', 'width'))
        assert model_base.components['TestGaussian1Name'].width.fixed is True
        assert not any(p.fixed for p in model_base.get_parameters('TestGaussian1Name', 'area'))
        assert not any(p.fixed for p in model_base.get_parameters('TestLorentzian1Name', 'width'))

    def test_fix_parameter_with_value(self, model_base):
        # WHEN THEN
        model_base.fix_parameter('TestGaussian1Name', 'width', value=0.25)

        # EXPECT the value is set at every Q and in the template
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        widths.append(model_base.components['TestGaussian1Name'].width)
        assert all(p.fixed for p in widths)
        assert all(p.value == pytest.approx(0.25) for p in widths)

    def test_fix_parameter_with_Q_index(self, model_base):
        # WHEN THEN
        model_base.fix_parameter('TestGaussian1Name', 'width', Q_index=1, value=0.25)

        # EXPECT only that Q is changed; the other Q and the template are left alone
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.fixed for p in widths] == [False, True, False]
        assert [p.value for p in widths] == pytest.approx([1.0, 0.25, 1.0])
        assert model_base.components['TestGaussian1Name'].width.fixed is False

    def test_fix_parameter_survives_rebuild(self, model_base):
        # WHEN
        model_base.fix_parameter('TestGaussian1Name', 'width', value=0.25)

        # THEN the per-Q collections are rebuilt from the template
        model_base.clear_Q(confirm=True)
        model_base.Q = np.array([1.0, 2.0])

        # EXPECT
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert len(widths) == 2
        assert all(p.fixed for p in widths)
        assert all(p.value == pytest.approx(0.25) for p in widths)

    def test_fix_parameter_before_Q_applies_to_later_collections(self):
        # WHEN
        model = ModelBase(components=Gaussian(name='G'))

        # THEN
        model.fix_parameter('G', 'width')
        model.Q = np.array([1.0, 2.0])

        # EXPECT
        assert all(p.fixed for p in model.get_parameters('G', 'width'))
        assert not any(p.fixed for p in model.get_parameters('G', 'area'))

    def test_fix_parameter_unknown_component_without_Q_raises(self):
        # WHEN
        model = ModelBase(components=Gaussian(name='G'))

        # THEN EXPECT
        with pytest.raises(KeyError, match='No item with name "Missing" found'):
            model.fix_parameter('Missing', 'width')

    def test_free_parameter(self, model_base):
        # WHEN
        model_base.fix_all_parameters()

        # THEN
        model_base.free_parameter('TestGaussian1Name', 'width')

        # EXPECT only that Parameter is freed, at every Q and in the template
        assert not any(p.fixed for p in model_base.get_parameters('TestGaussian1Name', 'width'))
        assert model_base.components['TestGaussian1Name'].width.fixed is False
        assert all(p.fixed for p in model_base.get_parameters('TestGaussian1Name', 'area'))

    def test_free_parameter_with_Q_index(self, model_base):
        # WHEN
        model_base.fix_all_parameters()

        # THEN
        model_base.free_parameter('TestGaussian1Name', 'width', Q_index=2)

        # EXPECT only that Q is freed; the other Q and the template stay fixed
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.fixed for p in widths] == [True, True, False]
        assert model_base.components['TestGaussian1Name'].width.fixed is True

    #############
    # Q constraints (issue #70)
    #############

    def test_make_parameter_Q_independent(self, model_base):
        # WHEN THEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

        # EXPECT the other Q follow Q index 0, and other Parameters are untouched
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.independent for p in widths] == [True, False, False]
        widths[0].value = 0.7
        assert [p.value for p in widths] == pytest.approx([0.7, 0.7, 0.7])
        assert all(p.independent for p in model_base.get_parameters('TestGaussian1Name', 'area'))

    def test_make_parameter_Q_independent_twice_is_a_no_op(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

        # THEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

        # EXPECT
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.independent for p in widths] == [True, False, False]

    def test_make_parameter_Q_independent_survives_rebuild(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

        # THEN the per-Q collections are rebuilt from the template
        model_base.clear_Q(confirm=True)
        model_base.Q = np.array([1.0, 2.0])

        # EXPECT
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.independent for p in widths] == [True, False]

    def test_make_parameter_Q_independent_survives_append_component(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')
        widths = model_base.get_parameters('TestGaussian1Name', 'width')

        # THEN the per-Q collections are patched in place
        model_base.append_component(Gaussian(name='NewGaussian'))

        # EXPECT the same Parameters are still tied
        assert model_base.get_parameters('TestGaussian1Name', 'width') == widths
        assert [p.independent for p in widths] == [True, False, False]

    def test_make_parameter_Q_independent_before_Q_applies_to_later_collections(self):
        # WHEN
        model = ModelBase(components=Gaussian(name='G'))

        # THEN
        model.make_parameter_Q_independent('G', 'width')
        model.Q = np.array([1.0, 2.0])

        # EXPECT
        assert [p.independent for p in model.get_parameters('G', 'width')] == [True, False]

    def test_make_parameter_Q_independent_dependent_parameter_raises(self, model_base):
        # WHEN the Parameter already depends on another Parameter at one Q
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        widths[1].make_dependent_on('a', {'a': widths[2]})

        # THEN EXPECT
        with pytest.raises(ValueError, match='already depends on other Parameters'):
            model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

    def test_make_parameter_Q_independent_unknown_component_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(KeyError, match='No item with name "Missing" found'):
            model_base.make_parameter_Q_independent('Missing', 'width')

    def test_remove_component_drops_its_Q_constraint(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

        # THEN a component of the same name is removed and added again
        model_base.remove_component('TestGaussian1Name')
        model_base.append_component(Gaussian(name='TestGaussian1Name'))

        # EXPECT the new component is not tied
        assert all(p.independent for p in model_base.get_parameters('TestGaussian1Name', 'width'))

    def test_fix_and_free_parameter_of_Q_independent_parameter(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

        # THEN
        model_base.fix_parameter('TestGaussian1Name', 'width', value=0.25)

        # EXPECT Q index 0 and the template are fixed, and the other Q follow the value
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert widths[0].fixed is True
        assert model_base.components['TestGaussian1Name'].width.fixed is True
        assert [p.value for p in widths] == pytest.approx([0.25, 0.25, 0.25])

        # THEN
        model_base.free_parameter('TestGaussian1Name', 'width')

        # EXPECT
        assert widths[0].fixed is False
        assert model_base.components['TestGaussian1Name'].width.fixed is False

    def test_remove_parameter_Q_constraint(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')
        model_base.get_parameters('TestGaussian1Name', 'width')[0].value = 0.7

        # THEN
        model_base.remove_parameter_Q_constraint('TestGaussian1Name', 'width')

        # EXPECT every Q is independent again, keeps its value, and stays so after a rebuild
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert all(p.independent for p in widths)
        assert [p.value for p in widths] == pytest.approx([0.7, 0.7, 0.7])
        model_base.clear_Q(confirm=True)
        model_base.Q = np.array([1.0, 2.0])
        assert all(p.independent for p in model_base.get_parameters('TestGaussian1Name', 'width'))

    def test_remove_parameter_Q_constraint_without_constraint_is_a_no_op(self, model_base):
        # WHEN the Parameter is dependent, but not through a Q constraint
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        widths[1].make_dependent_on('a', {'a': widths[2]})

        # THEN
        model_base.remove_parameter_Q_constraint('TestGaussian1Name', 'width')

        # EXPECT the dependency is left alone
        assert widths[1].independent is False

    def test_set_parameter_Q_dependence(self, model_base):
        # WHEN THEN
        (D,) = model_base.set_parameter_Q_dependence(
            'TestGaussian1Name',
            'width',
            expression='hbar * D * Q**2',
            parameters={'D': 2.0e-9},
            parameter_units={'D': 'm**2/s'},
        )

        # EXPECT the width at every Q follows the expression, in its own unit
        assert D.name == 'TestGaussian1Name width D'
        assert D.unit == 'm^2/s'
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert not any(p.independent for p in widths)
        assert all(p.unit == 'meV' for p in widths)
        hbar_D = 6.582119569e-13 * 2.0e-9 * 1e20  # hbar in meV*s, D in angstrom^2/s
        expected = hbar_D * np.array([1.0, 2.0, 3.0]) ** 2
        assert [p.value for p in widths] == pytest.approx(expected)

        # THEN the shared Parameter changes
        D.value = 4.0e-9

        # EXPECT
        assert [p.value for p in widths] == pytest.approx(2 * expected)

    def test_set_parameter_Q_dependence_with_constant_and_default_unit(self, model_base):
        # WHEN THEN a symbol has no entry in parameter_units
        T, scale = model_base.set_parameter_Q_dependence(
            'TestGaussian1Name',
            'width',
            expression='scale * kb * T',
            parameters={'T': 100.0, 'scale': 2.0},
            parameter_units={'T': 'K'},
        )

        # EXPECT it is dimensionless, and kb is provided
        assert T.unit == 'K'
        assert scale.unit == 'dimensionless'
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.value for p in widths] == pytest.approx([2 * 8.617333262e-2 * 100.0] * 3)

    @pytest.mark.parametrize('Q_index', [None, 1])
    def test_get_all_variables_includes_Q_dependence_parameters(self, model_base, Q_index):
        # WHEN
        (D,) = model_base.set_parameter_Q_dependence(
            'TestGaussian1Name', 'width', 'hbar * D * Q**2', {'D': 2.0e-9}, {'D': 'm**2/s'}
        )

        # THEN
        variables = model_base.get_all_variables(Q_index=Q_index)

        # EXPECT the shared Parameter is listed once
        assert sum(variable is D for variable in variables) == 1

    def test_set_parameter_Q_dependence_survives_rebuild(self, model_base):
        # WHEN
        (D,) = model_base.set_parameter_Q_dependence(
            'TestGaussian1Name', 'width', 'hbar * D * Q**2', {'D': 2.0e-9}, {'D': 'm**2/s'}
        )
        D.value = 4.0e-9

        # THEN the per-Q collections are rebuilt from the template, with new Q values
        model_base.clear_Q(confirm=True)
        model_base.Q = np.array([0.5, 1.5])

        # EXPECT the same shared Parameter drives the new per-Q Parameters
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert not any(p.independent for p in widths)
        assert widths[1].value == pytest.approx(9 * widths[0].value)
        D.value = 8.0e-9
        assert widths[0].value == pytest.approx(6.582119569e-13 * 8.0e-9 * 1e20 * 0.25)

    def test_set_parameter_Q_dependence_before_Q_applies_to_later_collections(self):
        # WHEN
        model = ModelBase(components=Gaussian(name='G'))

        # THEN
        model.set_parameter_Q_dependence('G', 'width', 'a * Q', {'a': 0.1}, {'a': 'meV*angstrom'})
        model.Q = np.array([1.0, 2.0])

        # EXPECT
        widths = model.get_parameters('G', 'width')
        assert [p.value for p in widths] == pytest.approx([0.1, 0.2])

    def test_set_parameter_Q_dependence_replaces_existing_constraint(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')
        model_base.set_parameter_Q_dependence(
            'TestGaussian1Name', 'width', 'a * Q', {'a': 0.1}, {'a': 'meV*angstrom'}
        )

        # THEN
        (b,) = model_base.set_parameter_Q_dependence(
            'TestGaussian1Name', 'width', 'b * Q**2', {'b': 0.1}, {'b': 'meV*angstrom**2'}
        )

        # EXPECT only the last constraint is in effect
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.value for p in widths] == pytest.approx([0.1, 0.4, 0.9])
        variables = model_base.get_all_variables()
        assert [v.name for v in variables if v.name.startswith('TestGaussian1Name width ')] == [
            b.name
        ]

    def test_set_parameter_Q_dependence_wrong_unit_keeps_existing_constraint(self, model_base):
        # WHEN
        model_base.make_parameter_Q_independent('TestGaussian1Name', 'width')

        # THEN EXPECT an expression that does not evaluate to an energy is rejected
        with pytest.raises(UnitError, match='Failed to convert unit'):
            model_base.set_parameter_Q_dependence(
                'TestGaussian1Name', 'width', 'D * Q**2', {'D': 2.0e-9}, {'D': 'm**2/s'}
            )

        # EXPECT the model is unchanged
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.independent for p in widths] == [True, False, False]
        assert len(model_base.get_all_variables()) == 18

    @pytest.mark.parametrize('symbol', ['Q', 'hbar', 'kb', 'sinc'])
    def test_set_parameter_Q_dependence_reserved_symbol_raises(self, model_base, symbol):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='reserved symbols'):
            model_base.set_parameter_Q_dependence(
                'TestGaussian1Name', 'width', f'{symbol} * a', {symbol: 1.0, 'a': 1.0}
            )

    def test_set_parameter_Q_dependence_unknown_parameter_unit_raises(self, model_base):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='parameter_units given for unknown parameter'):
            model_base.set_parameter_Q_dependence(
                'TestGaussian1Name', 'width', 'a * Q', {'a': 1.0}, {'b': 'meV'}
            )

    def test_set_parameter_Q_dependence_dependent_parameter_raises(self, model_base):
        # WHEN the Parameter already depends on another Parameter at one Q
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        widths[1].make_dependent_on('a', {'a': widths[2]})

        # THEN EXPECT
        with pytest.raises(ValueError, match='already depends on other Parameters'):
            model_base.set_parameter_Q_dependence(
                'TestGaussian1Name', 'width', 'a * Q', {'a': 0.1}, {'a': 'meV*angstrom'}
            )

    def test_remove_parameter_Q_constraint_removes_Q_dependence(self, model_base):
        # WHEN
        (a,) = model_base.set_parameter_Q_dependence(
            'TestGaussian1Name', 'width', 'a * Q', {'a': 0.1}, {'a': 'meV*angstrom'}
        )

        # THEN
        model_base.remove_parameter_Q_constraint('TestGaussian1Name', 'width')

        # EXPECT every Q is independent, keeps its value, and the shared Parameter is gone
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert all(p.independent for p in widths)
        assert [p.value for p in widths] == pytest.approx([0.1, 0.2, 0.3])
        assert not any(variable is a for variable in model_base.get_all_variables())

    def test_remove_component_drops_its_Q_dependence(self, model_base):
        # WHEN
        (a,) = model_base.set_parameter_Q_dependence(
            'TestGaussian1Name', 'width', 'a * Q', {'a': 0.1}, {'a': 'meV*angstrom'}
        )

        # THEN
        model_base.remove_component('TestGaussian1Name')

        # EXPECT the shared Parameter is no longer a variable of the model
        assert not any(variable is a for variable in model_base.get_all_variables())

    def test_set_parameter_Q_dependence_survives_append_component(self, model_base):
        # WHEN
        model_base.set_parameter_Q_dependence(
            'TestGaussian1Name', 'width', 'a * Q', {'a': 0.1}, {'a': 'meV*angstrom'}
        )
        widths = model_base.get_parameters('TestGaussian1Name', 'width')

        # THEN the per-Q collections are patched in place
        model_base.append_component(Gaussian(name='NewGaussian'))

        # EXPECT the same Parameters still follow the expression
        assert model_base.get_parameters('TestGaussian1Name', 'width') == widths
        assert [p.value for p in widths] == pytest.approx([0.1, 0.2, 0.3])

    def test_set_parameter_Q_dependence_with_functions(self, model_base):
        # WHEN THEN the expression uses functions of dimensionless arguments
        model_base.set_parameter_Q_dependence(
            'TestGaussian1Name',
            'width',
            expression='G * (1 - sinc(Q * L)) * exp(-(Q**2) * u2)',
            parameters={'G': 0.5, 'L': 1.5, 'u2': 0.1},
            parameter_units={'G': 'meV', 'L': 'angstrom', 'u2': 'angstrom**2'},
        )

        # EXPECT
        Q = np.array([1.0, 2.0, 3.0])
        expected = 0.5 * (1 - np.sin(1.5 * Q) / (1.5 * Q)) * np.exp(-(Q**2) * 0.1)
        widths = model_base.get_parameters('TestGaussian1Name', 'width')
        assert [p.value for p in widths] == pytest.approx(expected)

    def test_set_parameter_Q_dependence_function_of_quantity_with_unit_raises(self, model_base):
        # WHEN THEN EXPECT exp(Q) is rejected, as Q is in 1/angstrom
        with pytest.raises(SyntaxError, match='to `dimensionless` is not valid'):
            model_base.set_parameter_Q_dependence(
                'TestGaussian1Name', 'width', 'G * exp(Q)', {'G': 0.5}, {'G': 'meV'}
            )
