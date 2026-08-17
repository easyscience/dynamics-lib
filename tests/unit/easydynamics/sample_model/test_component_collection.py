# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipy.integrate import simpson

from easydynamics.exceptions import AmbiguousNameError
from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import ExpressionComponent
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial


class TestComponentCollection:
    @pytest.fixture
    def component_collection(self):
        model = ComponentCollection(display_name='TestComponentCollection')
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
        model.append_component(component1)
        model.append_component(component2)
        return model

    def test_init(self):
        # WHEN THEN
        component_collection = ComponentCollection(display_name='InitModel')

        # EXPECT
        assert component_collection.display_name == 'InitModel'
        assert not component_collection
        assert component_collection.x_unit == 'meV'
        assert component_collection.y_unit == 'dimensionless'

    def test_get_fit_targets(self, component_collection):
        # WHEN
        targets = component_collection.get_fit_targets()

        # EXPECT: a single 'value' prediction wrapping the summed evaluate, stamped with the
        # collection's units and no default dataset key
        assert len(targets) == 1
        target = targets[0]
        assert target.name == 'value'
        assert target.dataset_key is None
        assert target.label == 'TestComponentCollection'
        assert target.x_unit == component_collection.x_unit
        assert target.y_unit == component_collection.y_unit
        x = np.linspace(-1, 1, 5)
        np.testing.assert_allclose(target.function(x), component_collection.evaluate(x))

    def test_init_with_component(self):
        # WHEN THEN
        component1 = Gaussian(name='TestGaussian1', area=1.0, center=0.0, width=1.0, x_unit='meV')
        component_collection = ComponentCollection(display_name='InitModel', components=component1)

        # EXPECT
        assert component_collection.display_name == 'InitModel'
        assert len(component_collection) == 1
        assert component_collection[0] is component1

    def test_init_with_components(self):
        # WHEN THEN
        component1 = Gaussian(name='TestGaussian1', area=1.0, center=0.0, width=1.0, x_unit='meV')
        component2 = Lorentzian(
            name='TestLorentzian1', area=2.0, center=1.0, width=0.5, x_unit='meV'
        )
        component_collection = ComponentCollection(
            display_name='InitModel', components=[component1, component2]
        )

        # EXPECT
        assert component_collection.display_name == 'InitModel'
        assert len(component_collection) == 2
        assert component_collection[0] is component1
        assert component_collection[1] is component2

    def test_init_with_invalid_components_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='All items in components must be instances of ModelComponent',
        ):
            ComponentCollection(components=['NotAComponent'])

    def test_init_with_invalid_list_of_components_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='components must be a ModelComponent or a list of ModelComponent',
        ):
            ComponentCollection(components='NotAList')

    def test_init_with_invalid_unit_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='unit must be'):
            ComponentCollection(x_unit=123)

    #############
    # Component Management
    #############

    def test_append_component(self, component_collection):
        # WHEN
        component = Gaussian(name='TestComponent', area=1.0, center=0.0, width=1.0, x_unit='meV')
        # THEN
        component_collection.append_component(component)
        # EXPECT
        assert component_collection[-1] is component

    def test_append_component_collection(self, component_collection):
        # WHEN
        component = Gaussian(name='TestComponent', area=1.0, center=0.0, width=1.0, x_unit='meV')
        component_collection2 = ComponentCollection()
        component_collection2.append_component(component)
        # THEN
        component_collection.append_component(component_collection2)
        # EXPECT
        assert component_collection[-1] is component

    def test_append_existing_component_warns(self, component_collection):
        # WHEN THEN
        component = component_collection[0]
        # EXPECT
        with pytest.warns(UserWarning, match='it will be ignored'):
            component_collection.append_component(component)

    def test_append_invalid_component_raises(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Value must be an instance of type'):
            component_collection.append_component('NotAComponent')

    def test_getitem(self, component_collection):
        # WHEN
        component = Gaussian(name='TestComponent', area=1.0, center=0.0, width=1.0, x_unit='meV')
        # THEN
        component_collection.append_component(component)
        # EXPECT
        assert component_collection[-1] is component

    def test_is_empty(self):
        # WHEN THEN
        component_collection = ComponentCollection(display_name='EmptyModel')
        # EXPECT
        assert component_collection.is_empty is True

        # WHEN THEN
        component = Gaussian(name='TestComponent', area=1.0, center=0.0, width=1.0, x_unit='meV')
        component_collection.append_component(component)
        # EXPECT
        assert component_collection.is_empty is False

    def test_is_empty_setter(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=r'is_empty is a read-only property.'):
            component_collection.is_empty = True

    def test_list_component_names(self, component_collection):
        # WHEN THEN
        components = component_collection.list_component_names()
        # EXPECT
        assert len(components) == 2
        assert components[0] == 'TestGaussian1Name'
        assert components[1] == 'TestLorentzian1Name'

    def test_convert_x_unit(self, component_collection):
        # WHEN THEN
        component_collection.convert_x_unit('eV')
        # EXPECT
        for component in component_collection:
            assert component.x_unit == 'eV'

    def test_convert_x_unit_incorrect_unit_raises(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'unit must be a string or sc.Unit'):
            component_collection.convert_x_unit(123)

    def test_convert_x_unit_failure_rolls_back(self, component_collection):
        # WHEN THEN
        # Introduce a faulty component that will fail conversion
        class FaultyComponent(Gaussian):
            def convert_x_unit(self, _unit: str) -> None:
                raise RuntimeError('Conversion failed.')

        faulty_component = FaultyComponent(
            name='FaultyComponent', area=1.0, center=0.0, width=1.0, x_unit='meV'
        )
        component_collection.append_component(faulty_component)

        original_units = {component.name: component.x_unit for component in component_collection}

        # EXPECT
        with pytest.raises(RuntimeError, match=r'Conversion failed.'):
            component_collection.convert_x_unit('eV')

        # Check that all components have their original units
        for component in component_collection:
            assert component.x_unit == original_units[component.name]

    def test_set_x_unit(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match=r'read-only',
        ):
            component_collection.x_unit = 'eV'

    def test_evaluate(self, component_collection):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN
        result = component_collection.evaluate(x)
        # EXPECT
        expected_result = component_collection[0].evaluate(x) + component_collection[1].evaluate(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_no_components_returns_zero(self):
        # WHEN
        component_collection = ComponentCollection(display_name='EmptyModel')
        x = np.linspace(-5, 5, 100)
        # THEN
        result = component_collection.evaluate(x)

        # EXPECT
        assert np.all(result == pytest.approx(0.0))
        assert result.shape == x.shape

    def test_evaluate_no_components_scipp_output(self):
        # WHEN
        component_collection = ComponentCollection(display_name='EmptyModel', y_unit='1/meV')
        x = np.linspace(-5, 5, 100)

        # THEN
        result = component_collection.evaluate(x, output='scipp')

        # EXPECT: an sc.Variable of zeros carrying the collection's y_unit
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('1/meV')
        assert np.all(result.values == pytest.approx(0.0))

    def test_evaluate_no_components_scipp_input(self):
        # WHEN
        component_collection = ComponentCollection(display_name='EmptyModel')
        x = sc.linspace('energy', -5.0, 5.0, 100, unit='meV')

        # THEN
        result = component_collection.evaluate(x, output='scipp')

        # EXPECT: zeros on the input grid, keeping the input's dimension name
        assert isinstance(result, sc.Variable)
        assert result.dims == ('energy',)
        assert np.all(result.values == pytest.approx(0.0))

    def test_evaluate_component(self, component_collection):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN
        result1 = component_collection.evaluate_component(x, 'TestGaussian1Name')
        result2 = component_collection.evaluate_component(x, 'TestLorentzian1Name')

        # EXPECT
        expected_result1 = component_collection[0].evaluate(x)
        expected_result2 = component_collection[1].evaluate(x)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-5)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-5)

    def test_evaluate_nonexistent_component_raises(self, component_collection):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN EXPECT
        with pytest.raises(KeyError, match="No component named 'NonExistentComponent' exists"):
            component_collection.evaluate_component(x, 'NonExistentComponent')

    def test_evaluate_component_no_components_raises(self):
        # WHEN THEN
        component_collection = ComponentCollection(display_name='EmptyModel')
        x = np.linspace(-5, 5, 100)
        # EXPECT
        with pytest.raises(ValueError, match=r'No components in the model to evaluate.'):
            component_collection.evaluate_component(x, 'AnyComponent')

    def test_evaluate_component_invalid_name_type_raises(self, component_collection):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN EXPECT
        with pytest.raises(
            TypeError,
            match=r"Component name must be a string, got <class 'int'> instead.",
        ):
            component_collection.evaluate_component(x, 123)

    #############
    # Utilities
    #############

    def test_normalize_area(self, component_collection):
        # WHEN THEN
        component_collection.normalize_area()
        # EXPECT
        x = np.linspace(-10000, 10000, 1000000)  # Lorentzians have long tails
        result = component_collection.evaluate(x)
        numerical_area = simpson(result, x)
        assert np.isclose(numerical_area, 1.0, rtol=1e-4)

    def test_normalize_area_no_components_raises(self):
        # WHEN THEN
        component_collection = ComponentCollection(display_name='EmptyModel')
        # EXPECT
        with pytest.raises(ValueError, match=r'No components in the model to normalize.'):
            component_collection.normalize_area()

    @pytest.mark.parametrize(
        'area_value',
        [np.nan, 0.0, np.inf],
        ids=['NaN area', 'Zero area', 'Infinite area'],
    )
    def test_normalize_area_not_finite_area_raises(self, component_collection, area_value):
        # WHEN THEN
        component_collection[0].area = area_value
        component_collection[1].area = area_value

        # EXPECT
        with pytest.raises(ValueError, match=r'cannot normalize'):
            component_collection.normalize_area()

    def test_normalize_area_non_area_component_warns(self, component_collection):
        # WHEN
        component1 = Polynomial(
            display_name='TestPolynomial', coefficients=[1, 2, 3], x_unit='meV'
        )
        component_collection.append_component(component1)

        # THEN EXPECT
        with pytest.warns(UserWarning, match="does not have an 'area' "):
            component_collection.normalize_area()

    def test_convert_x_unit_rollback_skipped_when_old_unit_none(self):
        # WHEN: a collection without an x_unit of its own
        collection = ComponentCollection(
            components=Gaussian(name='G', area=1.0, width=0.5, x_unit='meV'), x_unit=None
        )

        # THEN: an incompatible unit fails; the outer rollback is skipped (no old unit to
        # restore) but the component's own atomic rollback keeps it consistent
        with pytest.raises(sc.UnitError):
            collection.convert_x_unit('m')

        # EXPECT
        assert collection[0].x_unit == 'meV'

    def test_normalize_area_only_non_area_components_raises(self):
        # WHEN: no component in the collection has an area attribute
        collection = ComponentCollection(
            components=Polynomial(display_name='OnlyPolynomial', coefficients=[1, 2])
        )

        # THEN EXPECT
        with (
            pytest.warns(UserWarning, match="does not have an 'area' "),
            pytest.raises(ValueError, match='No components with an area attribute'),
        ):
            collection.normalize_area()

    def test_normalize_area_mixed_units(self):
        # WHEN: two Gaussians with compatible but different area units (meV and ueV)
        gaussian_mev = Gaussian(name='G1', area=1.0, width=1.0, x_unit='meV')
        gaussian_uev = Gaussian(name='G2', area=1000.0, width=500.0, x_unit='ueV')
        collection = ComponentCollection(components=[gaussian_mev, gaussian_uev])

        # THEN: both areas are physically 1 meV, so each should end up at half its value
        collection.normalize_area()

        # EXPECT: areas sum to 1 in the first component's unit (meV)
        total_mev = gaussian_mev.area.value + gaussian_uev.area.value / 1000.0
        assert total_mev == pytest.approx(1.0)
        assert gaussian_mev.area.value == pytest.approx(0.5)
        assert gaussian_uev.area.value == pytest.approx(500.0)

    def test_get_all_parameters(self, component_collection):
        # WHEN THEN
        parameters = component_collection.get_all_parameters()
        # EXPECT
        assert len(parameters) == 6

        expected_names = {
            'TestGaussian1Name area',
            'TestGaussian1Name center',
            'TestGaussian1Name width',
            'TestLorentzian1Name area',
            'TestLorentzian1Name center',
            'TestLorentzian1Name width',
        }
        actual_names = {param.name for param in parameters}
        assert actual_names == expected_names
        assert all(isinstance(param, Parameter) for param in parameters)

    def test_get_parameters_no_components(self):
        component_collection = ComponentCollection(display_name='EmptyModel')
        # WHEN THEN
        parameters = component_collection.get_all_parameters()
        # EXPECT
        assert len(parameters) == 0

    def test_get_fit_parameters(self, component_collection):
        # WHEN

        # Fix one parameter and make another dependent
        component_collection[0].area.fixed = True
        component_collection[1].width.make_dependent_on(
            'comp1_width',
            {'comp1_width': component_collection[0].width},
        )

        # THEN
        fit_parameters = component_collection.get_fit_parameters()

        # EXPECT
        assert len(fit_parameters) == 4

        expected_names = {
            'TestGaussian1Name center',
            'TestGaussian1Name width',
            'TestLorentzian1Name area',
            'TestLorentzian1Name center',
        }
        actual_names = {param.name for param in fit_parameters}
        assert actual_names == expected_names
        assert all(isinstance(param, Parameter) for param in fit_parameters)

    def test_fix_and_free_all_parameters(self, component_collection):
        # WHEN THEN
        component_collection.fix_all_parameters()

        # EXPECT
        for param in component_collection.get_all_parameters():
            assert param.fixed is True

        # WHEN
        component_collection.free_all_parameters()

        # THEN
        for param in component_collection.get_all_parameters():
            assert param.fixed is False

    def test_contains(self, component_collection):
        # WHEN THEN EXPECT: membership by name and by object
        assert 'TestGaussian1Name' in component_collection
        assert 'TestLorentzian1Name' in component_collection
        assert 'NonExistentComponent' not in component_collection

        gaussian_component = component_collection[0]
        lorentzian_component = component_collection[1]
        assert gaussian_component in component_collection
        assert lorentzian_component in component_collection

        # WHEN: a component not in the collection — THEN EXPECT
        fake_component = Gaussian(
            name='FakeGaussian', area=1.0, center=0.0, width=1.0, x_unit='meV'
        )
        assert fake_component not in component_collection
        assert 123 not in component_collection  # Invalid type

    def test_repr_contains_name_and_components(self, component_collection):
        # WHEN THEN
        rep = repr(component_collection)
        # EXPECT
        assert 'ComponentCollection' in rep
        assert 'TestGaussian1Name' in rep

    def test_to_dict(self, component_collection):
        # WHEN
        model_dict = component_collection.to_dict()

        # EXPECT
        assert model_dict['display_name'] == component_collection.display_name
        assert model_dict['x_unit'] == component_collection.x_unit
        assert model_dict['y_unit'] == component_collection.y_unit
        assert len(model_dict['components']) == len(component_collection)

        for comp, comp_dict in zip(component_collection, model_dict['components'], strict=True):
            assert comp_dict['@class'] == type(comp).__name__
            assert comp_dict['display_name'] == comp.display_name
            assert comp_dict['x_unit'] == comp.x_unit
            assert comp_dict['y_unit'] == comp.y_unit

    def test_from_dict(self, component_collection):
        # WHEN
        model_dict = component_collection.to_dict()

        # THEN
        new_model = ComponentCollection.from_dict(model_dict)

        # EXPECT
        assert new_model.display_name == component_collection.display_name
        assert new_model.x_unit == component_collection.x_unit
        assert new_model.y_unit == component_collection.y_unit
        assert len(new_model) == len(component_collection)

        for orig_comp, new_comp in zip(component_collection, new_model, strict=True):
            assert type(new_comp) is type(orig_comp)
            assert new_comp.display_name == orig_comp.display_name
            assert new_comp.x_unit == orig_comp.x_unit
            assert new_comp.y_unit == orig_comp.y_unit

            orig_params = orig_comp.get_all_parameters()
            new_params = new_comp.get_all_parameters()

            assert len(orig_params) == len(new_params)

            for param_orig, param_new in zip(orig_params, new_params, strict=True):
                assert param_new.name == param_orig.name
                assert param_new.value == param_orig.value
                assert param_new.fixed == param_orig.fixed

    @pytest.mark.parametrize('missing_key', ['x_unit', 'y_unit', 'components', 'name'])
    def test_from_dict_requires_all_keys(self, component_collection, missing_key):
        model_dict = component_collection.to_dict()
        del model_dict[missing_key]
        with pytest.raises(KeyError):
            ComponentCollection.from_dict(model_dict)

    def test_copy(self, component_collection):
        # WHEN
        component_collection[0].area.min = 0.5
        component_collection[0].area.fixed = True
        component_collection[0].area.max = 5.0
        component_collection[1].width.min = 0.1
        component_collection[1].width.fixed = True
        component_collection[1].width.max = 2.0

        # THEN
        model_copy = copy(component_collection)

        # EXPECT collection-level checks
        assert model_copy is not component_collection
        assert model_copy.display_name == component_collection.display_name
        assert len(model_copy) == len(component_collection)

        # EXPECT: deep copy, same order
        for orig_comp, copied_comp in zip(component_collection, model_copy, strict=True):
            # New object
            assert copied_comp is not orig_comp

            # Same type and display name
            assert type(copied_comp) is type(orig_comp)
            assert copied_comp.display_name == orig_comp.display_name
            assert copied_comp.x_unit == orig_comp.x_unit
            assert copied_comp.y_unit == orig_comp.y_unit

            # Parameters are deep-copied and equivalent
            orig_params = orig_comp.get_all_parameters()
            copied_params = copied_comp.get_all_parameters()

            assert len(orig_params) == len(copied_params)

            for param_orig, param_copy in zip(orig_params, copied_params, strict=True):
                assert param_copy is not param_orig
                assert param_copy.value == param_orig.value
                assert param_copy.min == param_orig.min
                assert param_copy.max == param_orig.max
                assert param_copy.fixed == param_orig.fixed

    def test_warns_on_duplicate_names_at_init(self):
        g1 = Gaussian(name='SameName', display_name='Display1', area=1.0)
        g2 = Gaussian(name='SameName', display_name='Display2', area=2.0)

        with pytest.warns(UserWarning, match='Duplicate component names'):
            ComponentCollection(components=[g1, g2])

    def test_warns_on_duplicate_names_on_append(self):
        g1 = Gaussian(name='SameName', display_name='Display1', area=1.0)
        g2 = Gaussian(name='SameName', display_name='Display2', area=2.0)
        collection = ComponentCollection(components=[g1])

        with pytest.warns(UserWarning, match='Duplicate component names'):
            collection.append_component(g2)

    def test_no_warning_with_unique_names(self, recwarn):
        g1 = Gaussian(name='Name1', display_name='Display1', area=1.0)
        g2 = Gaussian(name='Name2', display_name='Display2', area=2.0)
        ComponentCollection(components=[g1, g2])
        user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        assert not user_warnings

    def test_y_unit_custom(self):
        # WHEN THEN
        cc = ComponentCollection(y_unit='1/meV')
        # EXPECT
        assert cc.y_unit == '1/meV'

    def test_y_unit_setter_raises(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=r'read-only'):
            component_collection.y_unit = '1/meV'

    def test_convert_y_unit(self):
        # WHEN: components with y_unit='1/meV' so area_unit ≈ dimensionless
        g = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        lor = Lorentzian(area=1.0, x_unit='meV', y_unit='1/meV')
        cc = ComponentCollection(components=[g, lor])

        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        cc.convert_y_unit('1/eV')

        # EXPECT
        assert cc.y_unit == '1/eV'
        for component in cc:
            assert component.y_unit == '1/eV'
        assert g.area.value == pytest.approx(1e3)
        assert lor.area.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            component_collection.convert_y_unit(123)

    def test_convert_x_unit_rollback_on_failure(self):
        # WHEN: collection whose first Gaussian converts fine, but second has an
        # ExpressionComponent that raises NotImplementedError for convert_x_unit.
        g = Gaussian(area=1.0, x_unit='meV')
        expr = ExpressionComponent('A * x', parameters={'A': 1.0}, x_unit='meV')
        cc = ComponentCollection(components=[g, expr])
        original_area = g.area.value

        # THEN: attempt a unit conversion that will fail on the ExpressionComponent
        with pytest.raises(NotImplementedError):
            cc.convert_x_unit('microeV')

        # EXPECT: Gaussian is rolled back to its original state
        assert cc.x_unit == 'meV'
        assert g.x_unit == 'meV'
        assert g.area.value == pytest.approx(original_area)

    def test_convert_y_unit_rollback_on_failure(self):
        # WHEN: collection where first Gaussian converts successfully but second
        # ExpressionComponent always raises NotImplementedError for convert_y_unit.
        g = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        expr = ExpressionComponent('A * x', parameters={'A': 1.0}, x_unit='meV')
        cc = ComponentCollection(components=[g, expr], y_unit='1/meV')
        original_area = g.area.value

        # THEN: attempt y_unit conversion that will fail on the ExpressionComponent
        with pytest.raises(NotImplementedError):
            cc.convert_y_unit('1/eV')

        # EXPECT: collection y_unit and Gaussian are both rolled back
        assert cc.y_unit == '1/meV'
        assert g.y_unit == '1/meV'
        assert g.area.value == pytest.approx(original_area)

    def test_evaluate_scipp_output_with_y_unit(self):
        # WHEN
        g = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        cc = ComponentCollection(components=[g], y_unit='1/meV')
        x = np.linspace(-5, 5, 50)
        # THEN
        result = cc.evaluate(x, output='scipp')
        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('1/meV')

    #############
    # Versioning
    #############

    def test_version_starts_at_zero_and_bumps_on_mutation(self):
        # WHEN a freshly constructed collection with initial components
        collection = ComponentCollection(components=[Gaussian(name='G1'), Lorentzian(name='L1')])

        # EXPECT it starts at version 0
        assert collection.version == 0

        # THEN structural mutations bump the version
        collection.append_component(Gaussian(name='G2'))
        assert collection.version == 1
        collection.pop('G2')
        assert collection.version == 2

    #############
    # Slicing
    #############

    def test_getitem_slice_returns_working_collection(self, component_collection):
        "Regression: slicing used to crash because the base slice path called the wrong ctor"
        # WHEN THEN
        sliced = component_collection[:1]

        # EXPECT a working collection of the same class, carrying the units, sharing the
        # component objects
        assert type(sliced) is ComponentCollection
        assert len(sliced) == 1
        assert sliced[0] is component_collection[0]
        assert sliced.x_unit == component_collection.x_unit
        assert sliced.y_unit == component_collection.y_unit

        # EXPECT the slice is usable
        x = np.linspace(-5, 5, 11)
        np.testing.assert_allclose(sliced.evaluate(x), component_collection[0].evaluate(x))

    #############
    # Regression tests
    #############

    def test_normalize_area_negative_area_raises(self, component_collection):
        "Regression: negative areas used to be silently clamped by normalization"
        # WHEN
        component_collection[0].area.min = -10.0
        component_collection[0].area = -2.0

        # THEN EXPECT
        with pytest.raises(ValueError, match=r'Negative area'):
            component_collection.normalize_area()

    def test_evaluate_empty_invalid_output_raises(self):
        "Regression: the empty-collection path used to skip output validation"
        # WHEN
        collection = ComponentCollection(display_name='EmptyModel')

        # THEN EXPECT
        with pytest.raises(ValueError, match=r"output must be 'numpy' or 'scipp'"):
            collection.evaluate(np.linspace(-1, 1, 5), output='invalid')

    def test_evaluate_empty_scalar_shape_matches_non_empty_path(self):
        "Regression: empty and non-empty paths must agree on the output shape for scalar x"
        # WHEN an empty and a non-empty collection evaluated at a scalar
        empty = ComponentCollection(display_name='EmptyModel')
        non_empty = ComponentCollection(components=Gaussian(name='G'))

        # THEN
        empty_result = empty.evaluate(0.5)
        non_empty_result = non_empty.evaluate(0.5)

        # EXPECT both return 1D arrays of the same shape
        assert empty_result.shape == non_empty_result.shape == (1,)
        assert np.all(empty_result == pytest.approx(0.0))

    def test_evaluate_component_ambiguous_name_raises(self):
        "Regression: duplicate names used to silently evaluate the first match"
        # WHEN a collection with two components sharing a name
        with pytest.warns(UserWarning, match='Duplicate component names'):
            collection = ComponentCollection(
                components=[
                    Gaussian(name='SameName', area=1.0),
                    Gaussian(name='SameName', area=2.0),
                ]
            )

        # THEN EXPECT
        with pytest.raises(AmbiguousNameError, match=r"Ambiguous name 'SameName'"):
            collection.evaluate_component(np.linspace(-1, 1, 5), 'SameName')

    def test_evaluate_scipp_output_multi_component_does_not_raise(self, component_collection):
        # WHEN: collection with two components (Gaussian + Lorentzian)
        x = sc.Variable(dims=['energy'], values=np.linspace(-5.0, 5.0, 100), unit='meV')
        # THEN: evaluate with scipp output
        # Before the fix, sum() started from int 0 → '0 + sc.Variable' raised TypeError.
        result = component_collection.evaluate(x, output='scipp')
        # EXPECT: returns a Variable whose values are the sum of both components
        assert isinstance(result, sc.Variable)
        expected = component_collection[0].evaluate(x, output='scipp') + component_collection[
            1
        ].evaluate(x, output='scipp')
        assert sc.allclose(result, expected)
