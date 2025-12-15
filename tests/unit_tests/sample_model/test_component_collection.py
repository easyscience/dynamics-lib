from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.integrate import simpson

from easydynamics.sample_model import (
    ComponentCollection,
    Gaussian,
    Lorentzian,
    Polynomial,
)


class TestComponentCollection:
    @pytest.fixture
    def component_collection(self):
        model = ComponentCollection(display_name="TestComponentCollection")
        component1 = Gaussian(
            display_name="TestGaussian1", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        component2 = Lorentzian(
            display_name="TestLorentzian1", area=2.0, center=1.0, width=0.5, unit="meV"
        )
        model.add_component(component1)
        model.add_component(component2)
        return model

    def test_init(self):
        # WHEN THEN
        component_collection = ComponentCollection(display_name="InitModel")

        # EXPECT
        assert component_collection.display_name == "InitModel"
        assert component_collection.components == []

    # ───── Component Management ─────

    def test_add_component(self, component_collection):
        # WHEN
        component = Gaussian(
            display_name="TestComponent", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # THEN
        component_collection.add_component(component)
        # EXPECT
        assert component_collection.components[-1] is component

    def test_add_duplicate_component_name_raises(self, component_collection):
        # WHEN THEN
        component = Gaussian(
            display_name="TestGaussian1", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # EXPECT
        with pytest.raises(ValueError, match="is already in the collection"):
            component_collection.add_component(component)

    def test_add_existing_component_raises(self, component_collection):
        # WHEN THEN
        component = component_collection.components[0]
        # EXPECT
        with pytest.raises(ValueError, match="is already in the collection"):
            component_collection.add_component(component)

    def test_add_invalid_component_raises(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError, match="Component must be an instance of ModelComponent."
        ):
            component_collection.add_component("NotAComponent")

    def test_remove_component(self, component_collection):
        # WHEN THEN
        component_collection.remove_component("TestGaussian1")
        # EXPECT
        assert "TestGaussian1" not in component_collection.components

    def test_remove_component_raises(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="Component name must be a string"):
            component_collection.remove_component(123)

    def test_remove_nonexistent_component_raises(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(
            KeyError, match="No component named 'NonExistentComponent' exists"
        ):
            component_collection.remove_component("NonExistentComponent")

    def test_getitem(self, component_collection):
        # WHEN
        component = Gaussian(
            display_name="TestComponent", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # THEN
        component_collection.add_component(component)
        # EXPECT
        assert component_collection.components[-1] is component

    def test_list_component_names(self, component_collection):
        # WHEN THEN
        components = component_collection.list_component_names()
        # EXPECT
        assert len(components) == 2
        assert components[0] == "TestGaussian1"
        assert components[1] == "TestLorentzian1"

    def test_clear_components(self, component_collection):
        # WHEN THEN
        component_collection.clear_components()
        # EXPECT
        assert len(component_collection.components) == 0

    def test_convert_unit(self, component_collection):
        # WHEN THEN
        component_collection.convert_unit("eV")
        # EXPECT
        for component in component_collection.components:
            assert component.unit == "eV"

    def test_convert_unit_failure_rolls_back(self, component_collection):
        # WHEN THEN
        # Introduce a faulty component that will fail conversion
        class FaultyComponent(Gaussian):
            def convert_unit(self, unit: str) -> None:
                raise RuntimeError("Conversion failed.")

        faulty_component = FaultyComponent(
            display_name="FaultyComponent", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        component_collection.add_component(faulty_component)

        original_units = {
            component.display_name: component.unit
            for component in component_collection.components
        }

        # EXPECT
        with pytest.raises(RuntimeError, match="Conversion failed."):
            component_collection.convert_unit("eV")

        # Check that all components have their original units
        for component in component_collection.components:
            assert component.unit == original_units[component.display_name]

    def test_set_unit(self, component_collection):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match="Unit is read-only. Use convert_unit to change the unit",
        ):
            component_collection.unit = "eV"

    def test_evaluate(self, component_collection):
        # WHEN
        x = np.linspace(-5, 5, 100)
        result = component_collection.evaluate(x)
        # EXPECT
        expected_result = component_collection.components[0].evaluate(
            x
        ) + component_collection.components[1].evaluate(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_no_components_raises(self):
        # WHEN THEN
        component_collection = ComponentCollection(display_name="EmptyModel")
        x = np.linspace(-5, 5, 100)
        # EXPECT
        with pytest.raises(ValueError, match="No components in the model to evaluate."):
            component_collection.evaluate(x)

    def test_evaluate_component(self, component_collection):
        # WHEN  THEN
        x = np.linspace(-5, 5, 100)
        result1 = component_collection.evaluate_component(x, "TestGaussian1")
        result2 = component_collection.evaluate_component(x, "TestLorentzian1")

        # EXPECT
        expected_result1 = component_collection.components[0].evaluate(x)
        expected_result2 = component_collection.components[1].evaluate(x)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-5)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-5)

    def test_evaluate_nonexistent_component_raises(self, component_collection):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN EXPECT
        with pytest.raises(
            KeyError, match="No component named 'NonExistentComponent' exists"
        ):
            component_collection.evaluate_component(x, "NonExistentComponent")

    def test_evaluate_component_no_components_raises(self):
        # WHEN THEN
        component_collection = ComponentCollection(display_name="EmptyModel")
        x = np.linspace(-5, 5, 100)
        # EXPECT
        with pytest.raises(ValueError, match="No components in the model to evaluate."):
            component_collection.evaluate_component(x, "AnyComponent")

    def test_evaluate_component_invalid_name_type_raises(self, component_collection):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN EXPECT
        with pytest.raises(
            TypeError,
            match="Component name must be a string, got <class 'int'> instead.",
        ):
            component_collection.evaluate_component(x, 123)

    # ───── Utilities ─────

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
        component_collection = ComponentCollection(display_name="EmptyModel")
        # EXPECT
        with pytest.raises(
            ValueError, match="No components in the model to normalize."
        ):
            component_collection.normalize_area()

    @pytest.mark.parametrize(
        "area_value",
        [np.nan, 0.0, np.inf],
        ids=["NaN area", "Zero area", "Infinite area"],
    )
    def test_normalize_area_not_finite_area_raises(
        self, component_collection, area_value
    ):
        # WHEN THEN
        component_collection.components[0].area = area_value
        component_collection.components[1].area = area_value

        # EXPECT
        with pytest.raises(ValueError, match="cannot normalize."):
            component_collection.normalize_area()

    def test_normalize_area_non_area_component_warns(self, component_collection):
        # WHEN
        component1 = Polynomial(
            display_name="TestPolynomial", coefficients=[1, 2, 3], unit="meV"
        )
        component_collection.add_component(component1)

        # THEN EXPECT
        with pytest.warns(UserWarning, match="does not have an 'area' "):
            component_collection.normalize_area()

    def test_get_all_parameters(self, component_collection):
        # WHEN THEN
        parameters = component_collection.get_all_parameters()
        # EXPECT
        assert len(parameters) == 6

        expected_names = {
            "TestGaussian1 area",
            "TestGaussian1 center",
            "TestGaussian1 width",
            "TestLorentzian1 area",
            "TestLorentzian1 center",
            "TestLorentzian1 width",
        }
        actual_names = {param.name for param in parameters}
        assert actual_names == expected_names
        assert all(isinstance(param, Parameter) for param in parameters)

    def test_get_parameters_no_components(self):
        component_collection = ComponentCollection(display_name="EmptyModel")
        # WHEN THEN
        parameters = component_collection.get_all_parameters()
        # EXPECT
        assert len(parameters) == 0

    def test_get_fit_parameters(self, component_collection):
        # WHEN

        # Fix one parameter and make another dependent
        component_collection.components[0].area.fixed = True
        component_collection.components[1].width.make_dependent_on(
            "comp1_width",
            {"comp1_width": component_collection.components[0].width},
        )

        # THEN
        fit_parameters = component_collection.get_fit_parameters()

        # EXPECT
        assert len(fit_parameters) == 4

        expected_names = {
            "TestGaussian1 center",
            "TestGaussian1 width",
            "TestLorentzian1 area",
            "TestLorentzian1 center",
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
        assert "TestGaussian1" in component_collection
        assert "TestLorentzian1" in component_collection
        assert "NonExistentComponent" not in component_collection

        gaussian_component = component_collection.components[0]
        lorentzian_component = component_collection.components[1]
        assert gaussian_component in component_collection
        assert lorentzian_component in component_collection

        # WHEN THEN
        fake_component = Gaussian(
            display_name="FakeGaussian", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # EXPECT
        assert fake_component not in component_collection
        assert 123 not in component_collection  # Invalid type

    def test_repr_contains_name_and_components(self, component_collection):
        # WHEN THEN
        rep = repr(component_collection)
        # EXPECT
        assert "ComponentCollection" in rep
        assert "TestGaussian" in rep

    def test_copy(self, component_collection):
        # WHEN THEN
        component_collection.temperature = 300
        model_copy = copy(component_collection)
        # EXPECT
        assert model_copy is not component_collection
        assert model_copy.display_name == component_collection.display_name
        assert len(model_copy.components) == len(component_collection.components)
        for comp in component_collection.components:
            copied_comp = model_copy.components[
                model_copy.list_component_names().index(comp.display_name)
            ]
            assert copied_comp is not comp
            assert copied_comp.display_name == comp.display_name
            for param_orig, param_copy in zip(
                comp.get_all_parameters(), copied_comp.get_all_parameters()
            ):
                assert param_copy is not param_orig
                assert param_copy.name == param_orig.name
                assert param_copy.value == param_orig.value
                assert param_copy.fixed == param_orig.fixed

    def test_to_dict(self, component_collection):
        # WHEN THEN
        model_dict = component_collection.to_dict()
        # EXPECT
        assert model_dict["display_name"] == "TestComponentCollection"
        assert len(model_dict["components"]) == 2
        component_names = [
            comp_dict["display_name"] for comp_dict in model_dict["components"]
        ]
        assert "TestGaussian1" in component_names
        assert "TestLorentzian1" in component_names

    def test_from_dict(self, component_collection):
        # WHEN
        model_dict = component_collection.to_dict()
        # THEN
        new_model = ComponentCollection.from_dict(model_dict)
        # EXPECT
        assert new_model.display_name == component_collection.display_name
        assert len(new_model.components) == len(component_collection.components)
        for comp in component_collection.components:
            new_comp = new_model.components[
                new_model.list_component_names().index(comp.display_name)
            ]
            assert new_comp.display_name == comp.display_name
            for param_orig, param_new in zip(
                comp.get_all_parameters(), new_comp.get_all_parameters()
            ):
                assert param_new.name == param_orig.name
                assert param_new.value == param_orig.value
                assert param_new.fixed == param_orig.fixed
