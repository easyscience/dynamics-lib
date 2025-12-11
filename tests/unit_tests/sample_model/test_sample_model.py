from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.integrate import simpson

from easydynamics.sample_model import Gaussian, Lorentzian, Polynomial, SampleModel


class TestSampleModel:
    @pytest.fixture
    def sample_model(self):
        model = SampleModel(display_name="TestSampleModel")
        component1 = Gaussian(
            name="TestGaussian1", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        component2 = Lorentzian(
            display_name="TestLorentzian1", area=2.0, center=1.0, width=0.5, unit="meV"
        )
        model.add_component(component1)
        model.add_component(component2)
        return model

    def test_init(self):
        # WHEN THEN
        sample_model = SampleModel(display_name="InitModel")

        # EXPECT
        assert sample_model.name == "InitModel"
        assert sample_model.components == []

    # ───── Component Management ─────

    def test_add_component(self, sample_model):
        # WHEN
        component = Gaussian(
            name="TestComponent", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # THEN
        sample_model.add_component(component)
        # EXPECT
        assert sample_model.components[-1] is component

    def test_add_duplicate_component_raises(self, sample_model):
        # WHEN THEN
        component = Gaussian(
            name="TestGaussian1", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # EXPECT
        with pytest.raises(ValueError, match="already exists"):
            sample_model.add_component(component)

    def test_add_invalid_component_raises(self, sample_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError, match="Component must be an instance of ModelComponent."
        ):
            sample_model.add_component("NotAComponent")

    def test_remove_component(self, sample_model):
        # WHEN THEN
        sample_model.remove_component("TestGaussian1")
        # EXPECT
        assert "TestGaussian1" not in sample_model.components

    def test_remove_nonexistent_component_raises(self, sample_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            KeyError, match="No component named 'NonExistentComponent' exists"
        ):
            sample_model.remove_component("NonExistentComponent")

    def test_getitem(self, sample_model):
        # WHEN
        component = Gaussian(
            name="TestComponent", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # THEN
        sample_model.add_component(component)
        # EXPECT
        assert sample_model.components[-1] is component

    def test_list_component_names(self, sample_model):
        # WHEN THEN
        components = sample_model.list_component_names()
        # EXPECT
        assert len(components) == 2
        assert components[0] == "TestGaussian1"
        assert components[1] == "TestLorentzian1"

    def test_clear_components(self, sample_model):
        # WHEN THEN
        sample_model.clear_components()
        # EXPECT
        assert len(sample_model.components) == 0

    def test_convert_unit(self, sample_model):
        # WHEN THEN
        sample_model.convert_unit("eV")
        # EXPECT
        for component in sample_model.components:
            assert component.unit == "eV"

    def test_convert_unit_failure_rolls_back(self, sample_model):
        # WHEN THEN
        # Introduce a faulty component that will fail conversion
        class FaultyComponent(Gaussian):
            def convert_unit(self, unit: str) -> None:
                raise RuntimeError("Conversion failed.")

        faulty_component = FaultyComponent(
            name="FaultyComponent", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        sample_model.add_component(faulty_component)

        original_units = {
            component.name: component.unit for component in sample_model.components
        }

        # EXPECT
        with pytest.raises(RuntimeError, match="Conversion failed."):
            sample_model.convert_unit("eV")

        # Check that all components have their original units
        for component in sample_model.components:
            assert component.unit == original_units[component.name]

    def test_set_unit(self, sample_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match="Unit is read-only. Use convert_unit to change the unit",
        ):
            sample_model.unit = "eV"

    def test_evaluate(self, sample_model):
        # WHEN
        x = np.linspace(-5, 5, 100)
        result = sample_model.evaluate(x)
        # EXPECT
        expected_result = sample_model.components[0].evaluate(
            x
        ) + sample_model.components[1].evaluate(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_no_components_raises(self):
        # WHEN THEN
        sample_model = SampleModel(display_name="EmptyModel")
        x = np.linspace(-5, 5, 100)
        # EXPECT
        with pytest.raises(ValueError, match="No components in the model to evaluate."):
            sample_model.evaluate(x)

    def test_evaluate_component(self, sample_model):
        # WHEN  THEN
        x = np.linspace(-5, 5, 100)
        result1 = sample_model.evaluate_component(x, "TestGaussian1")
        result2 = sample_model.evaluate_component(x, "TestLorentzian1")

        # EXPECT
        expected_result1 = sample_model.components[0].evaluate(x)
        expected_result2 = sample_model.components[1].evaluate(x)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-5)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-5)

    def test_evaluate_nonexistent_component_raises(self, sample_model):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN EXPECT
        with pytest.raises(
            KeyError, match="No component named 'NonExistentComponent' exists"
        ):
            sample_model.evaluate_component(x, "NonExistentComponent")

    def test_evaluate_component_no_components_raises(self):
        # WHEN THEN
        sample_model = SampleModel(display_name="EmptyModel")
        x = np.linspace(-5, 5, 100)
        # EXPECT
        with pytest.raises(ValueError, match="No components in the model to evaluate."):
            sample_model.evaluate_component(x, "AnyComponent")

    def test_evaluate_component_invalid_name_type_raises(self, sample_model):
        # WHEN
        x = np.linspace(-5, 5, 100)

        # THEN EXPECT
        with pytest.raises(
            TypeError,
            match="Component name must be a string, got <class 'int'> instead.",
        ):
            sample_model.evaluate_component(x, 123)

    # ───── Utilities ─────

    def test_normalize_area(self, sample_model):
        # WHEN THEN
        sample_model.normalize_area()
        # EXPECT
        x = np.linspace(-10000, 10000, 1000000)  # Lorentzians have long tails
        result = sample_model.evaluate(x)
        numerical_area = simpson(result, x)
        assert np.isclose(numerical_area, 1.0, rtol=1e-4)

    def test_normalize_area_no_components_raises(self):
        # WHEN THEN
        sample_model = SampleModel(display_name="EmptyModel")
        # EXPECT
        with pytest.raises(
            ValueError, match="No components in the model to normalize."
        ):
            sample_model.normalize_area()

    @pytest.mark.parametrize(
        "area_value",
        [np.nan, 0.0, np.inf],
        ids=["NaN area", "Zero area", "Infinite area"],
    )
    def test_normalize_area_not_finite_area_raises(self, sample_model, area_value):
        # WHEN THEN
        sample_model.components[0].area = area_value
        sample_model.components[1].area = area_value

        # EXPECT
        with pytest.raises(ValueError, match="cannot normalize."):
            sample_model.normalize_area()

    def test_normalize_area_non_area_component_warns(self, sample_model):
        # WHEN
        component1 = Polynomial(
            display_name="TestPolynomial", coefficients=[1, 2, 3], unit="meV"
        )
        sample_model.add_component(component1)

        # THEN EXPECT
        with pytest.warns(UserWarning, match="does not have an 'area' "):
            sample_model.normalize_area()

    def test_get_parameters(self, sample_model):
        # WHEN THEN
        parameters = sample_model.get_parameters()
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
        sample_model = SampleModel(display_name="EmptyModel")
        # WHEN THEN
        parameters = sample_model.get_parameters()
        # EXPECT
        assert len(parameters) == 0

    def test_get_fit_parameters(self, sample_model):
        # WHEN

        # Fix one parameter and make another dependent
        sample_model.components[0].area.fixed = True
        sample_model.components[1].width.make_dependent_on(
            "comp1_width",
            {"comp1_width": sample_model.components[0].width},
        )

        # THEN
        fit_parameters = sample_model.get_fit_parameters()

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

    def test_fix_and_free_all_parameters(self, sample_model):
        # WHEN THEN
        sample_model.fix_all_parameters()

        # EXPECT
        for param in sample_model.get_parameters():
            assert param.fixed is True

        # WHEN
        sample_model.free_all_parameters()

        # THEN
        for param in sample_model.get_parameters():
            assert param.fixed is False

    def test_contains(self, sample_model):
        # WHEN THEN
        assert "TestGaussian1" in sample_model
        assert "TestLorentzian1" in sample_model
        assert "NonExistentComponent" not in sample_model

        gaussian_component = sample_model.components[0]
        lorentzian_component = sample_model.components[1]
        assert gaussian_component in sample_model
        assert lorentzian_component in sample_model

        fake_component = Gaussian(
            name="FakeGaussian", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        assert fake_component not in sample_model

    def test_repr_contains_name_and_components(self, sample_model):
        # WHEN THEN
        rep = repr(sample_model)
        # EXPECT
        assert "SampleModel" in rep
        assert "TestGaussian" in rep

    def test_copy(self, sample_model):
        # WHEN THEN
        sample_model.temperature = 300
        model_copy = copy(sample_model)
        # EXPECT
        assert model_copy is not sample_model
        assert model_copy.name == sample_model.name
        assert len(model_copy.components) == len(sample_model.components)
        for comp in sample_model.components:
            copied_comp = model_copy.components[
                model_copy.list_component_names().index(comp.name)
            ]
            assert copied_comp is not comp
            assert copied_comp.name == comp.name
            for param_orig, param_copy in zip(
                comp.get_parameters(), copied_comp.get_parameters()
            ):
                assert param_copy is not param_orig
                assert param_copy.name == param_orig.name
                assert param_copy.value == param_orig.value
                assert param_copy.fixed == param_orig.fixed
