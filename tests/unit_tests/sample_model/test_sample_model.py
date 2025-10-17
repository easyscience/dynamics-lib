from copy import copy

import numpy as np
import pytest
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson

from easydynamics.sample_model import Gaussian, Lorentzian, Polynomial, SampleModel
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils import _detailed_balance_factor as detailed_balance_factor


class TestSampleModel:
    @pytest.fixture
    def sample_model(self):
        model = SampleModel(name="TestSampleModel")
        component1 = Gaussian(
            name="TestGaussian1", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        component2 = Lorentzian(
            name="TestLorentzian1", area=2.0, center=1.0, width=0.5, unit="meV"
        )
        model.add_component(component1)
        model.add_component(component2)
        return model

    def test_init_no_temperature(self, sample_model):
        # WHEN THEN EXPECT
        assert sample_model.name == "TestSampleModel"
        assert isinstance(sample_model.components, dict)
        assert len(sample_model.components) == 2
        assert not sample_model.use_detailed_balance

    def test_init_with_temperature(self):
        # WHEN THEN
        sample_model = SampleModel(name="TempModel", temperature=100)

        # EXPECT
        assert sample_model.name == "TempModel"
        assert isinstance(sample_model.components, dict)
        assert len(sample_model.components) == 0
        assert sample_model.use_detailed_balance
        assert isinstance(sample_model.temperature, Parameter)
        assert sample_model.temperature.value == 100

    # ───── Component Management ─────

    def test_add_component(self, sample_model):
        # WHEN
        component = Gaussian(
            name="TestComponent", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        # THEN
        sample_model.add_component(component)
        # EXPECT
        assert "TestComponent" in sample_model.components

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
            TypeError, match="component must be an instance of ModelComponent."
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
        assert sample_model["TestComponent"] is component

    def test_setitem(self, sample_model):
        # WHEN
        component = ModelComponent(name="TestComponent")
        # THEN
        sample_model["TestComponent"] = component
        # EXPECT
        assert sample_model["TestComponent"] is component

    def test_contains_component(self, sample_model):
        # WHEN THEN EXPECT
        assert "TestGaussian1" in sample_model
        assert "NonExistentComponent" not in sample_model

    def test_list_components(self, sample_model):
        # WHEN THEN
        components = sample_model.list_components()
        # EXPECT
        assert len(components) == 2
        assert components[0] == "TestGaussian1"
        assert components[1] == "TestLorentzian1"

    def test_clear_components(self, sample_model):
        # WHEN THEN
        sample_model.clear_components()
        # EXPECT
        assert len(sample_model.components) == 0

    # ───── Temperature and Detailed Balance ─────

    def test_set_temperature(self, sample_model):
        # Set valid temperature
        # WHEN THEN
        sample_model.temperature = 300
        # EXPECT
        assert sample_model.temperature.value == 300
        assert sample_model.temperature.unit == "K"

        # WHEN THEN
        sample_model.temperature = 150.0
        # EXPECT
        assert sample_model.temperature.value == 150.0
        assert sample_model.temperature.unit == "K"

        # Set temperature to None
        # WHEN THEN
        sample_model.temperature = None
        # EXPECT
        assert sample_model.temperature is None
        assert not sample_model.use_detailed_balance

    def test_invalid_temperature_raises(self, sample_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="Temperature must be a number or None."):
            sample_model.temperature = "invalid"

    def test_negative_temperature_raises(self, sample_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            sample_model.temperature = -50

    def test_convert_temperature_unit(self, sample_model):
        # WHEN
        sample_model.temperature = 300  # Kelvin
        # THEN
        sample_model.convert_temperature_unit("mK")
        # EXPECT
        assert np.isclose(sample_model.temperature.value, 300000.0)
        assert sample_model.temperature.unit == "mK"

    def test_convert_temperature_unit_incompatible_unit_raises(self, sample_model):
        # WHEN
        sample_model.temperature = 300  # Kelvin
        # THEN EXPECT
        with pytest.raises(UnitError, match="Failed to convert temperature"):
            sample_model.convert_temperature_unit("m")

    def test_convert_temperature_unit_no_temperature_raises(self, sample_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match="cannot convert units"):
            sample_model.convert_temperature_unit("mK")

    def test_use_detailed_balance(self, sample_model):
        sample_model.temperature = 300
        # WHEN THEN EXPECT
        assert sample_model.use_detailed_balance is False
        sample_model.use_detailed_balance = True
        assert sample_model.use_detailed_balance is True
        sample_model.use_detailed_balance = False
        assert sample_model.use_detailed_balance is False

    def test_use_detailed_balance_no_temperature_raises(self, sample_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError,
            match="Temperature must be set to use detailed balance.",
        ):
            sample_model.use_detailed_balance = True

    # ───── Evaluation ─────

    def test_evaluate(self, sample_model):
        # WHEN
        x = np.linspace(-5, 5, 100)
        result = sample_model.evaluate(x)
        # EXPECT
        expected_result = sample_model["TestGaussian1"].evaluate(x) + sample_model[
            "TestLorentzian1"
        ].evaluate(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    @pytest.mark.parametrize(
        "normalize_db", [True, False], ids=["normalize DB", "Don't normalize DB"]
    )
    def test_evaluate_with_detailed_balance(self, sample_model, normalize_db):
        # WHEN
        sample_model.temperature = 300
        sample_model.use_detailed_balance = True
        sample_model.normalize_detailed_balance = normalize_db

        x = np.linspace(-5, 5, 100)

        # THEN
        result = sample_model.evaluate(x)

        # EXPECT
        expected_result = sample_model["TestGaussian1"].evaluate(x) + sample_model[
            "TestLorentzian1"
        ].evaluate(x)
        expected_result *= detailed_balance_factor(
            energy=x,
            temperature=sample_model.temperature,
            divide_by_temperature=normalize_db,
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_no_components_raises(self):
        # WHEN THEN
        sample_model = SampleModel(name="EmptyModel")
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
        expected_result1 = sample_model["TestGaussian1"].evaluate(x)
        expected_result2 = sample_model["TestLorentzian1"].evaluate(x)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-5)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-5)

    @pytest.mark.parametrize(
        "normalize_db", [True, False], ids=["normalize DB", "Don't normalize DB"]
    )
    def test_evaluate_component_with_detailed_balance(self, sample_model, normalize_db):
        # WHEN
        sample_model.temperature = 300
        sample_model.use_detailed_balance = True
        sample_model.normalize_detailed_balance = normalize_db

        # THEN
        x = np.linspace(-5, 5, 100)
        result1 = sample_model.evaluate_component(x, name="TestGaussian1")
        result2 = sample_model.evaluate_component(x, name="TestLorentzian1")

        # EXPECT
        expected_result1 = sample_model["TestGaussian1"].evaluate(x)
        expected_result2 = sample_model["TestLorentzian1"].evaluate(x)
        expected_result1 *= detailed_balance_factor(
            energy=x,
            temperature=sample_model.temperature,
            divide_by_temperature=normalize_db,
        )
        expected_result2 *= detailed_balance_factor(
            energy=x,
            temperature=sample_model.temperature,
            divide_by_temperature=normalize_db,
        )
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
        sample_model = SampleModel(name="EmptyModel")
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
        sample_model["TestGaussian1"].area = area_value
        sample_model["TestLorentzian1"].area = area_value

        # EXPECT
        with pytest.raises(ValueError, match="cannot normalize."):
            sample_model.normalize_area()

    def test_normalize_area_non_area_component_warns(self, sample_model):
        # WHEN
        component1 = Polynomial(
            name="TestPolynomial", coefficients=[1, 2, 3], unit="meV"
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

        # WHEN
        sample_model.temperature = 300
        # THEN
        parameters = sample_model.get_parameters()
        # EXPECT
        assert len(parameters) == 7
        expected_names.add("temperature")
        actual_names = {param.name for param in parameters}
        assert actual_names == expected_names

    def test_get_parameters_no_components(self):
        sample_model = SampleModel(name="EmptyModel")
        # WHEN THEN
        parameters = sample_model.get_parameters()
        # EXPECT
        assert len(parameters) == 0

        # WHEN THEN
        sample_model.temperature = 300
        parameters = sample_model.get_parameters()
        # EXPECT
        assert len(parameters) == 1
        assert parameters[0].name == "temperature"

    def test_get_fit_parameters(self, sample_model):
        # WHEN

        # Fix one parameter and make another dependent
        sample_model["TestGaussian1"].area.fixed = True
        sample_model["TestLorentzian1"].width.make_dependent_on(
            "comp1_width",
            {"comp1_width": sample_model["TestGaussian1"].width},
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

    def test_delitem(self, sample_model):
        # WHEN THEN
        del sample_model["TestGaussian1"]
        # EXPECT
        assert "TestGaussian1" not in sample_model.components

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
        assert model_copy.name == "copy of " + sample_model.name
        assert len(model_copy.components) == len(sample_model.components)
        assert model_copy.temperature is not sample_model.temperature
        assert model_copy.temperature.name == sample_model.temperature.name
        assert model_copy.temperature.value == sample_model.temperature.value
        assert model_copy.temperature.unit == sample_model.temperature.unit
        assert model_copy.use_detailed_balance == sample_model.use_detailed_balance
        assert (
            model_copy.normalize_detailed_balance
            == sample_model.normalize_detailed_balance
        )
        for name, comp in sample_model.components.items():
            copied_comp = model_copy.components[name]
            assert copied_comp is not comp
            assert copied_comp.name == comp.name
            for param_orig, param_copy in zip(
                comp.get_parameters(), copied_comp.get_parameters()
            ):
                assert param_copy is not param_orig
                assert param_copy.name == param_orig.name
                assert param_copy.value == param_orig.value
                assert param_copy.fixed == param_orig.fixed
