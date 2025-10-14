import pytest
import numpy as np
from scipy.integrate import simpson

from easyscience.variable import Parameter
from easydynamics.sample import SampleModel, Gaussian, Lorentzian
from easydynamics.sample.components import ModelComponent
from easydynamics.utils import detailed_balance_factor

class TestSampleModel:
    @pytest.fixture
    def sample_model(self):
        return SampleModel(name="TestSampleModel")

    # ───── Component Management ─────

    def test_add_component(self, sample_model):
        #When
        component = Gaussian(name="TestComponent", area=1.0, center=0.0, width=1.0, unit='meV')
        #Then
        sample_model.add_component(component)
        #Expect
        assert "TestComponent" in sample_model.components

    def test_add_duplicate_component_raises(self, sample_model):
        #When
        component = Gaussian(name="Dup", area=1.0, center=0.0, width=1.0, unit='meV')
        #Then
        sample_model.add_component(component)
        #Expect
        with pytest.raises(ValueError, match="already exists"):
            sample_model.add_component(component)

    def test_remove_component(self, sample_model):
        #When
        component = Gaussian(name="TestComponent", area=1.0, center=0.0, width=1.0, unit='meV')
        #Then
        sample_model.add_component(component)
        sample_model.remove_component("TestComponent")
        #Expect
        assert "TestComponent" not in sample_model.components

    def test_remove_nonexistent_component_raises(self, sample_model):
        #When Then Expect
        with pytest.raises(KeyError, match="No component named 'NonExistentComponent' exists"):
            sample_model.remove_component("NonExistentComponent")

    def test_getitem(self, sample_model):
        #When
        component = Gaussian(name="TestComponent", area=1.0, center=0.0, width=1.0, unit='meV')
        #Then
        sample_model.add_component(component)
        #Expect
        assert sample_model["TestComponent"] is component

    def test_setitem(self, sample_model):
        #When
        component = ModelComponent(name="TestComponent")
        #Then
        sample_model["TestComponent"] = component
        #Expect
        assert sample_model["TestComponent"] is component

    def test_contains_component(self, sample_model):
        #When
        component = Gaussian(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        #Then
        sample_model.add_component(component)
        #Expect
        assert "TestGaussian" in sample_model
        assert "NonExistentComponent" not in sample_model

    def test_list_components(self, sample_model):
        #When
        component1 = Gaussian(name="TestGaussian1", area=2.0, center=0.0, width=1.0, unit='meV')
        component2 = Gaussian(name="TestGaussian2", area=3.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)
        #Then
        components = sample_model.list_components()
        #Expect
        assert len(components) == 2
        assert components[0] == 'TestGaussian1'
        assert components[1] == 'TestGaussian2'

    def test_clear_components(self, sample_model):
        #when
        component1 = Gaussian(name="TestGaussian1", area=2.0, center=0.0, width=1.0, unit='meV')
        component2 = Gaussian(name="TestGaussian2", area=3.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)
        #Then
        sample_model.clear_components()
        #Expect
        assert len(sample_model.components) == 0

    # ───── Temperature and Detailed Balance ─────

    def test_temperature_init(self, sample_model):
        # When Then Expect
        assert sample_model._temperature is None
        assert sample_model._use_detailed_balance is False
        # assert sample_model._temperature.unit == 'K'

    def test_set_temperature(self, sample_model):
        # When Then 
        sample_model.temperature = 300
        # Expect
        assert sample_model._temperature.value == 300
        assert sample_model._temperature.unit == 'K'
        assert sample_model._use_detailed_balance is True

    def test_negative_temperature_throws(self, sample_model):
        # When
        sample_model.use_detailed_balance = True
        # Then Expect
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            sample_model.temperature = -50

    def test_use_detailed_balance(self, sample_model):
        # When Then Expect
        assert sample_model._use_detailed_balance is False
        sample_model._use_detailed_balance = True
        assert sample_model._use_detailed_balance is True
        sample_model._use_detailed_balance = False
        assert sample_model._use_detailed_balance is False

    # ───── Evaluation ─────

    def test_evaluate(self, sample_model):
        # When
        component1 = Gaussian(name="Gaussian1", area=1.0, center=0.0, width=1.0, unit='meV')
        component2 = Lorentzian(name="Lorentzian1", area=2.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)
        # Then
        x = np.linspace(-5, 5, 100)
        result = sample_model.evaluate(x)
        # Expect
        expected_result = component1.evaluate(x) + component2.evaluate(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_detailed_balance(self, sample_model):
        # When
        sample_model.temperature = 300
        component1 = Gaussian(name="Gaussian1", area=1.0, center=0.0, width=1.0, unit='meV')
        component2 = Lorentzian(name="Lorentzian1", area=2.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)
        x = np.linspace(-5, 5, 100)
        # Then
        result = sample_model.evaluate(x)
        # Expect
        expected_result = component1.evaluate(x) + component2.evaluate(x)
        expected_result *= detailed_balance_factor(x, sample_model._temperature.value)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_component(self, sample_model):
        # When
        component1 = Gaussian(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        component2 = Lorentzian(name="TestLorentzian", area=2.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)

        # Then
        x = np.linspace(-5, 5, 100)
        result1 = sample_model.evaluate_component("TestGaussian", x)
        result2 = sample_model.evaluate_component("TestLorentzian", x)
        # Expect
        expected_result1 = component1.evaluate(x)
        expected_result2 = component2.evaluate(x)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-5)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-5)

    def test_evaluate_component_with_detailed_balance(self, sample_model):
        # When
        component1 = Gaussian(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        component2 = Lorentzian(name="TestLorentzian", area=2.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)
        sample_model.temperature = 300
        # Then
        x = np.linspace(-5, 5, 100)
        result1 = sample_model.evaluate_component('TestGaussian', x)
        result2 = sample_model.evaluate_component('TestLorentzian', x)
        # Expect
        expected_result1 = component1.evaluate(x)
        expected_result2 = component2.evaluate(x)
        expected_result1 *= detailed_balance_factor(x, sample_model._temperature.value)
        expected_result2 *= detailed_balance_factor(x, sample_model._temperature.value)
        np.testing.assert_allclose(result1, expected_result1, rtol=1e-5)
        np.testing.assert_allclose(result2, expected_result2, rtol=1e-5)

    def test_evaluate_nonexistent_component_raises(self, sample_model):
        # When Then Expect
        x = np.linspace(-5, 5, 100)
        with pytest.raises(KeyError, match="No component named 'NonExistentComponent' exists"):
            sample_model.evaluate_component("NonExistentComponent", x)

    # ───── Utilities ─────

    def test_normalize_area(self, sample_model):
        # When
        component1 = Gaussian(name="TestGaussian1", area=2.0, center=0.0, width=1.0, unit='meV')
        component2 = Gaussian(name="TestGaussian2", area=3.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)
        # Then
        sample_model.normalize_area()
        # Expect
        x = np.linspace(-10, 10, 1000)
        result = sample_model.evaluate(x)
        numerical_area = simpson(result, x)
        assert np.isclose(numerical_area, 1.0)

    def test_get_parameters(self, sample_model):
        # When
        component = Gaussian(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)
        # Then
        parameters = sample_model.get_parameters()
        # Expect
        assert len(parameters) == 3
        assert parameters[0].name == 'TestGaussian area'
        assert parameters[1].name == 'TestGaussian center'
        assert parameters[2].name == 'TestGaussian width'
        assert all(isinstance(param, Parameter) for param in parameters)

    def test_get_parameters_no_components(self, sample_model):
        # When Then
        parameters = sample_model.get_parameters()
        # Expect
        assert len(parameters) == 0

    def test_repr_contains_name_and_components(self, sample_model):
        # When
        component = Gaussian(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)
        # Then
        rep = repr(sample_model)
        # Expect
        assert "SampleModel" in rep
        assert "TestGaussian" in rep

    # def test_repr_no_components(self, sample_model):
    #     # When Then
    #     rep = repr(sample_model)
    #     # Expect
    #     assert "SampleModel" in rep
    #     assert "Components: None" in rep

    def test_str_contains_name_and_components(self, sample_model):
        # When
        component = Gaussian(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)
        # Then
        str_repr = str(sample_model)
        # Expect
        assert "SampleModel" in str_repr
        assert "TestGaussian" in str_repr

