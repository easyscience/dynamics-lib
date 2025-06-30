import pytest

import numpy as np

from scipy.integrate import simpson

from easydynamics.sample import GaussianComponent, LorentzianComponent, VoigtComponent, DeltaFunctionComponent, DHOComponent, PolynomialComponent
from easydynamics.sample.components import ModelComponent

from easydynamics.sample import SampleModel

from easyscience.variable import Parameter

from scipy.special import voigt_profile

from easydynamics.utils import detailed_balance_factor

class TestSampleModel:
    @pytest.fixture
    def sample_model(self):
        return SampleModel(name="TestSampleModel")

    def test_add_component(self, sample_model):
        component = GaussianComponent(name="TestComponent",area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)
        assert "TestComponent" in sample_model.components

    def test_remove_component(self, sample_model):
        component = GaussianComponent(name="TestComponent",area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)
        sample_model.remove_component("TestComponent")
        assert "TestComponent" not in sample_model.components

    def test_getitem(self, sample_model):
        component = GaussianComponent(name="TestComponent",area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)
        assert sample_model["TestComponent"] is component

    def test_setitem(self, sample_model):
        component = ModelComponent(name="TestComponent")
        sample_model["TestComponent"] = component
        assert sample_model["TestComponent"] is component

    def test_evaluate(self, sample_model):
        component1 = GaussianComponent(name="Gaussian1", area=1.0, center=0.0, width=1.0, unit='meV')
        component2 = LorentzianComponent(name="Lorentzian1", area=2.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)

        x = np.linspace(-5, 5, 100)
        result = sample_model.evaluate(x)

        expected_result = component1.evaluate(x) + component2.evaluate(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_temperature_init(self, sample_model):
        # Check default temperature
        assert sample_model._temperature.value == -1
        assert sample_model._temperature.unit == 'K'

    def test_set_temperature(self, sample_model):
        # Set a valid temperature
        sample_model.temperature = 300
        assert sample_model._temperature.value == 300
        assert sample_model._temperature.unit == 'K'

        # Check if detailed balance is enabled
        assert sample_model.use_detailed_balance is True

    def test_evaluate_with_detailed_balance(self, sample_model):
        # Set a temperature for detailed balance
        sample_model.temperature = 300
        component1 = GaussianComponent(name="Gaussian1", area=1.0, center=0.0, width=1.0, unit='meV')
        component2 = LorentzianComponent(name="Lorentzian1", area=2.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)

        x = np.linspace(-5, 5, 100)
        result = sample_model.evaluate(x)

        expected_result = component1.evaluate(x) + component2.evaluate(x)
        # Apply detailed balance factor
        if sample_model.use_detailed_balance:
            omega_meV = x 
            expected_result *= detailed_balance_factor(omega_meV, sample_model._temperature.value)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)


    def test_evaluate_component(self, sample_model):
        # Add a Gaussian component
        component = GaussianComponent(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)

        x = np.linspace(-5, 5, 100)
        result = sample_model.evaluate(x)

        expected_result = component.evaluate(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_component_with_detailed_balance(self, sample_model):
        # Add a Gaussian component
        component = GaussianComponent(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)

        # Set a temperature for detailed balance
        sample_model.temperature = 300

        x = np.linspace(-5, 5, 100)
        result = sample_model.evaluate(x)

        expected_result = component.evaluate(x)
        # Apply detailed balance factor
        if sample_model.use_detailed_balance:
            omega_meV = x
            expected_result *= detailed_balance_factor(omega_meV, sample_model._temperature.value)

        np.testing.assert_allclose(result, expected_result, rtol=1e-5)



    def test_use_detailed_balance(self, sample_model):
        # Check default value
        assert sample_model.use_detailed_balance is False

        # Enable detailed balance
        sample_model.use_detailed_balance = True
        assert sample_model.use_detailed_balance is True

        # Disable detailed balance
        sample_model.use_detailed_balance = False
        assert sample_model.use_detailed_balance is False


    def test_normalize_area(self, sample_model):
        # Add a couple of Gaussian components with a specific area
        component1 = GaussianComponent(name="TestGaussian1", area=2.0, center=0.0, width=1.0, unit='meV')
        component2 = GaussianComponent(name="TestGaussian2", area=3.0, center=1.0, width=0.5, unit='meV')
        sample_model.add_component(component1)
        sample_model.add_component(component2)

        sample_model.normalize_area()

        # Evaluate the model at a range of x values
        x = np.linspace(-10, 10, 1000)
        result = sample_model.evaluate(x)

        # Check if the area matches the component's area
        numerical_area = simpson(result, x)

        assert np.isclose(numerical_area,1.0)

    def test_get_parameters(self, sample_model):
        # Add a Gaussian component
        component = GaussianComponent(name="TestGaussian", area=1.0, center=0.0, width=1.0, unit='meV')
        sample_model.add_component(component)

        parameters = sample_model.get_parameters()

        # Check if the parameters are correctly retrieved
        assert len(parameters) == 4  # area, center, width, temperature
        assert parameters[0].name=='temperature'
        assert parameters[1].name == 'TestGaussianarea'
        assert parameters[2].name == 'TestGaussiancenter'
        assert parameters[3].name == 'TestGaussianwidth'
        assert all(isinstance(param, Parameter) for param in parameters)

    
