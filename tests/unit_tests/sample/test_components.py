import pytest

import numpy as np

from scipy.integrate import simpson

from easydynamics.sample import GaussianComponent, LorentzianComponent, VoigtComponent, DeltaFunctionComponent, DHOComponent, PolynomialComponent
from easydynamics.sample.components import ModelComponent

from easyscience.variable import Parameter

from scipy.special import voigt_profile


class TestModelComponent:
    class DummyComponent(ModelComponent):
        def __init__(self):
            super().__init__(name="Dummy")
            self.area = Parameter(name="area", value=1.0, unit="meV")
            self.center = Parameter(name="center", value=2.0, unit="meV", fixed=True)
            self.width = Parameter(name="width", value=3.0, unit="meV", fixed=True)

        def get_parameters(self):
            return [self.area, self.center, self.width]

        def evaluate(self, x):
            return np.zeros_like(x)

    @pytest.fixture
    def dummy(self):
        return self.DummyComponent()

    def test_fix_all_parameters_sets_all_to_fixed(self, dummy):
        # WHEN
        dummy.fix_all_parameters()

        # THEN EXPECT
        assert all(p.fixed for p in dummy.get_parameters())

    def test_fit_all_parameters_sets_all_to_unfixed(self, dummy):
        # WHEN
        dummy.fit_all_parameters()

        # THEN EXPECT
        assert all(not p.fixed for p in dummy.get_parameters())


class TestGaussianComponent:

    @pytest.fixture
    def gaussian(self):
        return GaussianComponent(name='TestGaussian', area=2.0, center=0.5, width=0.6, unit='meV')
    
    def test_initialization(self, gaussian: GaussianComponent):
        assert gaussian.name == 'TestGaussian'
        assert gaussian.area.value == 2.0
        assert gaussian.center.value == 0.5
        assert gaussian.width.value == 0.6
        assert gaussian.unit == 'meV'

    def test_evaluate(self, gaussian: GaussianComponent):
        x = np.array([0.0, 0.5, 1.0])
        expected = gaussian.evaluate(x)
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_get_parameters(self, gaussian: GaussianComponent):
        params = gaussian.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestGaussianarea'
        assert params[1].name == 'TestGaussiancenter'
        assert params[2].name == 'TestGaussianwidth'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, gaussian: GaussianComponent):
        # WHEN
        x = np.linspace(gaussian.center.value - 10 * gaussian.width.value, gaussian.center.value + 10 * gaussian.width.value, 1000)
        y = gaussian.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert np.isclose(numerical_area, gaussian.area.value, rtol=1e-3)

class TestLorentzianComponent:

    @pytest.fixture
    def lorentzian(self):
        return LorentzianComponent(name='TestLorentzian', area=2.0, center=0.5, width=0.6, unit='meV')

    def test_initialization(self, lorentzian: LorentzianComponent):
        assert lorentzian.name == 'TestLorentzian'
        assert lorentzian.area.value == 2.0
        assert lorentzian.center.value == 0.5
        assert lorentzian.width.value == 0.6
        assert lorentzian.unit == 'meV'

    def test_evaluate(self, lorentzian: LorentzianComponent):
        x = np.array([0.0, 0.5, 1.0])
        expected = lorentzian.evaluate(x)
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_get_parameters(self, lorentzian: LorentzianComponent):
        params = lorentzian.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestLorentzianarea'
        assert params[1].name == 'TestLorentziancenter'
        assert params[2].name == 'TestLorentzianwidth'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, lorentzian: LorentzianComponent):
        # WHEN
        x = np.linspace(lorentzian.center.value - 500 * lorentzian.width.value, lorentzian.center.value + 500 * lorentzian.width.value, 20000) #Lorentzians have very long tails
        y = lorentzian.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(lorentzian.area.value, rel=2e-3)

class TestVoigtComponent:

    @pytest.fixture
    def voigt(self):
        return VoigtComponent(name='TestVoigt', area=2.0, center=0.5, Gwidth=0.6, Lwidth=0.7, unit='meV')

    def test_initialization(self, voigt: VoigtComponent):
        assert voigt.name == 'TestVoigt'
        assert voigt.area.value == 2.0
        assert voigt.center.value == 0.5
        assert voigt.Gwidth.value == 0.6
        assert voigt.Lwidth.value == 0.7
        assert voigt.unit == 'meV'

    def test_evaluate(self, voigt: VoigtComponent):
        x = np.array([0.0, 0.5, 1.0])
        expected = voigt.evaluate(x)
        expected_result = 2.0 * voigt_profile(x - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_get_parameters(self, voigt: VoigtComponent):
        params = voigt.get_parameters()
        assert len(params) == 4
        assert params[0].name == 'TestVoigtarea'
        assert params[1].name == 'TestVoigtcenter'
        assert params[2].name == 'TestVoigtGwidth'
        assert params[3].name == 'TestVoigtLwidth'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, voigt: VoigtComponent):   
        # WHEN
        x = np.linspace(voigt.center.value - 100 * voigt.Gwidth.value-300*voigt.Lwidth.value, voigt.center.value + 100 * voigt.Gwidth.value+300*voigt.Lwidth.value, 20000) #Voigts have very long tails
        y = voigt.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(voigt.area.value, rel=2e-3)

class TestDeltaFunctionComponent:

    @pytest.fixture
    def delta_function(self):
        return DeltaFunctionComponent(name='TestDeltaFunction', area=2.0, center=0.5, unit='meV')

    def test_initialization(self, delta_function: DeltaFunctionComponent):
        assert delta_function.name == 'TestDeltaFunction'
        assert delta_function.area.value == 2.0
        assert delta_function.center.value == 0.5
        assert delta_function.unit == 'meV'

    @pytest.mark.xfail(reason="DeltaFunctionComponent.evaluate is not implemented yet without resolution convolution")
    def test_evaluate(self, delta_function: DeltaFunctionComponent):
        x = np.array([0.0, 0.5, 1.0])
        expected = delta_function.evaluate(x)
        expected_result = np.zeros_like(x)
        # expected_result[x == 0.5] = 2.0
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_get_parameters(self, delta_function: DeltaFunctionComponent):
        params = delta_function.get_parameters()
        assert len(params) == 2
        assert params[0].name == 'TestDeltaFunctionarea'
        assert params[1].name == 'TestDeltaFunctioncenter'
        assert all(isinstance(param, Parameter) for param in params)


class TestDHOComponent: 
    @pytest.fixture
    def dho(self):  
        return DHOComponent(name='TestDHO', area=2.0, center=1.5, width=0.3, unit='meV')
    
    def test_initialization(self, dho: DHOComponent):
        assert dho.name == 'TestDHO'
        assert dho.area.value == 2.0
        assert dho.center.value == 1.5
        assert dho.width.value == 0.3
        assert dho.unit == 'meV'

    def test_evaluate(self, dho: DHOComponent):
        x = np.array([0.0, 1.5, 3.0])
        expected = dho.evaluate(x)
        expected_result = 2*2.0 * (1.5**2) * (0.3) / np.pi / (((x**2 - 1.5**2) ** 2 + (2*0.3 * x) ** 2))
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_get_parameters(self, dho: DHOComponent):
        params = dho.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestDHOarea'
        assert params[1].name == 'TestDHOcenter'
        assert params[2].name == 'TestDHOwidth'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, dho: DHOComponent):
        # WHEN
        x = np.linspace(-dho.center.value - 20 * dho.width.value, dho.center.value + 20 * dho.width.value, 5000)
        y = dho.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(dho.area.value, rel=2e-3)


class TestPolynomialComponent:
    @pytest.fixture
    def polynomial(self):
        return PolynomialComponent(name='TestPolynomial', coefficients=[1.0, -2.0, 3.0])


    def test_initialization(self, polynomial: PolynomialComponent):
        assert polynomial.name == 'TestPolynomial'
        assert polynomial.coefficients[0].value==1.0
        assert polynomial.coefficients[1].value==-2.0
        assert polynomial.coefficients[2].value==3.0

    def test_evaluate(self, polynomial: PolynomialComponent):
        x = np.array([0.0, 1.0, 2.0])
        expected = polynomial.evaluate(x)
        expected_result = 1.0 - 2.0 * x + 3.0 * x**2
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_get_parameters(self, polynomial: PolynomialComponent):
        params = polynomial.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestPolynomial_c0'
        assert params[1].name == 'TestPolynomial_c1'
        assert params[2].name == 'TestPolynomial_c2'
        assert all(isinstance(param, Parameter) for param in params)
