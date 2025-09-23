import pytest

import numpy as np
import scipp as sc

from scipy.integrate import simpson

from easydynamics.sample import Gaussian, Lorentzian, Voigt, DeltaFunction, DampedHarmonicOscillator, Polynomial
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

class TestGaussian:

    @pytest.fixture
    def gaussian(self):
        return Gaussian(name='TestGaussian', area=2.0, center=0.5, width=0.6, unit='meV')
    
    def test_initialization(self, gaussian: Gaussian):
        assert gaussian.name == 'TestGaussian'
        assert gaussian.area.value == 2.0
        assert gaussian.center.value == 0.5
        assert gaussian.width.value == 0.6
        assert gaussian.unit == 'meV'

    def test_evaluate(self, gaussian: Gaussian):
        x = np.array([0.0, 0.5, 1.0])
        expected = gaussian.evaluate(x)
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, gaussian: Gaussian):
        x = sc.array(dims=['x'], values=[0.0, 0.5, 1.0], unit='meV')
        expected = gaussian.evaluate(x)
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x.values - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, gaussian: Gaussian):
        x = sc.array(dims=['x'], values=[0.0, 500.0, 1000.0], unit='microeV')
        expected = gaussian.evaluate(x)
        expected_result = (2.0*1e3 / (0.6*1e3 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x.values - 500.0) / (0.6*1e3)) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)    

    def test_center_is_fixed_if_set_to_None(self):
        test_gaussian=Gaussian(name='TestGaussian', area=2.0, center=None, width=0.6, unit='meV')
        assert test_gaussian.center.value ==0.0
        assert test_gaussian.center.fixed is True

    def test_input_as_parameter(self):
        param_area=Parameter(name='area_param',value=2.0,unit='meV')
        param_center=Parameter(name='center_param',value=0.5,unit='meV')
        param_width=Parameter(name='width_param',value=0.6,unit='meV')
        test_gaussian=Gaussian(name='TestGaussian', area=param_area, center=param_center, width=param_width, unit='meV')
        assert test_gaussian.area==param_area
        assert test_gaussian.center==param_center
        assert test_gaussian.width==param_width

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="The width of a Gaussian must be greater than zero."):
            Gaussian(name='TestGaussian', area=2.0, center=0.5, width=-0.6, unit='meV')

    def test_get_parameters(self, gaussian: Gaussian):
        params = gaussian.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestGaussian area'
        assert params[1].name == 'TestGaussian center'
        assert params[2].name == 'TestGaussian width'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, gaussian: Gaussian):
        # WHEN
        x = np.linspace(gaussian.center.value - 10 * gaussian.width.value, gaussian.center.value + 10 * gaussian.width.value, 1000)
        y = gaussian.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert np.isclose(numerical_area, gaussian.area.value, rtol=1e-3)

class TestLorentzian:

    @pytest.fixture
    def lorentzian(self):
        return Lorentzian(name='TestLorentzian', area=2.0, center=0.5, width=0.6, unit='meV')

    def test_initialization(self, lorentzian: Lorentzian):
        assert lorentzian.name == 'TestLorentzian'
        assert lorentzian.area.value == 2.0
        assert lorentzian.center.value == 0.5
        assert lorentzian.width.value == 0.6
        assert lorentzian.unit == 'meV'

    def test_evaluate(self, lorentzian: Lorentzian):
        x = np.array([0.0, 0.5, 1.0])
        expected = lorentzian.evaluate(x)
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, lorentzian: Lorentzian):
        x = sc.array(dims=['x'], values=[0.0, 0.5, 1.0], unit='meV')
        expected = lorentzian.evaluate(x)
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x.values - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, lorentzian: Lorentzian):
        x = sc.array(dims=['x'], values=[0.0, 500.0, 1000.0], unit='microeV')
        expected = lorentzian.evaluate(x)
        expected_result = (2.0*1e3 / (np.pi * 0.6*1e3)) / (1 + ((x.values - 0.5*1e3) / (0.6*1e3)) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self):
        test_lorentzian=Lorentzian(name='TestLorentzian', area=2.0, center=None, width=0.6, unit='meV')
        assert test_lorentzian.center.value ==0.0
        assert test_lorentzian.center.fixed is True

    def test_input_as_parameter(self):
        param_area=Parameter(name='area_param',value=2.0,unit='meV')
        param_center=Parameter(name='center_param',value=0.5,unit='meV')
        param_width=Parameter(name='width_param',value=0.6,unit='meV')
        test_lorentzian=Lorentzian(name='TestLorentzian', area=param_area, center=param_center, width=param_width, unit='meV')
        assert test_lorentzian.area==param_area
        assert test_lorentzian.center==param_center
        assert test_lorentzian.width==param_width

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="The width of a Lorentzian must be greater than zero."):
            Lorentzian(name='TestLorentzian', area=2.0, center=0.5, width=-0.6, unit='meV')

    def test_get_parameters(self, lorentzian: Lorentzian):
        params = lorentzian.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestLorentzian area'
        assert params[1].name == 'TestLorentzian center'
        assert params[2].name == 'TestLorentzian width'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, lorentzian: Lorentzian):
        # WHEN
        x = np.linspace(lorentzian.center.value - 500 * lorentzian.width.value, lorentzian.center.value + 500 * lorentzian.width.value, 20000) #Lorentzians have very long tails
        y = lorentzian.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(lorentzian.area.value, rel=2e-3)

class TestVoigt:

    @pytest.fixture
    def voigt(self):
        return Voigt(name='TestVoigt', area=2.0, center=0.5, gaussian_width=0.6, lorentzian_width=0.7, unit='meV')

    def test_initialization(self, voigt: Voigt):
        assert voigt.name == 'TestVoigt'
        assert voigt.area.value == 2.0
        assert voigt.center.value == 0.5
        assert voigt.Gwidth.value == 0.6
        assert voigt.Lwidth.value == 0.7
        assert voigt.unit == 'meV'

    def test_evaluate(self, voigt: Voigt):
        x = np.array([0.0, 0.5, 1.0])
        expected = voigt.evaluate(x)
        expected_result = 2.0 * voigt_profile(x - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, voigt: Voigt):
        x = sc.array(dims=['x'], values=[0.0, 0.5, 1.0], unit='meV')
        expected = voigt.evaluate(x)
        expected_result = 2.0 * voigt_profile(x.values - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, voigt: Voigt):
        x = sc.array(dims=['x'], values=[0.0, 500.0, 1000.0], unit='microeV')
        expected = voigt.evaluate(x)
        expected_result = 2.0*1e3 * voigt_profile(x.values - 0.5*1e3, 0.6*1e3, 0.7*1e3)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self):
        test_voigt=Voigt(name='TestVoigt', area=2.0, center=None, gaussian_width=0.6, lorentzian_width=0.7, unit='meV')
        assert test_voigt.center.value ==0.0
        assert test_voigt.center.fixed is True

    def test_input_as_parameter(self):
        param_area=Parameter(name='area_param',value=2.0,unit='meV')
        param_center=Parameter(name='center_param',value=0.5,unit='meV')
        param_Gwidth=Parameter(name='Gwidth_param',value=0.6,unit='meV')
        param_Lwidth=Parameter(name='Lwidth_param',value=0.7,unit='meV')
        test_voigt=Voigt(name='TestVoigt', area=param_area, center=param_center, gaussian_width=param_Gwidth, lorentzian_width=param_Lwidth, unit='meV')
        assert test_voigt.area==param_area
        assert test_voigt.center==param_center
        assert test_voigt.Gwidth==param_Gwidth
        assert test_voigt.Lwidth==param_Lwidth

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="Gwidth must be greater than 0 for Voigt profile."):
            Voigt(name='TestVoigt', area=2.0, center=0.5, gaussian_width=-0.6, lorentzian_width=0.7, unit='meV')

        with pytest.raises(ValueError, match="Lwidth must be greater than 0 for Voigt profile."):
            Voigt(name='TestVoigt', area=2.0, center=0.5, gaussian_width=0.6, lorentzian_width=-0.7, unit='meV')

    def test_get_parameters(self, voigt: Voigt):
        params = voigt.get_parameters()
        assert len(params) == 4
        assert params[0].name == 'TestVoigt area'
        assert params[1].name == 'TestVoigt center'
        assert params[2].name == 'TestVoigt Gwidth'
        assert params[3].name == 'TestVoigt Lwidth'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, voigt: Voigt):   
        # WHEN
        x = np.linspace(voigt.center.value - 100 * voigt.Gwidth.value-300*voigt.Lwidth.value, voigt.center.value + 100 * voigt.Gwidth.value+300*voigt.Lwidth.value, 20000) #Voigts have very long tails
        y = voigt.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(voigt.area.value, rel=2e-3)

class TestDeltaFunction:

    @pytest.fixture
    def delta_function(self):
        return DeltaFunction(name='TestDeltaFunction', area=2.0, center=0.5, unit='meV')

    def test_initialization(self, delta_function: DeltaFunction):
        assert delta_function.name == 'TestDeltaFunction'
        assert delta_function.area.value == 2.0
        assert delta_function.center.value == 0.5
        assert delta_function.unit == 'meV'

    @pytest.mark.xfail(reason="DeltaFunction.evaluate is not implemented yet without resolution convolution")
    def test_evaluate(self, delta_function: DeltaFunction):
        x = np.array([0.0, 0.5, 1.0])
        expected = delta_function.evaluate(x)
        expected_result = np.zeros_like(x)
        # expected_result[x == 0.5] = 2.0
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)


    def test_center_is_fixed_if_set_to_None(self):
        test_delta=DeltaFunction(name='TestDeltaFunction', area=2.0, center=None, unit='meV')
        assert test_delta.center.value ==0.0
        assert test_delta.center.fixed is True

    def test_input_as_parameter(self):
        param_area=Parameter(name='area_param',value=2.0,unit='meV')
        param_center=Parameter(name='center_param',value=0.5,unit='meV')
        test_delta=DeltaFunction(name='TestDeltaFunction', area=param_area, center=param_center, unit='meV')
        assert test_delta.area==param_area
        assert test_delta.center==param_center

    def test_get_parameters(self, delta_function: DeltaFunction):
        params = delta_function.get_parameters()
        assert len(params) == 2
        assert params[0].name == 'TestDeltaFunction area'
        assert params[1].name == 'TestDeltaFunction center'
        assert all(isinstance(param, Parameter) for param in params)


class TestDampedHarmonicOscillator: 
    @pytest.fixture
    def dho(self):  
        return DampedHarmonicOscillator(name='TestDHO', area=2.0, center=1.5, width=0.3, unit='meV')
    
    def test_initialization(self, dho: DampedHarmonicOscillator):
        assert dho.name == 'TestDHO'
        assert dho.area.value == 2.0
        assert dho.center.value == 1.5
        assert dho.width.value == 0.3
        assert dho.unit == 'meV'

    def test_evaluate(self, dho: DampedHarmonicOscillator):
        x = np.array([0.0, 1.5, 3.0])
        expected = dho.evaluate(x)
        expected_result = 2*2.0 * (1.5**2) * (0.3) / np.pi / (((x**2 - 1.5**2) ** 2 + (2*0.3 * x) ** 2))
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)


    def test_evaluate_scipp_array(self, dho: DampedHarmonicOscillator):
        x = sc.array(dims=['x'], values=[0.0, 1.5, 3.0], unit='meV')
        expected = dho.evaluate(x)
        expected_result = 2*2.0 * (1.5**2) * (0.3) / np.pi / (((x.values**2 - 1.5**2) ** 2 + (2*0.3 * x.values) ** 2))
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, dho: DampedHarmonicOscillator):
        x = sc.array(dims=['x'], values=[0.0, 500.0, 1000.0], unit='microeV')
        expected = dho.evaluate(x)
        expected_result = 2*2.0*1e3 * ((1.5*1e3)**2) * (0.3*1e3) / np.pi / (((x.values**2 - (1.5*1e3)**2) ** 2 + (2*0.3*1e3 * x.values) ** 2))
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_input_as_parameter(self):
        param_area=Parameter(name='area_param',value=2.0,unit='meV')
        param_center=Parameter(name='center_param',value=0.5,unit='meV')
        param_width=Parameter(name='width_param',value=0.6,unit='meV')
        test_dho=DampedHarmonicOscillator(name='TestDHO', area=param_area, center=param_center, width=param_width, unit='meV')
        assert test_dho.area==param_area
        assert test_dho.center==param_center
        assert test_dho.width==param_width

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="The width of a DampedHarmonicOscillator must be greater than zero."):
            DampedHarmonicOscillator(name='TestDHO', area=2.0, center=0.5, width=-0.6, unit='meV')


    def test_get_parameters(self, dho: DampedHarmonicOscillator):
        params = dho.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestDHO area'
        assert params[1].name == 'TestDHO center'
        assert params[2].name == 'TestDHO width'
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, dho: DampedHarmonicOscillator):
        # WHEN
        x = np.linspace(-dho.center.value - 20 * dho.width.value, dho.center.value + 20 * dho.width.value, 5000)
        y = dho.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(dho.area.value, rel=2e-3)


class TestPolynomial:
    @pytest.fixture
    def polynomial(self):
        return Polynomial(name='TestPolynomial', coefficients=[1.0, -2.0, 3.0])

    def test_initialization(self, polynomial: Polynomial):
        assert polynomial.name == 'TestPolynomial'
        assert polynomial.coefficients[0].value==1.0
        assert polynomial.coefficients[1].value==-2.0
        assert polynomial.coefficients[2].value==3.0

    def test_evaluate(self, polynomial: Polynomial):
        x = np.array([0.0, 1.0, 2.0])
        expected = polynomial.evaluate(x)
        expected_result = 1.0 - 2.0 * x + 3.0 * x**2
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_get_parameters(self, polynomial: Polynomial):
        params = polynomial.get_parameters()
        assert len(params) == 3
        assert params[0].name == 'TestPolynomial_c0'
        assert params[1].name == 'TestPolynomial_c1'
        assert params[2].name == 'TestPolynomial_c2'
        assert all(isinstance(param, Parameter) for param in params)


    def test_convert_unit_raises_for_polynomial(self, polynomial):
        with pytest.raises(NotImplementedError, match="Unit conversion is not implemented for Polynomial components. The automatic unit converter does not like powers of units."):
            polynomial.convert_unit("eV")

@pytest.mark.skip(reason="UserDefinedComponent not implemented yet")
class TestUserDefinedComponent:
    def test_placeholder(self):
        pass