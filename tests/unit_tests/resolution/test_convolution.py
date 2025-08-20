import pytest
import numpy as np

from scipy.special import voigt_profile

from easyscience.variable import Parameter
from easydynamics.sample import SampleModel, GaussianComponent, LorentzianComponent, DeltaFunctionComponent


from easydynamics.resolution import ResolutionHandler

 
# Numerical convolutions are not very accurate
NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE = 1e-6
NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE = 1e-5

class TestConvolution:
    @pytest.fixture
    def sample_model(self):
        test_sample_model = SampleModel(name="TestSampleModel")
        test_sample_model.add_component(LorentzianComponent(center=0.1, width=0.2, area=2.0))
        return test_sample_model

    @pytest.fixture
    def resolution_model(self):
        test_resolution_model = SampleModel(name="TestResolutionModel")
        test_resolution_model.add_component(GaussianComponent(center=0.2, width=0.3, area=3.0))
        return test_resolution_model
       
    @pytest.fixture
    def x(self):
        return np.linspace(-50, 50, 50001)
    


# Test convolution of components
    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_gauss_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss = GaussianComponent(center=0.1, width=0.3, area=2)
        resolution_gauss = GaussianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample_gauss, resolution_model=resolution_gauss, offset=offset_obj, method=method)

        #EXPECT
        expected_width = np.sqrt(sample_gauss.width.value**2 + resolution_gauss.width.value**2)
        expected_area = sample_gauss.area.value * resolution_gauss.area.value
        expected_center = sample_gauss.center.value + resolution_gauss.center.value + expected_shift
        expected_result = expected_area * np.exp(-0.5 * ((x - expected_center) / expected_width)**2) / (np.sqrt(2 * np.pi) * expected_width)

        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_lorentzian_lorentzian(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_lorentzian = LorentzianComponent(center=0.1, width=0.3, area=2)
        resolution_lorentzian = LorentzianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_lorentzian, resolution_model=resolution_lorentzian,offset=offset_obj,method=method, upsample_factor=5)

        #EXPECT
        expected_width = sample_lorentzian.width.value + resolution_lorentzian.width.value
        expected_area = sample_lorentzian.area.value * resolution_lorentzian.area.value
        expected_center = sample_lorentzian.center.value + resolution_lorentzian.center.value + expected_shift
        expected_result = expected_area * expected_width / np.pi / ((x - expected_center)**2 + expected_width**2)

        np.testing.assert_allclose(convolution, expected_result, atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE, rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_gauss_lorentzian(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss = GaussianComponent(center=0.1, width=0.3, area=2)
        resolution_lorentzian = LorentzianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample_gauss, resolution_model=resolution_lorentzian, offset=offset_obj, method=method, upsample_factor=5)

        #EXPECT
        expected_center = sample_gauss.center.value + resolution_lorentzian.center.value + expected_shift
        expected_area = sample_gauss.area.value * resolution_lorentzian.area.value
        expected_result = expected_area * voigt_profile(
            x - expected_center,
            sample_gauss.width.value,
            resolution_lorentzian.width.value
        )

        np.testing.assert_allclose(convolution, expected_result, atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE, rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_lorentzian_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        resolution_gauss = GaussianComponent(center=0.1, width=0.3, area=2)
        sample_lorentzian = LorentzianComponent(center=0.2, width=0.4, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_lorentzian, resolution_model=resolution_gauss,offset=offset_obj,method=method,upsample_factor=5)

        #EXPECT
        expected_center = sample_lorentzian.center.value + resolution_gauss.center.value + expected_shift
        expected_area = sample_lorentzian.area.value * resolution_gauss.area.value
        expected_result = expected_area * voigt_profile(
            x - expected_center,
            resolution_gauss.width.value,
            sample_lorentzian.width.value
        )

        np.testing.assert_allclose(convolution, expected_result, atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE, rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_delta_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_delta = DeltaFunctionComponent(name="Delta", center=0.1, area=2)
        resolution_gauss = GaussianComponent(center=0.2, width=0.3, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_delta, resolution_model=resolution_gauss,offset = offset_obj, method=method)

        #EXPECT
        expected_center = sample_delta.center.value + resolution_gauss.center.value + expected_shift
        expected_area = sample_delta.area.value * resolution_gauss.area.value
        expected_result = expected_area * np.exp(-0.5 * ((x - expected_center) / resolution_gauss.width.value)**2) / (np.sqrt(2 * np.pi) * resolution_gauss.width.value)

        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_components_gauss_delta(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss = GaussianComponent(center=0.1, width=0.2, area=2)
        resolution_delta = DeltaFunctionComponent(name="Delta", center=0.2, area=3)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x,sample_model=sample_gauss, resolution_model=resolution_delta,offset = offset_obj, method=method)

        #EXPECT
        expected_center = sample_gauss.center.value + resolution_delta.center.value + expected_shift
        expected_area = sample_gauss.area.value * resolution_delta.area.value
        expected_result = expected_area * np.exp(-0.5 * ((x - expected_center) / sample_gauss.width.value)**2) / (np.sqrt(2 * np.pi) * sample_gauss.width.value)

        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

# Test convolution of SampleModel
    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"]
    )
    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_model_gauss_gauss_resolution_gauss(self, x, offset_obj, expected_shift, method):
        #WHEN
        sample_gauss1 = GaussianComponent(center=0.1, width=0.3, area=2,name="SampleGauss1")
        sample_gauss2 = GaussianComponent(center=0.2, width=0.4, area=3,name="SampleGauss2")
        resolution_gauss = GaussianComponent(center=0.3, width=0.5, area=4)

        sample= SampleModel(name="SampleModel")
        sample.add_component(sample_gauss1)
        sample.add_component(sample_gauss2)

        resolution = SampleModel(name="ResolutionModel")
        resolution.add_component(resolution_gauss)

        resolution_handler = ResolutionHandler()

        # THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample, resolution_model=resolution, offset=offset_obj, method=method)

        #EXPECT
        expected_width1 = np.sqrt(sample_gauss1.width.value**2 + resolution_gauss.width.value**2)
        expected_width2 = np.sqrt(sample_gauss2.width.value**2 + resolution_gauss.width.value**2)
        expected_area1 = sample_gauss1.area.value * resolution_gauss.area.value
        expected_area2 = sample_gauss2.area.value * resolution_gauss.area.value
        expected_center1 = sample_gauss1.center.value + resolution_gauss.center.value + expected_shift
        expected_center2 = sample_gauss2.center.value + resolution_gauss.center.value + expected_shift

        expected_result = (
            expected_area1 * np.exp(-0.5 * ((x - expected_center1) / expected_width1)**2) / (np.sqrt(2 * np.pi) * expected_width1) +
            expected_area2 * np.exp(-0.5 * ((x - expected_center2) / expected_width2)**2) / (np.sqrt(2 * np.pi) * expected_width2)
        )
        np.testing.assert_allclose(convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "method",
        ["analytical", "numerical"],
        ids=["analytical", "numerical"]
    )
    def test_model_lorentzian_delta_resolution_gauss(self, x, method):
        #WHEN
        sample_lorentzian = LorentzianComponent(center=0.1, width=0.3, area=2, name="SampleLorentzian")
        sample_delta = DeltaFunctionComponent(center=0.5, area=4, name="SampleDelta")
        resolution_gauss = GaussianComponent(center=-0.3, width=0.4, area=3, name="ResolutionGauss")
        sample = SampleModel(name="SampleModel")
        sample.add_component(sample_lorentzian)
        sample.add_component(sample_delta)
        resolution = SampleModel(name="ResolutionModel")
        resolution.add_component(resolution_gauss)
        resolution_handler = ResolutionHandler()
        # THEN
        x = np.linspace(-10, 10, 20001)
        convolution = resolution_handler.convolve(x=x, sample_model=sample, resolution_model=resolution, method=method, upsample_factor=5)

        #EXPECT: Combine Gaussian, Lorentzian, and Delta functions contributions
        expected_voigt = 2*3*voigt_profile(x-(0.1-0.3),0.4,0.3)
        expected_gauss_center = -0.3+0.5
        expected_gauss = 3 *4* np.exp(-0.5 * ((x - (expected_gauss_center)) / 0.4)**2) / (np.sqrt(2 * np.pi) * 0.4)
        expected_result = expected_voigt + expected_gauss
        np.testing.assert_allclose(convolution, expected_result, atol = NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE, rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE)

# Test numerical convolution
    @pytest.mark.parametrize(
        "x",
        [
            np.linspace(-10, 10, 5001),  # Odd length
            np.linspace(-10, 10, 5000)   # Even length
        ],
        ids=["odd_length", "even_length"]
    )
    def test_numerical_convolve_x_length_even_and_odd(self, x):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))

        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))

        resolution_handler = ResolutionHandler()

        #THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='numerical', upsample_factor=0)

        #EXPECT
        expected_convolution=resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='analytical', upsample_factor=0)

        np.testing.assert_allclose(convolution, expected_convolution, atol=1e-10)

    @pytest.mark.parametrize(
        "upsample_factor",
        [0, 2, 5, 10],
        ids=["no_upsample", "upsample_2", "upsample_5", "upsample_10"]
    )
    def test_numerical_convolve_upsample_factor(self, x, upsample_factor):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))

        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))

        resolution_handler = ResolutionHandler()

        #THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='numerical', upsample_factor=upsample_factor)

        #EXPECT
        expected_convolution = resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='analytical', upsample_factor=0)

        np.testing.assert_allclose(convolution, expected_convolution, atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE, rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE)


    @pytest.mark.parametrize(
        "x",
        [
            np.linspace(-5, 15, 20000),  
            np.linspace(5, 15, 20000)   
        ],
        ids=["asymmetric", "only_positive"]
    )
    @pytest.mark.parametrize(
        "upsample_factor",
        [0, 2, 5],
        ids=["no_upsample", "upsample_2", "upsample_5"]
    )
    def test_numerical_convolve_x_not_symmetric(self, x,upsample_factor):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=9, width=0.3, area=2))

        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))

        resolution_handler = ResolutionHandler()

        #THEN
        convolution = resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='numerical', upsample_factor=upsample_factor)

        #EXPECT
        expected_convolution = resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='analytical', upsample_factor=0)

        np.testing.assert_allclose(convolution, expected_convolution, atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE, rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE)

    def test_numerical_convolve_x_not_uniform(self):

        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))
        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))
        resolution_handler = ResolutionHandler()

        x_1=np.linspace(-2,0,1000)
        x_2=np.linspace(0.001,2,2000)
        x_non_uniform = np.concatenate([x_1, x_2])
        #THEN
        convolution = resolution_handler.convolve(x=x_non_uniform, sample_model=sample_model, resolution_model=resolution_model, method='numerical', upsample_factor=5)

        #EXPECT
        expected_convolution = resolution_handler.convolve(x=x_non_uniform, sample_model=sample_model, resolution_model=resolution_model, method='analytical')

        np.testing.assert_allclose(convolution, expected_convolution, atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE, rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE)

# Test error handling

    def test_analytical_convolution_fails_with_detailed_balance(self, x):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))
        sample_model.temperature=300

        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))

        resolution_handler = ResolutionHandler()

        #THEN
        with pytest.raises(ValueError, match="Analytical convolution is not supported with detailed balance."):
            resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='analytical')

    def test_convolution_only_accepts_analytical_and_numerical_methods(self, x):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))

        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))

        resolution_handler = ResolutionHandler()

        #THEN
        with pytest.raises(ValueError, match="Unknown convolution method: unknown_method. Choose from 'analytical', or 'numerical'."):
            resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='unknown_method')

    def test_x_must_be_1d_finite_array(self):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))

        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))

        resolution_handler = ResolutionHandler()

        #THEN
        with pytest.raises(ValueError, match="`x` must be a 1D finite array."):
            resolution_handler.convolve(x=np.array([[1, 2], [3, 4]]), sample_model=sample_model, resolution_model=resolution_model)

        with pytest.raises(ValueError, match="`x` must be a 1D finite array."):
            resolution_handler.convolve(x=np.array([1, 2, np.nan]), sample_model=sample_model, resolution_model=resolution_model)

        with pytest.raises(ValueError, match="`x` must be a 1D finite array."):
            resolution_handler.convolve(x=np.array([1, 2, np.inf]), sample_model=sample_model, resolution_model=resolution_model)

    def test_numerical_convolve_requires_uniform_grid_if_no_upsample(self):
        #WHEN
        x = np.array([0, 1, 2, 4, 5])  # Non-uniform grid
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))
        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.4, area=3))
        resolution_handler = ResolutionHandler()
        #THEN
        with pytest.raises(ValueError, match="Input array `x` must be uniformly spaced if upsample_factor = 0."):
            resolution_handler.convolve(x=x, sample_model=sample_model, resolution_model=resolution_model, method='numerical', upsample_factor=0)

    def test_sample_model_must_have_components(self):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(GaussianComponent(center=0.2, width=0.3, area=3))
        resolution_handler = ResolutionHandler()
        #THEN
        with pytest.raises(ValueError, match="SampleModel must have at least one component."):
            resolution_handler.convolve(x=np.array([0, 1, 2]), sample_model=sample_model, resolution_model=resolution_model)

    def test_resolution_model_must_have_components(self):
        #WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(GaussianComponent(center=0.1, width=0.3, area=2))
        resolution_model = SampleModel(name="ResolutionModel")
        resolution_handler = ResolutionHandler()
        #THEN
        with pytest.raises(ValueError, match="ResolutionModel must have at least one component."):
            resolution_handler.convolve(x=np.array([0, 1, 2]), sample_model=sample_model, resolution_model=resolution_model)
