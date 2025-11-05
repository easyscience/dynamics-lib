import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.signal import fftconvolve
from scipy.special import voigt_profile

from easydynamics.sample_model import (
    DampedHarmonicOscillator,
    DeltaFunction,
    Gaussian,
    Lorentzian,
    SampleModel,
)
from easydynamics.utils import convolution
from easydynamics.utils.detailed_balance import (
    _detailed_balance_factor as detailed_balance_factor,
)

# Numerical convolutions are not very accurate
NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE = 1e-6
NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE = 1e-5


class TestConvolution:
    @pytest.fixture
    def sample_model(self):
        test_sample_model = SampleModel(name="TestSampleModel")
        test_sample_model.add_component(Gaussian(center=0.1, width=0.3, area=2.0))
        return test_sample_model

    @pytest.fixture
    def resolution_model(self):
        test_resolution_model = SampleModel(name="TestResolutionModel")
        test_resolution_model.add_component(Gaussian(center=0.2, width=0.4, area=3.0))
        return test_resolution_model

    @pytest.fixture
    def gaussian_component(self):
        return Gaussian(center=0.1, width=0.3, area=2.0)

    @pytest.fixture
    def other_gaussian_component(self):
        return Gaussian(name="other Gaussian", center=0.2, width=0.4, area=3.0)

    @pytest.fixture
    def lorentzian_component(self):
        return Lorentzian(center=0.1, width=0.3, area=2.0)

    @pytest.fixture
    def other_lorentzian_component(self):
        return Lorentzian(center=0.2, width=0.4, area=3.0)

    @pytest.fixture
    def energy(self):
        return np.linspace(-50, 50, 50001)

    # Test convolution of components
    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"],
    )
    @pytest.mark.parametrize(
        "method", ["analytical", "numerical"], ids=["analytical", "numerical"]
    )
    def test_components_gauss_gauss(
        self,
        energy,
        gaussian_component,
        other_gaussian_component,
        offset_obj,
        expected_shift,
        method,
    ):
        "Test convolution of Gaussian sample and Gaussian resolution components without SampleModel."
        "Test with different offset types and methods."
        # WHEN
        sample_gauss = gaussian_component
        resolution_gauss = other_gaussian_component

        # THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample_gauss,
            resolution_model=resolution_gauss,
            offset=offset_obj,
            method=method,
        )

        # EXPECT
        # Convolution of two Gaussians is another Gaussian with width = sqrt(w1^2 + w2^2)
        expected_width = np.sqrt(
            sample_gauss.width.value**2 + resolution_gauss.width.value**2
        )
        expected_area = sample_gauss.area.value * resolution_gauss.area.value
        expected_center = (
            sample_gauss.center.value + resolution_gauss.center.value + expected_shift
        )
        expected_result = (
            expected_area
            * np.exp(-0.5 * ((energy - expected_center) / expected_width) ** 2)
            / (np.sqrt(2 * np.pi) * expected_width)
        )

        np.testing.assert_allclose(calculated_convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"],
    )
    @pytest.mark.parametrize("method", ["auto", "numerical"], ids=["auto", "numerical"])
    def test_components_DHO_gauss(
        self, energy, gaussian_component, offset_obj, expected_shift, method
    ):
        "Test convolution of DHO sample and Gaussian resolution components without SampleModel."
        "Test with different offset types and methods."
        # WHEN
        sample_dho = DampedHarmonicOscillator(center=1.5, width=0.3, area=2)
        resolution_gauss = gaussian_component

        # THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample_dho,
            resolution_model=resolution_gauss,
            offset=offset_obj,
            method=method,
        )

        # EXPECT
        # no simple analytical form, so compute expected result via direct convolution
        sample_values = sample_dho.evaluate(energy - expected_shift)
        resolution_values = resolution_gauss.evaluate(energy)
        expected_result = fftconvolve(sample_values, resolution_values, mode="same")
        expected_result *= energy[1] - energy[0]  # normalize

        np.testing.assert_allclose(calculated_convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"],
    )
    @pytest.mark.parametrize(
        "method", ["analytical", "numerical"], ids=["analytical", "numerical"]
    )
    def test_components_lorentzian_lorentzian(
        self,
        energy,
        lorentzian_component,
        other_lorentzian_component,
        offset_obj,
        expected_shift,
        method,
    ):
        "Test convolution of Lorentzian sample and Lorentzian resolution components without SampleModel."
        "Test with different offset types and methods."
        # WHEN
        sample_lorentzian = lorentzian_component
        resolution_lorentzian = other_lorentzian_component

        # THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample_lorentzian,
            resolution_model=resolution_lorentzian,
            offset=offset_obj,
            method=method,
            upsample_factor=5,
        )

        # EXPECT
        # Convolution of two Lorentzians is another Lorentzian with width = w1 + w2
        expected_width = (
            sample_lorentzian.width.value + resolution_lorentzian.width.value
        )
        expected_area = sample_lorentzian.area.value * resolution_lorentzian.area.value
        expected_center = (
            sample_lorentzian.center.value
            + resolution_lorentzian.center.value
            + expected_shift
        )
        expected_result = (
            expected_area
            * expected_width
            / np.pi
            / ((energy - expected_center) ** 2 + expected_width**2)
        )

        np.testing.assert_allclose(
            calculated_convolution,
            expected_result,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
        )

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"],
    )
    @pytest.mark.parametrize(
        "method", ["analytical", "numerical"], ids=["analytical", "numerical"]
    )
    @pytest.mark.parametrize(
        "sample_is_gauss",
        [True, False],
        ids=["gauss_sample__lorentz_resolution", "lorentz_sample__gauss_resolution"],
    )
    def test_components_gauss_lorentzian(
        self,
        energy,
        gaussian_component,
        lorentzian_component,
        offset_obj,
        expected_shift,
        method,
        sample_is_gauss,
    ):
        "Test convolution of Gaussian and Lorentzian components without SampleModel."
        "Test with different offset types and methods."
        # WHEN
        if sample_is_gauss:
            sample = gaussian_component
            resolution = lorentzian_component
        else:
            sample = lorentzian_component
            resolution = gaussian_component

        # THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample,
            resolution_model=resolution,
            offset=offset_obj,
            method=method,
            upsample_factor=5,
        )

        # EXPECT
        expected_center = sample.center.value + resolution.center.value + expected_shift
        expected_area = sample.area.value * resolution.area.value

        gaussian_width = (
            sample.width.value if sample_is_gauss else resolution.width.value
        )
        lorentzian_width = (
            resolution.width.value if sample_is_gauss else sample.width.value
        )

        expected_result = expected_area * voigt_profile(
            energy - expected_center,
            gaussian_width,
            lorentzian_width,
        )

        np.testing.assert_allclose(
            calculated_convolution,
            expected_result,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
        )

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"],
    )
    @pytest.mark.parametrize(
        "method", ["analytical", "numerical"], ids=["analytical", "numerical"]
    )
    @pytest.mark.parametrize(
        "sample_is_gauss",
        [True, False],
        ids=["gauss_sample__delta_resolution", "delta_sample__gauss_resolution"],
    )
    def test_components_delta_gauss(
        self,
        energy,
        gaussian_component,
        offset_obj,
        expected_shift,
        method,
        sample_is_gauss,
    ):
        "Test convolution of Delta function sample and Gaussian resolution components without SampleModel."
        "Test with different offset types and methods."
        # WHEN
        if sample_is_gauss:
            sample = gaussian_component
            resolution = DeltaFunction(name="Delta", center=0.1, area=2)
        else:
            sample = DeltaFunction(name="Delta", center=0.1, area=2)
            resolution = gaussian_component

        # THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample,
            resolution_model=resolution,
            offset=offset_obj,
            method=method,
        )

        # EXPECT
        expected_center = sample.center.value + resolution.center.value + expected_shift
        expected_area = sample.area.value * resolution.area.value
        width = sample.width.value if sample_is_gauss else resolution.width.value
        expected_result = (
            expected_area
            * np.exp(-0.5 * ((energy - expected_center) / width) ** 2)
            / (np.sqrt(2 * np.pi) * width)
        )

        np.testing.assert_allclose(calculated_convolution, expected_result, atol=1e-10)

    # Test convolution of SampleModel
    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"],
    )
    @pytest.mark.parametrize(
        "method", ["analytical", "numerical"], ids=["analytical", "numerical"]
    )
    def test_model_gauss_gauss_resolution_gauss(
        self,
        energy,
        sample_model,
        resolution_model,
        offset_obj,
        expected_shift,
        method,
    ):
        "Test convolution of Gaussian sample components in SampleModel and Gaussian resolution components in SampleModel."
        "Test with different offset types and methods."

        # WHEN
        sample_G2 = Gaussian(name="another Gaussian", center=0.3, width=0.5, area=4)
        sample_model.add_component(sample_G2)

        # THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset=offset_obj,
            method=method,
        )

        # EXPECT
        sample_G1 = sample_model["Gaussian"]
        resolution_G1 = resolution_model["Gaussian"]
        expected_width1 = np.sqrt(
            sample_G1.width.value**2 + resolution_G1.width.value**2
        )
        expected_width2 = np.sqrt(
            sample_G2.width.value**2 + resolution_G1.width.value**2
        )
        expected_area1 = sample_G1.area.value * resolution_G1.area.value
        expected_area2 = sample_G2.area.value * resolution_G1.area.value
        expected_center1 = (
            sample_G1.center.value + resolution_G1.center.value + expected_shift
        )
        expected_center2 = (
            sample_G2.center.value + resolution_G1.center.value + expected_shift
        )

        expected_result = expected_area1 * np.exp(
            -0.5 * ((energy - expected_center1) / expected_width1) ** 2
        ) / (np.sqrt(2 * np.pi) * expected_width1) + expected_area2 * np.exp(
            -0.5 * ((energy - expected_center2) / expected_width2) ** 2
        ) / (np.sqrt(2 * np.pi) * expected_width2)
        np.testing.assert_allclose(calculated_convolution, expected_result, atol=1e-10)

    @pytest.mark.parametrize(
        "offset_obj, expected_shift",
        [
            (None, 0.0),
            (0.4, 0.4),
            (Parameter("off", 0.4), 0.4),
        ],
        ids=["none", "float", "parameter"],
    )
    @pytest.mark.parametrize(
        "method", ["analytical", "numerical"], ids=["analytical", "numerical"]
    )
    def test_model_lorentzian_delta_resolution_gauss(
        self,
        energy,
        method,
        lorentzian_component,
        resolution_model,
        offset_obj,
        expected_shift,
    ):
        "Test convolution of Lorentzian and Delta function sample components in SampleModel and Gaussian resolution components in SampleModel."
        " Result is a combination of Voigt profile and Gaussian."
        # WHEN

        sample = SampleModel(name="SampleModel")
        sample.add_component(lorentzian_component)
        sample_delta = DeltaFunction(center=0.5, area=4, name="SampleDelta")
        sample.add_component(sample_delta)

        # THEN
        energy = np.linspace(-10, 10, 20001)
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample,
            resolution_model=resolution_model,
            offset=offset_obj,
            method=method,
            upsample_factor=5,
        )

        # EXPECT: Combine Gaussian, Lorentzian, and Delta functions contributions
        #
        gaussian_component = resolution_model["Gaussian"]

        expected_voigt_area = (
            lorentzian_component.area.value * gaussian_component.area.value
        )
        expected_voigt_center = (
            lorentzian_component.center.value
            + gaussian_component.center.value
            + expected_shift
        )
        expected_voigt = expected_voigt_area * voigt_profile(
            energy - expected_voigt_center,
            gaussian_component.width.value,
            lorentzian_component.width.value,
        )
        expected_gauss_area = sample_delta.area.value * gaussian_component.area.value
        expected_gauss_center = (
            sample_delta.center.value + gaussian_component.center.value + expected_shift
        )
        expected_gauss_width = gaussian_component.width.value
        expected_gauss = (
            expected_gauss_area
            * np.exp(
                -0.5 * ((energy - (expected_gauss_center)) / expected_gauss_width) ** 2
            )
            / (np.sqrt(2 * np.pi) * expected_gauss_width)
        )
        expected_result = expected_voigt + expected_gauss
        np.testing.assert_allclose(
            calculated_convolution,
            expected_result,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
        )

    def test_numerical_convolve_with_temperature(
        self, energy, sample_model, resolution_model
    ):
        "Test numerical convolution with detailed balance correction."
        # WHEN
        temperature = 300.0  # Kelvin

        # THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="numerical",
            upsample_factor=5,
            temperature=temperature,
        )

        sample_with_db = sample_model.evaluate(energy) * detailed_balance_factor(
            energy=energy, temperature=temperature
        )
        resolution = resolution_model.evaluate(energy)

        expected_convolution = fftconvolve(sample_with_db, resolution, mode="same")
        expected_convolution *= [energy[1] - energy[0]]  # normalize

        np.testing.assert_allclose(
            calculated_convolution,
            expected_convolution,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
        )

    @pytest.mark.parametrize(
        "x",
        [
            np.linspace(-10, 10, 5001),  # Odd length
            np.linspace(-10, 10, 5000),  # Even length
        ],
        ids=["odd_length", "even_length"],
    )
    def test_numerical_convolve_x_length_even_and_odd(
        self, x, sample_model, resolution_model
    ):
        "Test numerical convolution with both even and odd length x arrays. With even length the FFT shifts the signal by half a bin."

        # WHEN THEN
        calculated_convolution = convolution(
            energy=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="numerical",
            upsample_factor=0,
        )

        # EXPECT
        expected_convolution = convolution(
            energy=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="analytical",
            upsample_factor=0,
        )

        np.testing.assert_allclose(
            calculated_convolution, expected_convolution, atol=1e-10
        )

    @pytest.mark.parametrize(
        "upsample_factor",
        [0, 2, 5, 10],
        ids=["no_upsample", "upsample_2", "upsample_5", "upsample_10"],
    )
    def test_numerical_convolve_upsample_factor(
        self, energy, upsample_factor, sample_model, resolution_model
    ):
        "Test numerical convolution with different upsample factors."
        # WHEN THEN
        calculated_convolution = convolution(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="numerical",
            upsample_factor=upsample_factor,
        )

        # EXPECT
        expected_convolution = convolution(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="analytical",
            upsample_factor=0,
        )

        np.testing.assert_allclose(
            calculated_convolution,
            expected_convolution,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
        )

    @pytest.mark.parametrize(
        "x",
        [np.linspace(-5, 15, 20000), np.linspace(5, 15, 20000)],
        ids=["asymmetric", "only_positive"],
    )
    @pytest.mark.parametrize(
        "upsample_factor", [0, 2, 5], ids=["no_upsample", "upsample_2", "upsample_5"]
    )
    def test_numerical_convolve_x_not_symmetric(
        self, x, upsample_factor, resolution_model
    ):
        "Test numerical convolution with asymmetric and only positive x arrays."
        # WHEN
        sample_model = SampleModel(name="SampleModel")
        sample_model.add_component(Gaussian(center=9, width=0.3, area=2))

        # THEN
        calculated_convolution = convolution(
            energy=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="numerical",
            upsample_factor=upsample_factor,
        )

        # EXPECT
        expected_convolution = convolution(
            energy=x,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="analytical",
        )

        np.testing.assert_allclose(
            calculated_convolution,
            expected_convolution,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
        )

    def test_numerical_convolve_x_not_uniform(self, sample_model, resolution_model):
        "Test numerical convolution with non-uniform x arrays."
        # WHEN
        x_1 = np.linspace(-2, 0, 1000)
        x_2 = np.linspace(0.001, 2, 2000)
        x_non_uniform = np.concatenate([x_1, x_2])

        # THEN
        calculated_convolution = convolution(
            energy=x_non_uniform,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="numerical",
            upsample_factor=5,
        )

        # EXPECT
        expected_convolution = convolution(
            energy=x_non_uniform,
            sample_model=sample_model,
            resolution_model=resolution_model,
            method="analytical",
        )

        np.testing.assert_allclose(
            calculated_convolution,
            expected_convolution,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
        )

    # Test error handling
    def test_analytical_convolution_fails_with_detailed_balance(
        self, energy, sample_model, resolution_model
    ):
        # WHEN
        temperature = 300.0
        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match="Analytical convolution is not supported with detailed balance.",
        ):
            convolution(
                energy=energy,
                sample_model=sample_model,
                resolution_model=resolution_model,
                method="analytical",
                temperature=temperature,
            )

    def test_convolution_only_accepts_auto_analytical_and_numerical_methods(
        self, energy, sample_model, resolution_model
    ):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError,
            match="Unknown convolution method: unknown_method. Choose from 'auto', 'analytical', or 'numerical'.",
        ):
            convolution(
                energy=energy,
                sample_model=sample_model,
                resolution_model=resolution_model,
                method="unknown_method",
            )

    def test_energy_must_be_1d_finite_array(self, sample_model, resolution_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match="`energy` must be a 1D finite array."):
            convolution(
                energy=np.array([[1, 2], [3, 4]]),
                sample_model=sample_model,
                resolution_model=resolution_model,
            )

        with pytest.raises(ValueError, match="`energy` must be a 1D finite array."):
            convolution(
                energy=np.array([1, 2, np.nan]),
                sample_model=sample_model,
                resolution_model=resolution_model,
            )

        with pytest.raises(ValueError, match="`energy` must be a 1D finite array."):
            convolution(
                energy=np.array([1, 2, np.inf]),
                sample_model=sample_model,
                resolution_model=resolution_model,
            )

    def test_numerical_convolve_requires_uniform_grid_if_no_upsample(
        self, sample_model, resolution_model
    ):
        # WHEN
        x = np.array([0, 1, 2, 4, 5])  # Non-uniform grid

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match="Input array `energy` must be uniformly spaced if upsample_factor = 0.",
        ):
            convolution(
                energy=x,
                sample_model=sample_model,
                resolution_model=resolution_model,
                method="numerical",
                upsample_factor=0,
            )

    def test_sample_model_must_have_components(self, resolution_model):
        # WHEN
        sample_model = SampleModel(name="SampleModel")

        # THEN EXPECT
        with pytest.raises(
            ValueError, match="SampleModel must have at least one component."
        ):
            convolution(
                energy=np.array([0, 1, 2]),
                sample_model=sample_model,
                resolution_model=resolution_model,
            )

    def test_resolution_model_must_have_components(self, sample_model):
        # WHEN
        resolution_model = SampleModel(name="ResolutionModel")

        # THEN EXPECT
        with pytest.raises(
            ValueError, match="ResolutionModel must have at least one component."
        ):
            convolution(
                energy=np.array([0, 1, 2]),
                sample_model=sample_model,
                resolution_model=resolution_model,
            )

    def test_numerical_convolution_wide_sample_peak_gives_warning(
        self, resolution_model
    ):
        # WHEN
        x = np.linspace(-2, 2, 20001)

        sample_gauss = Gaussian(center=0.1, width=1.9, area=2, name="SampleGauss")
        sample = SampleModel(name="SampleModel")
        sample.add_component(sample_gauss)

        # #THEN EXPECT
        with pytest.warns(
            UserWarning,
            match=r"The width of the sample model component ",
        ):
            convolution(
                energy=x,
                sample_model=sample,
                resolution_model=resolution_model,
                method="numerical",
                upsample_factor=0,
            )

    def test_numerical_convolution_wide_resolution_peak_gives_warning(
        self, sample_model
    ):
        # WHEN
        x = np.linspace(-2, 2, 20001)

        resolution_gauss = Gaussian(
            center=0.3, width=1.9, area=4, name="ResolutionGauss"
        )

        resolution = SampleModel(name="ResolutionModel")
        resolution.add_component(resolution_gauss)

        # #THEN EXPECT
        with pytest.warns(
            UserWarning,
            match=r"The width of the resolution model component 'ResolutionGauss' \(1.9\) is large",
        ):
            convolution(
                energy=x,
                sample_model=sample_model,
                resolution_model=resolution,
                method="numerical",
                upsample_factor=0,
            )

    def test_numerical_convolution_narrow_sample_peak_gives_warning(
        self, resolution_model
    ):
        # WHEN
        x = np.linspace(-2, 2, 201)

        sample_gauss1 = Gaussian(center=0.1, width=1e-3, area=2, name="SampleGauss")

        sample = SampleModel(name="SampleModel")
        sample.add_component(sample_gauss1)

        # #THEN EXPECT
        with pytest.warns(
            UserWarning,
            match=r"The width of the sample model component 'SampleGauss' \(0.001\) is small",
        ):
            convolution(
                energy=x,
                sample_model=sample,
                resolution_model=resolution_model,
                method="numerical",
                upsample_factor=0,
            )

    def test_numerical_convolution_narrow_resolution_peak_gives_warning(
        self, sample_model
    ):
        # WHEN
        x = np.linspace(-2, 2, 201)

        resolution_gauss = Gaussian(
            center=0.3, width=1e-3, area=4, name="ResolutionGauss"
        )

        resolution = SampleModel(name="ResolutionModel")
        resolution.add_component(resolution_gauss)

        # #THEN EXPECT
        with pytest.warns(
            UserWarning,
            match=r"The width of the resolution model component 'ResolutionGauss' \(0.001\) is small",
        ):
            convolution(
                energy=x,
                sample_model=sample_model,
                resolution_model=resolution,
                method="numerical",
                upsample_factor=0,
            )
