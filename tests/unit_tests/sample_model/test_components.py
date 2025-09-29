import pytest

import numpy as np
import scipp as sc

from scipy.integrate import simpson

from easydynamics.sample_model import (
    Gaussian,
    Lorentzian,
    Voigt,
    DeltaFunction,
    DampedHarmonicOscillator,
    Polynomial,
)
from easydynamics.sample_model.components.model_component import ModelComponent

from easyscience.variable import Parameter

from scipy.special import voigt_profile


class TestModelComponent:
    class DummyComponent(ModelComponent):
        def __init__(self):
            super().__init__(name="Dummy")
            self.area = Parameter(name="area", value=1.0, unit="meV")
            self.center = Parameter(name="center", value=2.0, unit="meV", fixed=True)
            self.width = Parameter(name="width", value=3.0, unit="meV", fixed=True)
            self.second_area = Parameter(name="second_area", value=4.0, unit="meV")

        def get_parameters(self):
            return [self.area, self.center, self.width, self.second_area]

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

    def test_free_all_parameters_sets_all_to_unfixed(self, dummy):
        # WHEN
        dummy.free_all_parameters()

        # THEN EXPECT
        assert all(not p.fixed for p in dummy.get_parameters())

    def test_get_parameter_exact_match(self, dummy):
        # WHEN
        param = dummy.get_parameter("width")

        # THEN EXPECT
        assert param is dummy.width

    def test_get_parameter_partial_match(self, dummy):
        # WHEN
        param = dummy.get_parameter("wid")

        # THEN EXPECT
        assert param is dummy.width

    def test_get_parameter_no_match_raises(self, dummy):
        # WHEN / THEN EXPECT
        with pytest.raises(
            ValueError,
            match="not found",
        ):
            dummy.get_parameter("nonexistent")

    def test_get_parameter_ambiguous_match_raises(self, dummy):
        # WHEN / THEN EXPECT
        with pytest.raises(ValueError, match="Ambiguous parameter name"):
            dummy.get_parameter("are")

    def test_set_parameter_value(self, dummy):
        # WHEN
        dummy.set_parameter_value("width", 10.0)

        # THEN EXPECT
        assert dummy.width.value == 10.0

    def test_set_parameter_bounds_min(self, dummy):
        # WHEN
        dummy.set_parameter_bounds("width", min=1.0)

        # THEN EXPECT
        assert dummy.width.min == 1.0
        assert dummy.width.max == np.inf

    def test_set_parameter_bounds_max(self, dummy):
        # WHEN
        dummy.set_parameter_bounds("width", max=5.0)

        # THEN EXPECT
        assert dummy.width.min == -np.inf
        assert dummy.width.max == 5.0

    def test_set_parameter_bounds_min_max(self, dummy):
        # WHEN
        dummy.set_parameter_bounds("width", min=1.0, max=5.0)

        # THEN EXPECT
        assert dummy.width.min == 1.0
        assert dummy.width.max == 5.0

    def test_set_parameter_bounds_with_unit_conversion(self, dummy):
        # WHEN
        dummy.set_parameter_bounds("width", min=1000.0, max=5000.0, unit="microeV")

        # THEN EXPECT
        assert dummy.width.min == 1000
        assert dummy.width.max == 5000
        assert dummy.width.unit == "µeV"

    def test_fix_parameter(self, dummy):
        # WHEN
        dummy.fix_parameter("width")

        # THEN EXPECT
        assert dummy.width.fixed is True

    def test_free_parameter(self, dummy):
        # WHEN
        dummy.fix_parameter("width")
        # THEN
        dummy.free_parameter("width")
        # EXPECT
        assert dummy.width.fixed is False

    def test_repr(self, dummy):
        repr_str = repr(dummy)
        assert "DummyComponent" in repr_str


class TestGaussian:
    @pytest.fixture
    def gaussian(self):
        return Gaussian(
            name="TestGaussian", area=2.0, center=0.5, width=0.6, unit="meV"
        )

    def test_initialization(self, gaussian: Gaussian):
        assert gaussian.name == "TestGaussian"
        assert gaussian.area.value == 2.0
        assert gaussian.center.value == 0.5
        assert gaussian.width.value == 0.6
        assert gaussian.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number or a Parameter"):
            Gaussian(
                name="TestGaussian", area="invalid", center=0.5, width=0.6, unit="meV"
            )
        with pytest.raises(
            TypeError, match="center must be None, a number or a Parameter"
        ):
            Gaussian(
                name="TestGaussian", area=2.0, center="invalid", width=0.6, unit="meV"
            )
        with pytest.raises(TypeError, match="width must be a number or a Parameter"):
            Gaussian(
                name="TestGaussian", area=2.0, center=0.5, width="invalid", unit="meV"
            )
        with pytest.raises(TypeError, match="unit must be a string"):
            Gaussian(name="TestGaussian", area=2.0, center=0.5, width=0.6, unit=123)

    def test_negative_width_raises(self):
        with pytest.raises(
            ValueError, match="The width of a Gaussian must be greater than zero."
        ):
            Gaussian(name="TestGaussian", area=2.0, center=0.5, width=-0.6, unit="meV")

    def test_negative_width_raises_in_evaluate(self):
        test_gaussian = Gaussian(
            name="TestGaussian", area=2.0, center=0.5, width=0.6, unit="meV"
        )
        test_gaussian.width.value = -0.6
        with pytest.raises(
            ValueError, match="The width of a Gaussian must be greater than zero."
        ):
            test_gaussian.evaluate(np.array([0.0, 0.5, 1.0]))

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            Gaussian(name="TestGaussian", area=-2.0, center=0.5, width=0.6, unit="meV")

    def test_negative_area_warns_in_evaluate(self):
        test_gaussian = Gaussian(
            name="TestGaussian", area=2.0, center=0.5, width=0.6, unit="meV"
        )
        test_gaussian.area.value = -2.0
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            test_gaussian.evaluate(np.array([0.0, 0.5, 1.0]))

    def test_evaluate(self, gaussian: Gaussian):
        x = np.array([0.0, 0.5, 1.0])
        expected = gaussian.evaluate(x)
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - 0.5) / 0.6) ** 2
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, gaussian: Gaussian):
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")
        expected = gaussian.evaluate(x)
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x.values - 0.5) / 0.6) ** 2
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, gaussian: Gaussian):
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")
        expected = gaussian.evaluate(x)
        expected_result = (2.0 * 1e3 / (0.6 * 1e3 * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x.values - 500.0) / (0.6 * 1e3)) ** 2
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self):
        test_gaussian = Gaussian(
            name="TestGaussian", area=2.0, center=None, width=0.6, unit="meV"
        )
        assert test_gaussian.center.value == 0.0
        assert test_gaussian.center.fixed is True

    def test_input_as_parameter(self):
        param_area = Parameter(name="area_param", value=2.0, unit="meV")
        param_center = Parameter(name="center_param", value=0.5, unit="meV")
        param_width = Parameter(name="width_param", value=0.6, unit="meV")
        test_gaussian = Gaussian(
            name="TestGaussian",
            area=param_area,
            center=param_center,
            width=param_width,
            unit="meV",
        )
        assert test_gaussian.area == param_area
        assert test_gaussian.center == param_center
        assert test_gaussian.width == param_width

    def test_get_parameters(self, gaussian: Gaussian):
        params = gaussian.get_parameters()
        assert len(params) == 3
        assert params[0].name == "TestGaussian area"
        assert params[1].name == "TestGaussian center"
        assert params[2].name == "TestGaussian width"
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, gaussian: Gaussian):
        # WHEN
        x = np.linspace(
            gaussian.center.value - 10 * gaussian.width.value,
            gaussian.center.value + 10 * gaussian.width.value,
            1000,
        )
        y = gaussian.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert np.isclose(numerical_area, gaussian.area.value, rtol=1e-3)

    def test_convert_unit(self, gaussian: Gaussian):
        gaussian.convert_unit("microeV")

        assert gaussian.unit == "microeV"
        assert gaussian.area.value == 2 * 1e3
        assert gaussian.center.value == 0.5 * 1e3
        assert gaussian.width.value == 0.6 * 1e3

    def test_copy(self, gaussian: Gaussian):
        gaussian_copy = gaussian.copy()
        assert gaussian_copy is not gaussian
        assert gaussian_copy.name == gaussian.name

        assert gaussian_copy.area.value == gaussian.area.value
        assert gaussian_copy.area.fixed == gaussian.area.fixed

        assert gaussian_copy.center.value == gaussian.center.value
        assert gaussian_copy.center.fixed == gaussian.center.fixed

        assert gaussian_copy.width.value == gaussian.width.value
        assert gaussian_copy.width.fixed == gaussian.width.fixed

        assert gaussian_copy.unit == gaussian.unit

    def test_repr(self, gaussian: Gaussian):
        repr_str = repr(gaussian)
        assert "Gaussian" in repr_str
        assert "name = TestGaussian" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str


class TestLorentzian:
    @pytest.fixture
    def lorentzian(self):
        return Lorentzian(
            name="TestLorentzian", area=2.0, center=0.5, width=0.6, unit="meV"
        )

    def test_initialization(self, lorentzian: Lorentzian):
        assert lorentzian.name == "TestLorentzian"
        assert lorentzian.area.value == 2.0
        assert lorentzian.center.value == 0.5
        assert lorentzian.width.value == 0.6
        assert lorentzian.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number or a Parameter"):
            Lorentzian(
                name="TestLorentzian", area="invalid", center=0.5, width=0.6, unit="meV"
            )

        with pytest.raises(
            TypeError, match="center must be None, a number or a Parameter"
        ):
            Lorentzian(
                name="TestLorentzian", area=2.0, center="invalid", width=0.6, unit="meV"
            )

        with pytest.raises(TypeError, match="width must be a number or a Parameter"):
            Lorentzian(
                name="TestLorentzian", area=2.0, center=0.5, width="invalid", unit="meV"
            )

        with pytest.raises(TypeError, match="unit must be a string"):
            Lorentzian(name="TestLorentzian", area=2.0, center=0.5, width=0.6, unit=123)

    def test_negative_width_raises(self):
        with pytest.raises(
            ValueError, match="The width of a Lorentzian must be greater than zero."
        ):
            Lorentzian(
                name="TestLorentzian", area=2.0, center=0.5, width=-0.6, unit="meV"
            )

    def test_negative_width_raises_in_evaluate(self):
        test_lorentzian = Lorentzian(
            name="TestLorentzian", area=2.0, center=0.5, width=0.6, unit="meV"
        )
        test_lorentzian.width.value = -0.6
        with pytest.raises(
            ValueError, match="The width of a Lorentzian must be greater than zero."
        ):
            test_lorentzian.evaluate(np.array([0.0, 0.5, 1.0]))

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            Lorentzian(
                name="TestLorentzian", area=-2.0, center=0.5, width=0.6, unit="meV"
            )

    def test_negative_area_warns_in_evaluate(self):
        test_lorentzian = Lorentzian(
            name="TestLorentzian", area=2.0, center=0.5, width=0.6, unit="meV"
        )
        test_lorentzian.area.value = -2.0
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            test_lorentzian.evaluate(np.array([0.0, 0.5, 1.0]))

    def test_evaluate(self, lorentzian: Lorentzian):
        x = np.array([0.0, 0.5, 1.0])
        expected = lorentzian.evaluate(x)
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, lorentzian: Lorentzian):
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")
        expected = lorentzian.evaluate(x)
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x.values - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, lorentzian: Lorentzian):
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")
        expected = lorentzian.evaluate(x)
        expected_result = (2.0 * 1e3 / (np.pi * 0.6 * 1e3)) / (
            1 + ((x.values - 0.5 * 1e3) / (0.6 * 1e3)) ** 2
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self):
        test_lorentzian = Lorentzian(
            name="TestLorentzian", area=2.0, center=None, width=0.6, unit="meV"
        )
        assert test_lorentzian.center.value == 0.0
        assert test_lorentzian.center.fixed is True

    def test_input_as_parameter(self):
        param_area = Parameter(name="area_param", value=2.0, unit="meV")
        param_center = Parameter(name="center_param", value=0.5, unit="meV")
        param_width = Parameter(name="width_param", value=0.6, unit="meV")
        test_lorentzian = Lorentzian(
            name="TestLorentzian",
            area=param_area,
            center=param_center,
            width=param_width,
            unit="meV",
        )
        assert test_lorentzian.area == param_area
        assert test_lorentzian.center == param_center
        assert test_lorentzian.width == param_width

    def test_get_parameters(self, lorentzian: Lorentzian):
        params = lorentzian.get_parameters()
        assert len(params) == 3
        assert params[0].name == "TestLorentzian area"
        assert params[1].name == "TestLorentzian center"
        assert params[2].name == "TestLorentzian width"
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, lorentzian: Lorentzian):
        # WHEN
        x = np.linspace(
            lorentzian.center.value - 500 * lorentzian.width.value,
            lorentzian.center.value + 500 * lorentzian.width.value,
            20000,
        )  # Lorentzians have very long tails
        y = lorentzian.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(lorentzian.area.value, rel=2e-3)

    def test_convert_unit(self, lorentzian: Lorentzian):
        lorentzian.convert_unit("microeV")

        assert lorentzian.unit == "microeV"
        assert lorentzian.area.value == 2 * 1e3
        assert lorentzian.center.value == 0.5 * 1e3
        assert lorentzian.width.value == 0.6 * 1e3

    def test_copy(self, lorentzian: Lorentzian):
        lorentzian_copy = lorentzian.copy()
        assert lorentzian_copy is not lorentzian
        assert lorentzian_copy.name == lorentzian.name

        assert lorentzian_copy.area.value == lorentzian.area.value
        assert lorentzian_copy.area.fixed == lorentzian.area.fixed

        assert lorentzian_copy.center.value == lorentzian.center.value
        assert lorentzian_copy.center.fixed == lorentzian.center.fixed

        assert lorentzian_copy.width.value == lorentzian.width.value
        assert lorentzian_copy.width.fixed == lorentzian.width.fixed

        assert lorentzian_copy.unit == lorentzian.unit

    def test_repr(self, lorentzian: Lorentzian):
        repr_str = repr(lorentzian)
        assert "Lorentzian" in repr_str
        assert "name = TestLorentzian" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str


class TestVoigt:
    @pytest.fixture
    def voigt(self):
        return Voigt(
            name="TestVoigt",
            area=2.0,
            center=0.5,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit="meV",
        )

    def test_initialization(self, voigt: Voigt):
        assert voigt.name == "TestVoigt"
        assert voigt.area.value == 2.0
        assert voigt.center.value == 0.5
        assert voigt.gaussian_width.value == 0.6
        assert voigt.lorentzian_width.value == 0.7
        assert voigt.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number or a Parameter"):
            Voigt(
                name="TestVoigt",
                area="invalid",
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit="meV",
            )

        with pytest.raises(
            TypeError, match="center must be None, a number or a Parameter"
        ):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center="invalid",
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit="meV",
            )

        with pytest.raises(
            TypeError, match="gaussian_width must be a number or a Parameter"
        ):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width="invalid",
                lorentzian_width=0.7,
                unit="meV",
            )
        with pytest.raises(
            TypeError, match="lorentzian_width must be a number or a Parameter"
        ):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width="invalid",
                unit="meV",
            )
        with pytest.raises(TypeError, match="unit must be a string"):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit=123,
            )

    def test_negative_gaussian_width_raises(self):
        with pytest.raises(
            ValueError, match="The gaussian_width of a Voigt must be greater than."
        ):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width=-0.6,
                lorentzian_width=0.7,
                unit="meV",
            )

    def test_negative_gaussian_width_raises_in_evaluate(self):
        test_voigt = Voigt(
            name="TestVoigt",
            area=2.0,
            center=0.5,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit="meV",
        )
        test_voigt.gaussian_width.value = -0.6
        with pytest.raises(
            ValueError, match="The gaussian_width of a Voigt must be greater than."
        ):
            test_voigt.evaluate(np.array([0.0, 0.5, 1.0]))

    def test_negative_lorentzian_width_raises(self):
        with pytest.raises(
            ValueError,
            match="The lorentzian_width of a Voigt must be greater than zero.",
        ):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=-0.7,
                unit="meV",
            )

    def test_negative_lorentzian_width_raises_in_evaluate(self):
        test_voigt = Voigt(
            name="TestVoigt",
            area=2.0,
            center=0.5,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit="meV",
        )
        test_voigt.lorentzian_width.value = -0.7
        with pytest.raises(
            ValueError,
            match="The lorentzian_width of a Voigt must be greater than zero.",
        ):
            test_voigt.evaluate(np.array([0.0, 0.5, 1.0]))

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            Voigt(
                name="TestVoigt",
                area=-2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit="meV",
            )

    def test_negative_area_warns_in_evaluate(self):
        test_voigt = Voigt(
            name="TestVoigt",
            area=2.0,
            center=0.5,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit="meV",
        )
        test_voigt.area.value = -2.0
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            test_voigt.evaluate(np.array([0.0, 0.5, 1.0]))

    def test_evaluate(self, voigt: Voigt):
        x = np.array([0.0, 0.5, 1.0])
        expected = voigt.evaluate(x)
        expected_result = 2.0 * voigt_profile(x - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, voigt: Voigt):
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")
        expected = voigt.evaluate(x)
        expected_result = 2.0 * voigt_profile(x.values - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, voigt: Voigt):
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")
        expected = voigt.evaluate(x)
        expected_result = (
            2.0 * 1e3 * voigt_profile(x.values - 0.5 * 1e3, 0.6 * 1e3, 0.7 * 1e3)
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self):
        test_voigt = Voigt(
            name="TestVoigt",
            area=2.0,
            center=None,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit="meV",
        )
        assert test_voigt.center.value == 0.0
        assert test_voigt.center.fixed is True

    def test_input_as_parameter(self):
        param_area = Parameter(name="area_param", value=2.0, unit="meV")
        param_center = Parameter(name="center_param", value=0.5, unit="meV")
        param_gaussian_width = Parameter(
            name="gaussian_width_param", value=0.6, unit="meV"
        )
        param_lorentzian_width = Parameter(
            name="lorentzian_width_param", value=0.7, unit="meV"
        )
        test_voigt = Voigt(
            name="TestVoigt",
            area=param_area,
            center=param_center,
            gaussian_width=param_gaussian_width,
            lorentzian_width=param_lorentzian_width,
            unit="meV",
        )
        assert test_voigt.area == param_area
        assert test_voigt.center == param_center
        assert test_voigt.gaussian_width == param_gaussian_width
        assert test_voigt.lorentzian_width == param_lorentzian_width

    def test_convert_unit(self, voigt: Voigt):
        voigt.convert_unit("microeV")

        assert voigt.unit == "microeV"
        assert voigt.area.value == 2 * 1e3
        assert voigt.center.value == 0.5 * 1e3
        assert voigt.gaussian_width.value == 0.6 * 1e3
        assert voigt.lorentzian_width.value == 0.7 * 1e3

    def test_get_parameters(self, voigt: Voigt):
        params = voigt.get_parameters()
        assert len(params) == 4
        assert params[0].name == "TestVoigt area"
        assert params[1].name == "TestVoigt center"
        assert params[2].name == "TestVoigt gaussian_width"
        assert params[3].name == "TestVoigt lorentzian_width"
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, voigt: Voigt):
        # WHEN
        x = np.linspace(
            voigt.center.value
            - 100 * voigt.gaussian_width.value
            - 300 * voigt.lorentzian_width.value,
            voigt.center.value
            + 100 * voigt.gaussian_width.value
            + 300 * voigt.lorentzian_width.value,
            20000,
        )  # Voigts have very long tails
        y = voigt.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(voigt.area.value, rel=2e-3)

    def test_copy(self, voigt: Voigt):
        voigt_copy = voigt.copy()
        assert voigt_copy is not voigt
        assert voigt_copy.name == voigt.name

        assert voigt_copy.area.value == voigt.area.value
        assert voigt_copy.area.fixed == voigt.area.fixed

        assert voigt_copy.center.value == voigt.center.value
        assert voigt_copy.center.fixed == voigt.center.fixed

        assert voigt_copy.gaussian_width.value == voigt.gaussian_width.value
        assert voigt_copy.gaussian_width.fixed == voigt.gaussian_width.fixed

        assert voigt_copy.lorentzian_width.value == voigt.lorentzian_width.value
        assert voigt_copy.lorentzian_width.fixed == voigt.lorentzian_width.fixed

        assert voigt_copy.unit == voigt.unit

    def test_repr(self, voigt: Voigt):
        repr_str = repr(voigt)
        assert "Voigt" in repr_str
        assert "name = TestVoigt" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "gaussian_width =" in repr_str
        assert "lorentzian_width =" in repr_str


class TestDeltaFunction:
    @pytest.fixture
    def delta_function(self):
        return DeltaFunction(name="TestDeltaFunction", area=2.0, center=0.5, unit="meV")

    def test_initialization(self, delta_function: DeltaFunction):
        assert delta_function.name == "TestDeltaFunction"
        assert delta_function.area.value == 2.0
        assert delta_function.center.value == 0.5
        assert delta_function.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number or a Parameter"):
            DeltaFunction(
                name="TestDeltaFunction",
                area="invalid",
                center=0.5,
                unit="meV",
            )
        with pytest.raises(
            TypeError, match="center must be None, a number or a Parameter"
        ):
            DeltaFunction(
                name="TestDeltaFunction",
                area=2.0,
                center="invalid",
                unit="meV",
            )
        with pytest.raises(TypeError, match="unit must be a string"):
            DeltaFunction(name="TestDeltaFunction", area=2.0, center=0.5, unit=123)

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            DeltaFunction(name="TestDeltaFunction", area=-2.0, center=0.5, unit="meV")

    @pytest.mark.xfail(
        reason="DeltaFunction.evaluate is not implemented yet without resolution convolution"
    )
    def test_evaluate(self, delta_function: DeltaFunction):
        x = np.array([0.0, 0.5, 1.0])
        expected = delta_function.evaluate(x)
        expected_result = np.zeros_like(x)
        # expected_result[x == 0.5] = 2.0
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self):
        test_delta = DeltaFunction(
            name="TestDeltaFunction", area=2.0, center=None, unit="meV"
        )
        assert test_delta.center.value == 0.0
        assert test_delta.center.fixed is True

    def test_input_as_parameter(self):
        param_area = Parameter(name="area_param", value=2.0, unit="meV")
        param_center = Parameter(name="center_param", value=0.5, unit="meV")
        test_delta = DeltaFunction(
            name="TestDeltaFunction", area=param_area, center=param_center, unit="meV"
        )
        assert test_delta.area == param_area
        assert test_delta.center == param_center

    def test_get_parameters(self, delta_function: DeltaFunction):
        params = delta_function.get_parameters()
        assert len(params) == 2
        assert params[0].name == "TestDeltaFunction area"
        assert params[1].name == "TestDeltaFunction center"
        assert all(isinstance(param, Parameter) for param in params)

    def test_convert_unit(self, delta_function: DeltaFunction):
        delta_function.convert_unit("microeV")

        assert delta_function.unit == "microeV"
        assert delta_function.area.value == 2 * 1e3
        assert delta_function.center.value == 0.5 * 1e3

    def test_copy(self, delta_function: DeltaFunction):
        delta_copy = delta_function.copy()
        assert delta_copy is not delta_function
        assert delta_copy.name == delta_function.name

        assert delta_copy.area.value == delta_function.area.value
        assert delta_copy.area.fixed == delta_function.area.fixed

        assert delta_copy.center.value == delta_function.center.value
        assert delta_copy.center.fixed == delta_function.center.fixed

        assert delta_copy.unit == delta_function.unit

    def test_repr(self, delta_function: DeltaFunction):
        repr_str = repr(delta_function)
        assert "DeltaFunction" in repr_str
        assert "name = TestDeltaFunction" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str


class TestDampedHarmonicOscillator:
    @pytest.fixture
    def dho(self):
        return DampedHarmonicOscillator(
            name="TestDHO", area=2.0, center=1.5, width=0.3, unit="meV"
        )

    def test_initialization(self, dho: DampedHarmonicOscillator):
        assert dho.name == "TestDHO"
        assert dho.area.value == 2.0
        assert dho.center.value == 1.5
        assert dho.width.value == 0.3
        assert dho.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number or a Parameter"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area="invalid",
                center=0.5,
                width=0.6,
                unit="meV",
            )

        with pytest.raises(TypeError, match="center must be a number or a Parameter"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=2.0,
                center="invalid",
                width=0.6,
                unit="meV",
            )

        with pytest.raises(TypeError, match="width must be a number or a Parameter"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=2.0,
                center=0.5,
                width="invalid",
                unit="meV",
            )

        with pytest.raises(TypeError, match="unit must be a string"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=2.0,
                center=0.5,
                width=0.6,
                unit=123,
            )

    def test_negative_width_raises(self):
        with pytest.raises(
            ValueError,
            match="The width of a DampedHarmonicOscillator must be greater than zero.",
        ):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=2.0,
                center=0.5,
                width=-0.6,
                unit="meV",
            )

    def test_negative_width_raises_in_evaluate(self):
        test_dho = DampedHarmonicOscillator(
            name="TestDampedHarmonicOscillator",
            area=2.0,
            center=0.5,
            width=0.6,
            unit="meV",
        )
        test_dho.width.value = -0.6
        with pytest.raises(
            ValueError,
            match="The width of a DampedHarmonicOscillator must be greater than zero.",
        ):
            test_dho.evaluate(np.array([0.0, 1.5, 3.0]))

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=-2.0,
                center=0.5,
                width=0.6,
                unit="meV",
            )

    def test_negative_area_warns_in_evaluate(self):
        test_dho = DampedHarmonicOscillator(
            name="TestDampedHarmonicOscillator",
            area=2.0,
            center=0.5,
            width=0.6,
            unit="meV",
        )
        test_dho.area.value = -2.0
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            test_dho.evaluate(np.array([0.0, 1.5, 3.0]))

    def test_evaluate(self, dho: DampedHarmonicOscillator):
        x = np.array([0.0, 1.5, 3.0])
        expected = dho.evaluate(x)
        expected_result = (
            2
            * 2.0
            * (1.5**2)
            * (0.3)
            / np.pi
            / ((x**2 - 1.5**2) ** 2 + (2 * 0.3 * x) ** 2)
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, dho: DampedHarmonicOscillator):
        x = sc.array(dims=["x"], values=[0.0, 1.5, 3.0], unit="meV")
        expected = dho.evaluate(x)
        expected_result = (
            2
            * 2.0
            * (1.5**2)
            * (0.3)
            / np.pi
            / ((x.values**2 - 1.5**2) ** 2 + (2 * 0.3 * x.values) ** 2)
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, dho: DampedHarmonicOscillator):
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")
        expected = dho.evaluate(x)
        expected_result = (
            2
            * 2.0
            * 1e3
            * ((1.5 * 1e3) ** 2)
            * (0.3 * 1e3)
            / np.pi
            / ((x.values**2 - (1.5 * 1e3) ** 2) ** 2 + (2 * 0.3 * 1e3 * x.values) ** 2)
        )
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_input_as_parameter(self):
        param_area = Parameter(name="area_param", value=2.0, unit="meV")
        param_center = Parameter(name="center_param", value=0.5, unit="meV")
        param_width = Parameter(name="width_param", value=0.6, unit="meV")
        test_dho = DampedHarmonicOscillator(
            name="TestDHO",
            area=param_area,
            center=param_center,
            width=param_width,
            unit="meV",
        )
        assert test_dho.area == param_area
        assert test_dho.center == param_center
        assert test_dho.width == param_width

    def test_get_parameters(self, dho: DampedHarmonicOscillator):
        params = dho.get_parameters()
        assert len(params) == 3
        assert params[0].name == "TestDHO area"
        assert params[1].name == "TestDHO center"
        assert params[2].name == "TestDHO width"
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, dho: DampedHarmonicOscillator):
        # WHEN
        x = np.linspace(
            -dho.center.value - 20 * dho.width.value,
            dho.center.value + 20 * dho.width.value,
            5000,
        )
        y = dho.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(dho.area.value, rel=2e-3)

    def test_convert_unit(self, dho: DampedHarmonicOscillator):
        dho.convert_unit("microeV")

        assert dho.unit == "microeV"
        assert dho.area.value == 2 * 1e3
        assert dho.center.value == 1.5 * 1e3
        assert dho.width.value == 0.3 * 1e3

    def test_copy(self, dho: DampedHarmonicOscillator):
        dho_copy = dho.copy()
        assert dho_copy is not dho
        assert dho_copy.name == dho.name

        assert dho_copy.area.value == dho.area.value
        assert dho_copy.area.fixed == dho.area.fixed

        assert dho_copy.center.value == dho.center.value
        assert dho_copy.center.fixed == dho.center.fixed

        assert dho_copy.width.value == dho.width.value
        assert dho_copy.width.fixed == dho.width.fixed

        assert dho_copy.unit == dho.unit

    def test_repr(self, dho: DampedHarmonicOscillator):
        repr_str = repr(dho)
        assert "DampedHarmonicOscillator" in repr_str
        assert "name = TestDHO" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str


class TestPolynomial:
    @pytest.fixture
    def polynomial(self):
        return Polynomial(name="TestPolynomial", coefficients=[1.0, -2.0, 3.0])

    def test_initialization(self, polynomial: Polynomial):
        assert polynomial.name == "TestPolynomial"
        assert polynomial.coefficients[0].value == 1.0
        assert polynomial.coefficients[1].value == -2.0
        assert polynomial.coefficients[2].value == 3.0

    def test_input_type_validation_raises(self):
        with pytest.raises(
            TypeError, match="coefficients must be a list, tuple or ndarray of floats."
        ):
            Polynomial(name="TestPolynomial", coefficients="invalid")

        with pytest.raises(TypeError, match="All coefficients must be numbers."):
            Polynomial(name="TestPolynomial", coefficients=[1.0, "invalid", 3.0])

        with pytest.raises(TypeError, match="unit must be a string"):
            Polynomial(name="TestPolynomial", coefficients=[1.0, -2.0, 3.0], unit=123)

        with pytest.raises(
            ValueError, match="At least one coefficient must be provided"
        ):
            Polynomial(name="TestPolynomial", coefficients=[])

    def test_negative_value_warns_in_evaluate(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            test_polynomial = Polynomial(name="TestPolynomial", coefficients=[-1.0])
            test_polynomial.evaluate(np.array([0.0, 1.0, 2.0]))

    def test_evaluate(self, polynomial: Polynomial):
        x = np.array([0.0, 1.0, 2.0])
        expected = polynomial.evaluate(x)
        expected_result = 1.0 - 2.0 * x + 3.0 * x**2
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, polynomial: Polynomial):
        x = sc.array(dims=["x"], values=[0.0, 1.0, 2.0], unit="meV")
        expected = polynomial.evaluate(x)
        expected_result = 1.0 - 2.0 * x.values + 3.0 * x.values**2
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit_error(self, polynomial: Polynomial):
        x = sc.array(dims=["x"], values=[0.0, 1.0, 2.0], unit="microeV")

        with pytest.raises(
            ValueError,
            match="Change the unit of the Polynomial and try again",
        ):
            polynomial.evaluate(x)

    def test_degree(self, polynomial: Polynomial):
        assert polynomial.degree() == 2

    def test_get_parameters(self, polynomial: Polynomial):
        params = polynomial.get_parameters()
        assert len(params) == 3
        assert params[0].name == "TestPolynomial_c0"
        assert params[1].name == "TestPolynomial_c1"
        assert params[2].name == "TestPolynomial_c2"
        assert all(isinstance(param, Parameter) for param in params)

    def test_convert_unit_raises_for_polynomial(self, polynomial):
        with pytest.raises(
            NotImplementedError,
            match="Unit conversion is not implemented for Polynomial components. The automatic unit converter does not like powers of units.",
        ):
            polynomial.convert_unit("eV")

    def test_copy(self, polynomial: Polynomial):
        polynomial_copy = polynomial.copy()
        assert polynomial_copy is not polynomial
        assert polynomial_copy.name == polynomial.name
        assert len(polynomial_copy.coefficients) == len(polynomial.coefficients)
        for original_coeff, copied_coeff in zip(
            polynomial.coefficients, polynomial_copy.coefficients
        ):
            assert copied_coeff.value == original_coeff.value
            assert copied_coeff.fixed == original_coeff.fixed

    def test_repr(self, polynomial: Polynomial):
        repr_str = repr(polynomial)
        assert "Polynomial" in repr_str
        assert "name = TestPolynomial" in repr_str
        assert "coefficients =" in repr_str


# @pytest.mark.skip(reason="UserDefinedComponent not implemented yet")
# class TestUserDefinedComponent:
#     def test_placeholder(self):
#         pass
