import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson
from scipy.special import voigt_profile

from easydynamics.sample_model import Voigt


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
        # WHEN THEN EXPECT
        assert voigt.name == "TestVoigt"
        assert voigt._area.value == 2.0
        assert voigt._center.value == 0.5
        assert voigt._gaussian_width.value == 0.6
        assert voigt._lorentzian_width.value == 0.7
        assert voigt.unit == "meV"

    def test_input_type_validation_area_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="area must be a number"):
            Voigt(
                name="TestVoigt",
                area="invalid",
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit="meV",
            )

    def test_input_type_validation_center_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="center must be None or a number"):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center="invalid",
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit="meV",
            )

    def test_input_type_validation_gaussian_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="gaussian_width must be a number"):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width="invalid",
                lorentzian_width=0.7,
                unit="meV",
            )

    def test_input_type_validation_lorentzian_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="lorentzian_width must be a number"):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width="invalid",
                unit="meV",
            )

    def test_input_type_validation_unit_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="unit must be a string or a scipp unit"):
            Voigt(
                name="TestVoigt",
                area=2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit=123,
            )

    def test_negative_gaussian_width_raises(self):
        # WHEN THEN EXPECT
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

    def test_negative_lorentzian_width_raises(self):
        # WHEN THEN EXPECT
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

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            Voigt(
                name="TestVoigt",
                area=-2.0,
                center=0.5,
                gaussian_width=0.6,
                lorentzian_width=0.7,
                unit="meV",
            )

    def test_area_property_getter(self, voigt: Voigt):
        # WHEN THEN EXPECT
        assert voigt.area.value == 2.0

    def test_area_property_setter(self, voigt: Voigt):
        # WHEN
        voigt.area = 3.0

        # THEN EXPECT
        assert voigt.area.value == 3.0
        with pytest.raises(TypeError, match="area must be a number."):
            voigt.area = "invalid"

    def test_center_property_getter(self, voigt: Voigt):
        # WHEN THEN EXPECT
        assert voigt.center.value == 0.5

    def test_center_property_setter(self, voigt: Voigt):
        # WHEN
        voigt.center = 0.6

        # THEN EXPECT
        assert voigt.center.value == 0.6
        with pytest.raises(TypeError, match="center must be a number."):
            voigt.center = "invalid"

    def test_gaussian_width_property_getter(self, voigt: Voigt):
        # WHEN THEN EXPECT
        assert voigt.gaussian_width.value == 0.6

    def test_gaussian_width_property_setter(self, voigt: Voigt):
        # WHEN THEN
        voigt.gaussian_width = 0.7

        # EXPECT
        assert voigt.gaussian_width.value == 0.7
        with pytest.raises(TypeError, match="width must be a number."):
            voigt.gaussian_width = "invalid"

    def test_lorentzian_width_property_getter(self, voigt: Voigt):
        # WHEN THEN EXPECT
        assert voigt.lorentzian_width.value == 0.7

    def test_lorentzian_width_property_setter(self, voigt: Voigt):
        # WHEN THEN
        voigt.lorentzian_width = 0.8

        # EXPECT
        assert voigt.lorentzian_width.value == 0.8

        with pytest.raises(TypeError, match="width must be a number."):
            voigt.lorentzian_width = "invalid"

    def test_evaluate(self, voigt: Voigt):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = voigt.evaluate(x)

        # EXPECT
        expected_result = 2.0 * voigt_profile(x - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, voigt: Voigt):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")

        # THEN
        result = voigt.evaluate(x)

        # EXPECT
        expected_result = 2.0 * voigt_profile(x.values - 0.5, 0.6, 0.7)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    @pytest.mark.filterwarnings(
        "ignore:Input x has unit µeV, but Voigt component has unit meV.*:UserWarning"
    )
    def test_evaluate_with_different_unit(self, voigt: Voigt):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")

        # THEN
        result = voigt.evaluate(x)

        # EXPECT
        expected_result = (
            2.0 * 1e3 * voigt_profile(x.values - 0.5 * 1e3, 0.6 * 1e3, 0.7 * 1e3)
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit_warns(self, voigt: Voigt):
        # WHEN THEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")

        # EXPECT
        with pytest.warns(
            UserWarning,
            match="Input x has unit µeV, but Voigt component has unit meV. Converting Voigt to µeV.",
        ):
            voigt.evaluate(x)

    def test_evaluate_with_incompatible_unit(self, voigt: Voigt):
        # WHEN THEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="nm")

        # EXPECT
        with pytest.raises(
            UnitError,
            match="Input x has unit nm, but Voigt component has unit meV. Failed to convert Voigt to nm.",
        ):
            voigt.evaluate(x)

    def test_evaluate_with_nan_input(self, voigt: Voigt):
        # WHEN THEN
        x = np.array([0.0, np.nan, 1.0])

        # EXPECT
        with pytest.raises(ValueError, match="Input x contains NaN values."):
            voigt.evaluate(x)

    def test_evaluate_with_infinite_input(self, voigt: Voigt):
        # WHEN THEN
        x = np.array([0.0, np.inf, 1.0])

        # EXPECT
        with pytest.raises(ValueError, match="Input x contains infinite values."):
            voigt.evaluate(x)

    def test_center_is_fixed_if_set_to_None(self):
        # WHEN THEN
        test_voigt = Voigt(
            name="TestVoigt",
            area=2.0,
            center=None,
            gaussian_width=0.6,
            lorentzian_width=0.7,
            unit="meV",
        )

        # EXPECT
        assert test_voigt._center.value == 0.0
        assert test_voigt._center.fixed is True

    def test_convert_unit(self, voigt: Voigt):
        # WHEN THEN
        voigt.convert_unit("microeV")

        # EXPECT
        assert voigt.unit == "microeV"
        assert voigt._area.value == 2 * 1e3
        assert voigt._center.value == 0.5 * 1e3
        assert voigt._gaussian_width.value == 0.6 * 1e3
        assert voigt._lorentzian_width.value == 0.7 * 1e3

    def test_get_parameters(self, voigt: Voigt):
        # WHEN THEN
        params = voigt.get_parameters()

        # EXPECT
        assert len(params) == 4
        assert params[0].name == "TestVoigt area"
        assert params[1].name == "TestVoigt center"
        assert params[2].name == "TestVoigt gaussian_width"
        assert params[3].name == "TestVoigt lorentzian_width"
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, voigt: Voigt):
        # WHEN THEN
        x = np.linspace(
            voigt._center.value
            - 100 * voigt._gaussian_width.value
            - 300 * voigt._lorentzian_width.value,
            voigt._center.value
            + 100 * voigt._gaussian_width.value
            + 300 * voigt._lorentzian_width.value,
            20000,
        )  # Voigts have very long tails
        y = voigt.evaluate(x)
        numerical_area = simpson(y, x)

        # EXPECT
        assert numerical_area == pytest.approx(voigt._area.value, rel=2e-3)

    def test_copy(self, voigt: Voigt):
        # WHEN THEN
        voigt_copy = voigt.copy()

        # EXPECT
        assert voigt_copy is not voigt
        assert voigt_copy.name == "copy of " + voigt.name

        assert voigt_copy._area.value == voigt._area.value
        assert voigt_copy._area.fixed == voigt._area.fixed

        assert voigt_copy._center.value == voigt._center.value
        assert voigt_copy._center.fixed == voigt._center.fixed

        assert voigt_copy._gaussian_width.value == voigt._gaussian_width.value
        assert voigt_copy._gaussian_width.fixed == voigt._gaussian_width.fixed

        assert voigt_copy._lorentzian_width.value == voigt._lorentzian_width.value
        assert voigt_copy._lorentzian_width.fixed == voigt._lorentzian_width.fixed

        assert voigt_copy.unit == voigt.unit

    def test_repr(self, voigt: Voigt):
        # WHEN THEN
        repr_str = repr(voigt)

        # EXPECT
        assert "Voigt" in repr_str
        assert "name = TestVoigt" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "gaussian_width =" in repr_str
        assert "lorentzian_width =" in repr_str
