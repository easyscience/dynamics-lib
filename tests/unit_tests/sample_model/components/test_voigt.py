import pytest

import numpy as np
import scipp as sc

from scipy.integrate import simpson

from easydynamics.sample_model import Voigt

from easyscience.variable import Parameter

from scipy.special import voigt_profile


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
