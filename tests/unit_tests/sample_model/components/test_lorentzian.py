import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson

from easydynamics.sample_model import Lorentzian


class TestLorentzian:
    @pytest.fixture
    def lorentzian(self):
        return Lorentzian(
            name="TestLorentzian", area=2.0, center=0.5, width=0.6, unit="meV"
        )

    def test_initialization(self, lorentzian: Lorentzian):
        # WHEN THEN EXPECT
        assert lorentzian.name == "TestLorentzian"
        assert lorentzian.area.value == 2.0
        assert lorentzian.center.value == 0.5
        assert lorentzian.width.value == 0.6
        assert lorentzian.unit == "meV"

    @pytest.mark.parametrize(
        "kwargs, expected_message",
        [
            (
                {"area": "invalid", "center": 0.5, "width": 0.6, "unit": "meV"},
                "area must be a number",
            ),
            (
                {"area": 2.0, "center": "invalid", "width": 0.6, "unit": "meV"},
                "center must be None or a number",
            ),
            (
                {"area": 2.0, "center": 0.5, "width": "invalid", "unit": "meV"},
                "width must be a number",
            ),
            (
                {"area": 2.0, "center": 0.5, "width": 0.6, "unit": 123},
                "unit must be a string or a scipp unit",
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        with pytest.raises(TypeError, match=expected_message):
            Lorentzian(name="TestLorentzian", **kwargs)

    def test_negative_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError, match="The width of a Lorentzian must be greater than zero."
        ):
            Lorentzian(
                name="TestLorentzian", area=2.0, center=0.5, width=-0.6, unit="meV"
            )

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            Lorentzian(
                name="TestLorentzian", area=-2.0, center=0.5, width=0.6, unit="meV"
            )

    def test_area_property_setter(self, lorentzian: Lorentzian):
        # WHEN
        lorentzian.area = 3.0

        # THEN EXPECT
        assert lorentzian.area.value == 3.0
        with pytest.raises(TypeError, match="area must be a number."):
            lorentzian.area = "invalid"

    def test_center_property_setter(self, lorentzian: Lorentzian):
        # WHEN THEN
        lorentzian.center = 0.6

        # EXPECT
        assert lorentzian.center.value == 0.6
        with pytest.raises(TypeError, match="center must be a number."):
            lorentzian.center = "invalid"

    def test_width_property_setter(self, lorentzian: Lorentzian):
        # WHEN THEN
        lorentzian.width = 0.7

        # EXPECT
        assert lorentzian.width.value == 0.7
        with pytest.raises(TypeError, match="width must be a number."):
            lorentzian.width = "invalid"

    def test_evaluate(self, lorentzian: Lorentzian):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = lorentzian.evaluate(x)

        # EXPECT
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, lorentzian: Lorentzian):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")

        # THEN
        result = lorentzian.evaluate(x)

        # EXPECT
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x.values - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    @pytest.mark.filterwarnings(
        "ignore:Input x has unit µeV, but Lorentzian component has unit meV.*:UserWarning"
    )
    def test_evaluate_with_different_unit(self, lorentzian: Lorentzian):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")

        # THEN
        result = lorentzian.evaluate(x)

        # EXPECT
        expected_result = (2.0 * 1e3 / (np.pi * 0.6 * 1e3)) / (
            1 + ((x.values - 0.5 * 1e3) / (0.6 * 1e3)) ** 2
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit_warns(self, lorentzian: Lorentzian):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")

        # THEN EXPECT
        with pytest.warns(
            UserWarning,
            match="Input x has unit µeV, but Lorentzian component has unit meV. Converting Lorentzian to µeV.",
        ):
            lorentzian.evaluate(x)

    def test_evaluate_with_incompatible_unit_raises(self, lorentzian: Lorentzian):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="nm")

        # THEN EXPECT
        with pytest.raises(
            UnitError,
            match="Input x has unit nm, but Lorentzian component has unit meV. Failed to convert Lorentzian to nm.",
        ):
            lorentzian.evaluate(x)

    @pytest.mark.parametrize(
        "x, expected_message",
        [
            (np.array([0.0, np.nan, 1.0]), "Input x contains NaN values."),
            (np.array([0.0, np.inf, 1.0]), "Input x contains infinite values."),
        ],
    )
    def test_evaluate_with_invalid_input_raises(
        self, lorentzian: Lorentzian, x, expected_message
    ):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=expected_message):
            lorentzian.evaluate(x)

    def test_center_is_fixed_if_set_to_None(self):
        # WHEN THEN
        test_lorentzian = Lorentzian(
            name="TestLorentzian", area=2.0, center=None, width=0.6, unit="meV"
        )

        # EXPECT
        assert test_lorentzian.center.value == 0.0
        assert test_lorentzian.center.fixed is True

    def test_get_parameters(self, lorentzian: Lorentzian):
        # WHEN THEN
        params = lorentzian.get_parameters()

        # EXPECT
        assert len(params) == 3
        assert params[0].name == "TestLorentzian area"
        assert params[1].name == "TestLorentzian center"
        assert params[2].name == "TestLorentzian width"
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, lorentzian: Lorentzian):
        # WHEN THEN
        x = np.linspace(
            lorentzian.center.value - 500 * lorentzian.width.value,
            lorentzian.center.value + 500 * lorentzian.width.value,
            20000,
        )  # Lorentzians have very long tails
        y = lorentzian.evaluate(x)
        numerical_area = simpson(y, x)

        # EXPECT
        assert numerical_area == pytest.approx(lorentzian._area.value, rel=2e-3)

    def test_convert_unit(self, lorentzian: Lorentzian):
        # WHEN THEN
        lorentzian.convert_unit("microeV")

        # EXPECT
        assert lorentzian.unit == "microeV"
        assert lorentzian.area.value == 2 * 1e3
        assert lorentzian.center.value == 0.5 * 1e3
        assert lorentzian.width.value == 0.6 * 1e3

    def test_copy(self, lorentzian: Lorentzian):
        # WHEN THEN
        lorentzian_copy = lorentzian.copy()

        # EXPECT
        assert lorentzian_copy is not lorentzian
        assert lorentzian_copy.name == "copy of " + lorentzian.name

        assert lorentzian_copy.area.value == lorentzian.area.value
        assert lorentzian_copy.area.fixed == lorentzian.area.fixed

        assert lorentzian_copy.center.value == lorentzian.center.value
        assert lorentzian_copy.center.fixed == lorentzian.center.fixed

        assert lorentzian_copy.width.value == lorentzian.width.value
        assert lorentzian_copy.width.fixed == lorentzian.width.fixed

        assert lorentzian_copy.unit == lorentzian.unit

    def test_repr(self, lorentzian: Lorentzian):
        # WHEN THEN
        repr_str = repr(lorentzian)

        # EXPECT
        assert "Lorentzian" in repr_str
        assert "name = TestLorentzian" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str
