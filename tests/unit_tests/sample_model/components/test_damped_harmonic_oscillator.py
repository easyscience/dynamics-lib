import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson

from easydynamics.sample_model import DampedHarmonicOscillator


class TestDampedHarmonicOscillator:
    @pytest.fixture
    def dho(self):
        return DampedHarmonicOscillator(
            name="TestDHO", area=2.0, center=1.5, width=0.3, unit="meV"
        )

    def test_initialization(self, dho: DampedHarmonicOscillator):
        # WHEN THEN EXPECT
        assert dho.name == "TestDHO"
        assert dho.area.value == 2.0
        assert dho.center.value == 1.5
        assert dho.width.value == 0.3
        assert dho.unit == "meV"

    @pytest.mark.parametrize(
        "kwargs, expected_message",
        [
            (
                {"area": "invalid", "center": 0.5, "width": 0.6, "unit": "meV"},
                "area must be a number",
            ),
            (
                {"area": 2.0, "center": "invalid", "width": 0.6, "unit": "meV"},
                "center must be a number",
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
            DampedHarmonicOscillator(name="DampedHarmonicOscillator", **kwargs)

    def test_negative_width_raises(self):
        # WHEN THEN EXPECT
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

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=-2.0,
                center=0.5,
                width=0.6,
                unit="meV",
            )

    @pytest.mark.parametrize(
        "prop, valid_value, invalid_value, invalid_message",
        [
            ("area", 3.0, "invalid", r"area must be a number\."),
            ("center", 0.6, "invalid", r"center must be a number\."),
            ("width", 0.7, "invalid", r"width must be a number\."),
        ],
    )
    def test_property_setters_validate(
        self,
        dho: DampedHarmonicOscillator,
        prop,
        valid_value,
        invalid_value,
        invalid_message,
    ):
        # set valid
        setattr(dho, prop, valid_value)
        assert getattr(dho, prop).value == valid_value

        # invalid
        with pytest.raises(TypeError, match=invalid_message):
            setattr(dho, prop, invalid_value)

    def test_evaluate(self, dho: DampedHarmonicOscillator):
        # WHEN
        x = np.array([0.0, 1.5, 3.0])

        # THEN
        result = dho.evaluate(x)

        # EXPECT
        expected_result = (
            2
            * 2.0
            * (1.5**2)
            * (0.3)
            / np.pi
            / ((x**2 - 1.5**2) ** 2 + (2 * 0.3 * x) ** 2)
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, dho: DampedHarmonicOscillator):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 1.5, 3.0], unit="meV")

        # THEN
        result = dho.evaluate(x)

        # EXPECT
        expected_result = (
            2
            * 2.0
            * (1.5**2)
            * (0.3)
            / np.pi
            / ((x.values**2 - 1.5**2) ** 2 + (2 * 0.3 * x.values) ** 2)
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    @pytest.mark.filterwarnings(
        "ignore:Input x has unit µeV, but DampedHarmonicOscillator component has unit meV.*:UserWarning"
    )
    def test_evaluate_with_different_unit(self, dho: DampedHarmonicOscillator):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")

        # THEN
        result = dho.evaluate(x)

        # EXPECT
        expected_result = (
            2
            * 2.0
            * 1e3
            * ((1.5 * 1e3) ** 2)
            * (0.3 * 1e3)
            / np.pi
            / ((x.values**2 - (1.5 * 1e3) ** 2) ** 2 + (2 * 0.3 * 1e3 * x.values) ** 2)
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit_warning(self, dho: DampedHarmonicOscillator):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")

        # THEN EXPECT
        with pytest.warns(
            UserWarning,
            match="Input x has unit µeV, but DampedHarmonicOscillator component has unit meV. Converting DampedHarmonicOscillator to µeV.",
        ):
            dho.evaluate(x)

    def test_evaluate_with_incompatible_unit_raises(
        self, dho: DampedHarmonicOscillator
    ):
        # WHEN THEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="nm")

        # EXPECT
        with pytest.raises(
            UnitError,
            match="Input x has unit nm, but DampedHarmonicOscillator component has unit meV. Failed to convert DampedHarmonicOscillator to nm.",
        ):
            dho.evaluate(x)

    @pytest.mark.parametrize(
        "x, expected_message",
        [
            (np.array([0.0, np.nan, 1.0]), "Input x contains NaN values."),
            (np.array([0.0, np.inf, 1.0]), "Input x contains infinite values."),
        ],
    )
    def test_evaluate_with_invalid_input_raises(
        self, dho: DampedHarmonicOscillator, x, expected_message
    ):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=expected_message):
            dho.evaluate(x)

    def test_get_parameters(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        params = dho.get_parameters()

        # EXPECT
        assert len(params) == 3
        assert params[0].name == "TestDHO area"
        assert params[1].name == "TestDHO center"
        assert params[2].name == "TestDHO width"
        assert all(isinstance(param, Parameter) for param in params)

    def test_area_matches_parameter(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        x = np.linspace(
            -dho._center.value - 20 * dho._width.value,
            dho._center.value + 20 * dho._width.value,
            5000,
        )
        y = dho.evaluate(x)
        numerical_area = simpson(y, x)

        # EXPECT
        assert numerical_area == pytest.approx(dho._area.value, rel=2e-3)

    def test_convert_unit(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        dho.convert_unit("microeV")

        # EXPECT
        assert dho.unit == "microeV"
        assert dho.area.value == 2 * 1e3
        assert dho.center.value == 1.5 * 1e3
        assert dho.width.value == 0.3 * 1e3

    def test_copy(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        dho_copy = dho.copy()

        # EXPECT
        assert dho_copy is not dho
        assert dho_copy.name == "copy of " + dho.name

        assert dho_copy.area.value == dho.area.value
        assert dho_copy.area.fixed == dho.area.fixed

        assert dho_copy.center.value == dho.center.value
        assert dho_copy.center.fixed == dho.center.fixed

        assert dho_copy.width.value == dho.width.value
        assert dho_copy.width.fixed == dho.width.fixed

        assert dho_copy.unit == dho.unit

    def test_repr(self, dho: DampedHarmonicOscillator):
        # WHEN THEN
        repr_str = repr(dho)

        # EXPECT
        assert "DampedHarmonicOscillator" in repr_str
        assert "name = TestDHO" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str
