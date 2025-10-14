import numpy as np
import pytest
from easyscience.variable import Parameter
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

    @pytest.mark.parametrize(
        "prop, valid_value, invalid_value, invalid_message",
        [
            ("area", 3.0, "invalid", r"area must be a number\."),
            ("center", 0.6, "invalid", r"center must be a number\."),
            ("width", 0.7, "invalid", r"width must be a number\."),
        ],
    )
    def test_property_setters(
        self, lorentzian: Lorentzian, prop, valid_value, invalid_value, invalid_message
    ):
        # set valid
        setattr(lorentzian, prop, valid_value)
        assert getattr(lorentzian, prop).value == valid_value

        # invalid
        with pytest.raises(TypeError, match=invalid_message):
            setattr(lorentzian, prop, invalid_value)

    def test_evaluate(self, lorentzian: Lorentzian):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = lorentzian.evaluate(x)

        # EXPECT
        expected_result = (2.0 / (np.pi * 0.6)) / (1 + ((x - 0.5) / 0.6) ** 2)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

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
