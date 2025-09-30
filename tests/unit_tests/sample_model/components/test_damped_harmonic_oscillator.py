import pytest

import numpy as np
import scipp as sc

from scipy.integrate import simpson

from easydynamics.sample_model import DampedHarmonicOscillator

from easyscience.variable import Parameter


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

        with pytest.raises(TypeError, match="unit must be a string or a scipp unit"):
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
