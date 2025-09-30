import pytest

import numpy as np
import scipp as sc
from scipp import UnitError

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
        assert dho._name == "TestDHO"
        assert dho._area.value == 2.0
        assert dho._center.value == 1.5
        assert dho._width.value == 0.3
        assert dho._unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area="invalid",
                center=0.5,
                width=0.6,
                unit="meV",
            )

        with pytest.raises(TypeError, match="center must be a number"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=2.0,
                center="invalid",
                width=0.6,
                unit="meV",
            )

        with pytest.raises(TypeError, match="width must be a number"):
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

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            DampedHarmonicOscillator(
                name="TestDampedHarmonicOscillator",
                area=-2.0,
                center=0.5,
                width=0.6,
                unit="meV",
            )

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

    def test_evaluate_with_incompatible_unit(self, dho: DampedHarmonicOscillator):
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="nm")
        with pytest.raises(
            UnitError,
            match="Input x has unit nm, but DampedHarmonicOscillator component has unit meV. Failed to convert DampedHarmonicOscillator to nm.",
        ):
            dho.evaluate(x)

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
            -dho._center.value - 20 * dho._width.value,
            dho._center.value + 20 * dho._width.value,
            5000,
        )
        y = dho.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert numerical_area == pytest.approx(dho._area.value, rel=2e-3)

    def test_convert_unit(self, dho: DampedHarmonicOscillator):
        dho.convert_unit("microeV")

        assert dho._unit == "microeV"
        assert dho._area.value == 2 * 1e3
        assert dho._center.value == 1.5 * 1e3
        assert dho._width.value == 0.3 * 1e3

    def test_copy(self, dho: DampedHarmonicOscillator):
        dho_copy = dho.copy()
        assert dho_copy is not dho
        assert dho_copy.name == "copy of " + dho._name

        assert dho_copy._area.value == dho._area.value
        assert dho_copy._area.fixed == dho._area.fixed

        assert dho_copy._center.value == dho._center.value
        assert dho_copy._center.fixed == dho._center.fixed

        assert dho_copy._width.value == dho._width.value
        assert dho_copy._width.fixed == dho._width.fixed

        assert dho_copy._unit == dho._unit

    def test_repr(self, dho: DampedHarmonicOscillator):
        repr_str = repr(dho)
        assert "DampedHarmonicOscillator" in repr_str
        assert "name = TestDHO" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str
