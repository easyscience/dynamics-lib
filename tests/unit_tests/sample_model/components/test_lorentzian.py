import pytest

import numpy as np
import scipp as sc

from scipy.integrate import simpson

from easydynamics.sample_model import Lorentzian
from easyscience.variable import Parameter


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

    def test_evaluate_with_incompatible_unit(self, lorentzian: Lorentzian):
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="nm")
        with pytest.raises(
            ValueError,
            match="Input x has unit nm, but Lorentzian component has unit meV. Failed to convert Lorentzian to nm.",
        ):
            lorentzian.evaluate(x)

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
