import pytest

import numpy as np
import scipp as sc
from scipp import UnitError

from scipy.integrate import simpson

from easydynamics.sample_model import Gaussian

from easyscience.variable import Parameter


class TestGaussian:
    @pytest.fixture
    def gaussian(self):
        return Gaussian(
            name="TestGaussian", area=2.0, center=0.5, width=0.6, unit="meV"
        )

    def test_initialization(self, gaussian: Gaussian):
        assert gaussian.name == "TestGaussian"
        assert gaussian._area.value == 2.0
        assert gaussian._center.value == 0.5
        assert gaussian._width.value == 0.6
        assert gaussian.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number"):
            Gaussian(
                name="TestGaussian", area="invalid", center=0.5, width=0.6, unit="meV"
            )
        with pytest.raises(TypeError, match="center must be None or a number"):
            Gaussian(
                name="TestGaussian", area=2.0, center="invalid", width=0.6, unit="meV"
            )
        with pytest.raises(TypeError, match="width must be a number"):
            Gaussian(
                name="TestGaussian", area=2.0, center=0.5, width="invalid", unit="meV"
            )
        with pytest.raises(TypeError, match="unit must be a string or a scipp unit"):
            Gaussian(name="TestGaussian", area=2.0, center=0.5, width=0.6, unit=123)

    def test_negative_width_raises(self):
        with pytest.raises(
            ValueError, match="The width of a Gaussian must be greater than zero."
        ):
            Gaussian(name="TestGaussian", area=2.0, center=0.5, width=-0.6, unit="meV")

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            Gaussian(name="TestGaussian", area=-2.0, center=0.5, width=0.6, unit="meV")

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

    def test_evaluate_with_incompatible_unit(self, gaussian: Gaussian):
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="nm")
        with pytest.raises(
            UnitError,
            match="Input x has unit nm, but Gaussian component has unit meV. Failed to convert Gaussian to nm.",
        ):
            gaussian.evaluate(x)

    def test_center_is_fixed_if_set_to_None(self):
        test_gaussian = Gaussian(
            name="TestGaussian", area=2.0, center=None, width=0.6, unit="meV"
        )
        assert test_gaussian._center.value == 0.0
        assert test_gaussian._center.fixed is True

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
            gaussian._center.value - 10 * gaussian._width.value,
            gaussian._center.value + 10 * gaussian._width.value,
            1000,
        )
        y = gaussian.evaluate(x)
        numerical_area = simpson(y, x)

        # THEN EXPECT
        assert np.isclose(numerical_area, gaussian._area.value, rtol=1e-3)

    def test_convert_unit(self, gaussian: Gaussian):
        gaussian.convert_unit("microeV")

        assert gaussian.unit == "microeV"
        assert gaussian._area.value == 2 * 1e3
        assert gaussian._center.value == 0.5 * 1e3
        assert gaussian._width.value == 0.6 * 1e3

    def test_copy(self, gaussian: Gaussian):
        gaussian_copy = gaussian.copy()
        assert gaussian_copy is not gaussian
        assert gaussian_copy.name == "copy of " + gaussian.name

        assert gaussian_copy._area.value == gaussian._area.value
        assert gaussian_copy._area.fixed == gaussian._area.fixed

        assert gaussian_copy._center.value == gaussian._center.value
        assert gaussian_copy._center.fixed == gaussian._center.fixed

        assert gaussian_copy._width.value == gaussian._width.value
        assert gaussian_copy._width.fixed == gaussian._width.fixed

        assert gaussian_copy.unit == gaussian.unit

    def test_repr(self, gaussian: Gaussian):
        repr_str = repr(gaussian)
        assert "Gaussian" in repr_str
        assert "name = TestGaussian" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str
