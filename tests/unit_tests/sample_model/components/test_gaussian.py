import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson

from easydynamics.sample_model import Gaussian


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

    def test_input_type_validation_area_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="area must be a number"):
            Gaussian(
                name="TestGaussian", area="invalid", center=0.5, width=0.6, unit="meV"
            )

    def test_input_type_validation_center_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="center must be None or a number"):
            Gaussian(
                name="TestGaussian", area=2.0, center="invalid", width=0.6, unit="meV"
            )

    def test_input_type_validation_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="width must be a number"):
            Gaussian(
                name="TestGaussian", area=2.0, center=0.5, width="invalid", unit="meV"
            )

    def test_input_type_validation_unit_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match="unit must be a string or a scipp unit"):
            Gaussian(name="TestGaussian", area=2.0, center=0.5, width=0.6, unit=123)

    def test_negative_width_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError, match="The width of a Gaussian must be greater than zero."
        ):
            Gaussian(name="TestGaussian", area=2.0, center=0.5, width=-0.6, unit="meV")

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            Gaussian(name="TestGaussian", area=-2.0, center=0.5, width=0.6, unit="meV")

    def test_area_property_getter(self, gaussian: Gaussian):
        # WHEN THEN EXPECT
        assert gaussian.area.value == 2.0

    def test_area_property_setter(self, gaussian: Gaussian):
        # WHEN
        gaussian.area = 3.0

        # THEN EXPECT
        assert gaussian.area.value == 3.0
        with pytest.raises(TypeError, match="area must be a number."):
            gaussian.area = "invalid"

    def test_center_property_getter(self, gaussian: Gaussian):
        # WHEN THEN EXPECT
        assert gaussian.center.value == 0.5

    def test_center_property_setter(self, gaussian: Gaussian):
        # WHEN
        gaussian.center = 0.6

        # THEN EXPECT
        assert gaussian.center.value == 0.6
        with pytest.raises(TypeError, match="center must be a number."):
            gaussian.center = "invalid"

    def test_width_property_getter(self, gaussian: Gaussian):
        # WHEN THEN EXPECT
        assert gaussian.width.value == 0.6

    def test_width_property_setter(self, gaussian: Gaussian):
        # WHEN
        gaussian.width = 0.7

        # THEN EXPECT
        assert gaussian.width.value == 0.7
        with pytest.raises(TypeError, match="width must be a number."):
            gaussian.width = "invalid"

    def test_evaluate(self, gaussian: Gaussian):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = gaussian.evaluate(x)

        # EXPECT
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - 0.5) / 0.6) ** 2
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, gaussian: Gaussian):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")

        # THEN
        result = gaussian.evaluate(x)

        # EXPECT
        expected_result = (2.0 / (0.6 * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x.values - 0.5) / 0.6) ** 2
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, gaussian: Gaussian):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="microeV")

        # THEN
        result = gaussian.evaluate(x)

        # EXPECT
        expected_result = (2.0 * 1e3 / (0.6 * 1e3 * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x.values - 500.0) / (0.6 * 1e3)) ** 2
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_incompatible_unit(self, gaussian: Gaussian):
        # WHEN THEN
        x = sc.array(dims=["x"], values=[0.0, 500.0, 1000.0], unit="nm")

        # EXPECT
        with pytest.raises(
            UnitError,
            match="Input x has unit nm, but Gaussian component has unit meV. Failed to convert Gaussian to nm.",
        ):
            gaussian.evaluate(x)

    def test_evaluate_with_nan_input(self, gaussian: Gaussian):
        # WHEN THEN
        x = np.array([0.0, np.nan, 1.0])

        # EXPECT
        with pytest.raises(ValueError, match="Input x contains NaN values."):
            gaussian.evaluate(x)

    def test_evaluate_with_infinite_input(self, gaussian: Gaussian):
        # WHEN THEN
        x = np.array([0.0, np.inf, 1.0])

        # EXPECT
        with pytest.raises(ValueError, match="Input x contains infinite values."):
            gaussian.evaluate(x)

    def test_center_is_fixed_if_set_to_None(self):
        # WHEN THEN
        test_gaussian = Gaussian(
            name="TestGaussian", area=2.0, center=None, width=0.6, unit="meV"
        )
        # EXPECT
        assert test_gaussian._center.value == 0.0
        assert test_gaussian._center.fixed is True

    def test_get_parameters(self, gaussian: Gaussian):
        # WHEN THEN
        params = gaussian.get_parameters()

        # EXPECT
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

        # THEN
        y = gaussian.evaluate(x)

        # EXPECT
        numerical_area = simpson(y, x)
        assert np.isclose(numerical_area, gaussian._area.value, rtol=1e-3)

    def test_convert_unit(self, gaussian: Gaussian):
        # WHEN THEN
        gaussian.convert_unit("microeV")

        # EXPECT
        assert gaussian.unit == "microeV"
        assert gaussian._area.value == 2 * 1e3
        assert gaussian._center.value == 0.5 * 1e3
        assert gaussian._width.value == 0.6 * 1e3

    def test_copy(self, gaussian: Gaussian):
        # WHEN THEN
        gaussian_copy = gaussian.copy()
        # EXPECT
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
        # WHEN THEN
        repr_str = repr(gaussian)
        # EXPECT
        assert "Gaussian" in repr_str
        assert "name = TestGaussian" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
        assert "width =" in repr_str
