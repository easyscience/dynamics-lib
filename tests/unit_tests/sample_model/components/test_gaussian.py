import numpy as np
import pytest
from easyscience.variable import Parameter
from scipy.integrate import simpson

from easydynamics.sample_model import Gaussian


class TestGaussian:
    @pytest.fixture
    def gaussian(self):
        return Gaussian(
            name="TestGaussian", area=2.0, center=0.5, width=0.6, unit="meV"
        )

    def test_initialization(self, gaussian: Gaussian):
        # WHEN THEN EXPECT
        assert gaussian.name == "TestGaussian"
        assert gaussian.area.value == 2.0
        assert gaussian.center.value == 0.5
        assert gaussian.width.value == 0.6
        assert gaussian.unit == "meV"

    def test_init_with_parameters(self):
        # WHEN
        area_param = Parameter(name="area_param", value=3.0, unit="meV")
        center_param = Parameter(name="center_param", value=1.0, unit="meV")
        width_param = Parameter(name="width_param", value=0.8, unit="meV")

        # THEN
        gaussian = Gaussian(
            name="ParamGaussian",
            area=area_param,
            center=center_param,
            width=width_param,
            unit="meV",
        )

        # EXPECT
        assert gaussian.name == "ParamGaussian"
        assert gaussian.area is area_param
        assert gaussian.center is center_param
        assert gaussian.width is width_param
        assert gaussian.unit == "meV"

    # @pytest.mark.parametrize(
    #     "kwargs, expected_message",
    #     [
    #         (
    #             {"area": "invalid", "center": 0.5, "width": 0.6, "unit": "meV"},
    #             "area must be a number",
    #         ),
    #         (
    #             {"area": 2.0, "center": "invalid", "width": 0.6, "unit": "meV"},
    #             "center must be None, a number",
    #         ),
    #         (
    #             {"area": 2.0, "center": 0.5, "width": "invalid", "unit": "meV"},
    #             "width must be a number",
    #         ),
    #         (
    #             {"area": 2.0, "center": 0.5, "width": 0.6, "unit": 123},
    #             "unit must be None",
    #         ),
    #     ],
    # )
    # def test_input_type_validation_raises(self, kwargs, expected_message):
    #     with pytest.raises(TypeError, match=expected_message):
    #         Gaussian(name="TestGaussian", **kwargs)

    # def test_negative_width_raises(self):
    #     # WHEN THEN EXPECT
    #     with pytest.raises(
    #         ValueError, match="The width of a Gaussian must be greater than zero."
    #     ):
    #         Gaussian(name="TestGaussian", area=2.0, center=0.5, width=-0.6, unit="meV")

    # def test_negative_area_warns(self):
    #     # WHEN THEN EXPECT
    #     with pytest.warns(UserWarning, match="may not be physically meaningful"):
    #         Gaussian(name="TestGaussian", area=-2.0, center=0.5, width=0.6, unit="meV")

    # @pytest.mark.parametrize(
    #     "prop, valid_value, invalid_value, invalid_message",
    #     [
    #         ("area", 3.0, "invalid", r"must be a number"),
    #         ("center", 0.6, "invalid", r"must be a number"),
    #         ("width", 0.7, "invalid", r"must be a number"),
    #     ],
    # )
    # def test_property_setters(
    #     self, gaussian: Gaussian, prop, valid_value, invalid_value, invalid_message
    # ):
    #     # set valid
    #     setattr(gaussian, prop, valid_value)
    #     assert getattr(gaussian, prop).value == valid_value

    #     # invalid
    #     with pytest.raises(TypeError, match=invalid_message):
    #         setattr(gaussian, prop, invalid_value)

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

    # def test_center_is_fixed_if_set_to_None(self):
    #     # WHEN THEN
    #     test_gaussian = Gaussian(
    #         name="TestGaussian", area=2.0, center=None, width=0.6, unit="meV"
    #     )
    #     # EXPECT
    #     assert test_gaussian.center.value == 0.0
    #     assert test_gaussian.center.fixed is True

    # def test_get_parameters(self, gaussian: Gaussian):
    #     # WHEN THEN
    #     params = gaussian.get_parameters()

    #     # EXPECT
    #     assert len(params) == 3
    #     assert all(isinstance(param, Parameter) for param in params)

    #     expected_names = {
    #         "TestGaussian area",
    #         "TestGaussian center",
    #         "TestGaussian width",
    #     }
    #     actual_names = {param.name for param in params}
    #     assert actual_names == expected_names

    def test_area_matches_parameter(self, gaussian: Gaussian):
        # WHEN
        x = np.linspace(
            gaussian.center.value - 10 * gaussian.width.value,
            gaussian.center.value + 10 * gaussian.width.value,
            1000,
        )

        # THEN
        y = gaussian.evaluate(x)

        # EXPECT
        numerical_area = simpson(y, x)
        assert np.isclose(numerical_area, gaussian.area.value, rtol=1e-3)

    # def test_convert_unit(self, gaussian: Gaussian):
    #     # WHEN THEN
    #     gaussian.convert_unit("microeV")

    #     # EXPECT
    #     assert gaussian.unit == "microeV"
    #     assert gaussian.area.value == 2 * 1e3
    #     assert gaussian.center.value == 0.5 * 1e3
    #     assert gaussian.width.value == 0.6 * 1e3

    # def test_copy(self, gaussian: Gaussian):
    #     # WHEN THEN
    #     gaussian_copy = copy(gaussian)
    #     # EXPECT
    #     assert gaussian_copy is not gaussian
    #     assert gaussian_copy.name == gaussian.name

    #     assert gaussian_copy.area.value == gaussian.area.value
    #     assert gaussian_copy.area.fixed == gaussian.area.fixed

    #     assert gaussian_copy.center.value == gaussian.center.value
    #     assert gaussian_copy.center.fixed == gaussian.center.fixed

    #     assert gaussian_copy.width.value == gaussian.width.value
    #     assert gaussian_copy.width.fixed == gaussian.width.fixed

    #     assert gaussian_copy.unit == gaussian.unit

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
