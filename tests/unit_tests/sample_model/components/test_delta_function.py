import pytest

import numpy as np
import scipp as sc
from scipp import UnitError
from easydynamics.sample_model import DeltaFunction
from easyscience.variable import Parameter


class TestDeltaFunction:
    @pytest.fixture
    def delta_function(self):
        return DeltaFunction(name="TestDeltaFunction", area=2.0, center=0.5, unit="meV")

    def test_initialization(self, delta_function: DeltaFunction):
        assert delta_function.name == "TestDeltaFunction"
        assert delta_function._area.value == 2.0
        assert delta_function._center.value == 0.5
        assert delta_function.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number"):
            DeltaFunction(
                name="TestDeltaFunction",
                area="invalid",
                center=0.5,
                unit="meV",
            )
        with pytest.raises(TypeError, match="center must be None or a number"):
            DeltaFunction(
                name="TestDeltaFunction",
                area=2.0,
                center="invalid",
                unit="meV",
            )
        with pytest.raises(TypeError, match="unit must be a string or a scipp unit"):
            DeltaFunction(name="TestDeltaFunction", area=2.0, center=0.5, unit=123)

    def test_negative_area_warns(self):
        with pytest.warns(UserWarning, match="may not be physically meaningful"):
            DeltaFunction(name="TestDeltaFunction", area=-2.0, center=0.5, unit="meV")

    def test_area_property_getter(self, delta_function: DeltaFunction):
        assert delta_function.area.value == 2.0

    def test_area_property_setter(self, delta_function: DeltaFunction):
        # WHEN
        delta_function.area = 3.0

        # THEN EXPECT
        assert delta_function.area.value == 3.0
        with pytest.raises(TypeError, match="area must be a number."):
            delta_function.area = "invalid"

    def test_center_property_getter(self, delta_function: DeltaFunction):
        # WHEN THEN EXPECT
        assert delta_function.center.value == 0.5

    def test_center_property_setter(self, delta_function: DeltaFunction):
        # WHEN
        delta_function.center = 0.6

        # THEN EXPECT
        assert delta_function.center.value == 0.6
        with pytest.raises(TypeError, match="center must be a number."):
            delta_function.center = "invalid"

    def test_evaluate(self, delta_function: DeltaFunction):
        # WHEN
        x = np.array([0.0, 0.5, 1.0])

        # THEN
        result = delta_function.evaluate(x)

        # EXPECT
        expected_result = np.zeros_like(x)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_scipp_array(self, delta_function: DeltaFunction):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")

        # THEN
        result = delta_function.evaluate(x)

        # EXPECT
        expected_result = np.zeros_like(x.values)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_different_unit(self, delta_function: DeltaFunction):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="microeV")

        # THEN
        result = delta_function.evaluate(x)

        # EXPECT
        expected_result = np.zeros_like(x.values)
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

    def test_evaluate_with_incompatible_unit(self, delta_function: DeltaFunction):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="nm")
        # THEN EXPECT
        with pytest.raises(
            UnitError,
            match="Input x has unit nm, but DeltaFunction component has unit meV. Failed to convert DeltaFunction to nm.",
        ):
            delta_function.evaluate(x)

    def test_evaluate_with_nan_input(self, delta_function: DeltaFunction):
        # WHEN
        x = np.array([0.0, np.nan, 1.0])

        # THEN EXPECT
        with pytest.raises(ValueError, match="Input x contains NaN values."):
            delta_function.evaluate(x)

    def test_evaluate_with_infinite_input(self, delta_function: DeltaFunction):
        # WHEN
        x = np.array([0.0, np.inf, 1.0])

        # THEN EXPECT
        with pytest.raises(ValueError, match="Input x contains infinite values."):
            delta_function.evaluate(x)

    def test_center_is_fixed_if_set_to_None(self):
        # WHEN THEN
        test_delta = DeltaFunction(
            name="TestDeltaFunction", area=2.0, center=None, unit="meV"
        )

        # EXPECT
        assert test_delta._center.value == 0.0
        assert test_delta._center.fixed is True

    def test_get_parameters(self, delta_function: DeltaFunction):
        # WHEN THEN
        params = delta_function.get_parameters()

        # EXPECT
        assert len(params) == 2
        assert params[0].name == "TestDeltaFunction area"
        assert params[1].name == "TestDeltaFunction center"
        assert all(isinstance(param, Parameter) for param in params)

    def test_convert_unit(self, delta_function: DeltaFunction):
        # WHEN THEN
        delta_function.convert_unit("microeV")

        # EXPECT
        assert delta_function.unit == "microeV"
        assert delta_function._area.value == 2 * 1e3
        assert delta_function._center.value == 0.5 * 1e3

    def test_copy(self, delta_function: DeltaFunction):
        # WHEN THEN
        delta_copy = delta_function.copy()

        # EXPECT
        assert delta_copy is not delta_function
        assert delta_copy.name == "copy of " + delta_function.name

        assert delta_copy._area.value == delta_function._area.value
        assert delta_copy._area.fixed == delta_function._area.fixed

        assert delta_copy._center.value == delta_function._center.value
        assert delta_copy._center.fixed == delta_function._center.fixed

        assert delta_copy.unit == delta_function.unit

    def test_repr(self, delta_function: DeltaFunction):
        # WHEN THEN
        repr_str = repr(delta_function)

        # EXPECT
        assert "DeltaFunction" in repr_str
        assert "name = TestDeltaFunction" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
