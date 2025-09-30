import pytest

import numpy as np

from easydynamics.sample_model import DeltaFunction
from easyscience.variable import Parameter


class TestDeltaFunction:
    @pytest.fixture
    def delta_function(self):
        return DeltaFunction(name="TestDeltaFunction", area=2.0, center=0.5, unit="meV")

    def test_initialization(self, delta_function: DeltaFunction):
        assert delta_function.name == "TestDeltaFunction"
        assert delta_function.area.value == 2.0
        assert delta_function.center.value == 0.5
        assert delta_function.unit == "meV"

    def test_input_type_validation_raises(self):
        with pytest.raises(TypeError, match="area must be a number or a Parameter"):
            DeltaFunction(
                name="TestDeltaFunction",
                area="invalid",
                center=0.5,
                unit="meV",
            )
        with pytest.raises(
            TypeError, match="center must be None, a number or a Parameter"
        ):
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

    @pytest.mark.xfail(
        reason="DeltaFunction.evaluate is not implemented yet without resolution convolution"
    )
    def test_evaluate(self, delta_function: DeltaFunction):
        x = np.array([0.0, 0.5, 1.0])
        expected = delta_function.evaluate(x)
        expected_result = np.zeros_like(x)
        # expected_result[x == 0.5] = 2.0
        np.testing.assert_allclose(expected, expected_result, rtol=1e-5)

    def test_center_is_fixed_if_set_to_None(self):
        test_delta = DeltaFunction(
            name="TestDeltaFunction", area=2.0, center=None, unit="meV"
        )
        assert test_delta.center.value == 0.0
        assert test_delta.center.fixed is True

    def test_input_as_parameter(self):
        param_area = Parameter(name="area_param", value=2.0, unit="meV")
        param_center = Parameter(name="center_param", value=0.5, unit="meV")
        test_delta = DeltaFunction(
            name="TestDeltaFunction", area=param_area, center=param_center, unit="meV"
        )
        assert test_delta.area == param_area
        assert test_delta.center == param_center

    def test_get_parameters(self, delta_function: DeltaFunction):
        params = delta_function.get_parameters()
        assert len(params) == 2
        assert params[0].name == "TestDeltaFunction area"
        assert params[1].name == "TestDeltaFunction center"
        assert all(isinstance(param, Parameter) for param in params)

    def test_convert_unit(self, delta_function: DeltaFunction):
        delta_function.convert_unit("microeV")

        assert delta_function.unit == "microeV"
        assert delta_function.area.value == 2 * 1e3
        assert delta_function.center.value == 0.5 * 1e3

    def test_copy(self, delta_function: DeltaFunction):
        delta_copy = delta_function.copy()
        assert delta_copy is not delta_function
        assert delta_copy.name == delta_function.name

        assert delta_copy.area.value == delta_function.area.value
        assert delta_copy.area.fixed == delta_function.area.fixed

        assert delta_copy.center.value == delta_function.center.value
        assert delta_copy.center.fixed == delta_function.center.fixed

        assert delta_copy.unit == delta_function.unit

    def test_repr(self, delta_function: DeltaFunction):
        repr_str = repr(delta_function)
        assert "DeltaFunction" in repr_str
        assert "name = TestDeltaFunction" in repr_str
        assert "unit = meV" in repr_str
        assert "area =" in repr_str
        assert "center =" in repr_str
