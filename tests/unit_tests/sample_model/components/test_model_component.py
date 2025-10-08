import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent


class TestModelComponent:
    class DummyComponent(ModelComponent):
        def __init__(self):
            super().__init__(name="Dummy")
            self._area = Parameter(name="area", value=1.0, unit="meV")
            self._center = Parameter(name="center", value=2.0, unit="meV", fixed=True)
            self._width = Parameter(name="width", value=3.0, unit="meV", fixed=True)
            self._unit = "meV"

        def get_parameters(self):
            return [self._area, self._center, self._width]

        def evaluate(self, x):
            return np.zeros_like(x)

        def convert_unit(self, unit):
            self._area.convert_unit(unit)
            self._center.convert_unit(unit)
            self._width.convert_unit(unit)
            self._unit = unit

    @pytest.fixture
    def dummy(self):
        return self.DummyComponent()

    def test_unit_cannot_be_set_directly(self, dummy: ModelComponent):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match="Unit is read-only"):
            dummy.unit = "K"

    def test_fix_all_parameters_sets_all_to_fixed(self, dummy):
        # WHEN
        dummy.fix_all_parameters()

        # THEN EXPECT
        assert all(p.fixed for p in dummy.get_parameters())

    def test_free_all_parameters_sets_all_to_unfixed(self, dummy):
        # WHEN
        dummy.free_all_parameters()

        # THEN EXPECT
        assert all(not p.fixed for p in dummy.get_parameters())

    def test_repr(self, dummy):
        # WHEN THEN EXPECT
        repr_str = repr(dummy)
        assert "DummyComponent" in repr_str

    def test_prepare_x_for_evaluate_with_numeric(self, dummy):
        # WHEN THEN
        x_prepared = dummy._prepare_x_for_evaluate(5.0)

        # EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (1,)
        assert x_prepared == 5.0

    def test_prepare_x_for_evaluate_with_list(self, dummy):
        # WHEN
        x = [1.0, 2.0, 3.0]

        # THEN
        x_prepared = dummy._prepare_x_for_evaluate(x)

        # EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (3,)
        np.testing.assert_array_equal(x_prepared, [1.0, 2.0, 3.0])

    def test_prepare_x_for_evaluate_with_numpy_array(self, dummy):
        # WHEN
        x = np.array([1.0, 2.0, 3.0])

        # THEN
        x_prepared = dummy._prepare_x_for_evaluate(x)

        # THEN EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (3,)
        np.testing.assert_array_equal(x_prepared, [1.0, 2.0, 3.0])

    def test_prepare_x_for_evaluate_with_scipp_scalar(self, dummy):
        # WHEN
        x_scipp = sc.scalar(5.0, unit="meV")

        # THEN
        x_prepared = dummy._prepare_x_for_evaluate(x_scipp)

        # EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (1,)
        assert x_prepared == 5.0

    def test_prepare_x_for_evaluate_with_scipp_variable(self, dummy):
        # WHEN
        x_scipp = sc.array(dims=["x"], values=[1.0, 2.0, 3.0], unit="meV")

        # THEN
        x_prepared = dummy._prepare_x_for_evaluate(x_scipp)

        # EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (3,)
        np.testing.assert_array_equal(x_prepared, [1.0, 2.0, 3.0])

    def test_prepare_x_for_evaluate_with_scipp_data_array(self, dummy):
        # WHEN
        x_scipp = sc.array(dims=["x"], values=[1.0, 2.0, 3.0], unit="meV")
        data_array = sc.DataArray(data=x_scipp, coords={"x": x_scipp})

        # THEN
        x_prepared = dummy._prepare_x_for_evaluate(data_array)

        # EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (3,)
        np.testing.assert_array_equal(x_prepared, [1.0, 2.0, 3.0])

    def test_prepare_x_for_evaluate_with_scipp_data_array_multiple_coords_raises(
        self, dummy
    ):
        # WHEN
        x = sc.array(dims=["x"], values=[0.0, 0.5, 1.0], unit="meV")
        y = sc.array(dims=["y"], values=[0.0, 1.0, 2.0], unit="meV")
        var = sc.array(
            dims=["x", "y"],
            values=[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
        )
        array = sc.DataArray(data=var, coords={"x": x, "y": y})

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match="must have exactly one coordinate",
        ):
            dummy._prepare_x_for_evaluate(array)

    def test_prepare_x_for_evaluate_with_nan_raises(self, dummy):
        # WHEN
        x = np.array([1.0, np.nan, 3.0])

        # THEN EXPECT
        with pytest.raises(ValueError, match="contains NaN values"):
            dummy._prepare_x_for_evaluate(x)

    def test_prepare_x_for_evaluate_with_infinite_raises(self, dummy):
        # WHEN
        x = np.array([1.0, np.inf, 3.0])

        # THEN EXPECT
        with pytest.raises(ValueError, match="contains infinite values"):
            dummy._prepare_x_for_evaluate(x)

    def test_prepare_x_for_evaluate_with_incompatible_unit_raises(self, dummy):
        # WHEN
        x = sc.array(dims=["x"], values=[1.0, 2.0, 3.0], unit="nm")

        # THEN EXPECT
        with pytest.raises(
            Exception,
            match="Input x has unit nm, but DummyComponent component has unit meV. Failed to convert DummyComponent to nm.",
        ):
            dummy._prepare_x_for_evaluate(x)

    def test_prepare_x_for_evaluate_with_different_unit_warns(self, dummy):
        # WHEN
        x = sc.array(dims=["x"], values=[1.0, 2.0, 3.0], unit="microeV")

        # THEN EXPECT
        with pytest.warns(
            UserWarning,
            match="Input x has unit µeV, but DummyComponent component has unit meV. Converting DummyComponent to µeV.",
        ):
            x_prepared = dummy._prepare_x_for_evaluate(x)

        # EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (3,)
        np.testing.assert_array_equal(x_prepared, [1.0, 2.0, 3.0])
        assert dummy.unit == "µeV"
        assert dummy._area.value == 1.0 * 1e3
        assert dummy._center.value == 2.0 * 1e3
        assert dummy._width.value == 3.0 * 1e3
