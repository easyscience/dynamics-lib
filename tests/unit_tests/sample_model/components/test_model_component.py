import pytest

import numpy as np

from easydynamics.sample_model.components.model_component import ModelComponent

from easyscience.variable import Parameter


class TestModelComponent:
    class DummyComponent(ModelComponent):
        def __init__(self):
            super().__init__(name="Dummy")
            self.area = Parameter(name="area", value=1.0, unit="meV")
            self.center = Parameter(name="center", value=2.0, unit="meV", fixed=True)
            self.width = Parameter(name="width", value=3.0, unit="meV", fixed=True)
            self.second_area = Parameter(name="second_area", value=4.0, unit="meV")

        def get_parameters(self):
            return [self.area, self.center, self.width, self.second_area]

        def evaluate(self, x):
            return np.zeros_like(x)

    @pytest.fixture
    def dummy(self):
        return self.DummyComponent()

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

    def test_get_parameter_exact_match(self, dummy):
        # WHEN
        param = dummy.get_parameter("width")

        # THEN EXPECT
        assert param is dummy.width

    def test_get_parameter_partial_match(self, dummy):
        # WHEN
        param = dummy.get_parameter("wid")

        # THEN EXPECT
        assert param is dummy.width

    def test_get_parameter_no_match_raises(self, dummy):
        # WHEN / THEN EXPECT
        with pytest.raises(
            ValueError,
            match="not found",
        ):
            dummy.get_parameter("nonexistent")

    def test_get_parameter_ambiguous_match_raises(self, dummy):
        # WHEN / THEN EXPECT
        with pytest.raises(ValueError, match="Ambiguous parameter name"):
            dummy.get_parameter("are")

    def test_repr(self, dummy):
        repr_str = repr(dummy)
        assert "DummyComponent" in repr_str
