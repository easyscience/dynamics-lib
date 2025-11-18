import pytest

from easydynamics.sample_model.diffusion_model.diffusion_model import DiffusionModel


class TestDiffusionModel:
    @pytest.fixture
    def diffusion_model(self):
        return DiffusionModel(name="TestDiffusionModel", unit="meV")

    def test_init_default(self, diffusion_model):
        # WHEN THEN EXPECT
        assert diffusion_model.name == "TestDiffusionModel"
        assert diffusion_model.unit == "meV"

    def test_unit_setter_raises(self, diffusion_model):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match="Unit is read-only. Use convert_unit to change the unit between allowed types",
        ):
            diffusion_model.unit = "eV"
