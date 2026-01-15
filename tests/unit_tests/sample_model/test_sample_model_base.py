import numpy as np
import pytest

from easydynamics.sample_model import (
    ComponentCollection,
    Gaussian,
    Lorentzian,
)
from easydynamics.sample_model.sample_model_base import SampleModelBase


class TestSampleModelBase:
    @pytest.fixture
    def sample_model_base(self):
        component1 = Gaussian(
            display_name="TestGaussian1", area=1.0, center=0.0, width=1.0, unit="meV"
        )
        component2 = Lorentzian(
            display_name="TestLorentzian1", area=2.0, center=1.0, width=0.5, unit="meV"
        )
        component_collection = ComponentCollection()
        component_collection.append_component(component1)
        component_collection.append_component(component2)
        sample_model_base = SampleModelBase(
            display_name="InitModel",
            components=component_collection,
            unit="meV",
            Q=np.array([1.0, 2.0, 3.0]),
        )

        return sample_model_base

    def test_init(self, sample_model_base):
        # WHEN THEN
        model = sample_model_base

        # EXPECT
        assert model.display_name == "InitModel"
        assert model.unit == "meV"
        assert len(model.components) == 2
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))
