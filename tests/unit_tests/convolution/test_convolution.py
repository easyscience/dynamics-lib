import numpy as np
import pytest
import scipp as sc

from easydynamics.convolution.convolution import (
    Convolution,
)
from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.sample_model import (
    DampedHarmonicOscillator,
    DeltaFunction,
    Gaussian,
    SampleModel,
)


class TestConvolution:
    @pytest.fixture
    def default_convolution(self):
        energy = np.linspace(-10, 10, 5001)
        sample_model = SampleModel(name="SampleModel")

        sample_model.add_component(
            Gaussian(name="Gaussian1", area=2.0, center=0.1, width=0.4)
        )

        sample_model.add_component(
            DampedHarmonicOscillator(name="DHO1", area=2.0, center=1.0, width=0.1)
        )

        sample_model.add_component(DeltaFunction(name="Delta1", area=2.0, center=0.3))

        resolution_model = SampleModel(name="ResolutionModel")
        resolution_model.add_component(
            Gaussian(name="GaussianRes", area=3.0, center=0.2, width=0.5)
        )

        return Convolution(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
        )

    def test_init(self, default_convolution):
        "Test initialization of Convolution with default parameters."
        # WHEN THEN EXPECT
        assert isinstance(default_convolution, Convolution)
        assert isinstance(default_convolution.energy, sc.Variable)
        assert np.allclose(
            default_convolution.energy.values, np.linspace(-10, 10, 5001)
        )
        assert isinstance(default_convolution._sample_model, SampleModel)
        assert isinstance(default_convolution._resolution_model, SampleModel)
        assert default_convolution.upsample_factor == 5
        assert default_convolution.extension_factor == 0.2
        assert default_convolution.temperature is None
        assert default_convolution.energy_unit == "meV"
        assert default_convolution.normalize_detailed_balance is True
        assert isinstance(default_convolution._energy_grid, EnergyGrid)

        assert isinstance(default_convolution._analytical_sample_model, SampleModel)
        assert (
            default_convolution._analytical_sample_model.components[0]
            is default_convolution.sample_model.components[0]
        )
        assert isinstance(default_convolution._numerical_sample_model, SampleModel)
        assert (
            default_convolution._numerical_sample_model.components[0]
            is default_convolution.sample_model.components[1]
        )

        assert isinstance(default_convolution._delta_sample_model, SampleModel)
        assert (
            default_convolution._delta_sample_model.components[0]
            is default_convolution.sample_model.components[2]
        )
        assert default_convolution._convolution_plan_is_valid is True
        assert default_convolution._reactions_enabled is True

    def test_plan_is_built_when_invalid(mocker, default_convolution):
        conv = default_convolution
        conv._convolution_plan_is_valid = False

        build_plan = mocker.spy(conv, "_build_convolution_plan")

        conv.convolution()

        build_plan.assert_called_once()
