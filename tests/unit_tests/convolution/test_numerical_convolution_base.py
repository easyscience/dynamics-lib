import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.convolution.numerical_convolution_base import (
    NumericalConvolutionBase,
)
from easydynamics.sample_model import SampleModel


class TestNumericalConvolutionBase:
    @pytest.fixture
    def default_numerical_convolution_base(self):
        energy = np.linspace(-10, 10, 100)
        sample_model = SampleModel(name="SampleModel")
        resolution_model = SampleModel(name="ResolutionModel")

        return NumericalConvolutionBase(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
        )

    def test_init(self, default_numerical_convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(default_numerical_convolution_base, NumericalConvolutionBase)
        assert isinstance(default_numerical_convolution_base.energy, sc.Variable)
        assert np.allclose(
            default_numerical_convolution_base.energy.values, np.linspace(-10, 10, 100)
        )
        assert isinstance(default_numerical_convolution_base._sample_model, SampleModel)
        assert isinstance(
            default_numerical_convolution_base._resolution_model, SampleModel
        )
        assert isinstance(default_numerical_convolution_base.offset, Parameter)
        assert default_numerical_convolution_base.offset.value == 0.0
        assert default_numerical_convolution_base.offset.unit == "meV"
        assert default_numerical_convolution_base.upsample_factor == 5
        assert default_numerical_convolution_base.extension_factor == 0.2
        assert default_numerical_convolution_base.temperature is None
        # assert default_numerical_convolution_base.temperature_unit == "K"
        assert default_numerical_convolution_base.energy_unit == "meV"
        assert default_numerical_convolution_base.normalize_detailed_balance is True
        assert isinstance(default_numerical_convolution_base._energy_grid, EnergyGrid)
