import numpy as np
import pytest
import scipp as sc
from scipy.signal import fftconvolve

from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.convolution.numerical_convolution import NumericalConvolution
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.utils.detailed_balance import _detailed_balance_factor as detailed_balance_factor


class TestNumericalConvolution:
    @pytest.fixture
    def default_numerical_convolution(self):
        energy = np.linspace(-10, 10, 5001)
        sample_components = ComponentCollection(display_name='ComponentCollection')
        sample_components.append_component(
            Gaussian(display_name='Gaussian1', area=2.0, center=0.1, width=0.4)
        )
        resolution_components = ComponentCollection(display_name='ResolutionModel')
        resolution_components.append_component(
            Gaussian(display_name='GaussianRes', area=3.0, center=0.2, width=0.5)
        )

        return NumericalConvolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

    def test_init(self, default_numerical_convolution):
        """
        Test initialization of NumericalConvolution with
        default parameters.
        """
        # WHEN THEN EXPECT
        assert isinstance(default_numerical_convolution, NumericalConvolution)
        assert isinstance(default_numerical_convolution.energy, sc.Variable)
        assert np.allclose(default_numerical_convolution.energy.values, np.linspace(-10, 10, 5001))
        assert isinstance(default_numerical_convolution._sample_components, ComponentCollection)
        assert isinstance(
            default_numerical_convolution._resolution_components, ComponentCollection
        )
        assert default_numerical_convolution.upsample_factor == 5
        assert default_numerical_convolution.extension_factor == 0.2
        assert default_numerical_convolution.temperature is None
        assert default_numerical_convolution.energy_unit == 'meV'
        assert default_numerical_convolution.normalize_detailed_balance is True
        assert isinstance(default_numerical_convolution._energy_grid, EnergyGrid)

    @pytest.mark.parametrize('upsample_factor', [None, 5])
    def test_convolution(self, default_numerical_convolution, upsample_factor):
        """
        Test that convolution of two Gaussians produces the
        expected result.
        """
        # WHEN THEN
        default_numerical_convolution.upsample_factor = upsample_factor
        result = default_numerical_convolution.convolution()

        # EXPECT
        expected_area = 2.0 * 3.0  # area of sample_components * area of resolution_components
        expected_center = (
            0.1 + 0.2
        )  # center of sample_components + center of resolution_components
        expected_width = np.sqrt(0.4**2 + 0.5**2)  # sqrt(width_sample^2 + width_res^2)
        expected_result = Gaussian(
            display_name='ExpectedConvolution',
            area=expected_area,
            center=expected_center,
            width=expected_width,
        ).evaluate(default_numerical_convolution.energy)
        assert np.allclose(result, expected_result, rtol=1e-4)

    def test_convolution_with_temperature(
        self,
        default_numerical_convolution,
    ):
        """
        Test that convolution includes detailed balance correction
        when temperature is provided.
        """

        # WHEN
        default_numerical_convolution.temperature = 5.0  # Kelvin

        # THEN
        result = default_numerical_convolution.convolution()

        # EXPECT
        sample_valds = default_numerical_convolution._sample_components.evaluate(
            default_numerical_convolution.energy.values
        )
        resolution_vals = default_numerical_convolution._resolution_components.evaluate(
            default_numerical_convolution.energy.values
        )
        DBF = detailed_balance_factor(energy=default_numerical_convolution.energy, temperature=5.0)
        expected_result = fftconvolve(sample_valds * DBF, resolution_vals, mode='same') * (
            default_numerical_convolution.energy.values[1]
            - default_numerical_convolution.energy.values[0]
        )

        assert np.allclose(result, expected_result, rtol=1e-4)
