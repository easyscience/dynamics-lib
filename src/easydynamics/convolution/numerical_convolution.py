# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.signal import fftconvolve

from easydynamics.convolution.convolution_settings import ConvolutionSettings
from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.detailed_balance_settings import DetailedBalanceSettings
from easydynamics.utils.detailed_balance import detailed_balance_factor
from easydynamics.utils.utils import Numeric


class NumericalConvolution(NumericalConvolutionBase):
    """
    Numerical convolution of a ComponentCollection with a ComponentCollection using FFT.

    Includes optional upsampling and extended range to improve accuracy. Warns about very wide or
    very narrow peaks in the models. If temperature is provided, detailed balance correction is
    applied to the sample model.
    """

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_components: ComponentCollection | ModelComponent,
        resolution_components: ComponentCollection | ModelComponent,
        energy_offset: Numeric | Parameter = 0.0,
        convolution_settings: ConvolutionSettings | None = None,
        temperature: Parameter | Numeric | None = None,
        temperature_unit: str | sc.Unit = 'K',
        detailed_balance_settings: DetailedBalanceSettings | None = None,
        unit: str | sc.Unit = 'meV',
        display_name: str | None = 'MyConvolution',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the NumericalConvolution object.

        Parameters
        ----------
        energy : np.ndarray | sc.Variable
            1D array of energy values where the convolution is evaluated.
        sample_components : ComponentCollection | ModelComponent
            The sample model to be convolved.
        resolution_components : ComponentCollection | ModelComponent
            The resolution model to convolve with.
        energy_offset : Numeric | Parameter, default=0.0
            An energy offset to apply to the energy values before convolution.
        convolution_settings : ConvolutionSettings | None, default=None
            The settings for the convolution.
        temperature : Parameter | Numeric | None, default=None
            The temperature to use for detailed balance correction.
        temperature_unit : str | sc.Unit, default='K'
            The unit of the temperature parameter.
        detailed_balance_settings : DetailedBalanceSettings | None, default=None
            The settings for detailed balance. If None, default settings will be used.
        unit : str | sc.Unit, default='meV'
            The unit of the energy.
        display_name : str | None, default='MyConvolution'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        """
        super().__init__(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy_offset=energy_offset,
            convolution_settings=convolution_settings,
            temperature=temperature,
            temperature_unit=temperature_unit,
            detailed_balance_settings=detailed_balance_settings,
            unit=unit,
            display_name=display_name,
            unique_name=unique_name,
        )

    def convolution(
        self,
    ) -> np.ndarray:
        """
        Calculate the convolution of the sample and resolution models at the values given in
        energy. Includes detailed balance correction if temperature is provided.

        Returns
        -------
        np.ndarray
            The convolved values evaluated at energy.
        """
        # Make sure the convolver is updated with the latest convolution
        # settings before convolution.
        if not self.convolution_settings.convolution_plan_is_valid:
            self._energy_grid = self._create_energy_grid()

        # Give warnings if peaks are very wide or very narrow
        self._check_width_thresholds(
            model=self.sample_components,
            model_name='sample model',
        )
        self._check_width_thresholds(
            model=self.resolution_components,
            model_name='resolution model',
        )

        # Evaluate sample model. If called via the Convolution class,
        # delta functions are already filtered out.
        sample_vals = self.sample_components.evaluate(
            self._energy_grid.energy_dense
            - self._energy_grid.energy_even_length_offset
            - self.energy_offset.value
        )

        # Detailed balance correction
        if self.temperature is not None and self.detailed_balance_settings.use_detailed_balance:
            detailed_balance_factor_correction = detailed_balance_factor(
                energy=self._energy_grid.energy_dense - self.energy_offset.value,
                temperature=self.temperature,
                energy_unit=self.energy.unit,
                divide_by_temperature=self.detailed_balance_settings.normalize_detailed_balance,
            )
            sample_vals *= detailed_balance_factor_correction

        # Evaluate resolution model
        resolution_vals = self.resolution_components.evaluate(
            self._energy_grid.energy_dense_centered
        )

        # Convolution
        convolved = fftconvolve(sample_vals, resolution_vals, mode='same')
        convolved *= self._energy_grid.energy_dense_step  # normalize

        if self.upsample_factor is not None:
            # interpolate back to original energy grid
            convolved = np.interp(
                self.energy.values,
                self._energy_grid.energy_dense,
                convolved,
                left=0.0,
                right=0.0,
            )

        return convolved
