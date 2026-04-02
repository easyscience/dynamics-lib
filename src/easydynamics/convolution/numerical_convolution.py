# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.signal import fftconvolve

from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.detailed_balance import detailed_balance_factor
from easydynamics.utils.utils import Numeric


class NumericalConvolution(NumericalConvolutionBase):
    """Numerical convolution of a ComponentCollection with a
    ComponentCollection using FFT.

    Includes optional upsampling and extended range to improve accuracy.
    Warns about very wide or very narrow peaks in the models. If
    temperature is provided, detailed balance correction is applied to
    the sample model.
    """

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_components: ComponentCollection | ModelComponent,
        resolution_components: ComponentCollection | ModelComponent,
        energy_offset: Numeric | Parameter = 0.0,
        upsample_factor: Numeric | None = 5,
        extension_factor: Numeric | None = 0.2,
        temperature: Parameter | Numeric | None = None,
        temperature_unit: str | sc.Unit = 'K',
        energy_unit: str | sc.Unit = 'meV',
        normalize_detailed_balance: bool = True,
    ) -> None:
        """Initialize the NumericalConvolution object.

        Parameters
        ----------
        energy : np.ndarray | sc.Variable
            1D array of energy values
            where the convolution is evaluated.
        sample_components : ComponentCollection | ModelComponent
            The sample model to be convolved.
        resolution_components : ComponentCollection | ModelComponent
            The resolution model to convolve with.
        energy_offset : Numeric | Parameter, optional
            An energy
            offset to apply to the energy values before convolution. By default, 0.0.
        upsample_factor : Numeric | None, optional
            The factor by which to
            upsample the input data before convolution. By default, 5.
        extension_factor : Numeric | None, optional
            The factor by which to
            extend the input data range before convolution. By default, 0.2.
        temperature : Parameter | Numeric | None, optional
            The
            temperature to use for detailed balance correction. By default, None.
        temperature_unit : str | sc.Unit, optional
            The unit of the
            temperature parameter. By default, 'K'.
        energy_unit : str | sc.Unit, optional
            The unit of the
            energy. By default, 'meV'.
        normalize_detailed_balance : bool, optional
            Whether to
            normalize the detailed balance correction. Default is
            True. By default, True.
        """
        super().__init__(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy_offset=energy_offset,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
            temperature=temperature,
            temperature_unit=temperature_unit,
            energy_unit=energy_unit,
            normalize_detailed_balance=normalize_detailed_balance,
        )

    def convolution(
        self,
    ) -> np.ndarray:
        """Calculate the convolution of the sample and resolution models
        at the values given in energy. Includes detailed balance
        correction if temperature is provided.

        Returns
        -------
        np.ndarray
            The convolved values evaluated at energy.
        """

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
        if self.temperature is not None:
            detailed_balance_factor_correction = detailed_balance_factor(
                energy=self._energy_grid.energy_dense - self.energy_offset.value,
                temperature=self.temperature,
                energy_unit=self.energy.unit,
                divide_by_temperature=self.normalize_detailed_balance,
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
