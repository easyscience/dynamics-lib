# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from scipy.signal import fftconvolve

from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.utils.detailed_balance import detailed_balance_factor


class NumericalConvolution(NumericalConvolutionBase):
    """
    Numerical convolution of a ComponentCollection with a ComponentCollection using FFT.

    Includes optional upsampling and extended range to improve accuracy. Warns about very wide or
    very narrow peaks in the models. If temperature is provided, detailed balance correction is
    applied to the sample model.
    """

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
            self.convolution_settings.convolution_plan_is_valid = True

        # Give warnings if peaks are very wide or very narrow
        if not self.convolution_settings.suppress_warnings:
            self._check_width_thresholds(
                model=self.sample_components,
                model_name='sample model',
            )
            self._check_width_thresholds(
                model=self.resolution_components,
                model_name='resolution model',
            )

        # Unit-convert the energy offset to match the energy grid unit.
        # sc.to_unit returns a new scalar — self.energy_offset is never mutated.
        offset_value = sc.to_unit(self.energy_offset.full_value, self.energy.unit).value

        # Evaluate sample model. If called via the Convolution class,
        # delta functions are already filtered out.
        sample_vals = self.sample_components.evaluate(
            self._energy_grid.energy_dense
            - self._energy_grid.energy_even_length_offset
            - offset_value
        )

        # Detailed balance correction
        if self.temperature is not None and self.detailed_balance_settings.use_detailed_balance:
            detailed_balance_factor_correction = detailed_balance_factor(
                energy=self._energy_grid.energy_dense
                - self._energy_grid.energy_even_length_offset
                - offset_value,
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

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'display_name={self.display_name!r}, '
            f'unique_name={self.unique_name!r}, '
            f'x_unit={self.x_unit}, '
            f'y_unit={self.y_unit}, '
            f'energy_len={len(self.energy)}, '
            f'temperature={self.temperature})'
        )
