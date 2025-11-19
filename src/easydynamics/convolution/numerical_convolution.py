from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.signal import fftconvolve

from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.sample_model import (
    SampleModel,
)
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.detailed_balance import (
    _detailed_balance_factor as detailed_balance_factor,
)

Numerical = Union[float, int]


class NumericalConvolution(NumericalConvolutionBase):
    """ "
    Args:
    energy : np.ndarray
        1D array of energy values where the convolution is evaluated.
    sample_model : SampleModel or ModelComponent
        The sample model to be convolved.
    resolution_model : SampleModel or ModelComponent
        The resolution model to convolve with.
    offset_float : float, or None, optional
        The offset to apply to the input array.
    upsample_factor : int, optional
        The factor by which to upsample the input data before convolution. Default is 5.
    extension_factor : float, optional
        The factor by which to extend the input data range before convolution. Default is 0.2.
    temperature : Parameter, float, or None, optional
        The temperature to use for detailed balance correction. Default is None.
    temperature_unit : str or sc.Unit, optional
        The unit of the temperature parameter. Default is 'K'.
    energy_unit : str or sc.Unit, optional
        The unit of the energy. Default is 'meV'.
    normalize_detailed_balance : bool, optional
        Whether to normalize the detailed balance factor. Default is True.
    """

    def __init__(
        self,
        energy: Union[np.ndarray, sc.Variable],
        sample_model: Union[SampleModel, ModelComponent],
        resolution_model: Union[SampleModel, ModelComponent],
        offset: Optional[Union[Numerical, Parameter]] = 0.0,
        upsample_factor: Optional[Numerical] = 5,
        extension_factor: Optional[float] = 0.2,
        temperature: Optional[Union[Parameter, float]] = None,
        temperature_unit: Optional[Union[str, sc.Unit]] = "K",
        energy_unit: Optional[Union[str, sc.Unit]] = "meV",
        normalize_detailed_balance: Optional[bool] = True,
    ):
        super().__init__(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            offset=offset,
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
        """
        Numerical convolution using FFT with optional upsampling + extended range.
        Includes detailed balance correction if temperature is provided.



        Returns:
            np.ndarray
                The convolved values evaluated at energy.
        """

        # Give warnings if peaks are very wide or very narrow
        self._check_width_thresholds(
            model=self.sample_model,
            model_name="sample model",
        )
        self._check_width_thresholds(
            model=self.resolution_model,
            model_name="resolution model",
        )

        # Evaluate sample model. Delta functions are already filtered out
        sample_vals = self.sample_model.evaluate(
            self._energy_grid.energy_dense
            - self._offset.value
            - self._energy_grid.energy_even_length_offset
        )

        # Detailed balance correction
        if self.temperature is not None:
            detailed_balance_factor_correction = detailed_balance_factor(
                energy=self._energy_grid.energy_dense - self._offset.value,
                temperature=self.temperature,
                energy_unit=self.energy.unit,
                divide_by_temperature=self.normalize_detailed_balance,
            )
            sample_vals *= detailed_balance_factor_correction

        # Evaluate resolution model
        resolution_vals = self.resolution_model.evaluate(
            self._energy_grid.energy_dense_centered
        )

        # Convolution
        convolved = fftconvolve(sample_vals, resolution_vals, mode="same")
        convolved *= self._energy_grid.energy_step  # normalize

        if self.upsample_factor > 0:
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
            f"NumericalConvolution(energy_unit={self._energy_unit}, "
            f"offset={self.offset}, upsample_factor={self.upsample_factor}, "
            f"extension_factor={self.extension_factor}, "
            f"temperature={self.temperature}, "
            f"temperature_unit={self.temperature_unit}, "
            f"normalize_detailed_balance={self.normalize_detailed_balance})"
        )
