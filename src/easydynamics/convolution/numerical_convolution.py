import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.signal import fftconvolve

from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.sample_model.component_collection import (
    ComponentCollection,
)
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.detailed_balance import (
    _detailed_balance_factor as detailed_balance_factor,
)

Numerical = float | int


class NumericalConvolution(NumericalConvolutionBase):
    """Numerical convolution of a ComponentCollection with a ComponentCollection using FFT.
        Includes optional upsampling and extended range to improve accuracy.
        Warns about very wide or very narrow peaks in the models.
        If temperature is provided, detailed balance correction is applied to the sample model.

    Args:
    energy : np.ndarray or scipp.Variable
        1D array of energy values where the convolution is evaluated.
    sample_components : ComponentCollection or ModelComponent
        The sample model to be convolved.
    resolution_components : ComponentCollection or ModelComponent
        The resolution model to convolve with.
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
        Whether to normalize the detailed balance correction. Default is True.
    """

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_components: ComponentCollection | ModelComponent,
        resolution_components: ComponentCollection | ModelComponent,
        upsample_factor: Numerical = 5,
        extension_factor: float = 0.2,
        temperature: Parameter | float | None = None,
        temperature_unit: str | sc.Unit = "K",
        energy_unit: str | sc.Unit = "meV",
        normalize_detailed_balance: bool = True,
    ):
        super().__init__(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
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
        Calculate the convolution of the sample and resolution models at the values
        given in energy.
        Includes detailed balance correction if temperature is provided.

        Returns:
            np.ndarray
                The convolved values evaluated at energy.
        """

        # Give warnings if peaks are very wide or very narrow
        self._check_width_thresholds(
            model=self.sample_components,
            model_name="sample model",
        )
        self._check_width_thresholds(
            model=self.resolution_components,
            model_name="resolution model",
        )

        # Evaluate sample model. If called via the Convolution class, delta functions are already filtered out.
        sample_vals = self.sample_components.evaluate(
            self._energy_grid.energy_dense - self._energy_grid.energy_even_length_offset
        )

        # Detailed balance correction
        if self.temperature is not None:
            detailed_balance_factor_correction = detailed_balance_factor(
                energy=self._energy_grid.energy_dense,
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
        convolved = fftconvolve(sample_vals, resolution_vals, mode="same")
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
