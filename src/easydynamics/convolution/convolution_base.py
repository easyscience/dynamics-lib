from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.model_component import ModelComponent

Numerical = Union[float, int]


class ConvolutionBase:
    """
    Base class for convolutions of sample and resolution models.
    Args:
    energy : np.ndarray or scipp.Variable
        1D array of energy values where the convolution is evaluated.
    sample_model : SampleModel or ModelComponent
        The sample model to be convolved.
    resolution_model : SampleModel or ModelComponent
        The resolution model to convolve with.
    energy_unit : str or sc.Unit, optional
        The unit of the energy. Default is 'meV'.
    offset_float : float, or None, optional
        The offset to apply to the input array.
    """

    def __init__(
        self,
        energy: Union[np.ndarray, sc.Variable],
        sample_model: Union[SampleModel, ModelComponent] = None,
        resolution_model: Union[SampleModel, ModelComponent] = None,
        energy_unit: str = "meV",
        offset: Optional[Union[Numerical, Parameter]] = 0.0,
    ):
        if isinstance(energy, Numerical):
            energy = np.array([energy])

        if not isinstance(energy, (np.ndarray, sc.Variable)):
            raise TypeError("Energy must be a numpy ndarray or a scipp Variable.")
        if isinstance(energy, np.ndarray):
            energy = sc.array(dims=["energy"], values=energy, unit=energy_unit)

        self._energy = energy
        self._sample_model = sample_model
        self._resolution_model = resolution_model

        if not isinstance(sample_model, SampleModel):
            raise TypeError(
                f"`sample_model` is an instance of {type(sample_model).__name__}, but must be SampleModel."
            )

        if not isinstance(resolution_model, SampleModel):
            raise TypeError(
                f"`resolution_model` is an instance of {type(resolution_model).__name__}, but must be SampleModel."
            )

        if offset is None:
            offset = 0.0

        if isinstance(offset, Numerical):
            offset = Parameter(value=offset, name="offset", unit=energy_unit)

        if not isinstance(offset, Parameter):
            raise TypeError("Offset must be a Number or Parameter.")

        self._offset = offset

    @property
    def energy(self) -> np.ndarray:
        return self._energy

    @energy.setter
    def energy(self, energy: np.ndarray) -> None:
        self._energy = energy

    @property
    def offset(self) -> Parameter:
        return self._offset

    @offset.setter
    def offset(self, offset: Union[Numerical, Parameter]) -> None:
        if not isinstance(offset, Parameter):
            raise TypeError("Offset must be a Number or Parameter.")

        if isinstance(offset, Numerical):
            self._offset.value = offset
        else:
            self._offset = offset

    @property
    def sample_model(self) -> Union[SampleModel, ModelComponent]:
        return self._sample_model

    @property
    def resolution_model(self) -> Union[SampleModel, ModelComponent]:
        return self._resolution_model
