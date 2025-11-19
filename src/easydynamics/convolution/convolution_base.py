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
    This base class has no convolution functionality.

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
        self._energy_unit = energy_unit
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
    def energy(self) -> sc.Variable:
        """Get the energy"""

        return self._energy

    @energy.setter
    def energy(self, energy: np.ndarray) -> None:
        """Set the energy.
         Args:
            energy : np.ndarray or scipp.Variable
                1D array of energy values where the convolution is evaluated.

        Raises:
            TypeError: If energy is not a numpy ndarray or a scipp Variable.
        """

        if isinstance(energy, Numerical):
            energy = np.array([energy])

        if not isinstance(energy, (np.ndarray, sc.Variable)):
            raise TypeError(
                "Energy must be a Number, a numpy ndarray or a scipp Variable."
            )

        if isinstance(energy, np.ndarray):
            self._energy = sc.array(
                dims=["energy"], values=energy, unit=self._energy.unit
            )

        if isinstance(energy, sc.Variable):
            self._energy = energy
            self._energy_unit = energy.unit

    @property
    def sample_model(self) -> Union[SampleModel, ModelComponent]:
        """Get the sample model"""
        return self._sample_model

    @sample_model.setter
    def sample_model(self, sample_model: Union[SampleModel, ModelComponent]) -> None:
        """Set the sample model.
        Args:
            sample_model : SampleModel or ModelComponent
                The sample model to be convolved.

        Raises:
            TypeError: If sample_model is not a SampleModel or ModelComponent.
        """
        if not isinstance(sample_model, (SampleModel, ModelComponent)):
            raise TypeError(
                f"`sample_model` is an instance of {type(sample_model).__name__}, but must be a SampleModel or ModelComponent."
            )
        self._sample_model = sample_model

    @property
    def resolution_model(self) -> Union[SampleModel, ModelComponent]:
        """Get the resolution model"""
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(
        self, resolution_model: Union[SampleModel, ModelComponent]
    ) -> None:
        """Set the resolution model.
        Args:
            resolution_model : SampleModel or ModelComponent
                The resolution model to convolve with.

        Raises:
            TypeError: If resolution_model is not a SampleModel or ModelComponent.
        """
        if not isinstance(resolution_model, (SampleModel, ModelComponent)):
            raise TypeError(
                f"`resolution_model` is an instance of {type(resolution_model).__name__}, but must be a SampleModel or ModelComponent."
            )
        self._resolution_model = resolution_model

    @property
    def offset(self) -> Parameter:
        """Get the offset"""
        return self._offset

    @offset.setter
    def offset(self, offset: Union[Numerical, Parameter]) -> None:
        """Set the offset.
        Args:
            offset : Number or Parameter
                The offset to apply to the input array.

        Raises:
            TypeError: If offset is not a Number or Parameter.
        """
        if not isinstance(offset, Parameter):
            raise TypeError("Offset must be a Number or Parameter.")

        if isinstance(offset, Numerical):
            self._offset.value = offset
        else:
            self._offset = offset
