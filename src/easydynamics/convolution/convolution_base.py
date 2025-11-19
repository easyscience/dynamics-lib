from typing import Optional, Union

import numpy as np
from easyscience.variable import Parameter

from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.model_component import ModelComponent

Numerical = Union[float, int]


class ConvolutionBase:
    def __init__(
        self,
        energy: np.ndarray,
        sample_model: Union[SampleModel, ModelComponent] = None,
        resolution_model: Union[SampleModel, ModelComponent] = None,
        energy_unit: str = "meV",
        offset: Optional[Union[Numerical, Parameter]] = 0.0,
    ):
        self._energy = energy
        self._sample_model = sample_model
        self._resolution_model = resolution_model
        self._energy_unit = energy_unit

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
    def energy_unit(self) -> str:
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )

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
