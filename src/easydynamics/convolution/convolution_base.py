import numpy as np
import scipp as sc

from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.model_component import ModelComponent

Numerical = float | int


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
    """

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_model: SampleModel | ModelComponent = None,
        resolution_model: SampleModel | ModelComponent = None,
        energy_unit: str | sc.Unit = "meV",
    ):
        if isinstance(energy, Numerical):
            energy = np.array([float(energy)])

        if not isinstance(energy, (np.ndarray, sc.Variable)):
            raise TypeError("Energy must be a numpy ndarray or a scipp Variable.")

        if not isinstance(energy_unit, (str, sc.Unit)):
            raise TypeError("Energy_unit must be a string or sc.Unit.")

        if isinstance(energy, np.ndarray):
            energy = sc.array(dims=["energy"], values=energy, unit=energy_unit)

        self._energy = energy
        self._energy_unit = energy_unit

        if sample_model is not None and not isinstance(sample_model, SampleModel):
            raise TypeError(
                f"`sample_model` is an instance of {type(sample_model).__name__}, but must be a SampleModel or ModelComponent."
            )
        self._sample_model = sample_model

        if resolution_model is not None and not isinstance(
            resolution_model, SampleModel
        ):
            raise TypeError(
                f"`resolution_model` is an instance of {type(resolution_model).__name__}, but must be a SampleModel or ModelComponent."
            )
        self._resolution_model = resolution_model

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
            energy = np.array([float(energy)])

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
    def energy_unit(self) -> str:
        """Get the energy unit"""
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )  # noqa: E501

    def convert_energy_unit(self, energy_unit: str | sc.Unit) -> None:
        """Convert the energy to the specified unit
        Args:
            energy_unit : str or sc.Unit
                The unit of the energy.

        Raises:
            TypeError: If energy_unit is not a string or scipp unit.
        """
        if not isinstance(energy_unit, (str, sc.Unit)):
            raise TypeError("Energy unit must be a string or scipp unit.")

        self.energy = sc.to_unit(self.energy, energy_unit)
        self._energy_unit = energy_unit

    @property
    def sample_model(self) -> SampleModel | ModelComponent:
        """Get the sample model"""
        return self._sample_model

    @sample_model.setter
    def sample_model(self, sample_model: SampleModel | ModelComponent) -> None:
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
    def resolution_model(self) -> SampleModel | ModelComponent:
        """Get the resolution model"""
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, resolution_model: SampleModel | ModelComponent) -> None:
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
