# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric


class ConvolutionBase:
    """
    Base class for convolutions of sample and resolution models.

    This base class has no convolution functionality.
    """

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_components: ComponentCollection | ModelComponent | None = None,
        resolution_components: ComponentCollection | ModelComponent | None = None,
        energy_unit: str | sc.Unit = 'meV',
        energy_offset: Numeric | Parameter = 0.0,
    ) -> None:
        """
        Initialize the ConvolutionBase.

        Parameters
        ----------
        energy : np.ndarray | sc.Variable
            1D array of energy values where the convolution is evaluated.
        sample_components : ComponentCollection | ModelComponent | None, default=None
            The sample model to be convolved. By default, None.
        resolution_components : ComponentCollection | ModelComponent | None, default=None
            The resolution model to convolve with. By default, None.
        energy_unit : str | sc.Unit, default='meV'
            The unit of the energy. By default, 'meV'.
        energy_offset : Numeric | Parameter, default=0.0
            The energy offset applied to the convolution. By default, 0.0.

        Raises
        ------
        TypeError :
            If energy is not a numpy ndarray or a scipp Variable or if energy_unit is not a string
            or scipp unit, or if energy_offset is not a number or a Parameter, or if
            sample_components is not a ComponentCollection or ModelComponent, or if
            resolution_components is not a ComponentCollection or ModelComponent.
        """
        if isinstance(energy, Numeric):
            energy = np.array([float(energy)])

        if not isinstance(energy, (np.ndarray, sc.Variable)):
            raise TypeError(f'Energy must be a numpy ndarray or a scipp Variable. Got {energy}')

        if not isinstance(energy_unit, (str, sc.Unit)):
            raise TypeError('Energy_unit must be a string or sc.Unit.')

        if isinstance(energy, np.ndarray):
            energy = sc.array(dims=['energy'], values=energy, unit=energy_unit)

        if isinstance(energy_offset, Numeric):
            energy_offset = Parameter(
                name='energy_offset', value=float(energy_offset), unit=energy_unit
            )

        if not isinstance(energy_offset, Parameter):
            raise TypeError('Energy_offset must be a number or a Parameter.')

        self._energy = energy
        self._energy_unit = energy_unit
        self._energy_offset = energy_offset

        if sample_components is not None and not (
            isinstance(sample_components, (ComponentCollection, ModelComponent))
        ):
            raise TypeError(
                f'`sample_components` is an instance of {type(sample_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # noqa: E501
            )
        if isinstance(sample_components, ModelComponent):
            sample_components = ComponentCollection(components=[sample_components])
        self._sample_components = sample_components

        if resolution_components is not None and not (
            isinstance(resolution_components, (ComponentCollection, ModelComponent))
        ):
            raise TypeError(
                f'`resolution_components` is an instance of {type(resolution_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # noqa: E501
            )
        if isinstance(resolution_components, ModelComponent):
            resolution_components = ComponentCollection(components=[resolution_components])
        self._resolution_components = resolution_components

    @property
    def energy_offset(self) -> Parameter:
        """
        Get the energy offset.

        Returns
        -------
        Parameter
            The energy offset applied to the convolution.
        """
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, energy_offset: Numeric | Parameter) -> None:
        """
        Set the energy offset.

        Parameters
        ----------
        energy_offset : Numeric | Parameter
            The energy offset to apply to the convolution.

        Raises
        ------
        TypeError :
            If energy_offset is not a number or a Parameter.
        """
        if not isinstance(energy_offset, Parameter | Numeric):
            raise TypeError('Energy_offset must be a number or a Parameter.')

        if isinstance(energy_offset, Numeric):
            self._energy_offset.value = float(energy_offset)

        if isinstance(energy_offset, Parameter):
            self._energy_offset = energy_offset

    @property
    def energy_with_offset(self) -> sc.Variable:
        """
        Get the energy with the offset applied.

        Returns
        -------
        sc.Variable
            The energy values with the offset applied.
        """
        energy_with_offset = self.energy.copy()
        energy_with_offset.values = self.energy.values - self.energy_offset.value
        return energy_with_offset

    @energy_with_offset.setter
    def energy_with_offset(self, _value: sc.Variable) -> None:
        """
        Energy with offset is a read-only property derived from energy and energy_offset.

        Parameters
        ----------
        _value : sc.Variable
            The value to set (ignored).

        Raises
        ------
        AttributeError :
            Always raised since energy_with_offset is read-only.
        """
        raise AttributeError(
            'Energy with offset is a read-only property derived from energy and energy_offset.'
        )

    @property
    def energy(self) -> sc.Variable:
        """
        Get the energy.

        Returns
        -------
        sc.Variable
            The energy values where the convolution is evaluated.
        """

        return self._energy

    @energy.setter
    def energy(self, energy: np.ndarray | sc.Variable) -> None:
        """
        Set the energy.

        Parameters
        ----------
        energy : np.ndarray | sc.Variable
            1D array of energy values where the convolution is evaluated.

        Raises
        ------
        TypeError :
            If energy is not a numpy ndarray or a scipp Variable.
        """

        if isinstance(energy, Numeric):
            energy = np.array([float(energy)])

        if not isinstance(energy, (np.ndarray, sc.Variable)):
            raise TypeError('Energy must be a Number, a numpy ndarray or a scipp Variable.')

        if isinstance(energy, np.ndarray):
            self._energy = sc.array(dims=['energy'], values=energy, unit=self._energy.unit)

        if isinstance(energy, sc.Variable):
            self._energy = energy
            self._energy_unit = energy.unit

    @property
    def energy_unit(self) -> str:
        """
        Get the energy unit.

        Returns
        -------
        str
            The unit of the energy.
        """
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, _unit_str: str) -> None:
        """Energy unit."""
        raise AttributeError(
            f'Unit is read-only. Use convert_unit to change the unit between allowed types '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )  # noqa: E501

    def convert_energy_unit(self, energy_unit: str | sc.Unit) -> None:
        """
        Convert the energy and energy_offset to the specified unit.

        Parameters
        ----------
        energy_unit : str | sc.Unit
            The unit of the energy.

        Raises
        ------
        TypeError :
            If energy_unit is not a string or scipp unit.
        Exception :
            If energy cannot be converted to the specified unit.
        """
        if not isinstance(energy_unit, (str, sc.Unit)):
            raise TypeError('Energy unit must be a string or scipp unit.')

        old_energy = self.energy.copy()
        try:
            self.energy = sc.to_unit(self.energy, energy_unit)
        except Exception as e:
            self.energy = old_energy
            raise e

        old_energy_offset = self.energy_offset
        try:
            self.energy_offset.convert_unit(energy_unit)
        except Exception as e:
            self.energy_offset = old_energy_offset
            raise e

        self._energy_unit = energy_unit

    @property
    def sample_components(self) -> ComponentCollection | ModelComponent:
        """
        Get the sample model.

        Returns
        -------
        ComponentCollection | ModelComponent
            The sample model to be convolved.
        """
        return self._sample_components

    @sample_components.setter
    def sample_components(self, sample_components: ComponentCollection | ModelComponent) -> None:
        """
        Set the sample model.

        Parameters
        ----------
        sample_components : ComponentCollection | ModelComponent
            The sample model to be convolved.

        Raises
        ------
        TypeError :
            If sample_components is not a ComponentCollection or ModelComponent.
        """
        if not isinstance(sample_components, (ComponentCollection, ModelComponent)):
            raise TypeError(
                f'`sample_components` is an instance of {type(sample_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # noqa: E501
            )

        if isinstance(sample_components, ModelComponent):
            sample_components = ComponentCollection(components=[sample_components])
        self._sample_components = sample_components

    @property
    def resolution_components(self) -> ComponentCollection | ModelComponent:
        """
        Get the resolution model.

        Returns
        -------
        ComponentCollection | ModelComponent
            The resolution model to be convolved.
        """
        return self._resolution_components

    @resolution_components.setter
    def resolution_components(
        self, resolution_components: ComponentCollection | ModelComponent
    ) -> None:
        """
        Set the resolution model.

        Parameters
        ----------
        resolution_components : ComponentCollection | ModelComponent
            The resolution model to be convolved. Can be a ComponentCollection or a single
            ModelComponent.

        Raises
        ------
        TypeError :
            If resolution_components is not a ComponentCollection or ModelComponent.
        """
        if not isinstance(resolution_components, (ComponentCollection, ModelComponent)):
            raise TypeError(
                f'`resolution_components` is an instance of {type(resolution_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # noqa: E501
            )

        if isinstance(resolution_components, ModelComponent):
            resolution_components = ComponentCollection(components=[resolution_components])
        self._resolution_components = resolution_components
