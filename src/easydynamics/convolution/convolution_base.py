# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.base_classes import EasyDynamicsModelBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.delta_function import DeltaFunction
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import convert_parameter_unit
from easydynamics.utils.utils import convert_units_with_rollback
from easydynamics.utils.utils import energy_to_scipp


class ConvolutionBase(EasyDynamicsModelBase):
    """
    Base class for convolutions of sample and resolution models.

    This base class has no convolution functionality.
    """

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_components: ComponentCollection | ModelComponent | None = None,
        resolution_components: ComponentCollection | ModelComponent | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        energy_offset: Numeric | Parameter = 0.0,
        display_name: str | None = 'MyConvolution',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the ConvolutionBase.

        Parameters
        ----------
        energy : np.ndarray | sc.Variable
            1D array of energy values where the convolution is evaluated.
        sample_components : ComponentCollection | ModelComponent | None, default=None
            The sample model to be convolved.
        resolution_components : ComponentCollection | ModelComponent | None, default=None
            The resolution model to convolve with.
        x_unit : str | sc.Unit, default='meV'
            The unit of the energy axis.
        y_unit : str | sc.Unit, default='dimensionless'
            The unit of the model output (intensity).
        energy_offset : Numeric | Parameter, default=0.0
            The energy offset applied to the convolution.
        display_name : str | None, default='MyConvolution'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If energy is not a numpy ndarray or a scipp Variable or if x_unit is not a string or
            scipp unit, or if energy_offset is not a number or a Parameter, or if sample_components
            is not a ComponentCollection or ModelComponent, or if resolution_components is not a
            ComponentCollection or ModelComponent.

        Notes
        -----
        A ``ValueError`` propagates from the validation helpers if resolution_components contains a
        DeltaFunction, or if the x_unit of the sample or resolution components does not match the
        convolver's x_unit.
        """

        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            display_name=display_name,
            unique_name=unique_name,
        )

        if isinstance(energy, Numeric):
            energy = np.array([float(energy)])

        if not isinstance(energy, (np.ndarray, sc.Variable)):
            raise TypeError(f'Energy must be a numpy ndarray or a scipp Variable. Got {energy}')

        if isinstance(energy, np.ndarray):
            energy = energy_to_scipp(energy, x_unit)

        if isinstance(energy_offset, Numeric):
            energy_offset = Parameter(
                name='energy_offset', value=float(energy_offset), unit=x_unit
            )

        if not isinstance(energy_offset, Parameter):
            raise TypeError('Energy_offset must be a number or a Parameter.')

        self._energy = energy
        self._energy_offset = energy_offset

        if sample_components is not None and not (
            isinstance(sample_components, (ComponentCollection, ModelComponent))
        ):
            raise TypeError(
                f'`sample_components` is an instance of {type(sample_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # ruff: ignore[line-too-long]
            )
        if isinstance(sample_components, ModelComponent):
            sample_components = ComponentCollection(
                components=[sample_components],
                x_unit=sample_components.x_unit,
                y_unit=sample_components.y_unit,
            )
        self._sample_components = sample_components

        if resolution_components is not None and not (
            isinstance(resolution_components, (ComponentCollection, ModelComponent))
        ):
            raise TypeError(
                f'`resolution_components` is an instance of {type(resolution_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # ruff: ignore[line-too-long]
            )
        if isinstance(resolution_components, ModelComponent):
            resolution_components = ComponentCollection(
                components=[resolution_components],
                x_unit=resolution_components.x_unit,
                y_unit=resolution_components.y_unit,
            )
        self._validate_no_delta_in_resolution(resolution_components)
        self._resolution_components = resolution_components

        self._validate_component_x_units()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_no_delta_in_resolution(
        resolution_components: ComponentCollection | None,
    ) -> None:
        """
        Validate that the resolution collection contains no DeltaFunction components.

        Convolving with a delta function in the resolution is not supported on any path
        (analytical, numerical, or delta), so the invariant is enforced when the resolution is
        bound to the convolver.

        Parameters
        ----------
        resolution_components : ComponentCollection | None
            The resolution collection to validate. None is skipped.

        Raises
        ------
        ValueError
            If resolution_components contains a DeltaFunction.
        """
        if resolution_components is None:
            return
        if any(isinstance(component, DeltaFunction) for component in resolution_components):
            raise ValueError(
                'resolution_components contains delta functions. This is not supported.'
            )

    def _validate_component_x_units(self) -> None:
        """
        Validate that sample and resolution collections use the convolver's x_unit.

        Components in a different (even compatible) x_unit would be evaluated with raw numbers in
        the wrong unit, silently producing wrong results.

        Raises
        ------
        ValueError
            If a collection's x_unit differs from the convolver's x_unit.
        """
        if self._x_unit is None:
            return
        for label, collection in (
            ('sample_components', self._sample_components),
            ('resolution_components', self._resolution_components),
        ):
            if collection is None or collection.x_unit is None:
                continue
            # Compare as sc.Unit so unit aliases (e.g. 'ueV' vs 'micro-eV') are not false
            # mismatches.
            if sc.Unit(str(collection.x_unit)) != sc.Unit(str(self._x_unit)):
                raise ValueError(
                    f'{label} has x_unit {str(collection.x_unit)!r}, which does not match the '
                    f'convolver x_unit {str(self._x_unit)!r}. Convert the components with '
                    f'convert_x_unit before constructing the convolver.'
                )

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
        TypeError
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
        offset_value = sc.to_unit(self.energy_offset.full_value, self._energy.unit).value
        energy_with_offset = self.energy.copy()
        energy_with_offset.values = self.energy.values - offset_value
        return energy_with_offset

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
            1D array of energy values where the convolution is evaluated. A scipp Variable must
            carry the convolver's x_unit; the x_unit itself can only be changed via convert_x_unit.

        Raises
        ------
        TypeError
            If energy is not a numpy ndarray or a scipp Variable.
        ValueError
            If energy is a scipp Variable whose unit differs from the convolver's x_unit.
        """

        if isinstance(energy, Numeric):
            energy = np.array([float(energy)])

        if not isinstance(energy, (np.ndarray, sc.Variable)):
            raise TypeError('Energy must be a Number, a numpy ndarray or a scipp Variable.')

        if isinstance(energy, np.ndarray):
            self._energy = energy_to_scipp(energy, self._energy.unit)

        if isinstance(energy, sc.Variable):
            # Compare as sc.Unit so unit aliases (e.g. 'ueV' vs 'micro-eV') are not false
            # mismatches.
            if self._x_unit is not None and energy.unit != sc.Unit(str(self._x_unit)):
                raise ValueError(
                    f'energy has unit {str(energy.unit)!r}, which does not match the convolver '
                    f'x_unit {str(self._x_unit)!r}. Use convert_x_unit to change the unit, or '
                    f'provide energy in {str(self._x_unit)!r}.'
                )
            self._energy = energy

    def convert_x_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the energy axis, energy_offset, and all components to the specified unit.

        If any conversion fails, the already-converted state is rolled back best-effort before the
        failing conversion's exception is re-raised.

        Parameters
        ----------
        unit : str | sc.Unit
            The unit of the energy.

        Raises
        ------
        TypeError
            If unit is not a string or scipp unit.
        """
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError('Energy unit must be a string or scipp unit.')

        old_x_unit = str(self.x_unit)
        old_offset_unit = str(self.energy_offset.unit)

        def _convert_energy(target_unit: str | sc.Unit) -> None:
            # Assign the backing field directly: the public setter rejects unit changes
            # (convert_x_unit is the one supported route for those).
            self._energy = sc.to_unit(self._energy, target_unit)

        conversions = [
            (_convert_energy, unit, old_x_unit),
            (partial(convert_parameter_unit, self._energy_offset), unit, old_offset_unit),
        ]
        if self.sample_components is not None:
            conversions.append((self.sample_components.convert_x_unit, unit, old_x_unit))
        if self.resolution_components is not None:
            conversions.append((self.resolution_components.convert_x_unit, unit, old_x_unit))
        convert_units_with_rollback(conversions)

        # Keep the str contract for x_unit even when an sc.Unit was passed.
        self._x_unit = str(unit) if isinstance(unit, sc.Unit) else unit

    def convert_y_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit of the sample components.

        Only propagates to sample components (resolution is normalised and unit-independent). If
        any component raises during unit conversion, the conversion is rolled back best-effort
        before the exception is re-raised.

        Parameters
        ----------
        unit : str | sc.Unit
            The new y-axis unit.

        Raises
        ------
        TypeError
            If unit is not a string or scipp unit.
        """
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError('y_unit must be a string or scipp unit.')
        old_y_unit = self.y_unit
        if self.sample_components is not None:
            convert_units_with_rollback([
                (self.sample_components.convert_y_unit, unit, old_y_unit)
            ])
        self._relabel_y_unit(unit)

    def _relabel_y_unit(self, unit: str | sc.Unit) -> None:
        """
        Update the y-unit label without converting any components.

        This is the contract for a parent convolver whose sub-convolvers share its component
        objects: the parent converts the components once, then relabels the sub-convolvers so their
        y_unit stays consistent without double-converting the shared components.

        Parameters
        ----------
        unit : str | sc.Unit
            The new y-axis unit. The caller is responsible for having converted the components.
        """
        self._y_unit = str(unit) if isinstance(unit, sc.Unit) else unit

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
        TypeError
            If sample_components is not a ComponentCollection or ModelComponent.
        """
        if not isinstance(sample_components, (ComponentCollection, ModelComponent)):
            raise TypeError(
                f'`sample_components` is an instance of {type(sample_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # ruff: ignore[line-too-long]
            )

        if isinstance(sample_components, ModelComponent):
            sample_components = ComponentCollection(
                components=[sample_components],
                x_unit=sample_components.x_unit,
                y_unit=sample_components.y_unit,
            )
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
        TypeError
            If resolution_components is not a ComponentCollection or ModelComponent.

        Notes
        -----
        A ``ValueError`` propagates from the validation helper if resolution_components contains a
        DeltaFunction.
        """
        if not isinstance(resolution_components, (ComponentCollection, ModelComponent)):
            raise TypeError(
                f'`resolution_components` is an instance of {type(resolution_components).__name__}, but must be a ComponentCollection or ModelComponent.'  # ruff: ignore[line-too-long]
            )

        if isinstance(resolution_components, ModelComponent):
            resolution_components = ComponentCollection(
                components=[resolution_components],
                x_unit=resolution_components.x_unit,
                y_unit=resolution_components.y_unit,
            )
        self._validate_no_delta_in_resolution(resolution_components)
        self._resolution_components = resolution_components
