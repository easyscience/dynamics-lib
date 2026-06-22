# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import scipp as sc
from easyscience.base_classes.new_base import NewBase
from easyscience.variable import Parameter

from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.resolution_model import ResolutionModel
from easydynamics.sample_model.sample_model import SampleModel
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import _validate_unit


class InstrumentModel(NewBase):
    """
    InstrumentModel represents a model of the instrument in an experiment at various Q.

    It can contain a model of the resolution function for convolutions, of the background and an
    offset in the energy axis.
    """

    def __init__(
        self,
        display_name: str = 'MyInstrumentModel',
        unique_name: str | None = None,
        Q: Q_type | None = None,
        resolution_model: ResolutionModel | SampleModel | None = None,
        background_model: BackgroundModel | None = None,
        energy_offset: Numeric | None = None,
        unit: str | sc.Unit = 'meV',
    ) -> None:
        """
        Initialize an InstrumentModel.

        Parameters
        ----------
        display_name : str, default='MyInstrumentModel'
            The display name of the InstrumentModel.
        unique_name : str | None, default=None
            The unique name of the InstrumentModel.
        Q : Q_type | None, default=None
            The Q values where the instrument is modelled.
        resolution_model : ResolutionModel | SampleModel | None, default=None
            The resolution model of the instrument. If a SampleModel it will be converted to a
            ResolutionModel. If None, an empty resolution model is created and no resolution
            convolution is carried out.
        background_model : BackgroundModel | None, default=None
            The background model of the instrument. If None, an empty background model is created,
            and the background evaluates to 0.
        energy_offset : Numeric | None, default=None
            Template energy offset of the instrument. Will be copied to each Q value. If None, the
            energy offset will be 0.
        unit : str | sc.Unit, default='meV'
            The unit of the energy axis.

        Raises
        ------
        TypeError
            If resolution_model is not a ResolutionModel or None, or if background_model is not a
            BackgroundModel or None, or if energy_offset is not a number or None.
        """
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        self._unit = _validate_unit(unit)

        if resolution_model is None:
            self._resolution_model = ResolutionModel()
        else:
            if not isinstance(resolution_model, (ResolutionModel, SampleModel)):
                raise TypeError(
                    f'resolution_model must be a ResolutionModel, a SampleModel or None, '
                    f'got {type(resolution_model).__name__}'
                )
            if isinstance(resolution_model, SampleModel):
                resolution_model = ResolutionModel.from_sample_model(resolution_model)
            self._resolution_model = resolution_model

        if background_model is None:
            self._background_model = BackgroundModel()
        else:
            if not isinstance(background_model, BackgroundModel):
                raise TypeError(
                    f'background_model must be a BackgroundModel or None, '
                    f'got {type(background_model).__name__}'
                )
            self._background_model = background_model

        if energy_offset is None:
            energy_offset = 0.0

        if not isinstance(energy_offset, Numeric):
            raise TypeError('energy_offset must be a number or None')

        self._energy_offset = Parameter(
            name='energy_offset',
            value=float(energy_offset),
            unit=self.unit,
            fixed=False,
        )
        self._energy_offsets: list = []
        self._energy_offsets_is_dirty = True
        self._Q = _validate_and_convert_Q(Q)
        self._on_Q_change()

    # -------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------

    @property
    def resolution_model(self) -> ResolutionModel:
        """
        Get the resolution model of the instrument.

        Returns
        -------
        ResolutionModel
            The resolution model of the instrument.
        """
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, value: ResolutionModel | SampleModel) -> None:
        """
        Set the resolution model of the instrument.

        Parameters
        ----------
        value : ResolutionModel | SampleModel
            The new resolution model of the instrument.

        Raises
        ------
        TypeError
            If value is not a ResolutionModel or SampleModel.
        """
        if not isinstance(value, (ResolutionModel, SampleModel)):
            raise TypeError(
                f'resolution_model must be a ResolutionModel or SampleModel, '
                f'got {type(value).__name__}'
            )
        if isinstance(value, SampleModel):
            value = ResolutionModel.from_sample_model(value)
        self._resolution_model = value
        self._on_resolution_model_change()

    @property
    def background_model(self) -> BackgroundModel:
        """
        Get the background model of the instrument.

        Returns
        -------
        BackgroundModel
            The background model of the instrument.
        """

        return self._background_model

    @background_model.setter
    def background_model(self, value: BackgroundModel) -> None:
        """
        Set the background model of the instrument.

        Parameters
        ----------
        value : BackgroundModel
            The new background model of the instrument.

        Raises
        ------
        TypeError
            If value is not a BackgroundModel.
        """

        if not isinstance(value, BackgroundModel):
            raise TypeError(
                f'background_model must be a BackgroundModel, got {type(value).__name__}'
            )
        self._background_model = value
        self._on_background_model_change()

    @property
    def Q(self) -> np.ndarray | None:
        """
        Get the Q values of the InstrumentModel.

        Returns
        -------
        np.ndarray | None
            The Q values of the InstrumentModel, or None if not set.
        """
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """
        Set the Q values of the InstrumentModel.

        If Q is already set, it raises an error if the new Q values are not similar to the old ones
        to prevent accidental changes to the background and resolution models. To change Q values,
        first run clear_Q().

        Parameters
        ----------
        value : Q_type | None
            The new Q values to set. If None, Q values are not changed.

        Raises
        ------
        ValueError
            If the new Q values are not similar to the old ones when Q is not None.
        """
        if value is None:
            return
        old_Q = self._Q
        new_Q = _validate_and_convert_Q(value)

        if old_Q is None:
            self._Q = new_Q
            self._on_Q_change()
            return

        if len(old_Q) != len(new_Q) or not np.allclose(old_Q, new_Q):
            raise ValueError(
                'New Q values are not similar to the old ones. '
                'To change Q values, first run clear_Q().'
            )

    @property
    def unit(self) -> str | sc.Unit:
        """
        Get the unit of the InstrumentModel.

        Returns
        -------
        str | sc.Unit
            The unit of the InstrumentModel.
        """
        return self._unit

    @unit.setter
    def unit(self, _unit_str: str) -> None:
        """
        Set the unit of the InstrumentModel.

        The unit is read-only and cannot be set directly. Use convert_unit to change the unit
        between allowed types or create a new InstrumentModel with the desired unit.

        Parameters
        ----------
        _unit_str : str
            The new unit for the InstrumentModel (ignored).

        Raises
        ------
        AttributeError
            Always, as the unit is read-only.
        """
        raise AttributeError(
            f'Unit is read-only. Use convert_unit to change the unit between allowed types '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    @property
    def energy_offset(self) -> Parameter:
        """
        Get the energy offset template parameter of the instrument model.

        Returns
        -------
        Parameter
            The energy offset template parameter of the instrument model.
        """
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, value: Numeric) -> None:
        """
        Set the offset parameter of the instrument model.

        Parameters
        ----------
        value : Numeric
            The new value for the energy offset parameter. Will be copied to all Q values.

        Raises
        ------
        TypeError
            If value is not a number.
        """
        if not isinstance(value, Numeric):
            raise TypeError(f'energy_offset must be a number, got {type(value).__name__}')
        self._energy_offset.value = value

        self._on_energy_offset_change()

    # --------------------------------------------------------------
    # Other methods
    # --------------------------------------------------------------

    def clear_Q(self, confirm: bool = False) -> None:
        """
        Clear the Q values of the InstrumentModel and any associated ResolutionModel and
        BackgroundModel, removing all component collections and their associated Parameters.

        Parameters
        ----------
        confirm : bool, default=False
            Confirmation to clear Q values.

        Raises
        ------
        ValueError
            If confirm is not True.
        """
        if not confirm:
            raise ValueError(
                'Clearing Q values requires confirmation. Set confirm=True to proceed.'
            )
        self._Q = None
        self.background_model.clear_Q(confirm=True)
        self.resolution_model.clear_Q(confirm=True)
        self._on_Q_change()

    def convert_unit(self, unit_str: str | sc.Unit) -> None:
        """
        Convert the unit of the InstrumentModel.

        Parameters
        ----------
        unit_str : str | sc.Unit
            The unit to convert to.

        Raises
        ------
        ValueError
            If unit_str is not a valid unit string or scipp Unit.
        """
        unit = _validate_unit(unit_str)
        if unit is None:
            raise ValueError('unit_str must be a valid unit string or scipp Unit')

        self._background_model.convert_unit(unit)
        self._resolution_model.convert_unit(unit)
        self._energy_offset.convert_unit(unit)
        self._ensure_energy_offsets_current()
        for offset in self._energy_offsets:
            offset.convert_unit(unit)

        self._unit = unit

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all variables in the InstrumentModel.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the Q value to get variables for. If None, get variables for all Q values.


        Raises
        ------
        TypeError
            If Q_index is not an int or None.
        IndexError
            If Q_index is out of bounds for the Q values in the InstrumentModel.

        Returns
        -------
        list[Parameter]
            A list of all variables in the InstrumentModel. If Q_index is specified, only variables
            from the ComponentCollection at the given Q index are included. Otherwise, all
            variables in the InstrumentModel are included.
        """
        if self._Q is None:
            return []

        self._ensure_energy_offsets_current()
        if Q_index is None:
            variables = [self._energy_offsets[i] for i in range(len(self._Q))]
        else:
            if not isinstance(Q_index, int):
                raise TypeError(f'Q_index must be an int or None, got {type(Q_index).__name__}')
            if Q_index < 0 or Q_index >= len(self._Q):
                raise IndexError(
                    f'Q_index {Q_index} is out of bounds for Q of length {len(self._Q)}'
                )
            variables = [self._energy_offsets[Q_index]]

        variables.extend(self._background_model.get_all_variables(Q_index=Q_index))
        variables.extend(self._resolution_model.get_all_variables(Q_index=Q_index))

        return variables

    def fix_resolution_parameters(self) -> None:
        """Fix all parameters in the resolution model."""
        self.resolution_model.fix_all_parameters()

    def free_resolution_parameters(self) -> None:
        """Free all parameters in the resolution model."""
        self.resolution_model.free_all_parameters()

    def normalize_resolution(self) -> None:
        """Normalize the resolution model to have area 1."""
        self.resolution_model.normalize_area()

    def get_energy_offset(
        self,
        Q_index: int | None = None,
    ) -> Parameter | list[Parameter]:
        """
        Get the energy offset Parameter at a specific Q index.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the Q value to get the energy offset for. If None, get the energy offset
            for all Q values.

        Raises
        ------
        ValueError
            If no Q values are set in the InstrumentModel.
        IndexError
            If Q_index is out of bounds.
        TypeError
            If Q_index is not an int or None.

        Returns
        -------
        Parameter | list[Parameter]
            The energy offset Parameter at the specified Q index, or a list of Parameters if
            Q_index is None.
        """
        if self._Q is None:
            raise ValueError('No Q values are set in the InstrumentModel.')

        self._ensure_energy_offsets_current()
        if Q_index is None:
            return self._energy_offsets

        if not isinstance(Q_index, int):
            raise TypeError(f'Q_index must be an int or None, got {type(Q_index).__name__}')

        if Q_index < 0 or Q_index >= len(self._Q):
            raise IndexError(f'Q_index {Q_index} is out of bounds for Q of length {len(self._Q)}')

        return self._energy_offsets[Q_index]

    def fix_energy_offset(self, Q_index: int | None = None) -> None:
        """
        Fix energy offset parameters.

        If Q_index is specified, only fix the energy offset for that Q value. If Q_index is None,
        fix energy offsets for all Q values.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the Q value to fix the energy offset for. If None, fix energy offsets for
            all Q values.
        """
        self._fix_or_free_energy_offset(Q_index, fixed=True)

    def free_energy_offset(self, Q_index: int | None = None) -> None:
        """
        Free energy offset parameters.

        If Q_index is specified, only free the energy offset for that Q value. If Q_index is None,
        free energy offsets for all Q values.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the Q value to free the energy offset for. If None, free energy offsets
            for all Q values.
        """
        self._fix_or_free_energy_offset(Q_index, fixed=False)

    # --------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------
    def _fix_or_free_energy_offset(self, Q_index: int | None = None, fixed: bool = True) -> None:
        """
        Fix or free energy offset parameters.

        If Q_index is specified, only fix or free the energy offset for that Q value. If Q_index is
        None, fix or free energy offsets for all Q values.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the Q value to fix or free the energy offset for. If None, fix or free
            energy offsets for all Q values.
        fixed : bool, default=True
            Whether to fix (True) or free (False) the energy offset.

        Raises
        ------
        TypeError
            If Q_index is not an int or None.
        IndexError
            If Q_index is out of bounds for the Q values in the InstrumentModel.
        """

        self._ensure_energy_offsets_current()
        if Q_index is None:
            for offset in self._energy_offsets:
                offset.fixed = fixed
        else:
            if not isinstance(Q_index, int):
                raise TypeError(f'Q_index must be an int or None, got {type(Q_index).__name__}')

            if Q_index < 0 or Q_index >= len(self._Q):
                raise IndexError(
                    f'Q_index {Q_index} is out of bounds for Q of length {len(self._Q)}'
                )
            self._energy_offsets[Q_index].fixed = fixed

    def _ensure_energy_offsets_current(self) -> None:
        """Rebuild energy offset Parameters if Q has changed since they were last built."""
        if self._energy_offsets_is_dirty:
            self._generate_energy_offsets()
            self._energy_offsets_is_dirty = False

    def _generate_energy_offsets(self) -> None:
        """Generate energy offset Parameters for each Q value."""
        if self._Q is None:
            self._energy_offsets = []
            return

        self._energy_offsets = [copy(self._energy_offset) for _ in self._Q]

    def _on_Q_change(self) -> None:
        """Handle changes to the Q values."""
        self._energy_offsets_is_dirty = True
        self.resolution_model.Q = self.Q
        self.background_model.Q = self.Q

    def _on_energy_offset_change(self) -> None:
        """Handle changes to the energy offset."""
        self._ensure_energy_offsets_current()
        for offset in self._energy_offsets:
            offset.value = self._energy_offset.value

    def _on_resolution_model_change(self) -> None:
        """Handle changes to the resolution model."""
        self.resolution_model.Q = self.Q

    def _on_background_model_change(self) -> None:
        """Handle changes to the background model."""
        self.background_model.Q = self.Q

    # -------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a string representation of the InstrumentModel.

        Returns
        -------
        str
            A string representation of the InstrumentModel.
        """

        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'unit={self.unit}, '
            f'Q_len={None if self._Q is None else len(self._Q)}, '
            f'resolution_model={self._resolution_model!r}, '
            f'background_model={self._background_model!r})'
        )
