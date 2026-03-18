# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import scipp as sc
from easyscience.base_classes.new_base import NewBase
from easyscience.variable import Parameter

from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.resolution_model import ResolutionModel
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import _validate_unit


class InstrumentModel(NewBase):
    """InstrumentModel represents a model of the instrument in an
    experiment at various Q.

    It can contain a model of the resolution function for convolutions,
    of the background and an offset in the energy axis.
    """

    def __init__(
        self,
        display_name: str = 'MyInstrumentModel',
        unique_name: str | None = None,
        Q: Q_type | None = None,
        resolution_model: ResolutionModel | None = None,
        background_model: BackgroundModel | None = None,
        energy_offset: Numeric | None = None,
        unit: str | sc.Unit = 'meV',
    ) -> None:
        """Initialize an InstrumentModel.

        Args:
            display_name (str, default="MyInstrumentModel"): The display name of the
                InstrumentModel. Default is "MyInstrumentModel".
            unique_name (str | None, default=None): The unique name of the
                InstrumentModel.
            Q (Q_type | None, default=None): The Q values where the instrument is modelled.
            resolution_model (ResolutionModel | None, default=None): The resolution
                model of the instrument. If None, an empty resolution
                model is created and no resolution convolution is
                carried out.
            background_model (BackgroundModel | None, default=None): The background
                model of the instrument. If None, an empty background
                model is created, and the background evaluates to 0.
            energy_offset (Numeric | None, default=None): Template energy offset
                of the instrument. Will be copied to each Q value. If
                None, the energy offset will be 0.
            unit (str | sc.Unit, default="meV"): The unit of the energy axis.

        Raises:
            TypeError: If resolution_model is not a ResolutionModel or
                None, or if background_model is not a BackgroundModel or None, or
                if energy_offset is not a number or None.
        """
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        self._unit = _validate_unit(unit)

        if resolution_model is None:
            self._resolution_model = ResolutionModel()
        else:
            if not isinstance(resolution_model, ResolutionModel):
                raise TypeError(
                    f'resolution_model must be a ResolutionModel or None, '
                    f'got {type(resolution_model).__name__}'
                )
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
        self._Q = _validate_and_convert_Q(Q)
        self._on_Q_change()

    # -------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------

    @property
    def resolution_model(self) -> ResolutionModel:
        """Get the resolution model of the instrument.

        Returns:
            ResolutionModel: The resolution model of the instrument.
        """
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, value: ResolutionModel) -> None:
        """Set the resolution model of the instrument.

        Args:
            value (ResolutionModel): The new resolution model of the
                instrument.

        Raises:
            TypeError: If value is not a ResolutionModel.
        """
        if not isinstance(value, ResolutionModel):
            raise TypeError(
                f'resolution_model must be a ResolutionModel, got {type(value).__name__}'
            )
        self._resolution_model = value
        self._on_resolution_model_change()

    @property
    def background_model(self) -> BackgroundModel:
        """Get the background model of the instrument.

        Returns:
            BackgroundModel: The background model of the instrument.
        """

        return self._background_model

    @background_model.setter
    def background_model(self, value: BackgroundModel) -> None:
        """Set the background model of the instrument.

        Args:
            value (BackgroundModel): The new background model of the
                instrument.

        Raises:
            TypeError: If value is not a BackgroundModel.
        """

        if not isinstance(value, BackgroundModel):
            raise TypeError(
                f'background_model must be a BackgroundModel, got {type(value).__name__}'
            )
        self._background_model = value
        self._on_background_model_change()

    @property
    def Q(self) -> np.ndarray | None:
        """Get the Q values of the InstrumentModel.

        Returns:
            np.ndarray | None: The Q values of the InstrumentModel, or
                None if not set
        """
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """Set the Q values of the InstrumentModel.

        Args:
            value (Q_type | None): The new Q values for the
                InstrumentModel.
        """
        self._Q = _validate_and_convert_Q(value)
        self._on_Q_change()

    @property
    def unit(self) -> str | sc.Unit:
        """Get the unit of the InstrumentModel.

        Returns:
            str | sc.Unit: The unit of the InstrumentModel.
        """
        return self._unit

    @unit.setter
    def unit(self, unit_str: str) -> None:
        """Set the unit of the InstrumentModel. The unit is read-only
        and cannot be set directly. Use convert_unit to change the unit
        between allowed types or create a new InstrumentModel with the
        desired unit.

        Args:
            unit_str (str): The new unit for the InstrumentModel
                (ignored)

        Raises:
            AttributeError: Always, as the unit is read-only.
        """
        raise AttributeError(
            f'Unit is read-only. Use convert_unit to change the unit between allowed types '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )  # noqa: E501

    @property
    def energy_offset(self) -> Parameter:
        """Get the energy offset template parameter of the instrument
        model.

        Returns:
            Parameter: The energy offset template parameter of the
                instrument model.
        """
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, value: Numeric) -> None:
        """Set the offset parameter of the instrument model.

        Args:
            value (Numeric): The new value for the energy offset
                parameter. Will be copied to all Q values.

        Raises:
            TypeError: If value is not a number.
        """
        if not isinstance(value, Numeric):
            raise TypeError(f'energy_offset must be a number, got {type(value).__name__}')
        self._energy_offset.value = value

        self._on_energy_offset_change()

    # --------------------------------------------------------------
    # Other methods
    # --------------------------------------------------------------

    def convert_unit(self, unit_str: str | sc.Unit) -> None:
        """Convert the unit of the InstrumentModel.

        Args:
            unit_str (str | sc.Unit): The unit to convert to.

        Raises:
            ValueError: If unit_str is not a valid unit string or
                scipp Unit.
        """
        unit = _validate_unit(unit_str)
        if unit is None:
            raise ValueError('unit_str must be a valid unit string or scipp Unit')

        self._background_model.convert_unit(unit)
        self._resolution_model.convert_unit(unit)
        self._energy_offset.convert_unit(unit)
        for offset in self._energy_offsets:
            offset.convert_unit(unit)

        self._unit = unit

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """Get all variables in the InstrumentModel.

        Args:
            Q_index (int | None, default=None): The index of the Q value to get
                variables for. If None, get variables for all Q values.

        Returns:
            list[Parameter]: A list of all variables in the
                InstrumentModel. If Q_index is specified, only variables
                from the ComponentCollection at the given Q index are
                included. Otherwise, all variables in the
                InstrumentModel are included.

        Raises:
            TypeError: If Q_index is not an int or None.
            IndexError: If Q_index is out of bounds for the Q values in
                the InstrumentModel.
        """
        if self._Q is None:
            return []

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

    def get_energy_offset_at_Q(self, Q_index: int) -> Parameter:
        """Get the energy offset Parameter at a specific Q index.

        Args:
            Q_index (int): The index of the Q value to get the energy
                offset for.

        Returns:
            Parameter: The energy offset Parameter at the specified Q
                index.

        Raises:
            ValueError: If no Q values are set in the InstrumentModel.
            IndexError: If Q_index is out of bounds.
        """
        if self._Q is None:
            raise ValueError('No Q values are set in the InstrumentModel.')

        if Q_index < 0 or Q_index >= len(self._Q):
            raise IndexError(f'Q_index {Q_index} is out of bounds for Q of length {len(self._Q)}')

        return self._energy_offsets[Q_index]

    # --------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------

    def _generate_energy_offsets(self) -> None:
        """Generate energy offset Parameters for each Q value."""
        if self._Q is None:
            self._energy_offsets = []
            return

        self._energy_offsets = [copy(self._energy_offset) for _ in self._Q]

    def _on_Q_change(self) -> None:
        """Handle changes to the Q values."""
        self._generate_energy_offsets()
        self._resolution_model.Q = self._Q
        self._background_model.Q = self._Q

    def _on_energy_offset_change(self) -> None:
        """Handle changes to the energy offset."""
        for offset in self._energy_offsets:
            offset.value = self._energy_offset.value

    def _on_resolution_model_change(self) -> None:
        """Handle changes to the resolution model."""
        self._resolution_model.Q = self._Q

    def _on_background_model_change(self) -> None:
        """Handle changes to the background model."""
        self._background_model.Q = self._Q

    # -------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation of the InstrumentModel.

        Returns:
            str: A string representation of the InstrumentModel.
        """

        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'unit={self.unit}, '
            f'Q_len={None if self._Q is None else len(self._Q)}, '
            f'resolution_model={self._resolution_model!r}, '
            f'background_model={self._background_model!r}'
            f')'
        )
