# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
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
    experiment at various Q. It can contain a model of the resolution
    function for convolutions, of the background and an offset in the
    energy axis.

    Parameters
    ----------
    display_name : str, optional
        The display name of the InstrumentModel. Default is
        "MyInstrumentModel".
    unique_name : str or None, optional
        The unique name of the InstrumentModel. Default is None.
    Q : np.ndarray, list, scipp Variable or None, optional
        The Q values where the instrument is modelled.
    resolution_model : ResolutionModel or None, optional
        The resolution model of the instrument. If None, an empty
        resolution model is created and no resolution convolution is
        carried out. Default is None.
    background_model : BackgroundModel or None, optional
        The background model of the instrument. If None, an empty
        background model is created, and the background evaluates to 0.
        Default is None.
    energy_offset : float, int or None, optional
        Template energy offset of the instrument. Will be copied to each
        Q value. If None, the energy offset will be 0. Default is None.
    unit : str or sc.Unit, optional
        The unit of the energy axis. Default is 'meV'.
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
    ):
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
        """The resolution model of the instrument."""
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, value: ResolutionModel):
        """Set the resolution model of the instrument."""
        if not isinstance(value, ResolutionModel):
            raise TypeError(
                f'resolution_model must be a ResolutionModel, got {type(value).__name__}'
            )
        self._resolution_model = value
        self._on_resolution_model_change()

    @property
    def background_model(self) -> BackgroundModel:
        """The background model of the instrument."""
        return self._background_model

    @background_model.setter
    def background_model(self, value: BackgroundModel):
        """Set the background model of the instrument."""
        if not isinstance(value, BackgroundModel):
            raise TypeError(
                f'background_model must be a BackgroundModel, got {type(value).__name__}'
            )
        self._background_model = value
        self._on_background_model_change()

    @property
    def Q(self) -> np.ndarray | None:
        """Get the Q values of the InstrumentModel."""
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """Set the Q values of the InstrumentModel."""
        self._Q = _validate_and_convert_Q(value)
        self._on_Q_change()

    @property
    def unit(self) -> sc.Unit:
        """Get the unit of the InstrumentModel.

        Returns
        -------
        str or sc.Unit or None
        """
        return self._unit

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f'Unit is read-only. Use convert_unit to change the unit between allowed types '
                f'or create a new {self.__class__.__name__} with the desired unit.'
            )
        )  # noqa: E501

    @property
    def energy_offset(self) -> Parameter:
        """The energy offset template parameter of the instrument
        model.
        """
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, value: Numeric):
        """Set the offset parameter of the instrument model.".

        Parameters
        ----------
        value : float or int
            The new value for the energy offset parameter. Will be
            copied to all Q values.
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

    def convert_unit(self, unit_str: str | sc.Unit) -> None:
        """Convert the unit of the InstrumentModel.

        Parameters
        ----------
        unit_str : str or sc.Unit
            The unit to convert to.

        Raises
        ------
        TypeError
            If unit_str is not a string or scipp Unit.
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

        Parameters
        ----------
        Q_index : int | None
            The index of the Q value to get variables for. If None, get
            variables for all Q values.
        Returns
        -------
        list of Parameter
            All variables in the InstrumentModel.
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

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'unit={self.unit}, '
            f'Q_len={None if self._Q is None else len(self._Q)}, '
            f'resolution_model={self._resolution_model!r}, '
            f'background_model={self._background_model!r}'
            f')'
        )
