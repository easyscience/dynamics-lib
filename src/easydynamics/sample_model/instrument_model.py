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
    experiment at various Q.
    """

    def __init__(
        self,
        display_name: str = 'MySampleModel',
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

        # TODO: Think very carefully about units.

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
        self._update_models()

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
        self._update_models()

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
        self._update_models()

    @property
    def Q(self) -> np.ndarray | None:
        """Get the Q values of the InstrumentModel."""
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """Set the Q values of the InstrumentModel."""
        self._Q = _validate_and_convert_Q(value)
        self._update_models()

    @property
    def unit(self) -> str | sc.Unit:
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
        """The offset parameter of the instrument model."""
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, value: Numeric):
        "set the offset parameter of the instrument model."
        if not isinstance(value, Numeric):
            raise TypeError(f'energy_offset must be a number, got {type(value).__name__}')
        self._energy_offset.value = value

    @property
    def energy_offsets(self) -> Parameter:
        """The offset parameters of the instrument model."""
        return self._energy_offsets

    @energy_offsets.setter
    def energy_offsets(self, value: list[Numeric]):
        """Set the offset parameters of the instrument model.

        Args:
            value : list of numbers
                The offset parameters to set.
        Raises:
            TypeError: If value is not a list of numbers.
        """
        if not isinstance(value, list) or not all(isinstance(v, Numeric) for v in value):
            raise TypeError(
                f'energy_offsets must be a list of numbers, got {type(value).__name__}'
            )
        self._energy_offsets = value

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

        self._unit = unit
        self._background_model.convert_unit(unit)
        self._resolution_model.convert_unit(unit)
        self._energy_offset.unit = unit

    def get_all_variables(self, Q_index) -> list[Parameter]:
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
        if Q_index is None:
            variables = [self._energy_offset]
            if self._background_model is not None:
                variables.extend(self._background_model.get_all_variables())
            if self._resolution_model is not None:
                variables.extend(self._resolution_model.get_all_variables())
        else:
            variables = [self._energy_offsets[Q_index]]
            if self._background_model is not None:
                variables.extend(
                    self._background_model._component_collections[Q_index].get_all_variables()
                )
            if self._resolution_model is not None:
                variables.extend(
                    self._resolution_model._component_collections[Q_index].get_all_variables()
                )
        return variables

    # --------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------

    def _generate_energy_offsets(self) -> None:
        """Generate energy offsets for each Q value."""
        if self._Q is None:
            self._energy_offsets = []
            return

        self._energy_offsets = [0.0] * len(self._Q)
        for i in range(len(self._Q)):
            self._energy_offsets[i] = copy(self._energy_offset)

    def _update_models(self) -> None:
        """Update the Q values of the resolution and background
        models.
        """
        self._generate_energy_offsets()
        if self._Q is None:
            return
        if self._resolution_model is not None:
            self._resolution_model.Q = self._Q
        if self._background_model is not None:
            self._background_model.Q = self._Q

    # -------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------

    def __repr__(self):
        repr_string = f'{self.__class__.__name__}(unique_name={self.unique_name}, '
        repr_string += f'unit={self.unit}),'
        repr_string += f'resolution_model = {self.resolution_model},'
        repr_string += f'background_model = {self.background_model}, '
        repr_string += f'offset = {self.energy_offsets}'
        return repr_string
