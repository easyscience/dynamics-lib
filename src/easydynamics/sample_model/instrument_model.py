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
        x_unit: str | sc.Unit = 'meV',
    ) -> None:
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        self._x_unit = _validate_unit(x_unit)

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
            unit=self.x_unit,
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
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, value: ResolutionModel | SampleModel) -> None:
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
        return self._background_model

    @background_model.setter
    def background_model(self, value: BackgroundModel) -> None:
        if not isinstance(value, BackgroundModel):
            raise TypeError(
                f'background_model must be a BackgroundModel, got {type(value).__name__}'
            )
        self._background_model = value
        self._on_background_model_change()

    @property
    def Q(self) -> np.ndarray | None:
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
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
    def x_unit(self) -> str | sc.Unit:
        return self._x_unit

    @x_unit.setter
    def x_unit(self, _: str) -> None:
        raise AttributeError(
            f'x_unit is read-only. Use convert_x_unit to change the unit between allowed types '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    @property
    def energy_offset(self) -> Parameter:
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, value: Numeric) -> None:
        if not isinstance(value, Numeric):
            raise TypeError(f'energy_offset must be a number, got {type(value).__name__}')
        self._energy_offset.value = value
        self._on_energy_offset_change()

    # --------------------------------------------------------------
    # Other methods
    # --------------------------------------------------------------

    def clear_Q(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError(
                'Clearing Q values requires confirmation. Set confirm=True to proceed.'
            )
        self._Q = None
        self.background_model.clear_Q(confirm=True)
        self.resolution_model.clear_Q(confirm=True)
        self._on_Q_change()

    def convert_x_unit(self, x_unit: str | sc.Unit) -> None:
        unit = _validate_unit(x_unit)
        if unit is None:
            raise ValueError('x_unit must be a valid unit string or scipp Unit')

        self._background_model.convert_x_unit(unit)
        self._resolution_model.convert_x_unit(unit)
        self._energy_offset.convert_unit(unit)
        self._ensure_energy_offsets_current()
        for offset in self._energy_offsets:
            offset.convert_unit(unit)

        self._x_unit = unit

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
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
        self.resolution_model.fix_all_parameters()

    def free_resolution_parameters(self) -> None:
        self.resolution_model.free_all_parameters()

    def normalize_resolution(self) -> None:
        self.resolution_model.normalize_area()

    def get_energy_offset(
        self,
        Q_index: int | None = None,
    ) -> Parameter | list[Parameter]:
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
        self._fix_or_free_energy_offset(Q_index, fixed=True)

    def free_energy_offset(self, Q_index: int | None = None) -> None:
        self._fix_or_free_energy_offset(Q_index, fixed=False)

    # --------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------
    def _fix_or_free_energy_offset(self, Q_index: int | None = None, fixed: bool = True) -> None:
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
        if self._energy_offsets_is_dirty:
            self._generate_energy_offsets()
            self._energy_offsets_is_dirty = False

    def _generate_energy_offsets(self) -> None:
        if self._Q is None:
            self._energy_offsets = []
            return

        self._energy_offsets = [copy(self._energy_offset) for _ in self._Q]

    def _on_Q_change(self) -> None:
        self._energy_offsets_is_dirty = True
        self.resolution_model.Q = self.Q
        self.background_model.Q = self.Q

    def _on_energy_offset_change(self) -> None:
        self._ensure_energy_offsets_current()
        for offset in self._energy_offsets:
            offset.value = self._energy_offset.value

    def _on_resolution_model_change(self) -> None:
        self.resolution_model.Q = self.Q

    def _on_background_model_change(self) -> None:
        self.background_model.Q = self.Q

    # -------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'x_unit={self.x_unit}, '
            f'Q_len={None if self._Q is None else len(self._Q)}, '
            f'resolution_model={self._resolution_model!r}, '
            f'background_model={self._background_model!r}'
            f')'
        )
