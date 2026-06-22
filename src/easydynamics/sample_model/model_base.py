# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q


class ModelBase(EasyDynamicsModelBase):
    """
    Base class for Sample Models.

    Contains common functionality for models with components and Q dependence.
    """

    def __init__(
        self,
        display_name: str = 'MyModelBase',
        unique_name: str | None = None,
        x_unit: str | sc.Unit | None = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            display_name=display_name,
            unique_name=unique_name,
        )
        self._Q = _validate_and_convert_Q(Q)

        if components is not None and not isinstance(
            components, (ModelComponent, ComponentCollection)
        ):
            raise TypeError(
                f'Components must be a ModelComponent, a ComponentCollection or None, '
                f'got {type(components).__name__}'
            )

        self._components = ComponentCollection()
        self._component_collections: list[ComponentCollection] = []
        self._component_collections_is_dirty = True
        if isinstance(components, (ModelComponent, ComponentCollection)):
            self.append_component(components)

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> list[np.ndarray] | list[sc.Variable]:
        """Evaluate the model at all Q for the given x values."""
        self._ensure_component_collections_current()
        if not self._component_collections:
            raise ValueError('No components in the model to evaluate.')
        return [
            collection.evaluate(x, output=output) for collection in self._component_collections
        ]

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------
    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        self._components.append_component(component)
        self._on_components_change()

    def remove_component(self, name: str) -> None:
        self._components.pop(name)
        self._on_components_change()

    def clear_components(self) -> None:
        self._components.clear()
        self._on_components_change()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self) -> list[ModelComponent]:
        return self._components

    @components.setter
    def components(self, value: ModelComponent | ComponentCollection | None) -> None:
        if not isinstance(value, (ModelComponent, ComponentCollection, type(None))):
            raise TypeError('Components must be a ModelComponent or a ComponentCollection')

        self.clear_components()
        if value is not None:
            self.append_component(value)

    @property
    def component_collections_is_dirty(self) -> bool:
        return self._component_collections_is_dirty

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

    def clear_Q(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError(
                'Clearing Q values requires confirmation. Set confirm=True to proceed.'
            )
        self._Q = None
        self._on_Q_change()

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def convert_x_unit(self, unit: str | sc.Unit) -> None:
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError(f'Unit must be a string or sc.Unit, got {type(unit).__name__}')

        old_unit = self._x_unit
        try:
            for component in self.components:
                component.convert_x_unit(unit)
            self._x_unit = str(unit) if isinstance(unit, sc.Unit) else unit
        except Exception as e:
            try:
                for component in self.components:
                    component.convert_x_unit(old_unit)
            except Exception:  # noqa: S110
                pass
            raise e
        self._on_components_change()

    def convert_y_unit(self, unit: str | sc.Unit) -> None:
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError(f'Unit must be a string or sc.Unit, got {type(unit).__name__}')

        old_unit = self._y_unit
        try:
            for component in self.components:
                component.convert_y_unit(unit)
            self._y_unit = str(unit) if isinstance(unit, sc.Unit) else unit
        except Exception as e:
            try:
                for component in self.components:
                    component.convert_y_unit(old_unit)
            except Exception:  # noqa: S110
                pass
            raise e
        self._on_components_change()

    def fix_all_parameters(self) -> None:
        for par in self.get_all_variables():
            par.fixed = True

    def free_all_parameters(self) -> None:
        for par in self.get_all_variables():
            par.fixed = False

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        self._ensure_component_collections_current()
        if Q_index is None:
            all_vars = [
                var
                for collection in self._component_collections
                for var in collection.get_all_variables()
            ]
        else:
            if not isinstance(Q_index, int):
                raise TypeError(f'Q_index must be an int or None, got {type(Q_index).__name__}')
            if Q_index < 0 or Q_index >= len(self._component_collections):
                raise IndexError(
                    f'Q_index {Q_index} is out of bounds for component collections '
                    f'of length {len(self._component_collections)}'
                )
            all_vars = self._component_collections[Q_index].get_all_variables()
        return all_vars

    def get_component_collection(self, Q_index: int) -> ComponentCollection:
        self._ensure_component_collections_current()
        if not isinstance(Q_index, int):
            raise TypeError(f'Q_index must be an int, got {type(Q_index).__name__}')
        if Q_index < 0 or Q_index >= len(self._component_collections):
            raise IndexError(
                f'Q_index {Q_index} is out of bounds for component collections '
                f'of length {len(self._component_collections)}'
            )
        return self._component_collections[Q_index]

    def normalize_area(self) -> None:
        self._ensure_component_collections_current()
        for collection in self._component_collections:
            collection.normalize_area()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _ensure_component_collections_current(self) -> None:
        if self._component_collections_is_dirty:
            self._generate_component_collections()
            self._component_collections_is_dirty = False

    def _generate_component_collections(self) -> None:
        if self.Q is None:
            self._component_collections = []
            return

        self._component_collections = []
        for _ in self.Q:
            self._component_collections.append(copy(self._components))

    def _on_Q_change(self) -> None:
        self._component_collections_is_dirty = True

    def _on_components_change(self) -> None:
        self._component_collections_is_dirty = True

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(unique_name={self.unique_name}, '
            f'x_unit={self.x_unit}), Q = {self.Q}, components = {self.components}'
        )
