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
from easydynamics.utils.utils import verify_Q_index


class ModelBase(EasyDynamicsModelBase):
    """
    Base class for Sample Models.

    Contains common functionality for models with components and Q dependence.
    """

    def __init__(
        self,
        display_name: str = 'MyModelBase',
        unique_name: str | None = None,
        unit: str | sc.Unit | None = 'meV',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        """
        Initialize the ModelBase.

        Parameters
        ----------
        display_name : str, default='MyModelBase'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        unit : str | sc.Unit | None, default='meV'
            Unit of the model.
        components : ModelComponent | ComponentCollection | None, default=None
            Template components of the model. If None, no components are added. These components
            are copied into ComponentCollections for each Q value.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.

        Raises
        ------
        TypeError
            If components is not a ModelComponent or ComponentCollection.
        """
        super().__init__(
            unit=unit,
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
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> list[np.ndarray]:
        """
        Evaluate the sample model at all Q for the given x values.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Energy axis values to evaluate the model at. If a scipp Variable or DataArray is
            provided, the unit of the model will be converted to match the unit of x for
            evaluation, and the result will be returned in the same unit as x.

        Raises
        ------
        ValueError
            If there are no components in the model to evaluate.

        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays containing the evaluated model values for each Q. The length of
            the list will match the number of Q values in the model.
        """

        self._ensure_component_collections_current()
        if not self._component_collections:
            raise ValueError('No components in the model to evaluate.')
        return [collection.evaluate(x) for collection in self._component_collections]

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------
    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """
        Append a ModelComponent or ComponentCollection to the SampleModel.

        Parameters
        ----------
        component : ModelComponent | ComponentCollection
            The ModelComponent or ComponentCollection to append.
        """
        self._components.append_component(component)
        self._on_components_change()

    def remove_component(self, name: str) -> None:
        """
        Remove a ModelComponent from the SampleModel by its name.

        Parameters
        ----------
        name : str
            The name of the ModelComponent to remove.
        """
        self._components.pop(name)
        self._on_components_change()

    def clear_components(self) -> None:
        """Clear all ModelComponents from the SampleModel."""
        self._components.clear()
        self._on_components_change()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self) -> list[ModelComponent]:
        """
        Get the components of the SampleModel.

        Returns
        -------
        list[ModelComponent]
            The components of the SampleModel.
        """
        return self._components

    @components.setter
    def components(self, value: ModelComponent | ComponentCollection | None) -> None:
        """
        Set the components of the SampleModel.

        Parameters
        ----------
        value : ModelComponent | ComponentCollection | None
            The new components to set. If None, all components will be cleared.

        Raises
        ------
        TypeError
            If value is not a ModelComponent, ComponentCollection, or None.
        """
        if not isinstance(value, (ModelComponent, ComponentCollection, type(None))):
            raise TypeError('Components must be a ModelComponent or a ComponentCollection')

        self.clear_components()
        if value is not None:
            self.append_component(value)

    @property
    def component_collections_is_dirty(self) -> bool:
        """
        Return whether component collections need to be rebuilt before use.

        Returns
        -------
        bool
            ``True`` if component collections have not been built yet or are stale.
        """
        return self._component_collections_is_dirty

    @property
    def Q(self) -> sc.Variable | None:
        """
        Get the Q values of the SampleModel.

        Returns
        -------
        sc.Variable | None
            The Q values of the SampleModel in 1/angstrom, or None if not set.
        """
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """
        Set the Q values of the SampleModel.

        If Q is already set, it throws an error if the new Q values are not similar to the old
        ones. To change Q values, first run clear_Q().

        Parameters
        ----------
        value : Q_type | None
            The new Q values to set. If None, Q values are not changed.

        Raises
        ------
        ValueError
            If the new Q values are not similar to the old ones when Q is already set.
        """
        if value is None:
            return
        old_Q = self._Q
        new_Q = _validate_and_convert_Q(value)

        if old_Q is None:
            self._Q = new_Q
            self._on_Q_change()
            return

        if len(old_Q) != len(new_Q) or not sc.allclose(old_Q, new_Q):
            raise ValueError(
                'New Q values are not similar to the old ones. '
                'To change Q values, first run clear_Q().'
            )

    def clear_Q(self, confirm: bool = False) -> None:
        """
        Clear the Q values of the SampleModel, removing all component collections and their
        associated Parameters.

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
        self._on_Q_change()

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the ComponentCollection and all its components.

        Parameters
        ----------
        unit : str | sc.Unit
            The new unit to convert to.

        Raises
        ------
        TypeError
            If the provided unit is not a string or sc.Unit.
        Exception
            If the provided unit is not compatible with the current unit.
        """

        old_unit = self._unit

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError(f'Unit must be a string or sc.Unit, got {type(unit).__name__}')
        try:
            for component in self.components:
                component.convert_unit(unit)
            self._unit = unit
        except Exception as e:
            # Attempt to rollback on failure
            try:
                for component in self.components:
                    component.convert_unit(old_unit)
            except Exception:  # noqa: S110
                pass  # Best effort rollback
            raise e
        self._on_components_change()

    def fix_all_parameters(self) -> None:
        """Fix all Parameters in all ComponentCollections."""
        for par in self.get_all_variables():
            par.fixed = True

    def free_all_parameters(self) -> None:
        """Free all Parameters in all ComponentCollections."""
        for par in self.get_all_variables():
            par.fixed = False

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all Parameters and Descriptors from all ComponentCollections in the ModelBase.

        Ignores the Parameters and Descriptors in self._components as these are just templates.

        Parameters
        ----------
        Q_index : int | None, default=None
            If None, get variables for all ComponentCollections. If int, get variables for the
            ComponentCollection at this index.

        Returns
        -------
        list[Parameter]
            A list of all Parameters and Descriptors from the ComponentCollections in the
            ModelBase.
        """
        self._ensure_component_collections_current()
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)
        if Q_index is None:
            all_vars = [
                var
                for collection in self._component_collections
                for var in collection.get_all_variables()
            ]
        else:
            all_vars = self._component_collections[Q_index].get_all_variables()
        return all_vars

    def get_component_collection(self, Q_index: int) -> ComponentCollection:
        """
        Get the ComponentCollection at the given Q index.

        Parameters
        ----------
        Q_index : int
            The index of the desired ComponentCollection.

        Returns
        -------
        ComponentCollection
            The ComponentCollection at the given Q index.
        """
        self._ensure_component_collections_current()
        verify_Q_index(Q_index=Q_index, Q=self.Q)
        return self._component_collections[Q_index]

    def normalize_area(self) -> None:
        """Normalize the area of the model across all Q values."""
        self._ensure_component_collections_current()
        for collection in self._component_collections:
            collection.normalize_area()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _ensure_component_collections_current(self) -> None:
        """
        Rebuild component collections if any dependency has changed since they were last built.
        """
        if self._component_collections_is_dirty:
            self._generate_component_collections()
            self._component_collections_is_dirty = False

    def _generate_component_collections(self) -> None:
        """Generate ComponentCollections for each Q value."""

        if self.Q is None:
            self._component_collections = []
            return

        self._component_collections = []
        for _ in range(len(self.Q)):
            self._component_collections.append(copy(self._components))

    def _on_Q_change(self) -> None:
        """Handle changes to the Q values."""
        self._component_collections_is_dirty = True

    def _on_components_change(self) -> None:
        """Handle changes to the components."""
        self._component_collections_is_dirty = True

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a string representation of the ModelBase.

        Returns
        -------
        str
            A string representation of the ModelBase.
        """
        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'unit={self.unit}, '
            f'Q={None if self.Q is None else self.Q.values}, '
            f'components={self.components})'
        )
