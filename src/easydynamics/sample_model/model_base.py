# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase as EasyScienceModelBase
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import _validate_unit


class ModelBase(EasyScienceModelBase):
    """Base class for Sample Models.

    Contains common functionality for models with components and Q
    dependence.
    """

    def __init__(
        self,
        display_name: str = 'MyModelBase',
        unique_name: str | None = None,
        unit: str | sc.Unit | None = 'meV',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        """Initialize the ModelBase.

        Args:
            display_name (str, default="MyModelBase"): Display name of the model.
            unique_name (str | None, default=None): Unique name of the model. If None,
                a unique name will be generated.
            unit (str | sc.Unit | None, default="meV"): Unit of the model.
            components (ModelComponent | ComponentCollection | None, default=None):
                Template components of the model. If None, no components
                are added. These components are copied into
                ComponentCollections for each Q value.
            Q (Q_type | None, default=None): Q values for the model.
                If None, Q is not set.

        Raises:
            TypeError: If components is not a ModelComponent or ComponentCollection.
        """
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )
        self._unit = _validate_unit(unit)
        self._Q = _validate_and_convert_Q(Q)

        if components is not None and not isinstance(
            components, (ModelComponent, ComponentCollection)
        ):
            raise TypeError(
                f'Components must be a ModelComponent, a ComponentCollection or None, '
                f'got {type(components).__name__}'
            )

        self._components = ComponentCollection()
        if isinstance(components, (ModelComponent, ComponentCollection)):
            self.append_component(components)

        self._generate_component_collections()

    def evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> list[np.ndarray]:
        """Evaluate the sample model at all Q for the given x values.

        Args:
            x (Numeric | list | np.ndarray | sc.Variable | sc.DataArray):
                Energy axis values to evaluate the model at. If a scipp
                Variable or DataArray is provided, the unit of the model
                will be converted to match the unit of x for evaluation, and
                the result will be returned in the same unit as x.

        Returns:
            list[np.ndarray]: A list of numpy arrays containing the
                evaluated model values for each Q. The length of the
                list will match the number of Q values in the model.

        Raises:
            ValueError: If there are no components in the model to
                evaluate.
        """

        if not self._component_collections:
            raise ValueError(
                'No components in the model to evaluate. '
                'Run generate_component_collections() first'
            )
        return [collection.evaluate(x) for collection in self._component_collections]

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------
    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """Append a ModelComponent or ComponentCollection to the
        SampleModel.

        Args:
            component (ModelComponent | ComponentCollection): The
                ModelComponent or ComponentCollection to append.
        """
        self._components.append_component(component)
        self._on_components_change()

    def remove_component(self, unique_name: str) -> None:
        """Remove a ModelComponent from the SampleModel by its unique
        name.

        Args:
            unique_name (str): The unique name of the ModelComponent
                to remove.
        """
        self._components.remove_component(unique_name)
        self._on_components_change()

    def clear_components(self) -> None:
        """Clear all ModelComponents from the SampleModel."""
        self._components.clear_components()
        self._on_components_change()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def unit(self) -> str | sc.Unit | None:
        """Get the unit of the ComponentCollection.

        Returns:
            str | sc.Unit | None: The unit of the ComponentCollection.
        """

        return self._unit

    @unit.setter
    def unit(self, _unit_str: str) -> None:
        """Unit is read-only and cannot be set directly.

        Args:
            _unit_str (str): The new unit to set (ignored).

        Raises:
            AttributeError: Always raised to indicate that the unit is
                read-only.
        """
        raise AttributeError(
            f'Unit is read-only. Use convert_unit to change the unit between allowed types '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )  # noqa: E501

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """Convert the unit of the ComponentCollection and all its
        components.

        Args:
            unit (str | sc.Unit): The new unit to convert to.

        Raises:
            TypeError: If the provided unit is not a string or sc.Unit.
            Exception: If the provided unit is not compatible with the
                current unit.
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

    @property
    def components(self) -> list[ModelComponent]:
        """Get the components of the SampleModel.

        Returns:
            list[ModelComponent]: The components of the SampleModel.
        """
        return self._components.components

    @components.setter
    def components(self, value: ModelComponent | ComponentCollection | None) -> None:
        """Set the components of the SampleModel.

        Args:
            value (ModelComponent | ComponentCollection | None): The new
                components to set. If None, all components will be cleared.

        Raises:
            TypeError: If value is not a ModelComponent,
                ComponentCollection, or None.
        """
        if not isinstance(value, (ModelComponent, ComponentCollection, type(None))):
            raise TypeError('Components must be a ModelComponent or a ComponentCollection')

        self.clear_components()
        if value is not None:
            self.append_component(value)

    @property
    def Q(self) -> np.ndarray | None:
        """Get the Q values of the SampleModel.

        Returns:
            np.ndarray | None: The Q values of the SampleModel, or None
                if not set.
        """
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """Set the Q values of the SampleModel. If Q is already set, it
        throws an error if the new Q values are not similar to the old
        ones. To change Q values, first run clear_Q().

        Args:
            value (Q_type | None): The new Q values to set.
                If None, Q values are not changed.

        Raises:
            ValueError: If the new Q values are not similar to the old
                ones when Q is already set.
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

    def clear_Q(self, confirm: bool = False) -> None:
        """Clear the Q values of the SampleModel, removing all component
        collections and their associated Parameters.

        Args:
            confirm (bool, default=False): Confirmation to clear Q values.

        Raises:
            ValueError: If confirm is not True.
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
    def fix_all_parameters(self) -> None:
        """Fix all Parameters in all ComponentCollections."""
        for par in self.get_all_variables():
            par.fixed = True

    def free_all_parameters(self) -> None:
        """Free all Parameters in all ComponentCollections."""
        for par in self.get_all_variables():
            par.fixed = False

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """Get all Parameters and Descriptors from all
        ComponentCollections in the ModelBase. Parameters Ignores the
        Parameters and Descriptors in self._components as these are just
        templates.

        Args:
            Q_index  (int | None, default=None): If None, get variables for all
                ComponentCollections. If int, get variables for the
                ComponentCollection at this index. Defaults to None.

        Returns:
            list[Parameter]: A list of all Parameters and Descriptors
                from the ComponentCollections in the ModelBase.

        Raises:
            TypeError: If Q_index is not an int or None.
            IndexError: If Q_index is out of bounds for the number of
                ComponentCollections.
        """

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
        """Get the ComponentCollection at the given Q index.

        Args:
            Q_index (int): The index of the desired ComponentCollection.

        Returns:
            ComponentCollection: The ComponentCollection at the
            specified Q index.

        Raises:
            TypeError: If Q_index is not an int.
            IndexError: If Q_index is out of bounds for the number of
                ComponentCollections.
        """
        if not isinstance(Q_index, int):
            raise TypeError(f'Q_index must be an int, got {type(Q_index).__name__}')
        if Q_index < 0 or Q_index >= len(self._component_collections):
            raise IndexError(
                f'Q_index {Q_index} is out of bounds for component collections '
                f'of length {len(self._component_collections)}'
            )
        return self._component_collections[Q_index]

    def normalize_area(self) -> None:
        """Normalize the area of the model across all Q values."""
        for collection in self._component_collections:
            collection.normalize_area()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _generate_component_collections(self) -> None:
        """Generate ComponentCollections for each Q value."""

        if self._Q is None:
            self._component_collections = []
            return

        self._component_collections = []
        for _ in self._Q:
            self._component_collections.append(copy(self._components))

    def _on_Q_change(self) -> None:
        """Handle changes to the Q values."""
        self._generate_component_collections()

    def _on_components_change(self) -> None:
        """Handle changes to the components."""
        self._generate_component_collections()

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation of the ModelBase.

        Returns:
            str: A string representation of the ModelBase.
        """
        return (
            f'{self.__class__.__name__}(unique_name={self.unique_name}, '
            f'unit={self.unit}), Q = {self.Q}, components = {self.components}'
        )
