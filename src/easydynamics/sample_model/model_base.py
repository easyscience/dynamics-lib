# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
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

    Contains common functionality for models with components and
    Q dependence.

    Parameters
    ----------
    display_name : str
        Display name of the model.
    unique_name : str | None
        Unique name of the model. If None, a unique name will be
        generated.
    unit : str | sc.Unit | None
        Unit of the model. If None, unitless.
    components : ModelComponent | ComponentCollection | None
        Template components of the model. If None, no components
        are added.
        These components are copied into ComponentCollections for each
        Q value.
    Q : Q_type | None
        Q values for the model. If None, Q is not set.
    """

    def __init__(
        self,
        display_name: str = 'MyModelBase',
        unique_name: str | None = None,
        unit: str | sc.Unit | None = 'meV',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ):
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

        Parameters
        ----------
        x : Number, list, np.ndarray, sc.Variable, or sc.DataArray
            Energy axis.

        Returns
        -------
        list[np.ndarray]
            Evaluated model values.
        """

        if not self._component_collections:
            raise ValueError(
                'No components in the model to evaluate. '
                'Run generate_component_collections() first'
            )
        y = [collection.evaluate(x) for collection in self._component_collections]

        return y

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------
    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """Append a ModelComponent or ComponentCollection to the
        SampleModel.

        Args:
            component (ModelComponent | ComponentCollection):
            The ModelComponent or ComponentCollection to append.
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
    def unit(self) -> str | sc.Unit:
        """Get the unit of the ComponentCollection.

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

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """Convert the unit of the ComponentCollection and all its
        components.
        """

        old_unit = self._unit

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
        """Get the components of the SampleModel."""
        return self._components.components

    @components.setter
    def components(self, value: ModelComponent | ComponentCollection | None) -> None:
        """Set the components of the SampleModel."""
        if not isinstance(value, (ModelComponent, ComponentCollection, type(None))):
            raise TypeError('Components must be a ModelComponent or a ComponentCollection')

        self.clear_components()
        if value is not None:
            self.append_component(value)

    @property
    def Q(self) -> np.ndarray | None:
        """Get the Q values of the SampleModel."""
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """Set the Q values of the SampleModel."""
        old_Q = self._Q
        new_Q = _validate_and_convert_Q(value)

        if (
            old_Q is not None
            and new_Q is not None
            and len(old_Q) == len(new_Q)
            and all(np.isclose(old_Q, new_Q))
        ):
            return  # No change in Q, so do nothing
        self._Q = new_Q
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

        Parameters
        ----------
        Q_index : int | None
            If int, get variables for the ComponentCollection at
            this index. If None, get variables for all
            ComponentCollections.
        Returns
        -------
        list[Parameter]
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

        Parameters
        ----------
        Q_index : int
            The index of the desired ComponentCollection.

        Returns
        -------
        ComponentCollection
            The ComponentCollection at the specified Q index.
        """
        if not isinstance(Q_index, int):
            raise TypeError(f'Q_index must be an int, got {type(Q_index).__name__}')
        if Q_index < 0 or Q_index >= len(self._component_collections):
            raise IndexError(
                f'Q_index {Q_index} is out of bounds for component collections '
                f'of length {len(self._component_collections)}'
            )
        return self._component_collections[Q_index]

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

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(unique_name={self.unique_name}, '
            f'unit={self.unit}), Q = {self.Q}, components = {self.components}'
        )
