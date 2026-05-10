# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorBase
from easyscience.variable import Parameter

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.components.model_component import ModelComponent

if TYPE_CHECKING:
    from easydynamics.utils.utils import Numeric


class ComponentCollection(EasyDynamicsModelBase):
    """
    Collection of model components representing a sample, background or resolution model.
    """

    def __init__(
        self,
        components: list[ModelComponent] | None = None,
        unit: str | sc.Unit = 'meV',
        name: str = 'ComponentCollection',
        display_name: str | None = 'MyComponentCollection',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new ComponentCollection.

        Parameters
        ----------
        components : list[ModelComponent] | None, default=None
            Initial model components to add to the ComponentCollection.
        unit : str | sc.Unit, default='meV'
            Unit of the collection.
        name : str, default='ComponentCollection'
            Name of the collection.
        display_name : str | None, default='MyComponentCollection'
            Display name of the collection.
        unique_name : str | None, default=None
            Unique name of the collection.

        Raises
        ------
        TypeError
            If unit is not a string or sc.Unit, or if components is not a list of ModelComponent.
        """

        super().__init__(
            unit=unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        self._components = []

        # Add initial components if provided. Used for serialization.
        if components is not None:
            if not isinstance(components, list):
                raise TypeError('components must be a list of ModelComponent instances.')
            for comp in components:
                self.append_component(comp)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self) -> list[ModelComponent]:
        """
        Get the list of components in the collection.

        Returns
        -------
        list[ModelComponent]
            The components in the collection.
        """

        return list(self._components)

    @components.setter
    def components(self, components: list[ModelComponent]) -> None:
        """
        Set the list of components in the collection.

        Parameters
        ----------
        components : list[ModelComponent]
            The new list of components.

        Raises
        ------
        TypeError
            If components is not a list of ModelComponent.
        """

        if not isinstance(components, list):
            raise TypeError('components must be a list of ModelComponent instances.')
        for comp in components:
            if not isinstance(comp, ModelComponent):
                raise TypeError(
                    'All items in components must be instances of ModelComponent. '
                    f'Got {type(comp).__name__} instead.'
                )

        self._components = components

    @property
    def is_empty(self) -> bool:
        """
        Check if the ComponentCollection has no components.

        Returns
        -------
        bool
            True if the collection has no components, False otherwise.
        """
        return not self._components

    @is_empty.setter
    def is_empty(self, _value: bool) -> None:
        """
        Is_empty is a read-only property that indicates whether the collection has components.

        Parameters
        ----------
        _value : bool
            The value to set (ignored).

        Raises
        ------
        AttributeError
            Always raised since is_empty is read-only.
        """
        raise AttributeError(
            'is_empty is a read-only property that indicates '
            'whether the collection has components.'
        )

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the ComponentCollection and all its components.

        Parameters
        ----------
        unit : str | sc.Unit
            The target unit to convert to.

        Raises
        ------
        TypeError
            If unit is not a string or sc.Unit.
        Exception
            If any component cannot be converted to the specified unit.
        """

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError(f'Unit must be a string or sc.Unit, got {type(unit).__name__}')

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

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """
        Append a model component or the components from another ComponentCollection to this
        ComponentCollection.

        Parameters
        ----------
        component : ModelComponent | ComponentCollection
            The component to append. If a ComponentCollection is provided, all of its components
            will be appended.

        Raises
        ------
        TypeError
            If component is not a ModelComponent or ComponentCollection.
        ValueError
            If a component with the same unique name already exists in the collection.
        """
        if not isinstance(component, (ModelComponent, ComponentCollection)):
            raise TypeError(
                'Component must be an instance of ModelComponent or ComponentCollection. '
                f'Got {type(component).__name__} instead.'
            )
        if isinstance(component, ModelComponent):
            components = (component,)
        if isinstance(component, ComponentCollection):
            components = component.components

        for comp in components:
            if comp in self._components:
                raise ValueError(
                    f"Component '{comp.unique_name}' is already in the collection. "
                    f'Existing components: {self.list_component_names()}'
                )

            self._components.append(comp)

    def remove_component(self, unique_name: str) -> None:
        """
        Remove a component from the collection by its unique name.

        Parameters
        ----------
        unique_name : str
            Unique name of the component to remove.

        Raises
        ------
        TypeError
            If unique_name is not a string.
        KeyError
            If no component with the given unique name exists in the collection.
        """

        if not isinstance(unique_name, str):
            raise TypeError('Component name must be a string.')

        for comp in self._components:
            if comp.unique_name == unique_name:
                self._components.remove(comp)
                return

        raise KeyError(
            f"No component named '{unique_name}' exists. "
            f'Did you accidentally use the display_name? '
            f'Here is a list of the components in the collection: {self.list_component_names()}'
        )

    @property
    def components(self) -> list[ModelComponent]:
        """
        Get the list of components in the collection.

        Returns
        -------
        list[ModelComponent]
            The components in the collection.
        """
        return list(self._components)

    @components.setter
    def components(self, components: list[ModelComponent]) -> None:
        """
        Set the components in the collection.

        Parameters
        ----------
        components : list[ModelComponent]
            The new components in the collection.

        Raises
        ------
        TypeError
            If components is not a list of ModelComponent.
        """
        if not isinstance(components, list):
            raise TypeError('components must be a list of ModelComponent instances.')
        for comp in components:
            if not isinstance(comp, ModelComponent):
                raise TypeError(
                    'All items in components must be instances of ModelComponent. '
                    f'Got {type(comp).__name__} instead.'
                )

        self._components = components

    @property
    def is_empty(self) -> bool:
        """
        Returns True if the collection has no components, otherwise False.

        Returns
        -------
        bool
            True if the collection has no components, otherwise False.
        """
        return not self._components

    @is_empty.setter
    def is_empty(self, _value: bool) -> None:
        """
        Is_empty is read-only.

        Parameters
        ----------
        _value : bool
            Ignored.

        Raises
        ------
        AttributeError
            Always raised since is_empty is read-only.
        """
        raise AttributeError(
            'is_empty is a read-only property that indicates '
            'whether the collection has components.'
        )

    def list_component_names(self) -> list[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        list[str]
            List of unique names of the components in the collection.
        """

        return [component.unique_name for component in self._components]

    def clear_components(self) -> None:
        """Remove all components."""
        self._components.clear()

    def normalize_area(self) -> None:
        """
        Normalize the areas of all components so they sum to 1.

        This is useful for convolutions.

        Raises
        ------
        ValueError
            If there are no components in the model or if the total area is zero or not finite,
            which would prevent normalization.
        """
        if not self.components:
            raise ValueError('No components in the model to normalize.')

        area_params = []
        total_area = Parameter(name='total_area', value=0.0, unit=self._unit)

        for component in self.components:
            if hasattr(component, 'area'):
                area_params.append(component.area)
                total_area += component.area
            else:
                warnings.warn(
                    f"Component '{component.unique_name}' does not have an 'area' attribute "
                    f'and will be skipped in normalization.',
                    UserWarning,
                    stacklevel=2,
                )

        if total_area.value == 0:
            raise ValueError('Total area is zero; cannot normalize.')

        if not np.isfinite(total_area.value):
            raise ValueError('Total area is not finite; cannot normalize.')

        for param in area_params:
            param.value /= total_area.value

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def get_all_variables(self) -> list[DescriptorBase]:
        """
        Get all parameters from the model component.

        Returns
        -------
        list[DescriptorBase]
            List of parameters in the component.
        """

        return [var for component in self.components for var in component.get_all_variables()]

    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        """
        Evaluate the sum of all components.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Energy axis.

        Returns
        -------
        np.ndarray
            Evaluated model values.
        """

        if not self.components:
            return np.zeros_like(x)
        return sum(component.evaluate(x) for component in self.components)

    def evaluate_component(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        unique_name: str,
    ) -> np.ndarray:
        """
        Evaluate a single component by name.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Energy axis.
        unique_name : str
            Component unique name.

        Raises
        ------
        ValueError
            If there are no components in the model.
        TypeError
            If unique_name is not a string.
        KeyError
            If no component with the given unique name exists in the collection.

        Returns
        -------
        np.ndarray
            Evaluated values for the specified component.
        """
        if not self.components:
            raise ValueError('No components in the model to evaluate.')

        if not isinstance(unique_name, str):
            raise TypeError(
                f'Component unique name must be a string, got {type(unique_name)} instead.'
            )

        matches = [comp for comp in self.components if comp.unique_name == unique_name]
        if not matches:
            raise KeyError(f"No component named '{unique_name}' exists.")

        component = matches[0]

        return component.evaluate(x)

    def fix_all_parameters(self) -> None:
        """Fix all free parameters in the model."""
        for param in self.get_fittable_parameters():
            param.fixed = True

    def free_all_parameters(self) -> None:
        """Free all fixed parameters in the model."""
        for param in self.get_fittable_parameters():
            param.fixed = False

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __contains__(self, item: str | ModelComponent) -> bool:
        """
        Check if a component with the given name or instance exists in the ComponentCollection.

        Parameters
        ----------
        item : str | ModelComponent
            The component name or instance to check for.

        Returns
        -------
        bool
            True if the component exists, False otherwise.
        """

        if isinstance(item, str):
            # Check by component unique name
            return any(comp.unique_name == item for comp in self.components)
        if isinstance(item, ModelComponent):
            # Check by component instance
            return any(comp is item for comp in self.components)
        return False

    def __repr__(self) -> str:
        """
        Return a string representation of the ComponentCollection.

        Returns
        -------
        str
            String representation of the ComponentCollection.
        """
        comp_names = ', '.join(c.unique_name for c in self.components) or 'No components'

        return f"<ComponentCollection unique_name='{self.unique_name}' | Components: {comp_names}>"
