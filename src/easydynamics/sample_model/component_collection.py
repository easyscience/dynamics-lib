# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorBase
from easyscience.variable import Parameter

from easydynamics.base_classes.easydynamics_list import EasyDynamicsList
from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.components.model_component import ModelComponent

if TYPE_CHECKING:
    from easydynamics.utils.utils import Numeric


class ComponentCollection(EasyDynamicsList, EasyDynamicsModelBase):
    """
    Collection of model components.

    Examples
    --------
    Create a ComponentCollection with two components:
    >>> import easydynamics.sample_model as sm
    >>> component1 = sm.Gaussian(name='Gaussian1', area=1.0, width=1.0)
    >>> component2 = sm.Lorentzian(name='Lorentzian1', area=2.0, width=0.5)
    >>> collection = sm.ComponentCollection(components=[component1, component2])

    Append a component to the collection:
    >>> component3 = sm.Gaussian(name='Gaussian2', area=0.5, width=0.8)
    >>> collection.append(component3)

    Evaluate the collection at a given energy axis:
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> values = collection.evaluate(x)

    Remove a component by name:
    >>> collection.remove('Gaussian1')

    List component names:
    >>> collection.list_component_names()
    ['Lorentzian1', 'Gaussian2']
    """

    def __init__(
        self,
        components: ModelComponent | list[ModelComponent] | None = None,
        unit: str | sc.Unit = "meV",
        name: str = "ComponentCollection",
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new ComponentCollection.

        Parameters
        ----------
        components : ModelComponent | list[ModelComponent] | None, default=None
            Initial model components to add to the ComponentCollection.
        unit : str | sc.Unit, default='meV'
            Unit of the collection.
        name : str, default='ComponentCollection'
            Name of the collection.
        display_name : str | None, default=None
            Display name of the collection.
        unique_name : str | None, default=None
            Unique name of the collection.

        Raises
        ------
        TypeError
            If components is not a list of ModelComponent.
        """
        if components is None:
            components = []
        if isinstance(components, ModelComponent):
            components = [components]
        elif not isinstance(components, list):
            raise TypeError(
                f"components must be a ModelComponent or a list of ModelComponent, got {type(components).__name__} instead."  # noqa: E501
            )
        for comp in components:
            if not isinstance(comp, ModelComponent):
                raise TypeError(
                    f"All items in components must be instances of ModelComponent, got {type(comp).__name__} instead."  # noqa: E501
                )

        EasyDynamicsList.__init__(
            self,
            *components,
            protected_types=ModelComponent,
        )

        EasyDynamicsModelBase.__init__(
            self,
            unit=unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        """
        Check if the ComponentCollection has no components.

        Returns
        -------
        bool
            True if the collection has no components, False otherwise.
        """
        return not self

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
            "is_empty is a read-only property that indicates "
            "whether the collection has components."
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
            raise TypeError(
                f"Unit must be a string or sc.Unit, got {type(unit).__name__}"
            )

        old_unit = self._unit

        try:
            for component in self:
                component.convert_unit(unit)
            self._unit = unit
        except Exception as e:
            # Attempt to rollback on failure
            try:
                for component in self:
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
        """
        if isinstance(component, ComponentCollection):
            self.extend(component)
        else:
            self.append(component)

    def list_component_names(self) -> list[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        list[str]
            List of names of the components in the collection.
        """

        return [component.name for component in self]

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
        if not self:
            raise ValueError("No components in the model to normalize.")

        area_params = []
        total_area = Parameter(name="total_area", value=0.0, unit=self._unit)

        for component in self:
            if hasattr(component, "area"):
                area_params.append(component.area)
                total_area += component.area
            else:
                warnings.warn(
                    f"Component '{component.name}' does not have an 'area' attribute "
                    f"and will be skipped in normalization.",
                    UserWarning,
                    stacklevel=2,
                )

        if total_area.value == 0:
            raise ValueError("Total area is zero; cannot normalize.")

        if not np.isfinite(total_area.value):
            raise ValueError("Total area is not finite; cannot normalize.")

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

        return [var for component in self for var in component.get_all_variables()]

    def evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> np.ndarray:
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

        if not self:
            return np.zeros_like(x)
        return sum(component.evaluate(x) for component in self)

    def evaluate_component(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        name: str,
    ) -> np.ndarray:
        """
        Evaluate a single component by name.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Energy axis.
        name : str
            Component name.

        Raises
        ------
        ValueError
            If there are no components in the model.
        TypeError
            If name is not a string.
        KeyError
            If no component with the given name exists in the collection.

        Returns
        -------
        np.ndarray
            Evaluated values for the specified component.
        """
        if not self:
            raise ValueError("No components in the model to evaluate.")

        if not isinstance(name, str):
            raise TypeError(
                f"Component name must be a string, got {type(name)} instead."
            )

        matches = [comp for comp in self if comp.name == name]
        if not matches:
            raise KeyError(f"No component named '{name}' exists.")

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

    def __repr__(self) -> str:
        """
        Return a string representation of the ComponentCollection.

        Returns
        -------
        str
            String representation of the ComponentCollection.
        """
        comp_names = ", ".join(c.name for c in self) or "No components"

        return (
            f"ComponentCollection(name='{self.name}', unit='{self.unit}', \n"
            f"Components: {comp_names})"
        )

    def to_dict(self) -> dict:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "unit": str(self.unit),
            "name": self.name,
            "display_name": self.display_name,
            "components": [c.to_dict() for c in self._data],
        }

    @classmethod
    def from_dict(cls, obj_dict: dict) -> ComponentCollection:

        def deserialise_component(d: dict) -> ModelComponent:
            """
            Deserialise a component from its dictionary representation.
            Parameters
            ----------
            d : dict
                The dictionary representation of the component.
            Returns
            -------
            ModelComponent
                The deserialised component.
            """
            module = importlib.import_module(d["@module"])
            cls = getattr(module, d["@class"])
            return cls.from_dict(d)

        components = [deserialise_component(c) for c in obj_dict.get("components", [])]

        return cls(
            components=components,
            unit=obj_dict.get("unit", "meV"),
            name=obj_dict.get("name", "ComponentCollection"),
            display_name=obj_dict.get("display_name"),
        )

    def __copy__(self) -> ComponentCollection:
        """
        Create a deep copy of the ComponentCollection.

        Returns
        -------
        ComponentCollection
            A deep copy of the ComponentCollection.
        """

        return self.from_dict(self.to_dict())
