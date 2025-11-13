import warnings
from typing import List, Optional, Union

import numpy as np
import scipp as sc
from easyscience.global_object.undo_redo import NotarizedDict
from easyscience.job.theoreticalmodel import TheoreticalModelBase

from .components.model_component import ModelComponent

Numeric = Union[float, int]


class SampleModel(TheoreticalModelBase):
    """
    A model of the scattering from a sample, combining multiple model components.

    Attributes
    ----------
    name : str
        Name of the SampleModel.
    unit : str or sc.Unit
        Unit of the SampleModel.

    """

    def __init__(
        self,
        name: str = "MySampleModel",
        unit: Optional[Union[str, sc.Unit]] = "meV",
        **kwargs,
    ):
        """
        Initialize a new SampleModel.

        Parameters
        ----------
        name : str
            Name of the sample model.
        unit : str or sc.Unit, optional
            Unit of the sample model. Defaults to "meV".
        **kwargs : ModelComponent
            Initial model components to add to the SampleModel. Keys are component names, values are ModelComponent instances.
        """

        super().__init__(name=name)
        if not isinstance(self._kwargs, NotarizedDict):
            self._kwargs = NotarizedDict()

        self._unit = unit
        self._components = []

        # Add initial components if provided. Used for serialization.
        for key, comp in list(kwargs.items()):
            self._add_component(key, comp)

    def add_component(
        self, component: ModelComponent, name: Optional[str] = None
    ) -> None:
        """
        Add a model component to the SampleModel. Component names must be unique.
        Parameters
        ----------
        component : ModelComponent
            The model component to add.
        name : str, optional
            Name to assign to the component. If None, uses the component's own name. Renames the component if a different name is provided.
        """

        if not isinstance(component, ModelComponent):
            raise TypeError("Component must be an instance of ModelComponent.")

        if name is None:
            name = component.name

        if not isinstance(name, str):
            raise TypeError("Component name must be a string.")
        if name in getattr(self, "_kwargs", {}):
            raise ValueError(f"Component with name '{name}' already exists.")

        # Use ObjBase to add component so Global Object is updated correctly
        self._add_component(name, component)

    def remove_component(self, name: str) -> None:
        """
        Remove a model component from the SampleModel by name.
        Parameters
        ----------
        name : str
            Name of the component to remove.
        """

        if not isinstance(name, str):
            raise TypeError("Component name must be a string.")

        for key, item in list(self._kwargs.items()):
            if item.name == name:
                del self._kwargs[key]
                return

        raise KeyError(f"No component named '{name}' exists in the model.")

    def list_component_names(self) -> List[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        List[str]
            Component names.
        """

        return [item.name for item in self.components]

    def clear_components(self) -> None:
        """Remove all components."""
        for key in list(self._kwargs.keys()):
            del self._kwargs[key]

    def normalize_area(self) -> None:
        # Useful for convolutions.
        """
        Normalize the areas of all components so they sum to 1.
        """
        if not self.components:
            raise ValueError("No components in the model to normalize.")

        area_params = []
        total_area = 0.0

        for component in self.components:
            if hasattr(component, "area"):
                area_params.append(component.area)
                total_area += component.area.value
            else:
                warnings.warn(
                    f"Component '{component.name}' does not have an 'area' attribute and will be skipped in normalization."
                )

        if total_area == 0:
            raise ValueError("Total area is zero; cannot normalize.")

        if not np.isfinite(total_area):
            raise ValueError("Total area is not finite; cannot normalize.")

        for param in area_params:
            param.value /= total_area

    @property
    def components(self) -> List[ModelComponent]:
        """
        Get the list of model components in the SampleModel.
        Returns
        -------
        List[ModelComponent]
            List of model components.
        """
        return list(self._kwargs.values())

    @property
    def unit(self) -> Optional[Union[str, sc.Unit]]:
        """
        Get the unit of the SampleModel.

        Returns
        -------
        str or sc.Unit or None
        """
        return self._unit

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )  # noqa: E501

    def convert_unit(self, unit: Union[str, sc.Unit]) -> None:
        """
        Convert the unit of the SampleModel and all its components.
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
            except Exception:
                pass  # Best effort rollback
            raise e

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """
        Evaluate the sum of all components.

        Parameters
        ----------
        x : Number, list, np.ndarray, sc.Variable, or sc.DataArray
            Energy axis.

        Returns
        -------
        np.ndarray
            Evaluated model values.
        """

        if not self.components:
            raise ValueError("No components in the model to evaluate.")
        return sum(component.evaluate(x) for component in self.components)

    def evaluate_component(
        self,
        x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray],
        name: str,
    ) -> np.ndarray:
        """
        Evaluate a single component by name.

        Parameters
        ----------
        x : Number, list, np.ndarray, sc.Variable, or sc.DataArray
            Energy axis.
        name : str
            Component name.

        Returns
        -------
        np.ndarray
            Evaluated values for the specified component.
        """
        if not self.components:
            raise ValueError("No components in the model to evaluate.")

        if not isinstance(name, str):
            raise TypeError(
                (f"Component name must be a string, got {type(name)} instead.")
            )

        matches = [comp for comp in self.components if comp.name == name]
        if not matches:
            raise KeyError(f"No component named '{name}' exists.")

        component = matches[0]

        result = component.evaluate(x)

        return result

    def fix_all_parameters(self) -> None:
        """
        Fix all free parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = True

    def free_all_parameters(self) -> None:
        """
        Free all fixed parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = False

    def __contains__(self, item: Union[str, ModelComponent]) -> bool:
        """
        Check if a component with the given name or instance exists in the SampleModel.
        Args:
        ----------
        item : str or ModelComponent
            The component name or instance to check for.
        Returns
        -------
        bool
            True if the component exists, False otherwise.
        """

        if isinstance(item, str):
            # Check by component name
            return any(comp.name == item for comp in self.components)
        elif isinstance(item, ModelComponent):
            # Check by component instance
            return any(comp is item for comp in self.components)
        else:
            return False

    def __repr__(self) -> str:
        """
        Return a string representation of the SampleModel.

        Returns
        -------
        str
        """
        comp_names = ", ".join(c.name for c in self.components) or "No components"

        return f"<SampleModel name='{self.name}' | Components: {comp_names}>"
