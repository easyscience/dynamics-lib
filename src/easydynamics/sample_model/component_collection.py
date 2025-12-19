import warnings
from typing import List

import numpy as np
import scipp as sc

# from easyscience.job.theoreticalmodel import TheoreticalModelBase
from easyscience.base_classes.model_base import ModelBase
from easyscience.variable import DescriptorBase, Parameter

from .components.model_component import ModelComponent

Numeric = float | int


class ComponentCollection(ModelBase):
    """
    A model of the scattering from a sample, combining multiple model components.

    Attributes
    ----------
    display_name : str
        Display name of the ComponentCollection.
    unit : str or sc.Unit
        Unit of the ComponentCollection.

    """

    def __init__(
        self,
        unit: str | sc.Unit = "meV",
        display_name: str = "MyComponentCollection",
        components: List[ModelComponent] | None = None,
    ):
        """
        Initialize a new ComponentCollection.

        Parameters
        ----------
        display_name : str
            Display name of the sample model.
        unit : str or sc.Unit, optional
            Unit of the sample model. Defaults to "meV".
        components : List[ModelComponent], optional
            Initial model components to add to the ComponentCollection.
        """

        super().__init__(display_name=display_name)

        if unit is not None and not isinstance(unit, (str, sc.Unit)):
            raise TypeError(
                f"unit must be None, a string, or a scipp Unit, got {type(unit).__name__}"
            )
        self._unit = unit
        self._components = []

        # Add initial components if provided. Used for serialization.
        if components is not None:
            if not isinstance(components, list):
                raise TypeError(
                    "components must be a list of ModelComponent instances."
                )
            for comp in components:
                self.add_component(comp)

    def add_component(self, component: ModelComponent) -> None:
        if not isinstance(component, ModelComponent):
            raise TypeError("Component must be an instance of ModelComponent.")

        if component in self._components:
            raise ValueError(
                f"Component '{component.display_name}' is already in the collection."
            )

        for comp in self._components:
            if comp.display_name == component.display_name:
                raise ValueError(
                    f"A component with the name '{component.display_name}' is already in the collection."
                )

        self._components.append(component)

    def remove_component(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("Component name must be a string.")

        for comp in self._components:
            if comp.display_name == name:
                self._components.remove(comp)
                return

        raise KeyError(f"No component named '{name}' exists.")

    @property
    def components(self) -> list[ModelComponent]:
        return list(self._components)

    def list_component_names(self) -> List[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        List[str]
            Component names.
        """

        return [component.display_name for component in self._components]

    def clear_components(self) -> None:
        """Remove all components."""
        self._components.clear()

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
            total_area = Parameter(name="total_area", value=0.0, unit=self._unit)
            if hasattr(component, "area"):
                area_params.append(component.area)
                total_area += component.area
            else:
                warnings.warn(
                    f"Component '{component.display_name}' does not have an 'area' attribute and will be skipped in normalization.",
                    UserWarning,
                )

        if total_area == 0:
            raise ValueError("Total area is zero; cannot normalize.")

        if not np.isfinite(total_area):
            raise ValueError("Total area is not finite; cannot normalize.")

        for param in area_params:
            param.value /= total_area

    def get_all_variables(self) -> list[DescriptorBase]:
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """

        return [
            var
            for component in self.components
            for var in component.get_all_variables()
        ]

    @property
    def unit(self) -> str | sc.Unit:
        """
        Get the unit of the ComponentCollection.

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

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the ComponentCollection and all its components.
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
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
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
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
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

        matches = [comp for comp in self.components if comp.display_name == name]
        if not matches:
            raise KeyError(f"No component named '{name}' exists.")

        component = matches[0]

        result = component.evaluate(x)

        return result

    def fix_all_parameters(self) -> None:
        """
        Fix all free parameters in the model.
        """
        for param in self.get_fittable_parameters():
            param.fixed = True

    def free_all_parameters(self) -> None:
        """
        Free all fixed parameters in the model.
        """
        for param in self.get_fittable_parameters():
            param.fixed = False

    def __contains__(self, item: str | ModelComponent) -> bool:
        """
        Check if a component with the given name or instance exists in the ComponentCollection.
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
            return any(comp.display_name == item for comp in self.components)
        elif isinstance(item, ModelComponent):
            # Check by component instance
            return any(comp is item for comp in self.components)
        else:
            return False

    def __repr__(self) -> str:
        """
        Return a string representation of the ComponentCollection.

        Returns
        -------
        str
        """
        comp_names = (
            ", ".join(c.display_name for c in self.components) or "No components"
        )

        return f"<ComponentCollection display_name='{self.display_name}' | Components: {comp_names}>"
