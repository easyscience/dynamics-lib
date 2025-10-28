import warnings
from typing import List, Optional, Union

import numpy as np
import scipp as sc
from easyscience.base_classes import CollectionBase
from easyscience.global_object.undo_redo import NotarizedDict
from easyscience.job.theoreticalmodel import TheoreticalModelBase

from .components.model_component import ModelComponent

Numeric = Union[float, int]


class SampleModel(CollectionBase, TheoreticalModelBase):
    """
    A model of the scattering from a sample, combining multiple model components.

    Attributes
    ----------
    name : str
        Name of the SampleModel.
    unit : str or sc.Unit
        Unit of the SampleModel.
    components : List[ModelComponent]
        List of model components in the SampleModel.

    """

    def __init__(
        self,
        name: str = "MySampleModel",
        unit: Optional[Union[str, sc.Unit]] = "meV",
        data: Optional[List] = None,
    ):
        """
        Initialize a new SampleModel.

        Parameters
        ----------
        name : str
            Name of the sample model.
        unit : str or sc.Unit, optional
            Unit of the sample model. Defaults to "meV".
        data : List[ModelComponent], optional
            Initial list of model components to include in the sample model.
        """

        CollectionBase.__init__(self, name=name)
        TheoreticalModelBase.__init__(self, name=name)
        if not isinstance(self._kwargs, NotarizedDict):
            self._kwargs = NotarizedDict()

        self._unit = unit

        # Add initial components if provided. Mostly used for serialization.
        if data:
            # Just to be safe
            self.clear_components()
            for item in data:
                # ensure item is a ModelComponent
                if not isinstance(item, ModelComponent):
                    raise TypeError("Data items must be instances of ModelComponent.")
                self.insert(index=len(self), value=item)

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
            Name to assign to the component. If None, uses the component's own name.
        """
        if not isinstance(component, ModelComponent):
            raise TypeError("component must be an instance of ModelComponent.")

        if name is None:
            name = component.name
        if name in self.list_component_names():
            raise ValueError(f"Component with name '{name}' already exists.")

        component.name = name

        self.insert(index=len(self), value=component)

    def remove_component(self, name: str):
        """
        Remove a model component by name.
        """
        # Find index where item.name == name
        indices = [i for i, item in enumerate(list(self)) if item.name == name]
        if not indices:
            raise KeyError(f"No component named '{name}' exists in the model.")
        del self[indices[0]]

    def list_component_names(self) -> List[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        List[str]
            Component names.
        """

        return [item.name for item in list(self)]

    def clear_components(self):
        """
        Remove all components from the model.
        """

        for _ in range(len(self)):
            del self[0]

    def normalize_area(self) -> None:
        # Useful for convolutions.
        """
        Normalize the areas of all components so they sum to 1.
        """
        if not self.components:
            raise ValueError("No components in the model to normalize.")

        area_params = []
        total_area = 0.0

        for component in list(self):
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
        Get the list of components in the SampleModel.

        Returns
        -------
        List[ModelComponent]
        """
        return list(self)

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
        self._unit = unit
        # for component in self.components.values():
        for component in list(self):
            component.convert_unit(unit)

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """
        Evaluate the sum of all components, optionally applying detailed balance.

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
        result = None
        for component in list(self):
            value = component.evaluate(x)
            result = value if result is None else result + value

        return result

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

        matches = [comp for comp in list(self) if comp.name == name]
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

    def __repr__(self) -> str:
        """
        Return a string representation of the SampleModel.

        Returns
        -------
        str
        """
        comp_names = ", ".join(c.name for c in self) or "No components"

        return f"<SampleModel name='{self.name}' | Components: {comp_names}>"
