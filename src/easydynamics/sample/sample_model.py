import warnings
from typing import Dict, List

import numpy as np

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

from easydynamics.utils import detailed_balance_factor
from .components import ModelComponent

class SampleModel(ObjBase):
    """
    Represents a combined model composed of multiple model components.

    This class allows components to be added, removed, and evaluated together.
    Optionally applies detailed balancing. 

    Attributes
    ----------
    components : dict
        Dictionary of model components keyed by name.
    """
    def __init__(self, name):
        """
        Initialize a new SampleModel.

        Parameters
        ----------
        name : str
            Name of the sample model.
        """
                
        self.components: Dict[str, ModelComponent] = {}
        super().__init__(name=name)
        self._temperature = Parameter(name="temperature", value=-1, unit='K', fixed=True)
        self._use_detailed_balance = False

    def add_component(self, component: ModelComponent):
        """
        Add a model component to the SampleModel.

        Raises
        ------
        ValueError
            If a component with the same name already exists.
        """
        if component.name in self.components:
            raise ValueError(f"Component with name '{component.name}' already exists.")
        self.components[component.name] = component

    def remove_component(self, name: str):
        """
        Remove a model component by name.

        Parameters
        ----------
        name : str
            Name of the component to remove.

        Raises
        ------
        KeyError
            If the component does not exist.
        """
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists in the model.")
        del self.components[name]

    def list_components(self) -> List[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        List[str]
            Component names.
        """
        return list(self.components.keys())

    def clear_components(self):
        """
        Remove all components from the model.
        """
        self.components.clear()

    def __getitem__(self, key: str) -> ModelComponent:
        """
        Access a component by name using dictionary-like indexing.

        Parameters
        ----------
        key : str
            Name of the component.

        Returns
        -------
        ModelComponent
        """
        return self.components[key]

    def __setitem__(self, key: str, value: ModelComponent):
        """
        Set or replace a component by name using dictionary-like syntax.

        Parameters
        ----------
        key : str
            Name of the component.
        value : ModelComponent
            The component to assign.
        """
        self.components[key] = value

    def __contains__(self, name: str) -> bool:
        """
        Check if a component exists in the model.

        Parameters
        ----------
        name : str
            Name of the component.

        Returns
        -------
        bool
        """
        return name in self.components

    def __repr__(self):
        """
        Return a string representation of the SampleModel.

        Returns
        -------
        str
        """
        comp_names = ", ".join(self.components.keys()) or "No components"
        temp_str = (f" | Temperature: {self._temperature.value} {self._temperature.unit}"
                    if self._use_detailed_balance else "")
        return (f"<SampleModel name='{self.name}' | "
                f"Components: {comp_names}{temp_str}>")

    @property
    def temperature(self) -> Parameter:
        """
        Access the temperature parameter.

        Returns
        -------
        Parameter
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """
        Set the temperature.

        If a negative value is provided, detailed balance is disabled and a warning is issued.

        Parameters
        ----------
        value : float
            Temperature value in Kelvin.
        """
        if value < 0:
            self._use_detailed_balance = False
            warnings.warn("Temperature is negative. Disabling detailed balance.")

        self._temperature.value = value

        if not self.use_detailed_balance:
            self.use_detailed_balance = value >= 0
            print(f"Detailed balance set to {self.use_detailed_balance} for temperature {value} K")

    @property
    def use_detailed_balance(self) -> bool:
        """
        Indicates whether detailed balance is enabled.

        Returns
        -------
        bool
        """
        return self._use_detailed_balance

    @use_detailed_balance.setter
    def use_detailed_balance(self, value: bool):
        """
        Enable or disable the use of detailed balance.

        Parameters
        ----------
        value : bool
            True to enable, False to disable.
        """
        self._use_detailed_balance = value

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the sum of all components, optionally applying detailed balance.

        Parameters
        ----------
        x : np.ndarray
            Energy axis (e.g., in meV).

        Returns
        -------
        np.ndarray
            Evaluated model values.
        """
        result = np.zeros_like(x, dtype=float)
        for component in self.components.values():
            result += component.evaluate(x)

        if self.use_detailed_balance and self._temperature.value >= 0:
            result *= detailed_balance_factor(x, self._temperature.value)

        return result

    def evaluate_component(self, name: str, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a single component by name, optionally applying detailed balance.

        Parameters
        ----------
        name : str
            Component name.
        x : np.ndarray
            Energy axis.

        Returns
        -------
        np.ndarray
            Evaluated values for the specified component.

        Raises
        ------
        KeyError
            If the component is not found.
        """
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists.")

        result = self.components[name].evaluate(x)
        if self._use_detailed_balance and self._temperature.value >= 0:
            result *= detailed_balance_factor(x, self._temperature.value)

        return result

    def normalize_area(self):
        """
        Normalize the areas of all components so they sum to 1.

        Modifies the 'area' parameters directly. Raises an error if total area is zero.
        """
        area_params = []
        total_area = 0.0

        for component in self.components.values():
            for param in component.get_parameters():
                if 'area' in param.name.lower():
                    area_params.append(param)
                    total_area += param.value

        if total_area == 0:
            raise ValueError("Total area is zero; cannot normalize.")

        for param in area_params:
            param.value /= total_area

    def get_parameters(self) -> List[Parameter]:
        """
        Return all parameters from the model, including temperature.

        Returns
        -------
        List[Parameter]
        """
        params = [self._temperature]
        for comp in self.components.values():
            params.extend(comp.get_parameters())
        return params