from typing import Dict
import numpy as np
from .components import ModelComponent


from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase




class SampleModel(ObjBase):
    """
    Represents a combined model composed of multiple model components.

    Methods:
        add_component(component): Add a new model component.
        evaluate(x): Sum and evaluate all components at given x.
    """
    def __init__(self,name):
        self.components: Dict[str, ModelComponent] = {}
        self.offset=Parameter(name='offset', value=0.0, unit='meV')
        super().__init__(name=name)


    def add_component(self, component: ModelComponent):
        """
        Add a model component to the sample model.

        Args:
            component (ModelComponent): An instance of a model component.
        """

        if component.name in self.components:
            raise ValueError(f"Component with name '{component.name}' already exists.")
        self.components[component.name] = component

    def remove_component(self, name: str):
        """
        Remove a model component by name.

        Args:
            name (str): Name of the component to remove.

        Raises:
            KeyError: If no component with the given name exists.
        """
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists in the model.")
        del self.components[name]


    def __getitem__(self, key: str) -> ModelComponent:
        """Allow access via SampleModel['ComponentName']"""
        return self.components[key]

    def __setitem__(self, key: str, value: ModelComponent):
        """Allow assignment via SampleModel['ComponentName'] = component"""
        self.components[key] = value


    def temperature(self, temperature: float):
        """
        Set the temperature for all components in the model.

        Args:
            temperature (float): Temperature value to set.
        """
        for comp in self.components:
            if hasattr(comp, 'temperature'):
                comp.temperature = temperature

    def set_offset(self, offset: float):
        # TODO: handle units properly
        
        self.offset.value= offset

    def fix_offset(self, fix: bool = True):
    
        self.offset.fixed = fix

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the full sample model by summing the contribution from all components.

        Args:
            x (np.ndarray): Input energy axis (e.g. in meV).

        Returns:
            np.ndarray: Total model output evaluated at `x`.
        """
        result = np.zeros_like(x, dtype=float)

        for component in self.components.values():
            result += component.evaluate( x - self.offset.value)

        return result

    
    def evaluate_component(self, name: str, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a specific component by name.

        Args:
            name (str): Name of the component to evaluate.
            x (np.ndarray): Input values (e.g. energy axis).

        Returns:
            np.ndarray: Evaluated component values.

        Raises:
            KeyError: If the named component does not exist.
        """
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists.")
        
        component = self.components[name]
        return component.evaluate(x - self.offset.value)

    
    def get_parameters(self):
        """
        Get all parameters from the model components.

        Returns:
            List[Parameter]: List of parameters from all components.
        """
        params = []
        for comp in self.components.values():
            params.extend(comp.get_parameters())
        params.append(self.offset)
        return params
    
    def name(self):
        """
        Get the name of the sample model.

        Returns:
            str: Name of the sample model.
        """
        return self.name
    
    def set_name(self, name: str):
        """
        Set the name of the sample model.

        Args:
            name (str): New name for the sample model.
        """
        self.name = name
