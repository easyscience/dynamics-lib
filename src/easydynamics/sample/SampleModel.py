from typing import List
import numpy as np
from .components import ModelComponent


from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase


from easyscience import global_object


class SampleModel(ObjBase):
    """
    Represents a combined model composed of multiple model components.

    Methods:
        add_component(component): Add a new model component.
        evaluate(x): Sum and evaluate all components at given x.
    """
    def __init__(self,name):
        self.components: List[ModelComponent] = []
        self.offset=Parameter(name='offset', value=0.0, unit='meV')
        self.unique_name = name

        self._global_object = global_object


    def add_component(self, component: ModelComponent):
        """
        Add a model component to the sample model.

        Args:
            component (ModelComponent): An instance of a model component.
        """
        self.components.append(component)

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
        Evaluate the full model by summing all components.

        Args:
            x (np.ndarray): Array of input values (e.g. energy).

        Returns:
            np.ndarray: Evaluated model at input x.
        """
        return sum(comp.evaluate(x-self.offset.value) for comp in self.components)
    
    def get_parameters(self):
        """
        Get all parameters from the model components.

        Returns:
            List[Parameter]: List of parameters from all components.
        """
        params = []
        for comp in self.components:
            params.extend(comp.get_parameters())
        params.append(self.offset)
        return params