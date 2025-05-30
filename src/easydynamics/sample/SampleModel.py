from typing import List
import numpy as np
from .components import ModelComponent

class SampleModel:
    """
    Represents a combined model composed of multiple model components.

    Methods:
        add_component(component): Add a new model component.
        evaluate(x): Sum and evaluate all components at given x.
    """
    def __init__(self):
        self.components: List[ModelComponent] = []

    def add_component(self, component: ModelComponent):
        """
        Add a model component to the sample model.

        Args:
            component (ModelComponent): An instance of a model component.
        """
        self.components.append(component)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the full model by summing all components.

        Args:
            x (np.ndarray): Array of input values (e.g. energy).

        Returns:
            np.ndarray: Evaluated model at input x.
        """
        return sum(comp.evaluate(x) for comp in self.components)