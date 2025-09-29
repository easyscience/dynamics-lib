from abc import abstractmethod

from typing import Union, List, Optional

import numpy as np

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

import scipp as sc

Numeric = Union[float, int]


class ModelComponent(ObjBase):
    """
    Abstract base class for all model components.
    """

    def __init__(self, name="ModelComponent"):
        super().__init__(name=name)
        self.unit = None

    def fix_all_parameters(self):
        """Fix all parameters in the model component."""

        pars = self.get_parameters()
        if pars is None or len(pars) == 0:
            raise ValueError("No parameters found to fix.")
        else:
            for p in pars:
                p.fixed = True

    def fit_all_parameters(self):
        """Fit all parameters in the model component."""
        for p in self.get_parameters():
            p.fixed = False

    def get_parameter(self, parameter_name: str) -> Parameter:
        """
        Get a specific parameter by name (explicit or partial match).

        Args:
            parameter_name (str): Name of the parameter, or partial name to match.

        Returns:
            Parameter: The matched parameter.

        Raises:
            ValueError: If no matching or ambiguous parameter is found.
        """
        # First, attempt exact match
        for p in self.get_parameters():
            if p.name == parameter_name:
                return p

        # If exact match is not found, attempt partial match
        matches = [p for p in self.get_parameters() if parameter_name in p.name]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous parameter name '{parameter_name}' matches multiple parameters: {[p.name for p in matches]}"
            )
        else:
            raise ValueError(f"Parameter '{parameter_name}' not found.")

    def set_parameter_value(
        self, parameter_name: str, value: float, unit: Optional[str] = None
    ):
        """
        Set the value of a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        if unit is not None:
            param.convert_unit(unit)
        param.value = value

    def set_parameter_bounds(
        self,
        parameter_name: str,
        min: Union[float, None] = None,
        max: Union[float, None] = None,
        unit: Optional[str] = None,
    ):
        """
        Set the bounds of a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        if unit is not None:
            param.convert_unit(unit)
        if min is not None:
            param.min = min
        if max is not None:
            param.max = max

    def fix_parameter(self, parameter_name: str):
        """
        Fix a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        param.fixed = True

    def free_parameter(self, parameter_name: str):
        """
        Free a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        param.fixed = False

    @abstractmethod
    def evaluate(self, x: Union[Numeric, sc.Variable]) -> np.ndarray:
        """
        Evaluate the model component at input x.

        Args:
            x (Union[Numeric, sc.Variable]): Input values.

        Returns:
            np.ndarray: Evaluated function values.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[Parameter]:
        """
        Get all parameters from the model component.

        Returns
        -------
        List[Parameter]
            List of parameters in the component.
        """
        pass

    @abstractmethod
    def copy(self) -> "ModelComponent":
        """
        Return a deep copy of this component with independent parameters.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
