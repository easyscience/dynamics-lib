from __future__ import annotations

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

    def __init__(self, name="ModelComponent", unit: Optional[str] = None):
        super().__init__(name=name)
        self._unit = unit

    @property
    def unit(self) -> str:
        """
        Get the unit.

        :return: Unit as a string.
        """
        return str(self._unit)

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )  # noqa: E501

    def fix_all_parameters(self):
        """Fix all parameters in the model component."""

        pars = self.get_parameters()
        for p in pars:
            p.fixed = True

    def free_all_parameters(self):
        """Free all parameters in the model component."""
        for p in self.get_parameters():
            p.fixed = False

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
    def copy(self) -> ModelComponent:
        """
        Return a deep copy of this component with independent parameters.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
