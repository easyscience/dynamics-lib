from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np
import scipp as sc
from easyscience.base_classes import ObjBase
from scipp import UnitError

Numeric = Union[float, int]


class ModelComponent(ObjBase):
    """
    Abstract base class for all model components.
    """

    def __init__(self, name="ModelComponent", unit: Optional[str] = "meV", **kwargs):
        self.validate_unit(unit)
        super().__init__(name=name, **kwargs)
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

    def _prepare_x_for_evaluate(
        self, x: Union[Numeric, List[Numeric], np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """ "Prepare the input x for evaluation by handling units and converting to a numpy array."""

        # Handle units
        if isinstance(x, sc.DataArray):
            # Check that there's exactly one coordinate
            coords = dict(x.coords)
            ncoords = len(coords)
            if ncoords != 1:
                coord_names = ", ".join(coords.keys())
                raise ValueError(
                    f"scipp.DataArray must have exactly one coordinate to be used as input `x`. "
                    f"Found {ncoords} coordinates: {coord_names}."
                )
            # get the coordinate, it's a sc.Variable
            coord_name, coord_obj = next(iter(coords.items()))
            x = coord_obj
        if isinstance(x, sc.Variable):
            # Need to check if the units are consistent, and convert if not.
            if x.sizes == {}:  # scalar
                x_in = x.value
            else:  # array
                x_in = x.values
            if self._unit is not None and x.unit != self._unit:
                self_unit_for_warning = self._unit
                try:
                    self.convert_unit(x.unit.name)
                except Exception as e:
                    raise UnitError(
                        f"Input x has unit {x.unit}, but {self.__class__.__name__} component has unit {self._unit}. Failed to convert {self.__class__.__name__} to {x.unit}."
                    ) from e

                warnings.warn(
                    f"Input x has unit {x.unit}, but {self.__class__.__name__} component has unit {self_unit_for_warning}. Converting {self.__class__.__name__} to {x.unit}."
                )
        else:
            x_in = x

        if isinstance(x_in, Numeric):
            x_in = np.array([x_in])
        elif isinstance(x_in, list):
            x_in = np.array(x_in)

        if any(np.isnan(x_in)):
            raise ValueError("Input x contains NaN values.")

        if any(np.isinf(x_in)):
            raise ValueError("Input x contains infinite values.")

        return np.sort(x_in)

    @staticmethod
    def validate_unit(unit) -> None:
        """Raise TypeError if unit is not allowed (string or sc.Unit)."""
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")

    @abstractmethod
    def convert_unit(self, unit: Union[str, sc.Unit]):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str or sc.Unit): The new unit to convert to.
        """
        pass

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
    def __copy__(self) -> ModelComponent:
        """
        Return a deep copy of this component with independent parameters.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
