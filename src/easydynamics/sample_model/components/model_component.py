# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import abstractmethod

import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase
from scipp import UnitError

from easydynamics.utils.utils import Numeric


class ModelComponent(ModelBase):
    """Abstract base class for all model components."""

    def __init__(
        self,
        unit: str | sc.Unit = 'meV',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the ModelComponent.

        Parameters
        ----------
        unit : str | sc.Unit, default='meV'
            The unit of the model component.
        display_name : str | None, default=None
            A human-readable name for the component.
        unique_name : str | None, default=None
            A unique identifier for the component.
        """
        self.validate_unit(unit)
        super().__init__(display_name=display_name, unique_name=unique_name)
        self._unit = unit

    @property
    def unit(self) -> str:
        """
        Get the unit.

        Returns
        -------
        str
            The unit of the model component.
        """
        return str(self._unit)

    @unit.setter
    def unit(self, _unit_str: str) -> None:
        """
        Unit is read-only.

        Use convert_unit to change the unit between allowed types or create a new ModelComponent
        with the desired unit.

        Parameters
        ----------
        _unit_str : str
            The new unit to set.

        Raises
        ------
        AttributeError
            Always raised since unit is read-only.
        """
        raise AttributeError(
            f'Unit is read-only. Use convert_unit to change the unit between allowed types '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    def fix_all_parameters(self) -> None:
        """Fix all parameters in the model component."""

        pars = self.get_fittable_parameters()
        for p in pars:
            p.fixed = True

    def free_all_parameters(self) -> None:
        """Free all parameters in the model component."""
        for p in self.get_fittable_parameters():
            p.fixed = False

    def _prepare_x_for_evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> np.ndarray:
        """
        Prepare the input x for evaluation by handling units and converting to a numpy array.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The input data to prepare.

        Raises
        ------
        ValueError
            If x contains NaN or infinite values, or if a sc.DataArray has more than one
            coordinate.
        UnitError
            If x has incompatible units that cannot be converted to the component's unit.

        Returns
        -------
        np.ndarray
            The prepared input data as a numpy array.
        """

        # Handle units
        if isinstance(x, sc.DataArray):
            # Check that there's exactly one coordinate
            coords = dict(x.coords)
            ncoords = len(coords)
            if ncoords != 1:
                coord_names = ', '.join(coords.keys())
                raise ValueError(
                    f'scipp.DataArray must have exactly one coordinate to be used as input `x`. '
                    f'Found {ncoords} coordinates: {coord_names}.'
                )
            # get the coordinate, it's a sc.Variable
            _, coord_obj = next(iter(coords.items()))
            x = coord_obj
        if isinstance(x, sc.Variable):
            # Need to check if the units are consistent,
            # and convert if not.
            x_in = x.value if x.sizes == {} else x.values
            if self._unit is not None and x.unit != self._unit:
                self_unit_for_warning = self._unit
                try:
                    self.convert_unit(x.unit.name)
                except Exception as e:
                    raise UnitError(
                        f'Input x has unit {x.unit}, but {self.__class__.__name__} component \
                            has unit {self._unit}. \
                                Failed to convert {self.__class__.__name__} to {x.unit}.'
                    ) from e

                warnings.warn(
                    f'Input x has unit {x.unit}, but {self.__class__.__name__} component \
                        has unit {self_unit_for_warning}. \
                            Converting {self.__class__.__name__} to {x.unit}.',
                    UserWarning,
                    stacklevel=3,
                )
        else:
            x_in = x

        if isinstance(x_in, Numeric):
            x_in = np.array([x_in])
        elif isinstance(x_in, list):
            x_in = np.array(x_in)

        if any(np.isnan(x_in)):
            raise ValueError('Input x contains NaN values.')

        if any(np.isinf(x_in)):
            raise ValueError('Input x contains infinite values.')

        return np.sort(x_in)

    @staticmethod
    def validate_unit(unit: str | sc.Unit | None) -> None:
        """
        Validate that the unit is either a string or a scipp Unit.

        Parameters
        ----------
        unit : str | sc.Unit | None
            The unit to validate.

        Raises
        ------
        TypeError
            If unit is not a string or scipp Unit.
        """
        if unit is not None and not isinstance(unit, (str, sc.Unit)):
            raise TypeError(
                f'unit must be None, a string, or a scipp Unit, got {type(unit).__name__}'
            )

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the Parameters in the component.

        Parameters
        ----------
        unit : str | sc.Unit
            The new unit to convert to.

        Raises
        ------
        TypeError
            If the provided unit is not a str or sc.Unit.
        Exception
            If the provided unit is invalid or incompatible with the component's parameters.
        """
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError(f'Unit must be a string or sc.Unit, got {type(unit).__name__}')

        old_unit = self._unit
        pars = self.get_all_parameters()
        try:
            for p in pars:
                p.convert_unit(unit)
            self._unit = unit
        except Exception as e:
            # Attempt to rollback on failure
            try:
                for p in pars:
                    if hasattr(p, 'convert_unit'):
                        p.convert_unit(old_unit)
            except Exception:  # noqa: S110
                pass  # Best effort rollback
            raise e

    @abstractmethod
    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        """
        Abstract method to evaluate the model component at input x.

        Must be implemented by subclasses.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the component.

        Returns
        -------
        np.ndarray
            Evaluated function values.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a string representation of the ModelComponent.

        Returns
        -------
        str
            A string representation of the ModelComponent.
        """

        return f'{self.__class__.__name__}(unique_name={self.unique_name}, unit={self._unit})'
