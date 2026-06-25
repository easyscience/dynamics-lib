# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import scipp as sc
from scipp import UnitError

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.utils.utils import Numeric

if TYPE_CHECKING:
    from easyscience.variable import Parameter


class ModelComponent(EasyDynamicsModelBase):
    """Abstract base class for all model components."""

    def __init__(
        self,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'ModelComponent',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        x_unit : str | sc.Unit, default='meV'
            Unit for the x-axis (independent variable) of this component.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit for the y-axis (dependent variable / output) of this component.
        name : str, default='ModelComponent'
            Internal name used for parameter labelling and logging.
        display_name : str | None, default=None
            Human-readable name shown in plots and reports. Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier. Auto-generated if None.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

    @property
    def x_unit(self) -> str:
        """
        Returns
        -------
        str
            The current x-axis unit as a string.
        """
        return str(self._x_unit)

    @x_unit.setter
    def x_unit(self, _: str) -> None:
        """
        Unit is read-only; raises AttributeError always.

        Use :meth:`convert_x_unit` to change the unit, or create a new instance with the desired
        unit.

        Raises
        ------
        AttributeError
            Always raised when this setter is called.
        """
        raise AttributeError(
            f'x_unit is read-only. Use convert_x_unit to change the unit '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    @property
    def y_unit(self) -> str:
        """
        Returns
        -------
        str
            The current y-axis unit as a string.
        """
        return str(self._y_unit)

    @y_unit.setter
    def y_unit(self, _: str) -> None:
        """
        Unit is read-only; raises AttributeError always.

        Use :meth:`convert_y_unit` to change the unit, or create a new instance with the desired
        unit.

        Raises
        ------
        AttributeError
            Always raised when this setter is called.
        """
        raise AttributeError(
            f'y_unit is read-only. Use convert_y_unit to change the unit '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    def fix_all_parameters(self) -> None:
        """
        Fix all parameters in the model component.

        Sets ``fixed=True`` on every fittable parameter returned by
        :meth:`get_fittable_parameters`.
        """
        for p in self.get_fittable_parameters():
            p.fixed = True

    def free_all_parameters(self) -> None:
        """
        Free all parameters in the model component.

        Sets ``fixed=False`` on every fittable parameter returned by
        :meth:`get_fittable_parameters`.
        """
        for p in self.get_fittable_parameters():
            p.fixed = False

    def _prepare_x_for_evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> tuple[np.ndarray, str | None, str]:
        """
        Validate x and extract its values, detected unit, and dimension name.

        x is never converted. When x carries a unit, the caller is responsible for resolving
        parameter values to that unit via _resolve_param_value.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Input x values to validate and extract.

        Returns
        -------
        tuple[np.ndarray, str | None, str]
            x_values : np.ndarray of raw float values (no unit conversion) detected_unit : str unit
            of x if scipp input, else None dim : scipp dimension name if scipp input, else 'x'

        Raises
        ------
        UnitError
            If x has a unit incompatible with the model's x_unit.
        ValueError
            If x contains NaN or infinite values, or if a DataArray has more than one coordinate.
        """
        detected_unit: str | None = None
        dim: str = 'x'
        dim_from_dataarray: bool = False

        if isinstance(x, sc.DataArray):
            coords = dict(x.coords)
            ncoords = len(coords)
            if ncoords != 1:
                coord_names = ', '.join(coords.keys())
                raise ValueError(
                    f'scipp.DataArray must have exactly one coordinate to be used as input `x`. '
                    f'Found {ncoords} coordinates: {coord_names}.'
                )
            dim, coord_obj = next(iter(coords.items()))
            x = coord_obj
            dim_from_dataarray = True

        if isinstance(x, sc.Variable):
            detected_unit = str(x.unit)
            if not dim_from_dataarray:
                dim = x.dims[0] if x.dims else 'x'
            x_in = x.value if x.sizes == {} else x.values

            # Validate that x's unit is compatible with model's x_unit
            if self._x_unit is not None and detected_unit != str(self._x_unit):
                try:
                    sc.to_unit(sc.scalar(1.0, unit=detected_unit), str(self._x_unit))
                except Exception as e:
                    raise UnitError(
                        f'Input x has unit {detected_unit}, which is incompatible with '
                        f'{self.__class__.__name__} x_unit {self._x_unit}.'
                    ) from e
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

        return x_in, detected_unit, dim

    def _resolve_param_value(self, param: Parameter, target_unit: str | None) -> float:
        """
        Return param's value converted to target_unit without mutating param.

        If target_unit is None or already matches param's unit, returns param.value directly. Uses
        a temporary scipp scalar for the conversion.

        Parameters
        ----------
        param : Parameter
            The parameter whose value should be resolved.
        target_unit : str | None
            The unit to which the parameter value should be converted.  When None (or equal to the
            parameter's own unit) the raw value is returned without any conversion.

        Returns
        -------
        float
            The parameter value expressed in *target_unit*.
        """
        if target_unit is None or str(param.unit) == str(target_unit):
            return param.value
        return sc.to_unit(sc.scalar(param.value, unit=str(param.unit)), target_unit).value

    def _convert_x_unit_area_based(
        self,
        new_x_unit: str | sc.Unit,
        x_params: list,
        area_param: Parameter,
    ) -> None:
        """
        Shared convert_x_unit logic for components with an area parameter (area = x_unit * y_unit).

        Validates the input type, converts all x-axis parameters and the area parameter to the new
        unit, and updates ``_x_unit``.  Rolls back all conversions if any step fails.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.
        x_params : list
            Parameters whose unit equals *x_unit* (e.g. center, width).
        area_param : Parameter
            The parameter whose unit equals ``x_unit * y_unit``.

        Raises
        ------
        TypeError
            If *new_x_unit* is not a ``str`` or ``sc.Unit``.
        Exception
            If the conversion fails; all parameters are rolled back to their original units.
        """
        if not isinstance(new_x_unit, (str, sc.Unit)):
            raise TypeError(f'x_unit must be a string or sc.Unit, got {type(new_x_unit).__name__}')
        old_x_unit = self._x_unit
        new_x_str = str(new_x_unit) if isinstance(new_x_unit, sc.Unit) else new_x_unit
        new_area_unit = str(sc.Unit(new_x_str) * sc.Unit(self._y_unit))
        try:
            for p in x_params:
                p.convert_unit(new_x_unit)
            area_param.convert_unit(new_area_unit)
            self._x_unit = new_x_str
        except Exception as e:
            try:
                old_area_unit = str(sc.Unit(old_x_unit) * sc.Unit(self._y_unit))
                for p in x_params:
                    p.convert_unit(old_x_unit)
                area_param.convert_unit(old_area_unit)
            except Exception:  # noqa: S110
                pass
            raise e

    def _convert_y_unit_area_based(
        self,
        new_y_unit: str | sc.Unit,
        area_param: Parameter,
    ) -> None:
        """
        Shared convert_y_unit logic for components with an area parameter (area = x_unit * y_unit).

        Validates the input type, rescales the area parameter from ``x_unit * old_y_unit`` to
        ``x_unit * new_y_unit``, and updates ``_y_unit``.  Rolls back on failure.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        area_param : Parameter
            The parameter whose unit equals ``x_unit * y_unit``.

        Raises
        ------
        TypeError
            If *new_y_unit* is not a ``str`` or ``sc.Unit``.
        Exception
            If the conversion fails; the area parameter is rolled back to its original unit.
        """
        if not isinstance(new_y_unit, (str, sc.Unit)):
            raise TypeError(f'y_unit must be a string or sc.Unit, got {type(new_y_unit).__name__}')
        old_y_unit = self._y_unit
        new_area_unit = str(sc.Unit(self._x_unit) * sc.Unit(new_y_unit))
        try:
            area_param.convert_unit(new_area_unit)
            self._y_unit = str(new_y_unit) if isinstance(new_y_unit, sc.Unit) else new_y_unit
        except Exception as e:
            try:
                old_area_unit = str(sc.Unit(self._x_unit) * sc.Unit(old_y_unit))
                area_param.convert_unit(old_area_unit)
            except Exception:  # noqa: S110
                pass
            raise e

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert the x-axis unit of the component.

        The base implementation converts all parameters. Subclasses with mixed-unit parameters
        (e.g. area ≠ x_unit) should override this method.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.

        Raises
        ------
        TypeError
            If *new_x_unit* is not a ``str`` or ``sc.Unit``.
        Exception
            If the conversion between the current unit and *new_x_unit* fails. On failure the
            component is rolled back to its original unit.
        """
        if not isinstance(new_x_unit, (str, sc.Unit)):
            raise TypeError(f'x_unit must be a string or sc.Unit, got {type(new_x_unit).__name__}')

        old_unit = self._x_unit
        pars = self.get_all_parameters()
        try:
            for p in pars:
                p.convert_unit(new_x_unit)
            self._x_unit = str(new_x_unit) if isinstance(new_x_unit, sc.Unit) else new_x_unit
        except Exception as e:
            try:
                for p in pars:
                    p.convert_unit(old_unit)
            except Exception:  # noqa: S110
                pass
            raise e

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis (output) unit. Subclasses with an area parameter should override this.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.

        Raises
        ------
        NotImplementedError
            Always raised in this base implementation.  Subclasses that carry an area parameter
            (area_unit = x_unit * y_unit) must override this method to rescale the area
            appropriately.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support convert_y_unit.')

    @abstractmethod
    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        """
        Evaluate the model component at input x.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
        output : str, default='numpy'
            'numpy' returns np.ndarray; 'scipp' returns sc.Variable with y_unit.

        Returns
        -------
        np.ndarray | sc.Variable
            Evaluated model values at x.
        """

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(unique_name={self.unique_name}, '
            f'x_unit={self._x_unit}, y_unit={self._y_unit})'
        )
