# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc
from easyscience.base_classes import ModelBase

from easydynamics.base_classes.name_mixin import NameMixin
from easydynamics.utils.utils import _validate_unit


class EasyDynamicsModelBase(NameMixin, ModelBase):
    """Base class for all EasyDynamics models."""

    def __init__(
        self,
        *args: object,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'MyEasyDynamicsModel',
        display_name: str | None = None,
        unique_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """
        Initialize the EasyDynamicsModelBase.

        Parameters
        ----------
        *args : object
            Positional arguments to pass to the parent class.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy, Q, etc.).
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity).
        name : str, default='MyEasyDynamicsModel'
            Name of the model.
        display_name : str | None, default=None
            Display name of the model. If None, the name will be used.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        **kwargs : object
             Additional keyword arguments to pass to the parent class.

        Raises
        ------
        TypeError
            If name is not a string.
        """

        if not isinstance(name, str):
            raise TypeError(f'Name must be a string, got {type(name)}')

        if display_name is None:
            display_name = name

        super().__init__(
            *args,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
            **kwargs,
        )

        self._x_unit = _validate_unit(x_unit)
        self._y_unit = _validate_unit(y_unit)

    @property
    def x_unit(self) -> str | sc.Unit | None:
        """
        Get the unit of the x-axis.

        Returns
        -------
        str | sc.Unit | None
            The unit of the x-axis (energy, Q, etc.).
        """
        return self._x_unit

    @x_unit.setter
    def x_unit(self, _: str) -> None:
        raise AttributeError(
            f'x_unit is read-only. Use convert_x_unit to change the unit '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    @property
    def y_unit(self) -> str | sc.Unit | None:
        """
        Get the unit of the model output.

        Returns
        -------
        str | sc.Unit | None
            The unit of the model output (intensity).
        """
        return self._y_unit

    @y_unit.setter
    def y_unit(self, _: str) -> None:
        raise AttributeError(
            f'y_unit is read-only. Use convert_y_unit to change the unit '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )
