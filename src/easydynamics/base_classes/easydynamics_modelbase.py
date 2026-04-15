# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc
from easyscience.base_classes import ModelBase

from easydynamics.utils.utils import _validate_unit


class EasyDynamicsModelBase(ModelBase):
    """Base class for all EasyDynamics models."""

    def __init__(
        self,
        unit: str | sc.Unit = 'meV',
        name: str | None = 'MyEasyDynamicsModel',
        display_name: str | None = 'MyEasyDynamicsModel',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the EasyDynamicsModelBase.

        Parameters
        ----------
        unit : str | sc.Unit, default='meV'
            Unit of the model.
        name : str | None, default='MyEasyDynamicsModel'
            Name of the model.
        display_name : str | None, default='MyEasyDynamicsModel'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If name is not a string or None.
        """
        super().__init__(display_name=display_name, unique_name=unique_name)
        self._unit = _validate_unit(unit)

        if name is not None and not isinstance(name, str):
            raise TypeError('Name must be a string or None.')
        self._name = name

    @property
    def unit(self) -> str | sc.Unit | None:
        """
        Get the unit of the model.

        Returns
        -------
        str | sc.Unit | None
             The unit of the model.
        """

        return self._unit

    @unit.setter
    def unit(self, _unit_str: str) -> None:
        """
        Unit is read-only and cannot be set directly.

        Parameters
        ----------
        _unit_str : str
            The new unit to set (ignored).

        Raises
        ------
        AttributeError
            Always raised to indicate that the unit is read-only.
        """
        raise AttributeError(
            f'Unit is read-only. Use convert_unit to change the unit between allowed types '
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    @property
    def name(self) -> str | None:
        """
        Get the name of the model.

        Returns
        -------
        str | None
            The name of the model.
        """
        return self._name

    @name.setter
    def name(self, name_str: str) -> None:
        """
        Set the name of the model.

        Parameters
        ----------
        name_str : str
            The new name to set.

        Raises
        ------
        TypeError
            If name_str is not a string or None.
        """

        if name_str is not None and not isinstance(name_str, str):
            raise TypeError('Name must be a string or None.')
        self._name = name_str
