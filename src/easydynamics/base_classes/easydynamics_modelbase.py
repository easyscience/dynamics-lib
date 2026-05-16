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
        unit: str | sc.Unit = "meV",
        name: str = "MyEasyDynamicsModel",
        display_name: str | None = None,
        unique_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """
        Initialize the EasyDynamicsModelBase.

        Parameters
        ----------
        unit : str | sc.Unit, default='meV'
            Unit of the model.
        name : str, default='MyEasyDynamicsModel'
            Name of the model.
        display_name : str | None, default=None
            Display name of the model. If None, the name will be used.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If name is not a string.
        """

        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name)}")

        if display_name is None:
            display_name = name

        super().__init__(
            *args,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
            **kwargs,
        )

        self._unit = _validate_unit(unit)

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
            f"Unit is read-only. Use convert_unit to change the unit between allowed types "
            f"or create a new {self.__class__.__name__} with the desired unit."
        )
