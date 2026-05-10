# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience.base_classes.new_base import NewBase


class EasyDynamicsBase(NewBase):
    """Base class for all EasyDynamics classes."""

    def __init__(
        self,
        name: str = 'MyEasyDynamicsModel',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the EasyDynamicsBase.

        Parameters
        ----------
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
            raise TypeError('Name must be a string.')
        self._name = name

        if display_name is None:
            display_name = name

        super().__init__(display_name=display_name, unique_name=unique_name)

    @property
    def name(self) -> str:
        """
        Get the name of the model.

        Returns
        -------
        str
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
            If name_str is not a string.
        """

        if not isinstance(name_str, str):
            raise TypeError('Name must be a string.')
        self._name = name_str
