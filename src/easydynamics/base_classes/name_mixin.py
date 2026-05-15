# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


class NameMixin:
    """Mixin class to add name functionality to EasyDynamics classes."""

    def __init__(
        self,
        name: str = "MyEasyDynamicsModel",
    ) -> None:
        """
        Initialize the NameMixin.

        Parameters
        ----------
        name : str, default='MyEasyDynamicsModel'
            Name of the model.

        Raises
        ------
        TypeError
            If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Name must be a string.")
        self._name = name
        self._name_lock_count = 0

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

        if self._name_lock_count > 0:
            raise AttributeError("Cannot change name while object is in a list.")

        if not isinstance(name_str, str):
            raise TypeError("Name must be a string.")
        self._name = name_str

    def lock_name(self) -> None:
        """Prevent the name from being modified."""
        self._name_lock_count += 1

    def unlock_name(self) -> None:
        """Allow the name to be modified if no containers remain."""
        if self._name_lock_count == 0:
            raise RuntimeError("Name lock count is already zero.")

        self._name_lock_count -= 1
