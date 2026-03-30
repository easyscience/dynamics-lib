# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience.base_classes.new_base import NewBase


class EasyDynamicsBase(NewBase):
    """Base class for all EasyDynamics classes."""

    def __init__(
        self,
        name: str | None = 'MyEasyDynamicsModel',
        display_name: str | None = 'MyEasyDynamicsModel',
        unique_name: str | None = None,
    ) -> None:
        """Initialize the EasyDynamicsBase.

        Args:
            name (str | None, default="MyEasyDynamicsModel"): Name of the model.
            display_name (str, default="MyEasyDynamicsModel"): Display name of the model.
            unique_name (str | None, default=None): Unique name of the model. If None,
                a unique name will be generated.

        Raises:
            TypeError: If name is not a string or None.
        """
        super().__init__(display_name=display_name, unique_name=unique_name)

        if name is not None and not isinstance(name, str):
            raise TypeError('Name must be a string or None.')
        self._name = name

    @property
    def name(self) -> str | None:
        """Get the name of the model.

        Returns:
            str | None: The name of the model.
        """
        return self._name

    @name.setter
    def name(self, name_str: str | None) -> None:
        """Set the name of the model.

        Args:
            name_str (str | None): The new name to set.
        """

        if name_str is not None and not isinstance(name_str, str):
            raise TypeError('Name must be a string or None.')
        self._name = name_str
