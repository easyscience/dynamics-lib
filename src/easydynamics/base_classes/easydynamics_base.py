# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience.base_classes.new_base import NewBase

from easydynamics.base_classes.name_mixin import NameMixin


class EasyDynamicsBase(NewBase, NameMixin):
    """Base class for all EasyDynamics classes."""

    def __init__(
        self,
        name: str = "MyEasyDynamicsModel",
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
        NameMixin.__init__(self, name=name)

        if display_name is None:
            display_name = self.name

        NewBase.__init__(self, display_name=display_name, unique_name=unique_name)
