# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from easyscience.base_classes.new_base import NewBase

from easydynamics.base_classes.name_mixin import NameMixin


class EasyDynamicsBase(NameMixin, NewBase):
    """Base class for all EasyDynamics classes."""

    def __init__(
        self,
        *args: object,
        name: str = "MyEasyDynamicsModel",
        display_name: str | None = None,
        unique_name: str | None = None,
        **kwargs: object,
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
        **kwargs : object
             Additional keyword arguments to pass to the parent class.

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
