# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easydynamics.base_classes.unique_names import generate_unique_name


class NameMixin:
    """Mixin class to add name functionality to EasyDynamics classes."""

    def __init__(
        self,
        *args: object,
        name: str = 'MyEasyDynamicsModel',
        **kwargs: object,
    ) -> None:
        """
        Initialize the NameMixin.

        Parameters
        ----------
        *args : object
            Positional arguments to pass to the parent class.
        name : str, default='MyEasyDynamicsModel'
            Name of the model.
        **kwargs : object
            Keyword arguments to pass to the parent class.

        Raises
        ------
        TypeError
            If name is not a string.
        """

        # Validate before delegating to the parent class so an invalid name fails fast,
        # before the parent registers the object in the global registry.
        if not isinstance(name, str):
            raise TypeError('Name must be a string.')
        # When a subclass passes unique_name=None, hand the parent a name generated here rather
        # than letting EasyScience generate one: its generator scans every registered name, which
        # made building models O(N^2) in the number of objects. Flag it as a default name
        # afterwards, like EasyScience does for its own (EasyBase sets the flag False for any name
        # it is given), so serialization leaves it out and copies get fresh names.
        generated = 'unique_name' in kwargs and kwargs['unique_name'] is None
        if generated:
            kwargs['unique_name'] = generate_unique_name(type(self).__name__)
        super().__init__(*args, **kwargs)
        if generated:
            self._default_unique_name = True
        self._name = name

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
