# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TypeVar

from easyscience.base_classes.easy_list import EasyList

from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.exceptions import AmbiguousNameError

ProtectedType_ = TypeVar('T', bound=EasyDynamicsBase | EasyDynamicsModelBase)


class EasyDynamicsList(EasyList[ProtectedType_]):
    """Base class for all EasyDynamics lists."""

    def __init__(
        self,
        *args: ProtectedType_ | list[ProtectedType_],
        protected_types: (list[type[ProtectedType_]] | type[ProtectedType_] | None) = None,
        display_name: str | None = None,
        unique_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """
        Initialize the EasyDynamicsList.

        Parameters
        ----------
        *args : ProtectedType_ | list[ProtectedType_]
            Initial items to add to the list. Can be a single item or a list of items. Each item
            must be an instance of one of the protected types.
        protected_types : list[type[ProtectedType_]] | type[ProtectedType_] | None, default=None
            Types that are allowed in the list. Can be a single EasyDynamicsBase or
            EasyDynamicsModelBase subclass or a list of them. If None, defaults to
            [EasyDynamicsBase].
        display_name : str | None, default=None
            Display name of the list. If None, the name will be used.
        unique_name : str | None, default=None
            Unique name of the list. If None, a unique name will be generated.
        **kwargs : object
            Additional keyword arguments to pass to the EasyList constructor.
        """

        if display_name is None:
            display_name = unique_name

        super().__init__(
            *args,
            protected_types=protected_types,
            display_name=display_name,
            unique_name=unique_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # List methods
    # ------------------------------------------------------------------

    def insert(self, index: int, value: ProtectedType_) -> None:
        """
        Insert an item into the list at a specific index.

        Parameters
        ----------
        index : int
            The index at which to insert the item.
        value : ProtectedType_
            The item to insert. Must be an instance of one of the protected types.
        """

        # Overwritten to update warning
        self._validate_type(value)
        if value in self:
            warnings.warn(
                (
                    f'Item with name "{self._get_key(value)}" already '
                    f'in EasyDynamicsList, it will be ignored'
                ),
                UserWarning,
                stacklevel=2,
            )
            return

        super().insert(index, value)

    def append(self, value: ProtectedType_) -> None:
        """
        Append an item to the end of the list.
        Parameters
        ----------
        value : ProtectedType_
            The item to append. Must be an instance of one of the protected types.
        """
        self._validate_type(value)

        # append calls insert which checks for duplicates
        super().append(value)

    def pop(self, index: int | str = -1) -> ProtectedType_:
        """
        Remove and return an item at a specific index or name.

        Parameters
        ----------
        index : int | str, default=-1
            The index or name at which to pop the item.

        Returns
        -------
        ProtectedType_
            The item that was popped.

        Raises
        ------
        TypeError
            If index is not an int or str.
        KeyError
            If index is a str and no item with that name is found.
        """

        # Overwritten to update warning
        if isinstance(index, int):
            return self._data.pop(index)
        if isinstance(index, str):
            for i, item in enumerate(self._data):
                if self._get_key(item) == index:
                    return self._data.pop(i)
            raise KeyError(f'No item with name "{index}" found')
        raise TypeError('Index must be an int or str')

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def get_names(self) -> list[str]:
        """
        Get a list of the names of all items in the list.

        Returns
        -------
        list[str]
            A list of the names of all items in the list.
        """
        return [item.name for item in self._data]

    def get_duplicate_names(self) -> list[str]:
        """
        Get a list of duplicate names in the list.

        Returns
        -------
        list[str]
            A list of duplicate names in the list.
        """
        seen = set()
        duplicates = set()
        for item in self._data:
            name = item.name
            if name in seen:
                duplicates.add(name)
            else:
                seen.add(name)
        return list(duplicates)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _get_key(self, obj: EasyDynamicsBase | EasyDynamicsModelBase) -> str:
        """
        Get the name of an object.

        Parameters
        ----------
        obj : EasyDynamicsBase | EasyDynamicsModelBase
            Object to get the key for.

        Returns
        -------
        str
            The name of the object.
        """
        return obj.name

    def _validate_type(self, value: object) -> None:
        """
        Validate that a value is an instance of one of the protected types.

        Parameters
        ----------
        value : object
            The value to validate.

        Raises
        ------
        TypeError
             If the value is not an instance of one of the protected types.
        """

        if not isinstance(value, tuple(self._protected_types)):
            allowed = ', '.join(t.__name__ for t in self._protected_types)
            raise TypeError(
                f'Value must be an instance of type: {allowed}. Got {type(value).__name__} instead.'  # noqa: E501
            )

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------
    def __getitem__(
        self, idx: int | slice | str
    ) -> ProtectedType_ | EasyDynamicsList[ProtectedType_]:
        """
        Get an item by index, slice, or unique_name.

        Parameters
        ----------
        idx : int | slice | str
                Index, slice, or name of the item to get.

        Returns
        -------
        ProtectedType_ | EasyDynamicsList[ProtectedType_]
             The item at the specified index or name, or a new EasyDynamicsList if a slice is
             provided.

        Raises
        ------
        TypeError
            If idx is not an int, slice, or str.
        KeyError
            If idx is a str and no item with that name is found.
        AmbiguousNameError
            If idx is a str and multiple items with that name are found.
        """
        if isinstance(idx, int):
            return self._data[idx]
        if isinstance(idx, slice):
            return self.__class__(self._data[idx], protected_types=self._protected_types)
        if isinstance(idx, str):
            matches = [r for r in self._data if self._get_key(r) == idx]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise AmbiguousNameError(idx, matches)

            raise KeyError(f'No item with name "{idx}" found')
        raise TypeError('Index must be an int, slice, or str')
