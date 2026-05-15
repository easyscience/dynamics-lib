# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from typing import TypeVar

from easyscience.base_classes.easy_list import EasyList

from easydynamics.base_classes.name_mixin import NameMixin

from .new_base import NewBase

ProtectedType_ = TypeVar("ProtectedType", bound=NewBase)


class EasyDynamicsList(EasyList, NameMixin):
    """Base class for all EasyDynamics lists."""

    def __init__(
        self,
        *args: ProtectedType_ | list[ProtectedType_],
        protected_types: list[type[NewBase]] | type[NewBase] | None = None,
        name: str = "MyEasyDynamicsList",
        display_name: str | None = None,
        unique_name: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """
        Initialize the EasyDynamicsList.

        Parameters
        ----------
        args : ProtectedType_ | list[ProtectedType_]
            Initial items to add to the list. Can be a single item or a list of items. Each item must be an instance of one of the protected types.
        protected_types : list[type[NewBase]] | type[NewBase] | None, optional
            Types that are allowed in the list. Can be a single NewBase subclass or a list of them. If None, defaults to [NewBase].
        name : str, default='MyEasyDynamicsList'
            Name of the list.
        display_name : str | None, default=None
            Display name of the list. If None, the name will be used.
        unique_name : str | None, default=None
            Unique name of the list. If None, a unique name will be generated.
        kwargs : Any
            Additional keyword arguments to pass to the EasyList constructor.

        """
        NameMixin.__init__(self, name=name)

        if display_name is None:
            display_name = self.name

        EasyList.__init__(
            self,
            *args,
            protected_types=protected_types,
            display_name=display_name,
            unique_name=unique_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Methods
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
        self._check_name_unique(value)
        super().insert(index, value)
        value.lock_name()

    def append(self, value: ProtectedType_) -> None:
        """
        Append an item to the end of the list.
        Parameters
        ----------
        value : ProtectedType_
            The item to append. Must be an instance of one of the protected types.
        """
        self._check_name_unique(value)
        super().append(value)
        value.lock_name()

    def extend(self, values: Iterable[ProtectedType_]) -> None:
        """
        Extend the list by appending elements from the iterable.
        Parameters
        ----------
        values : Iterable[ProtectedType_]
            An iterable of items to append. Each item must be an instance of one of the protected types.
        """
        values = list(values)
        self._check_name_unique(values)
        for v in values:
            self.append(v)

    def pop(self, idx: int) -> ProtectedType_:
        """
        Remove and return an item at a specific index.

        Parameters
        ----------
        idx : int
            The index at which to pop the item.

        Returns
        -------
        ProtectedType_
            The item that was popped.
        """
        item = self[idx]
        item.unlock_name()
        return super().pop(idx)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _check_name_unique(self, obj: NewBase | Iterable[NewBase]) -> None:
        """Check that the name of an object is unique in the list.
        Parameters
        ----------
        obj : NewBase | Iterable[NewBase]
            Object or objects to check. Can be a single object or an iterable of objects.
        Raises
        ------
        ValueError
            If the name of the object is not unique in the list.
        """

        items = [obj] if isinstance(obj, NewBase) else list(obj)

        get_key = self._get_key
        new_names = [get_key(item) for item in items]

        if len(new_names) != len(set(new_names)):
            raise ValueError(f"Duplicate names in {obj} detected.")

        existing_names = {get_key(o) for o in self._data}

        conflict = existing_names.intersection(new_names)
        if conflict:
            name = next(iter(conflict))
            raise ValueError(f'Name "{name}" already exists in list.')

    def _get_key(self, obj: NewBase) -> str:
        """Get the name of an object.

        Parameters
        ----------
        obj : NewBase
            Object to get the key for.

        Returns
        -------
        str
            The name of the object.
        """
        return obj.name

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __setitem__(
        self, idx: int | slice, value: ProtectedType_ | Iterable[ProtectedType_]
    ) -> None:
        """
        Set an item in the list at a specific index.

        Parameters
        ----------
        idx : int | slice
            The index at which to set the item.
        value : ProtectedType_ | Iterable[ProtectedType_]
            The item or items to set. Must be an instance of one of the protected types or an iterable of protected types.
        """
        self._check_name_unique(value)
        super().__setitem__(idx, value)
        if isinstance(idx, slice):
            for v in value:
                v.lock_name()
        else:
            value.lock_name()

    def __delitem__(self, idx: int | slice) -> None:
        """
        Delete an item from the list at a specific index.

        Parameters
        ----------
        idx : int | slice
            The index at which to delete the item.
        """
        if isinstance(idx, int):
            self[idx].unlock_name()
        elif isinstance(idx, slice):
            for item in self[idx]:
                item.unlock_name()
        super().__delitem__(idx)
