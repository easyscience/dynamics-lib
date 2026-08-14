# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Naming the columns of an MCMC chain.

The sampler labels its columns with each parameter's ``unique_name`` -- ``Parameter_4`` and the
like -- which is not what a user recognises, and which is handed out per session so it does not
survive a saved chain either. This turns those columns back into readable labels.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from easyscience.variable import Parameter


class ParameterLabels:
    """
    Readable labels and units for the columns of a chain.

    Built once for a fixed set of parameters, so the name counts and lookups are computed a single
    time. Doing this per column instead is quadratic in the parameter count, which is seconds of
    work for an analysis with many Q values.

    Parameters
    ----------
    parameters : list[Parameter]
        The parameters that can appear as columns.
    qualify : Callable[[Parameter], str | None] | None, default=None
        Returns a qualifier for a parameter whose name is shared with another, for example its Q
        index. Only consulted when the bare name really is ambiguous, so an analysis with nothing
        to disambiguate keeps its short names. Returning None leaves the name unqualified.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        qualify: Callable[[Parameter], str | None] | None = None,
    ) -> None:
        self._parameters = list(parameters)
        self._qualify = qualify
        self._counts = Counter(parameter.name for parameter in self._parameters)
        self._by_unique_name = {p.unique_name: p for p in self._parameters}
        self._by_label = {self.label(p): p for p in self._parameters}

    @property
    def parameters(self) -> list[Parameter]:
        """
        The parameters these labels describe.

        Returns
        -------
        list[Parameter]
            The parameters given at construction.
        """
        return list(self._parameters)

    def label(self, parameter: Parameter) -> str:
        """
        Get the label a parameter is reported under.

        Parameters
        ----------
        parameter : Parameter
            The parameter to label.

        Returns
        -------
        str
            The parameter's name, qualified only where that name is shared with another parameter.
        """
        if self._counts[parameter.name] <= 1 or self._qualify is None:
            return parameter.name
        qualifier = self._qualify(parameter)
        return parameter.name if qualifier is None else f'{parameter.name} ({qualifier})'

    def name_map(self) -> dict[str, str]:
        """
        Map each parameter's ``unique_name`` to its label.

        Saved alongside a chain, because unique names are per-session: without this a reloaded
        chain cannot be matched back to any parameter.

        Returns
        -------
        dict[str, str]
            Mapping of unique name to label.
        """
        return {p.unique_name: self.label(p) for p in self._parameters}

    def resolve(
        self,
        column_names: list[str],
        saved_labels: dict[str, str] | None = None,
    ) -> list[Parameter | None]:
        """
        Match each column of a chain to a parameter.

        Columns are matched on ``unique_name`` first. That fails for a chain loaded from disk,
        where the saved labels are used instead.

        Parameters
        ----------
        column_names : list[str]
            The sampler's name for each column.
        saved_labels : dict[str, str] | None, default=None
            Mapping of unique name to label, as recorded when a chain was saved.

        Returns
        -------
        list[Parameter | None]
            The parameter for each column, or None where no match could be made.
        """
        saved_labels = saved_labels or {}
        resolved = []
        for unique_name in column_names:
            parameter = self._by_unique_name.get(unique_name)
            if parameter is None:
                parameter = self._by_label.get(saved_labels.get(unique_name, ''))
            resolved.append(parameter)
        return resolved

    def display_names(
        self,
        column_names: list[str],
        saved_labels: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Get a readable label for each column of a chain.

        Parameters
        ----------
        column_names : list[str]
            The sampler's name for each column.
        saved_labels : dict[str, str] | None, default=None
            Mapping of unique name to label, as recorded when a chain was saved.

        Returns
        -------
        list[str]
            One label per column, falling back to the saved label and then to the raw column name.
        """
        saved_labels = saved_labels or {}
        return [
            saved_labels.get(unique_name, unique_name)
            if parameter is None
            else self.label(parameter)
            for unique_name, parameter in zip(
                column_names, self.resolve(column_names, saved_labels), strict=True
            )
        ]

    def units(
        self,
        column_names: list[str],
        saved_labels: dict[str, str] | None = None,
    ) -> list[str]:
        """
        Get the unit of each column of a chain.

        Parameters
        ----------
        column_names : list[str]
            The sampler's name for each column.
        saved_labels : dict[str, str] | None, default=None
            Mapping of unique name to label, as recorded when a chain was saved.

        Returns
        -------
        list[str]
            One unit per column, empty where no parameter could be matched.
        """
        return [
            '' if parameter is None else str(parameter.unit)
            for parameter in self.resolve(column_names, saved_labels)
        ]
