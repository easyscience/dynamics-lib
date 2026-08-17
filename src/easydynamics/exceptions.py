# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


class AmbiguousNameError(Exception):
    """Raised when a name lookup matches more than one element."""

    def __init__(self, name: str, matches: list[object]) -> None:
        """
        Initialize the AmbiguousNameError.

        Parameters
        ----------
        name : str
            The ambiguous name that was looked up.
        matches : list[object]
            The elements whose name matched. The elements' unique names are used in the
            message so the matches can be told apart.
        """
        self.name = name
        self.matches = matches
        match_names = [
            match.unique_name if hasattr(match, 'unique_name') else str(match)
            for match in matches
        ]
        super().__init__(
            f"Ambiguous name '{name}' matches {len(matches)} elements: {match_names}"
        )
