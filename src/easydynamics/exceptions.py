# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


class AmbiguousNameError(Exception):
    def __init__(self, name: str, matches: list[str]) -> None:
        self.name = name
        self.matches = matches
        super().__init__(f"Ambiguous name '{name}' matches {len(matches)} elements: {matches}")
