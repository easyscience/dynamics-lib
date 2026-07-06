# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from easydynamics.exceptions import AmbiguousNameError


class TestAmbiguousNameError:
    def test_initialization(self):
        # WHEN
        name = 'test'
        matches = ['test1', 'test2', 'test3']

        # THEN
        error = AmbiguousNameError(name, matches)

        # EXPECT
        assert error.name == name
        assert error.matches == matches
        assert str(error) == (
            "Ambiguous name 'test' matches 3 elements: ['test1', 'test2', 'test3']"
        )

    def test_empty_matches(self):
        # WHEN
        name = 'unknown'
        matches = []

        # THEN
        error = AmbiguousNameError(name, matches)

        # EXPECT
        assert error.name == name
        assert error.matches == matches
        assert str(error) == ("Ambiguous name 'unknown' matches 0 elements: []")
