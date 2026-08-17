# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from easydynamics.exceptions import AmbiguousNameError
from easydynamics.sample_model import Gaussian


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

    def test_object_matches_print_their_names(self):
        "Regression: the message used to print raw objects instead of their names"
        # WHEN matches are objects with unique names, as raised by EasyDynamicsList
        matches = [
            Gaussian(name='SameName', unique_name='UniqueGaussian1'),
            Gaussian(name='SameName', unique_name='UniqueGaussian2'),
        ]

        # THEN
        error = AmbiguousNameError('SameName', matches)

        # EXPECT the message names the matches instead of dumping object reprs
        assert str(error) == (
            "Ambiguous name 'SameName' matches 2 elements: "
            "['UniqueGaussian1', 'UniqueGaussian2']"
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
