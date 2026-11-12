# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import gc

from easyscience.variable import Parameter as EasyScienceParameter

from easydynamics.base_classes.unique_names import generate_unique_name
from easydynamics.variable import Parameter


class TestGenerateUniqueName:
    """Tests for the per-prefix unique-name generator."""

    def test_names_have_the_easyscience_form(self):
        # WHEN THEN
        name = generate_unique_name('TestPrefixForm')

        # EXPECT
        prefix, number = name.rsplit('_', 1)
        assert prefix == 'TestPrefixForm'
        assert number.isdecimal()

    def test_names_are_never_repeated(self):
        # WHEN THEN
        names = [generate_unique_name('TestPrefixRepeat') for _ in range(100)]

        # EXPECT
        assert len(set(names)) == 100

    def test_skips_names_already_in_the_registry(self):
        # WHEN the next name of a prefix is taken by an explicitly named object
        first = generate_unique_name('TestPrefixSkip')
        prefix, number = first.rsplit('_', 1)
        taken = Parameter(value=1.0, unique_name=f'{prefix}_{int(number) + 1}')

        # THEN
        second = generate_unique_name('TestPrefixSkip')

        # EXPECT the taken name is skipped
        assert second != taken.unique_name
        assert second == f'{prefix}_{int(number) + 2}'

    def test_counts_per_prefix(self):
        # WHEN two names drawn for one prefix
        generate_unique_name('TestPrefixA')
        generate_unique_name('TestPrefixA')

        # THEN
        b = generate_unique_name('TestPrefixB')

        # EXPECT the other prefix still starts from zero
        assert b == 'TestPrefixB_0'

    def test_interleaves_with_easyscience_names_without_collision(self):
        # WHEN EasyDynamics objects, then EasyScience ones (which number from the highest live
        # name), then garbage collection (after which EasyScience reuses low numbers), then both
        # again
        ours = [Parameter(value=1.0, name='p') for _ in range(5)]
        theirs = [EasyScienceParameter(value=1.0) for _ in range(5)]
        del ours
        gc.collect()
        theirs.extend(EasyScienceParameter(value=1.0) for _ in range(5))
        ours = [Parameter(value=1.0, name='p') for _ in range(5)]

        # THEN
        names = [parameter.unique_name for parameter in theirs + ours]

        # EXPECT every live object has its own name
        assert len(set(names)) == len(names)
