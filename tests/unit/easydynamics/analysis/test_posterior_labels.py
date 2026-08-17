# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from easyscience.variable import Parameter

from easydynamics.analysis.posterior_labels import ParameterLabels


def make_parameter(name, unit='meV'):
    return Parameter(name=name, value=1.0, unit=unit)


class TestLabelling:
    def test_unique_names_are_left_alone(self):
        # WHEN nothing is ambiguous, a qualifier would only cost width
        parameters = [make_parameter('area'), make_parameter('width')]

        # THEN
        labels = ParameterLabels(parameters, qualify=lambda _p: 'Q_index=0')

        # EXPECT
        assert [labels.label(p) for p in parameters] == ['area', 'width']

    def test_shared_names_are_qualified(self):
        # WHEN two parameters share a name
        first, second = make_parameter('width'), make_parameter('width')
        owners = {first.unique_name: 'Q_index=0', second.unique_name: 'Q_index=1'}

        # THEN
        labels = ParameterLabels([first, second], qualify=lambda p: owners[p.unique_name])

        # EXPECT
        assert labels.label(first) == 'width (Q_index=0)'
        assert labels.label(second) == 'width (Q_index=1)'

    def test_a_qualifier_that_declines_leaves_the_name_alone(self):
        # WHEN the qualifier cannot identify an owner, as for a parameter shared across Q
        first, second = make_parameter('width'), make_parameter('width')

        # THEN
        labels = ParameterLabels([first, second], qualify=lambda _p: None)

        # EXPECT the plain name rather than an invented qualifier
        assert labels.label(first) == 'width'

    def test_without_a_qualifier_names_stay_bare(self):
        # WHEN
        first, second = make_parameter('width'), make_parameter('width')

        # THEN
        labels = ParameterLabels([first, second])

        # EXPECT
        assert labels.label(first) == 'width'


class TestChainColumns:
    def test_columns_resolve_by_unique_name(self):
        # WHEN
        parameters = [make_parameter('area'), make_parameter('width')]
        labels = ParameterLabels(parameters)
        columns = [p.unique_name for p in reversed(parameters)]

        # THEN EXPECT resolution follows the chain's order, not the parameter list's
        assert labels.resolve(columns) == list(reversed(parameters))
        assert labels.display_names(columns) == ['width', 'area']
        assert labels.units(columns) == ['meV', 'meV']

    def test_a_saved_chain_resolves_through_its_labels(self):
        # WHEN a chain was saved in another session, so its unique names mean nothing here
        original = make_parameter('width')
        saved = {original.unique_name: 'width'}
        current = make_parameter('width')

        # THEN
        labels = ParameterLabels([current])

        # EXPECT the saved label finds the parameter this session has
        assert labels.resolve([original.unique_name], saved) == [current]
        assert labels.display_names([original.unique_name], saved) == ['width']

    def test_an_unknown_column_is_reported_not_guessed(self):
        # THEN
        labels = ParameterLabels([make_parameter('area')])

        # EXPECT None rather than a wrong parameter, and the raw name to show something
        assert labels.resolve(['Parameter_999']) == [None]
        assert labels.display_names(['Parameter_999']) == ['Parameter_999']
        assert labels.units(['Parameter_999']) == ['']

    def test_colliding_labels_get_distinct_sidecar_entries(self):
        # WHEN two parameters end up with the same display label
        first, second = make_parameter('width'), make_parameter('width')

        # THEN
        labels = ParameterLabels([first, second])

        # EXPECT deterministic positional suffixes in the sidecar mapping, rather than a silent
        # last-write-wins, while the display label stays bare
        assert labels.name_map() == {
            first.unique_name: 'width [1]',
            second.unique_name: 'width [2]',
        }
        assert labels.label(first) == 'width'

    def test_colliding_labels_round_trip_to_their_own_parameters(self):
        # WHEN a chain of two same-labelled parameters was saved in another session
        old_first, old_second = make_parameter('width'), make_parameter('width')
        saved = ParameterLabels([old_first, old_second]).name_map()
        new_first, new_second = make_parameter('width'), make_parameter('width')

        # THEN
        labels = ParameterLabels([new_first, new_second])
        resolved = labels.resolve([old_first.unique_name, old_second.unique_name], saved)

        # EXPECT each column finds its own parameter, not both the same one
        assert resolved == [new_first, new_second]

    def test_name_map_records_labels_against_unique_names(self):
        # WHEN
        first, second = make_parameter('width'), make_parameter('width')
        owners = {first.unique_name: 'Q_index=0', second.unique_name: 'Q_index=1'}

        # THEN
        labels = ParameterLabels([first, second], qualify=lambda p: owners[p.unique_name])

        # EXPECT what save() writes alongside a chain
        assert labels.name_map() == {
            first.unique_name: 'width (Q_index=0)',
            second.unique_name: 'width (Q_index=1)',
        }


class TestCost:
    def test_labelling_does_not_rescan_per_parameter(self):
        # WHEN there are many parameters. Computing the name counts per parameter is quadratic,
        # which was seconds of work for an analysis with many Q values.
        parameters = [make_parameter(f'p{i // 2}') for i in range(400)]
        labels = ParameterLabels(parameters, qualify=lambda _p: 'q')

        # THEN EXPECT labelling all of them stays cheap
        import time

        start = time.perf_counter()
        names = [labels.label(p) for p in parameters]
        assert time.perf_counter() - start < 0.5
        assert len(names) == len(parameters)
        assert np.all([n.endswith('(q)') for n in names])
