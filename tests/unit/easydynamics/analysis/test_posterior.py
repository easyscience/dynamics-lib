# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
from easyscience.variable import Parameter

from easydynamics.analysis.posterior import BoundsSuggestion
from easydynamics.analysis.posterior import BoundsSuggestions
from easydynamics.analysis.posterior import degenerate_parameters
from easydynamics.analysis.posterior import parameters_at_bounds
from easydynamics.analysis.posterior import suggest_bounds_for_parameters
from easydynamics.analysis.posterior import summarize_draws
from easydynamics.analysis.posterior import unbounded_parameters


def make_parameter(name='p', value=1.0, error=0.0, minimum=-np.inf, maximum=np.inf, unit='meV'):
    parameter = Parameter(name=name, value=value, unit=unit)
    parameter.min = minimum
    parameter.max = maximum
    if error:
        parameter.variance = error**2
    return parameter


class TestSuggestBounds:
    def test_fills_in_both_infinite_sides(self):
        # WHEN
        parameter = make_parameter(value=10.0, error=0.5)

        # THEN
        suggestions = suggest_bounds_for_parameters([parameter], n_sigma=10.0, relative_pad=0.2)

        # EXPECT: 10 * 0.5 + 0.2 * 10 = 7
        suggestion = suggestions.suggestions[0]
        assert suggestion.suggested_min == pytest.approx(3.0)
        assert suggestion.suggested_max == pytest.approx(17.0)
        assert not suggestion.needs_attention

    def test_never_loosens_an_existing_finite_bound(self):
        # WHEN a physical lower bound is already set
        parameter = make_parameter(value=1.2, error=1.5, minimum=1e-10)

        # THEN
        suggestion = suggest_bounds_for_parameters([parameter]).suggestions[0]

        # EXPECT the finite side survives untouched, even though the sigma rule would go negative
        assert suggestion.suggested_min == pytest.approx(1e-10)
        assert suggestion.suggested_max > 1.2

    def test_fully_bounded_parameter_is_left_alone(self):
        # WHEN
        parameter = make_parameter(value=1.0, error=0.1, minimum=0.0, maximum=2.0)

        # THEN
        suggestion = suggest_bounds_for_parameters([parameter]).suggestions[0]

        # EXPECT
        assert suggestion.suggested_min == pytest.approx(0.0)
        assert suggestion.suggested_max == pytest.approx(2.0)
        assert not suggestion.changes_bounds

    def test_zero_error_falls_back_to_the_relative_pad(self):
        # WHEN a minimizer reports no uncertainty at all
        parameter = make_parameter(value=4.0, error=0.0)

        # THEN
        suggestion = suggest_bounds_for_parameters([parameter], relative_pad=0.25).suggestions[0]

        # EXPECT the pad still yields a usable width
        assert suggestion.suggested_min == pytest.approx(3.0)
        assert suggestion.suggested_max == pytest.approx(5.0)
        assert not suggestion.needs_attention

    def test_zero_value_and_zero_error_is_flagged_not_guessed(self):
        # WHEN there is no scale information anywhere
        parameter = make_parameter(value=0.0, error=0.0)

        # THEN
        suggestion = suggest_bounds_for_parameters([parameter]).suggestions[0]

        # EXPECT
        assert suggestion.needs_attention
        assert 'no scale information' in suggestion.reason
        assert not np.isfinite(suggestion.suggested_min)

    def test_absolute_floor_rescues_a_scaleless_parameter(self):
        # WHEN
        parameter = make_parameter(value=0.0, error=0.0)

        # THEN
        suggestion = suggest_bounds_for_parameters([parameter], absolute_floor=0.5).suggestions[0]

        # EXPECT
        assert not suggestion.needs_attention
        assert suggestion.suggested_min == pytest.approx(-0.5)
        assert suggestion.suggested_max == pytest.approx(0.5)

    def test_non_finite_error_is_flagged_not_silently_narrowed(self):
        # WHEN a degenerate fit reports a NaN uncertainty
        parameter = make_parameter(value=5.0)
        parameter.variance = np.nan

        # THEN
        suggestion = suggest_bounds_for_parameters([parameter]).suggestions[0]

        # EXPECT a flag, rather than deceptively tight bounds from the relative pad alone
        assert suggestion.needs_attention
        assert 'uncertainty is not finite' in suggestion.reason

    def test_non_finite_value_is_flagged(self):
        # WHEN
        parameter = make_parameter(value=1.0)
        parameter.value = np.inf

        # THEN
        suggestion = suggest_bounds_for_parameters([parameter]).suggestions[0]

        # EXPECT
        assert suggestion.needs_attention
        assert 'not finite' in suggestion.reason

    @pytest.mark.parametrize('kwargs', [{'n_sigma': -1.0}, {'relative_pad': -0.1}])
    def test_negative_settings_raise(self, kwargs):
        # THEN EXPECT
        with pytest.raises(ValueError):
            suggest_bounds_for_parameters([make_parameter()], **kwargs)

    def test_non_numeric_setting_raises(self):
        # THEN EXPECT
        with pytest.raises(TypeError):
            suggest_bounds_for_parameters([make_parameter()], n_sigma='wide')


class TestBoundsSuggestions:
    #############
    # Applying suggestions
    #############

    def test_apply_sets_bounds_and_reports_changes(self):
        # WHEN nothing has changed yet, since apply has not been called
        parameter = make_parameter(value=10.0, error=0.5)
        suggestions = suggest_bounds_for_parameters([parameter])
        assert parameter.max == np.inf

        # THEN
        changed = suggestions.apply()

        # EXPECT
        assert changed == [parameter]
        assert parameter.min == pytest.approx(3.0)
        assert parameter.max == pytest.approx(17.0)

    def test_apply_skips_parameters_needing_attention(self):
        # WHEN
        parameter = make_parameter(value=0.0, error=0.0)
        suggestions = suggest_bounds_for_parameters([parameter])

        # THEN
        changed = suggestions.apply()

        # EXPECT the unusable suggestion is skipped rather than written
        assert changed == []
        assert parameter.min == -np.inf

    #############
    # Absurd-bounds warning
    #############

    def test_applying_a_wildly_wide_bound_warns(self):
        # WHEN a fit returns an enormous uncertainty, which is what a degenerate parameter looks
        # like coming out of least squares
        parameter = make_parameter(name='Delta area', value=1.0, error=1e9)
        suggestions = suggest_bounds_for_parameters([parameter])

        # THEN EXPECT it is still applied, since it is what the fit implied, but not silently
        with pytest.warns(UserWarning, match='far wider than the parameter'):
            changed = suggestions.apply()
        assert changed == [parameter]

    def test_a_sane_bound_applies_without_warning(self):
        # WHEN
        parameter = make_parameter(name='sane', value=10.0, error=0.5)
        suggestions = suggest_bounds_for_parameters([parameter])

        # THEN EXPECT
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            suggestions.apply()

    def test_a_zero_valued_parameter_is_not_called_absurd(self):
        # WHEN there is no magnitude to compare the width against
        parameter = make_parameter(name='zero', value=0.0, error=1.0)
        suggestions = suggest_bounds_for_parameters([parameter])

        # THEN EXPECT no warning, since the ratio is meaningless rather than alarming
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            suggestions.apply()

    #############
    # Repr and iteration
    #############

    def test_repr_lists_parameters_and_flags_attention(self):
        # WHEN
        good = make_parameter(name='good', value=10.0, error=0.5)
        bad = make_parameter(name='bad', value=0.0, error=0.0)

        # THEN
        text = repr(suggest_bounds_for_parameters([good, bad]))

        # EXPECT
        assert 'good' in text
        assert 'bad' in text
        assert 'need bounds set by hand' in text

    def test_repr_with_no_parameters(self):
        # WHEN THEN EXPECT
        assert 'no free parameters' in repr(BoundsSuggestions([]))

    def test_len_and_iteration(self):
        # WHEN
        suggestions = suggest_bounds_for_parameters([make_parameter(), make_parameter()])

        # THEN EXPECT
        assert len(suggestions) == 2
        assert all(isinstance(s, BoundsSuggestion) for s in suggestions)


class TestUnboundedParameters:
    def test_finds_parameters_with_an_infinite_side(self):
        # WHEN
        bounded = make_parameter(name='bounded', minimum=0.0, maximum=1.0)
        half_open = make_parameter(name='half_open', minimum=0.0)

        # THEN
        result = unbounded_parameters([bounded, half_open])

        # EXPECT
        assert result == [half_open]


class TestDegenerateParameters:
    def test_finds_zero_width_ranges(self):
        # WHEN one parameter's finite bounds enclose no range at all. The setters refuse identical
        # bounds, but a deserialized or hand-built parameter can still carry them, so the internal
        # state is written directly.
        healthy = make_parameter(name='healthy', minimum=0.0, maximum=2.0)
        degenerate = make_parameter(name='degenerate', value=1.0, minimum=0.0, maximum=1.0)
        degenerate._min.value = 1.0

        # THEN
        result = degenerate_parameters([healthy, degenerate])

        # EXPECT
        assert result == [degenerate]

    def test_infinite_bounds_are_not_reported_as_degenerate(self):
        # WHEN a bound is infinite, that is unboundedness rather than degeneracy
        parameter = make_parameter()

        # THEN EXPECT
        assert degenerate_parameters([parameter]) == []


class TestParametersAtBounds:
    def test_uniform_posterior_across_the_bounds_is_reported(self):
        # WHEN a posterior fills its whole allowed range, the bound is setting the interval
        parameter = make_parameter(minimum=0.0, maximum=1.0)
        draws = np.linspace(0.0, 1.0, 1000).reshape(-1, 1)

        # THEN
        result = parameters_at_bounds(draws, [parameter])

        # EXPECT
        assert parameter.unique_name in result
        assert result[parameter.unique_name] == pytest.approx(0.1, abs=0.01)

    def test_posterior_well_inside_its_bounds_is_not_reported(self):
        # WHEN
        parameter = make_parameter(minimum=0.0, maximum=1.0)
        draws = np.random.default_rng(0).normal(0.5, 0.02, size=1000).reshape(-1, 1)

        # THEN
        result = parameters_at_bounds(draws, [parameter])

        # EXPECT
        assert result == {}

    def test_partly_clipped_posterior_is_reported(self):
        # WHEN a posterior fills most, but not all, of its allowed range. A real bound-limited
        # chain looks like this rather than perfectly uniform, so the threshold has to catch it.
        parameter = make_parameter(minimum=0.0, maximum=1.0)
        draws = np.linspace(0.02, 0.98, 1000).reshape(-1, 1)

        # THEN
        result = parameters_at_bounds(draws, [parameter])

        # EXPECT
        assert parameter.unique_name in result

    def test_posterior_pinned_at_one_bound_is_reported(self):
        # WHEN
        parameter = make_parameter(minimum=0.0, maximum=1.0)
        draws = np.abs(np.random.default_rng(0).normal(0.0, 0.02, size=1000)).reshape(-1, 1)

        # THEN
        result = parameters_at_bounds(draws, [parameter])

        # EXPECT
        assert result[parameter.unique_name] > 0.9

    def test_unmatched_and_unbounded_columns_are_skipped(self):
        # WHEN
        unbounded = make_parameter()
        draws = np.zeros((10, 2))

        # THEN
        result = parameters_at_bounds(draws, [None, unbounded])

        # EXPECT
        assert result == {}

    def test_same_named_parameters_do_not_collide(self):
        # WHEN two parameters share a name and both posteriors are pinned at a bound
        first = make_parameter(name='width', minimum=0.0, maximum=1.0)
        second = make_parameter(name='width', minimum=0.0, maximum=1.0)
        draws = np.zeros((100, 2))

        # THEN
        result = parameters_at_bounds(draws, [first, second])

        # EXPECT one entry per parameter, keyed so they cannot overwrite each other
        assert len(result) == 2
        assert set(result) == {first.unique_name, second.unique_name}

    def test_zero_row_draws_return_nothing_rather_than_dividing_by_zero(self):
        # WHEN
        parameter = make_parameter(minimum=0.0, maximum=1.0)
        draws = np.zeros((0, 1))

        # THEN EXPECT
        assert parameters_at_bounds(draws, [parameter]) == {}


class TestSummarizeDraws:
    def test_reports_parameter_names_units_and_percentiles(self):
        # WHEN
        parameter = make_parameter(name='Gaussian width', value=1.5, unit='meV')
        draws = np.linspace(0.0, 100.0, 101).reshape(-1, 1)

        # THEN
        summary = summarize_draws(draws, ['Gaussian width'], [parameter])

        # EXPECT
        entry = summary['Gaussian width']
        assert entry.unit == 'meV'
        assert entry.median == pytest.approx(50.0)
        assert entry.lower == pytest.approx(16.0)
        assert entry.upper == pytest.approx(84.0)
        assert entry.minus == pytest.approx(34.0)
        assert entry.plus == pytest.approx(34.0)
        assert entry.value == pytest.approx(1.5)

    def test_labels_qualified_by_q_are_kept_verbatim(self):
        # WHEN a multi-Q analysis supplies Q-qualified labels, since every Q holds a copy of the
        # same parameter and the bare name would repeat
        parameter = make_parameter(name='Gaussian width')

        # THEN
        summary = summarize_draws(np.zeros((5, 1)), ['Gaussian width (Q_index=2)'], [parameter])

        # EXPECT
        assert summary.entries[0].name == 'Gaussian width (Q_index=2)'
        assert summary.entries[0].unit == 'meV'

    def test_unmatched_column_falls_back_to_the_supplied_name(self):
        # WHEN
        draws = np.zeros((10, 1))

        # THEN
        summary = summarize_draws(draws, ['Parameter_7'], [None])

        # EXPECT
        entry = summary.entries[0]
        assert entry.name == 'Parameter_7'
        assert entry.unit == ''
        assert np.isnan(entry.value)


class TestPosteriorSummary:
    def test_lookup_of_missing_name_raises(self):
        # WHEN
        summary = summarize_draws(np.zeros((5, 1)), ['x'], [None])

        # THEN EXPECT
        with pytest.raises(KeyError):
            summary['not a parameter']

    def test_repr_contains_the_parameter_name(self):
        # WHEN
        parameter = make_parameter(name='Gaussian area')

        # THEN
        text = repr(summarize_draws(np.zeros((5, 1)), ['Gaussian area'], [parameter]))

        # EXPECT
        assert 'Gaussian area' in text
        assert 'median' in text

    def test_len_and_iteration(self):
        # WHEN
        parameters = [make_parameter(name='a'), make_parameter(name='b')]
        summary = summarize_draws(np.zeros((7, 2)), ['a', 'b'], parameters)

        # THEN EXPECT
        assert len(summary) == 2
        assert [entry.name for entry in summary] == ['a', 'b']
        assert len(summary.entries) == 2

    def test_repr_with_no_entries(self):
        # WHEN THEN EXPECT
        assert 'no parameters' in repr(summarize_draws(np.zeros((3, 0)), [], []))
