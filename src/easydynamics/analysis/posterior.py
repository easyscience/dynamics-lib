# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Bounds suggestions and posterior summaries for Bayesian sampling.

The helpers here are deliberately free of any Analysis or Fitter machinery: they operate on plain
``Parameter`` objects and on the ``(n_draws, n_parameters)`` array produced by the sampler, so they
can be unit-tested on their own and reused by every Analysis class.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from easyscience.variable import Parameter

# How many times wider than the parameter's own value a suggested range may be before it is
# reported as suspicious. A fit that returns an uncertainty this large is describing a flat
# direction rather than a measurement.
ABSURD_WIDTH_FACTOR = 1e4

# Fraction of the allowed range at each end that counts as "at the bound" when checking whether
# the posterior has piled up against a bound.
BOUND_EDGE_FRACTION = 0.05

# Fraction of draws inside those edge bands above which a pile-up is reported. A posterior spread
# uniformly across its bounds -- the signature of a bound, rather than the data, setting the
# credible interval -- puts 2 * BOUND_EDGE_FRACTION of its draws there. A posterior comfortably
# inside its bounds puts essentially none there, so the threshold sits well below the uniform value
# to stay sensitive to partly-clipped posteriors without risking false positives.
BOUND_OCCUPANCY_THRESHOLD = 0.05


@dataclass(frozen=True)
class BoundsSuggestion:
    """
    A proposed pair of bounds for a single parameter.

    Attributes
    ----------
    parameter : Parameter
        The parameter the suggestion applies to.
    label : str
        The name the parameter is reported under. For a multi-Q analysis this is qualified by Q,
        since every Q holds an identically named copy of each parameter.
    suggested_min : float
        The proposed lower bound. Equal to the parameter's current lower bound when that is already
        finite.
    suggested_max : float
        The proposed upper bound. Equal to the parameter's current upper bound when that is already
        finite.
    reason : str
        Empty when the suggestion is usable. Otherwise, why the parameter needs manual attention.
    """

    parameter: Parameter
    label: str
    suggested_min: float
    suggested_max: float
    reason: str

    @property
    def needs_attention(self) -> bool:
        """
        Whether this parameter could not be given a usable suggestion.

        Returns
        -------
        bool
            True when no usable bounds could be derived and the user must set them by hand.
        """
        return bool(self.reason)

    @property
    def changes_bounds(self) -> bool:
        """
        Whether applying this suggestion would actually change the parameter.

        Returns
        -------
        bool
            True when either bound differs from the parameter's current bound.
        """
        return self.suggested_min != self.parameter.min or self.suggested_max != self.parameter.max


class BoundsSuggestions:
    """
    The result of :func:`suggest_bounds_for_parameters`, rendered as a table.

    This is advisory: nothing is changed until :meth:`apply` is called. Suggestions only ever fill
    in an infinite bound; a bound that is already finite is never widened or narrowed, so physical
    limits such as a non-negative area survive untouched.
    """

    def __init__(self, suggestions: list[BoundsSuggestion]) -> None:
        """
        Initialize the collection.

        Parameters
        ----------
        suggestions : list[BoundsSuggestion]
            The per-parameter suggestions.
        """
        self._suggestions = list(suggestions)

    @property
    def suggestions(self) -> list[BoundsSuggestion]:
        """
        All suggestions, including those needing manual attention.

        Returns
        -------
        list[BoundsSuggestion]
            The per-parameter suggestions.
        """
        return list(self._suggestions)

    @property
    def needing_attention(self) -> list[BoundsSuggestion]:
        """
        The suggestions for which no usable bounds could be derived.

        Returns
        -------
        list[BoundsSuggestion]
            Suggestions whose parameters must be bounded by hand.
        """
        return [s for s in self._suggestions if s.needs_attention]

    def apply(self) -> list[Parameter]:
        """
        Set the suggested bounds on every parameter that has a usable suggestion.

        Parameters needing manual attention are skipped rather than guessed at. A suggestion that
        is absurdly wide is still applied -- it is what the fit implied -- but warned about, since
        reading the table first is easy to skip in a script.

        Returns
        -------
        list[Parameter]
            The parameters whose bounds were changed.
        """
        changed = []
        absurd = []
        for suggestion in self._suggestions:
            if suggestion.needs_attention or not suggestion.changes_bounds:
                continue
            suggestion.parameter.min = suggestion.suggested_min
            suggestion.parameter.max = suggestion.suggested_max
            changed.append(suggestion.parameter)
            if _is_absurdly_wide(suggestion):
                absurd.append(suggestion.label)

        if absurd:
            warnings.warn(
                (
                    f'Applied bounds far wider than the parameter itself for: '
                    f'{", ".join(absurd)}. That width comes from a very large fitted uncertainty, '
                    f'which usually means these parameters are degenerate with others, so the '
                    f'data cannot determine them separately. Sampling explores that whole range.'
                ),
                UserWarning,
                stacklevel=2,
            )
        return changed

    def __len__(self) -> int:
        """
        Return the number of suggestions.

        Returns
        -------
        int
            The number of suggestions.
        """
        return len(self._suggestions)

    def __iter__(self) -> iter:
        """
        Iterate over the suggestions.

        Returns
        -------
        iter
            An iterator over the suggestions.
        """
        return iter(self._suggestions)

    def __repr__(self) -> str:
        """
        Render the suggestions as a table.

        Returns
        -------
        str
            A table of current and suggested bounds, one row per parameter.
        """
        if not self._suggestions:
            return 'BoundsSuggestions(no free parameters)'

        width = max(len('parameter'), *(len(s.label) for s in self._suggestions))
        header = f'{"parameter":<{width}s} {"current":>26s} {"suggested":>26s}'
        lines = ['BoundsSuggestions', header, '-' * len(header)]
        for s in self._suggestions:
            current = f'({s.parameter.min:.4g}, {s.parameter.max:.4g})'
            if s.needs_attention:
                suggested = f'-- {s.reason}'
            else:
                suggested = f'({s.suggested_min:.4g}, {s.suggested_max:.4g})'
            lines.append(f'{s.label:<{width}s} {current:>26s} {suggested:>26s}')

        attention = self.needing_attention
        if attention:
            lines.append('')
            lines.append(
                f'{len(attention)} parameter(s) need bounds set by hand; .apply() will skip them.'
            )
        return '\n'.join(lines)


def _is_absurdly_wide(suggestion: BoundsSuggestion) -> bool:
    """
    Check whether a suggested range dwarfs the parameter it describes.

    Parameters
    ----------
    suggestion : BoundsSuggestion
        The suggestion to judge.

    Returns
    -------
    bool
        True when the range is more than ``ABSURD_WIDTH_FACTOR`` times the parameter's magnitude.
    """
    scale = abs(float(suggestion.parameter.value))
    if scale == 0:
        # No magnitude to compare against, so the ratio would be meaningless rather than alarming.
        return False
    width = suggestion.suggested_max - suggestion.suggested_min
    # An infinite width compares greater than any threshold, so it needs no separate check.
    return width > ABSURD_WIDTH_FACTOR * scale


def suggest_bounds_for_parameters(
    parameters: list[Parameter],
    labels: list[str] | None = None,
    n_sigma: float = 10.0,
    relative_pad: float = 0.2,
    absolute_floor: float | None = None,
) -> BoundsSuggestions:
    """
    Propose finite bounds for parameters that currently have an infinite one.

    The half-width of a proposed bound is ``n_sigma * error + relative_pad * abs(value)``, floored
    at ``absolute_floor`` when one is given. The ``relative_pad`` term matters because
    least-squares minimizers sometimes report a zero or absurdly small uncertainty; without it such
    a parameter would be given a zero-width bound. When the half-width still comes out as zero or
    non-finite, the parameter is flagged for manual attention rather than given an invented scale.

    In BUMPS' DREAM sampler the bounds act as a uniform prior, so a generous width is the safe
    choice: too narrow a bound truncates the posterior and understates the uncertainty. Hence the
    deliberately loose ``n_sigma`` default.

    A ``TypeError`` is raised if any of the three settings is not a number, and a ``ValueError`` if
    any is negative.

    Parameters
    ----------
    parameters : list[Parameter]
        The parameters to propose bounds for.
    labels : list[str] | None, default=None
        The name to report each parameter under, one per parameter. Defaults to the parameters' own
        names, which is ambiguous when several share a name, as the per-Q copies of a multi-Q
        analysis do.
    n_sigma : float, default=10.0
        How many standard deviations of the parameter's fitted uncertainty to allow on each side.
    relative_pad : float, default=0.2
        Extra half-width as a fraction of the absolute parameter value, guarding against
        artificially small uncertainties.
    absolute_floor : float | None, default=None
        A minimum half-width, in the parameter's own units. Use it when the natural scale is known
        but neither the uncertainty nor the value carries it.

    Returns
    -------
    BoundsSuggestions
        The proposed bounds, which must be applied explicitly.
    """
    _verify_nonneg_number(n_sigma, 'n_sigma')
    _verify_nonneg_number(relative_pad, 'relative_pad')
    if absolute_floor is not None:
        _verify_nonneg_number(absolute_floor, 'absolute_floor')

    if labels is None:
        labels = [parameter.name for parameter in parameters]
    suggestions = [
        _suggest_bounds_for_parameter(
            parameter=parameter,
            label=label,
            n_sigma=n_sigma,
            relative_pad=relative_pad,
            absolute_floor=absolute_floor,
        )
        for parameter, label in zip(parameters, labels, strict=True)
    ]
    return BoundsSuggestions(suggestions)


def _suggest_bounds_for_parameter(
    parameter: Parameter,
    label: str,
    n_sigma: float,
    relative_pad: float,
    absolute_floor: float | None,
) -> BoundsSuggestion:
    """
    Propose bounds for a single parameter.

    Parameters
    ----------
    parameter : Parameter
        The parameter to propose bounds for.
    label : str
        The name to report the parameter under.
    n_sigma : float
        How many standard deviations to allow on each side.
    relative_pad : float
        Extra half-width as a fraction of the absolute parameter value.
    absolute_floor : float | None
        A minimum half-width, or None.

    Returns
    -------
    BoundsSuggestion
        The proposal for this parameter.
    """
    current_min = float(parameter.min)
    current_max = float(parameter.max)
    min_is_finite = np.isfinite(current_min)
    max_is_finite = np.isfinite(current_max)

    # Nothing to fill in: a bound that is already finite is never touched.
    if min_is_finite and max_is_finite:
        return BoundsSuggestion(
            parameter=parameter,
            label=label,
            suggested_min=current_min,
            suggested_max=current_max,
            reason='',
        )

    value = float(parameter.value)
    error = float(parameter.error)
    if not np.isfinite(value):
        return BoundsSuggestion(
            parameter=parameter,
            label=label,
            suggested_min=current_min,
            suggested_max=current_max,
            reason='value is not finite',
        )

    # A NaN uncertainty is exactly the degenerate fit this helper exists to guard against, so it
    # is flagged rather than silently treated like a zero error, which would yield deceptively
    # tight bounds of value +/- relative_pad * |value|.
    if not np.isfinite(error):
        return BoundsSuggestion(
            parameter=parameter,
            label=label,
            suggested_min=current_min,
            suggested_max=current_max,
            reason='fitted uncertainty is not finite',
        )

    half_width = relative_pad * abs(value) + n_sigma * error
    if absolute_floor is not None:
        half_width = max(half_width, absolute_floor)

    if not np.isfinite(half_width) or half_width <= 0:
        return BoundsSuggestion(
            parameter=parameter,
            label=label,
            suggested_min=current_min,
            suggested_max=current_max,
            reason='no scale information (zero value and uncertainty)',
        )

    return BoundsSuggestion(
        parameter=parameter,
        label=label,
        suggested_min=current_min if min_is_finite else value - half_width,
        suggested_max=current_max if max_is_finite else value + half_width,
        reason='',
    )


def unbounded_parameters(parameters: list[Parameter]) -> list[Parameter]:
    """
    Find parameters with a non-finite lower or upper bound.

    Parameters
    ----------
    parameters : list[Parameter]
        The parameters to check.

    Returns
    -------
    list[Parameter]
        Those parameters that have at least one infinite bound.
    """
    return [
        parameter
        for parameter in parameters
        if not (np.isfinite(parameter.min) and np.isfinite(parameter.max))
    ]


def degenerate_parameters(parameters: list[Parameter]) -> list[Parameter]:
    """
    Find parameters whose finite bounds enclose no range at all.

    A zero-width range (``min >= max``) gives DREAM nothing to explore: as the prior it has zero
    volume, and letting it through surfaces only as NaNs deep inside the sampler, far from the
    cause.

    Parameters
    ----------
    parameters : list[Parameter]
        The parameters to check.

    Returns
    -------
    list[Parameter]
        Those parameters whose bounds are both finite with ``min >= max``.
    """
    return [
        parameter
        for parameter in parameters
        if np.isfinite(parameter.min)
        and np.isfinite(parameter.max)
        and float(parameter.min) >= float(parameter.max)
    ]


def parameters_at_bounds(
    draws: np.ndarray,
    parameters_by_column: list[Parameter | None],
) -> dict[str, float]:
    """
    Find parameters whose posterior has piled up against one of its bounds.

    A chain that spends much of its time hard against a bound is a sign that the bound, rather than
    the data, is setting the credible interval. That happens when a bound is too tight, and also
    when two parameters are degenerate and the pair drifts until it is stopped by a bound.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.
    parameters_by_column : list[Parameter | None]
        The parameter for each column of ``draws``, or None where no parameter could be matched.

    Returns
    -------
    dict[str, float]
        Mapping of the parameter's ``unique_name`` -- ``name`` is not used as the key because two
        same-named parameters would collide -- to the fraction of draws sitting in the outer
        ``BOUND_EDGE_FRACTION`` of its allowed range, for those parameters where that fraction
        exceeds ``BOUND_OCCUPANCY_THRESHOLD``. The caller resolves the unique names back to
        readable labels where the result is reported.
    """
    if draws.shape[0] == 0:
        return {}
    piled_up = {}
    for column, parameter in enumerate(parameters_by_column):
        if parameter is None:
            continue
        low = float(parameter.min)
        high = float(parameter.max)
        if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
            continue
        edge = BOUND_EDGE_FRACTION * (high - low)
        values = draws[:, column]
        at_edge = (values <= low + edge) | (values >= high - edge)
        fraction = float(np.count_nonzero(at_edge)) / len(values)
        if fraction > BOUND_OCCUPANCY_THRESHOLD:
            piled_up[parameter.unique_name] = fraction
    return piled_up


@dataclass(frozen=True)
class ParameterPosterior:
    """
    The marginal posterior of a single parameter.

    Attributes
    ----------
    name : str
        The parameter's name.
    unit : str
        The parameter's unit, as a string.
    median : float
        The 50th percentile of the marginal posterior.
    lower : float
        The 16th percentile.
    upper : float
        The 84th percentile.
    value : float
        The parameter's current value, for comparison with the median.
    """

    name: str
    unit: str
    median: float
    lower: float
    upper: float
    value: float

    @property
    def minus(self) -> float:
        """
        Distance from the median down to the 16th percentile.

        Returns
        -------
        float
            The lower half of the 68% credible interval.
        """
        return self.median - self.lower

    @property
    def plus(self) -> float:
        """
        Distance from the median up to the 84th percentile.

        Returns
        -------
        float
            The upper half of the 68% credible interval.
        """
        return self.upper - self.median


class PosteriorSummary:
    """
    Marginal posterior summaries for every sampled parameter, rendered as a table.
    """

    def __init__(self, entries: list[ParameterPosterior]) -> None:
        """
        Initialize the summary.

        Parameters
        ----------
        entries : list[ParameterPosterior]
            One entry per sampled parameter.
        """
        self._entries = list(entries)

    @property
    def entries(self) -> list[ParameterPosterior]:
        """
        The per-parameter summaries.

        Returns
        -------
        list[ParameterPosterior]
            One entry per sampled parameter.
        """
        return list(self._entries)

    def __len__(self) -> int:
        """
        Return the number of summarized parameters.

        Returns
        -------
        int
            The number of entries.
        """
        return len(self._entries)

    def __iter__(self) -> iter:
        """
        Iterate over the entries.

        Returns
        -------
        iter
            An iterator over the entries.
        """
        return iter(self._entries)

    def __getitem__(self, name: str) -> ParameterPosterior:
        """
        Look up a parameter's summary by name.

        Parameters
        ----------
        name : str
            The parameter name.

        Returns
        -------
        ParameterPosterior
            The summary for that parameter.

        Raises
        ------
        KeyError
            If no sampled parameter has that name.
        """
        for entry in self._entries:
            if entry.name == name:
                return entry
        raise KeyError(f'No sampled parameter named {name!r}.')

    def __repr__(self) -> str:
        """
        Render the summary as a table.

        Returns
        -------
        str
            A table with the median and 68% credible interval of each parameter.
        """
        if not self._entries:
            return 'PosteriorSummary(no parameters)'

        width = max(len('parameter'), *(len(e.name) for e in self._entries))
        header = (
            f'{"parameter":<{width}s} {"unit":>10s} {"median":>14s} '
            f'{"-":>12s} {"+":>12s} {"current":>14s}'
        )
        lines = ['PosteriorSummary', header, '-' * len(header)]
        lines.extend(
            f'{e.name:<{width}s} {e.unit:>10s} {e.median:>14.5g} '
            f'{e.minus:>12.4g} {e.plus:>12.4g} {e.value:>14.5g}'
            for e in self._entries
        )
        return '\n'.join(lines)


def summarize_draws(
    draws: np.ndarray,
    labels: list[str],
    parameters_by_column: list[Parameter | None],
) -> PosteriorSummary:
    """
    Summarize posterior draws under caller-supplied labels.

    The sampler labels its columns with each parameter's ``unique_name`` (``Parameter_4`` and the
    like), which is not what a user recognises, so the caller supplies readable labels instead. A
    plain parameter name is enough for a single dataset, but a multi-Q analysis holds one copy of
    each parameter per Q, all sharing a name, so those labels have to be qualified by Q.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.
    labels : list[str]
        The label to report each column under, one per column.
    parameters_by_column : list[Parameter | None]
        The parameter for each column of ``draws``, or None where none could be matched.

    Returns
    -------
    PosteriorSummary
        One entry per column of ``draws``, in column order.
    """
    entries = []
    for column, parameter in enumerate(parameters_by_column):
        lower, median, upper = (
            float(percentile) for percentile in np.percentile(draws[:, column], [16, 50, 84])
        )
        entries.append(
            ParameterPosterior(
                name=labels[column],
                unit='' if parameter is None else str(parameter.unit),
                median=median,
                lower=lower,
                upper=upper,
                value=float('nan') if parameter is None else float(parameter.value),
            )
        )
    return PosteriorSummary(entries)


def _verify_nonneg_number(value: object, name: str) -> None:
    """
    Raise if a value is not a non-negative number.

    Parameters
    ----------
    value : object
        The object to verify.
    name : str
        The name of the object, for the error message.

    Raises
    ------
    TypeError
        If value is not an int or float.
    ValueError
        If value is negative.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f'{name} must be a number. Got {type(value)}.')
    if value < 0:
        raise ValueError(f'{name} must be non-negative. Got {value}.')
