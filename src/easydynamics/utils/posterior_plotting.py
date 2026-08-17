# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Diagnostic plots for Bayesian posterior samples.

These take plain arrays rather than an Analysis, so they can be used on any chain, including one
loaded from disk. The Analysis classes wrap them in convenience methods.
"""

from __future__ import annotations

import io
import warnings
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from ipywidgets import VBox
    from matplotlib.figure import Figure
    from plopp.backends.matplotlib.figure import InteractiveFigure


def plot_trace(
    draws: np.ndarray,
    names: list[str],
    logp: np.ndarray | None = None,
    units: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot the chain trace of every sampled parameter.

    A converged chain looks like a "hairy caterpillar": noisy but stationary, with no drift or long
    excursions. A visible trend means the chain has not reached the typical set and needs a longer
    burn-in.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.
    names : list[str]
        One label per column of ``draws``.
    logp : np.ndarray | None, default=None
        Log-posterior values, one per draw, plotted in an extra panel when given.
    units : list[str] | None, default=None
        Unit of each column, appended to its label. Entries that are empty or dimensionless are
        skipped, since a bare "dimensionless" only adds clutter.
    title : str | None, default=None
        Figure title.
    figsize : tuple[float, float] | None, default=None
        Figure size in inches. Defaults to a height that scales with the number of panels.

    Returns
    -------
    Figure
        The matplotlib Figure.

    Raises
    ------
    ValueError
        If ``draws`` is not two-dimensional or is empty, if ``names`` does not have one entry per
        column, or if ``logp`` does not have one entry per draw.
    """
    draws = np.asarray(draws)
    _verify_draws(draws, names)
    if logp is not None:
        logp = np.asarray(logp)
        if logp.ndim != 1 or logp.shape[0] != draws.shape[0]:
            raise ValueError(
                f'logp must have one entry per draw. '
                f'Got shape {logp.shape} for {draws.shape[0]} draws.'
            )

    n_panels = draws.shape[1] + (1 if logp is not None else 0)
    if figsize is None:
        figsize = (10.0, max(2.0, 1.6 * n_panels))

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]

    for axis, column, name in zip(axes, range(draws.shape[1]), names, strict=False):
        axis.plot(draws[:, column], lw=0.5)
        axis.set_ylabel(_with_unit(name, units, column), fontsize=8)
        # A single draw would make (0, len - 1) a zero-width range; matplotlib's autoscaling
        # handles that case better than an explicit degenerate limit would.
        if len(draws) > 1:
            axis.set_xlim(0, len(draws) - 1)

    if logp is not None:
        axes[-1].plot(logp, lw=0.5, color='C4')
        axes[-1].set_ylabel('log-posterior', fontsize=8)

    axes[-1].set_xlabel('sample index')
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_corner(
    draws: np.ndarray,
    names: list[str],
    units: list[str] | None = None,
    title: str | None = None,
    bins: int = 40,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot marginal and pairwise posterior distributions.

    Diagonal panels show each parameter's marginal distribution. Off-diagonal panels show the joint
    distribution of a pair: a compact blob means the two are independent, while a narrow diagonal
    ridge means they are correlated and cannot be determined separately from this data.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.
    names : list[str]
        One label per column of ``draws``.
    units : list[str] | None, default=None
        Unit of each column, appended to its label. Entries that are empty or dimensionless are
        skipped, since a bare "dimensionless" only adds clutter.
    title : str | None, default=None
        Figure title.
    bins : int, default=40
        Number of bins for the marginal histograms.
    figsize : tuple[float, float] | None, default=None
        Figure size in inches. Defaults to a square that scales with the parameter count.

    Returns
    -------
    Figure
        The matplotlib Figure.

    Raises
    ------
    ValueError
        If ``draws`` is not two-dimensional or is empty, if ``names`` does not have one entry per
        column, or if any column contains non-finite values.
    """
    draws = np.asarray(draws)
    _verify_draws(draws, names)

    # Caught up front, because numpy would otherwise report it as an obscure
    # "range [nan, nan]" error from inside the histogram.
    finite_columns = np.isfinite(draws).all(axis=0)
    if not finite_columns.all():
        bad = ', '.join(name for name, ok in zip(names, finite_columns, strict=True) if not ok)
        raise ValueError(f'draws contain non-finite values (NaN or infinity) in: {bad}.')

    n = draws.shape[1]
    if figsize is None:
        side = max(4.0, 2.0 * n)
        figsize = (side, side)

    # One shared limit per column, applied to the diagonal histogram and every hexbin panel below
    # it, so the ticks of a column line up instead of each panel autoscaling on its own.
    limits = _column_limits(draws)

    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
    for row in range(n):
        for col in range(n):
            axis = axes[row, col]
            if col > row:
                axis.set_visible(False)
                continue
            if row == col:
                axis.hist(draws[:, row], bins=bins, color='C0', histtype='stepfilled', alpha=0.7)
                axis.set_yticks([])
            else:
                axis.hexbin(draws[:, col], draws[:, row], gridsize=30, cmap='Blues', mincnt=1)
                axis.set_ylim(limits[row])
            axis.set_xlim(limits[col])
            if row == n - 1:
                axis.set_xlabel(names[col], fontsize=8)
            else:
                axis.set_xticklabels([])
            if col == 0 and row != 0:
                axis.set_ylabel(names[row], fontsize=8)
            else:
                axis.set_yticklabels([])
            if row == 0 and col == 0:
                # The top-left panel is a histogram, so its vertical axis counts draws rather than
                # carrying a parameter. Say so, instead of leaving it blank as if by omission.
                axis.set_ylabel('counts', fontsize=8)
            axis.tick_params(labelsize=7)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if row != col:
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # Matplotlib parks the shared exponent ("1e-8") at the end of the axis, where it lands on top
    # of the axis label. Fold it into the label instead.
    fig.canvas.draw()
    for row in range(n):
        for col in range(row + 1):
            axis = axes[row, col]
            if row == n - 1:
                _absorb_offset(axis.xaxis, axis.set_xlabel, names[col], units, col)
            if col == 0 and row != 0:
                _absorb_offset(axis.yaxis, axis.set_ylabel, names[row], units, row)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_marginal(
    values: np.ndarray,
    name: str,
    unit: str | None = None,
    title: str | None = None,
    bins: int = 40,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Figure:
    """
    Plot the marginal posterior distribution of a single parameter.

    Shows a density-normalized histogram of the parameter's draws, with the median and the 16th and
    84th percentiles marked -- the same 68% credible interval the posterior summary reports.

    Parameters
    ----------
    values : np.ndarray
        The parameter's posterior draws, one-dimensional.
    name : str
        The label the parameter is reported under.
    unit : str | None, default=None
        The parameter's unit, appended to the axis label. Empty or dimensionless units are skipped,
        since a bare "dimensionless" only adds clutter.
    title : str | None, default=None
        Figure title.
    bins : int, default=40
        Number of histogram bins.
    figsize : tuple[float, float], default=(8.0, 5.0)
        Figure size in inches.

    Returns
    -------
    Figure
        The matplotlib Figure.

    Raises
    ------
    ValueError
        If ``values`` is not one-dimensional, is empty, or contains non-finite entries.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f'values must be one-dimensional. Got shape {values.shape}.')
    if values.size == 0:
        raise ValueError('values is empty: there are no samples to plot.')
    # Caught up front, because numpy would otherwise report it as an obscure
    # "range [nan, nan]" error from inside the histogram.
    if not np.isfinite(values).all():
        raise ValueError(f'values contain non-finite entries (NaN or infinity) for {name}.')

    lower, median, upper = np.percentile(values, [16.0, 50.0, 84.0])

    fig, axis = plt.subplots(figsize=figsize)
    axis.hist(values, bins=bins, density=True, color='C0', histtype='stepfilled', alpha=0.7)
    axis.axvline(median, color='C3', lw=1.5, label='Median')
    axis.axvline(lower, color='C3', lw=1.0, ls='--', label='68% credible interval')
    axis.axvline(upper, color='C3', lw=1.0, ls='--')
    axis.set_xlabel(_with_unit(name, [unit] if unit is not None else None, 0))
    axis.set_ylabel('Probability density')
    axis.legend()
    if title is not None:
        axis.set_title(title)
    fig.tight_layout()
    return fig


def plot_correlations(
    draws: np.ndarray,
    names: list[str],
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot the Pearson correlation matrix of the sampled parameters.

    A strongly correlated pair (an entry near +1 or -1) cannot be determined separately from this
    data: the chain trades one off against the other. The matrix condenses what the off-diagonal
    panels of the corner plot show, one number per pair, which scales better to many parameters.

    Correlations are dimensionless, so the labels carry no units. A constant column has no defined
    correlation with anything; its cells are shown greyed out and marked "n/a" rather than failing.
    A ``ValueError`` propagates from the input validation if ``draws`` is not two-dimensional or
    is empty, or if ``names`` does not have one entry per column.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.
    names : list[str]
        One label per column of ``draws``.
    title : str | None, default=None
        Figure title.
    figsize : tuple[float, float] | None, default=None
        Figure size in inches. Defaults to a square that scales with the parameter count, plus room
        for the colorbar.

    Returns
    -------
    Figure
        The matplotlib Figure.
    """
    draws = np.asarray(draws)
    _verify_draws(draws, names)

    matrix = _correlation_matrix(draws)
    n = draws.shape[1]
    if figsize is None:
        side = max(4.0, 0.9 * n + 2.0)
        figsize = (side + 1.5, side)

    # A diverging map centred on zero, so positive and negative correlations read as two hues
    # around a neutral midpoint. Cells with no defined correlation are greyed out.
    colormap = colormaps['RdBu_r'].with_extremes(bad='0.85')

    fig, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=colormap, vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(n), labels=names, rotation=45, ha='right', fontsize=8)
    axis.set_yticks(range(n), labels=names, fontsize=8)
    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            defined = bool(np.isfinite(value))
            axis.text(
                col,
                row,
                f'{value:.2f}' if defined else 'n/a',
                ha='center',
                va='center',
                fontsize=8,
                # Saturated cells at the ends of the map are too dark for black text.
                color='white' if defined and abs(value) > 0.6 else 'black',
            )
    fig.colorbar(image, ax=axis, label='Pearson correlation')
    if title is not None:
        axis.set_title(title)
    fig.tight_layout()
    return fig


def plot_posterior_predictive(
    x: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray,
    y_err: np.ndarray | None = None,
    title: str | None = None,
    credible_interval: float = 68.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Figure:
    """
    Plot the data against the credible band implied by the posterior.

    The band shows where the model says the data should lie, given the posterior. If the data
    strays outside it systematically, the model is missing something that no amount of parameter
    tuning will fix.

    Parameters
    ----------
    x : np.ndarray
        Independent variable of the data.
    y : np.ndarray
        Observed values.
    predictions : np.ndarray
        Model evaluations, shape ``(n_draws, len(x))``, one row per posterior draw.
    y_err : np.ndarray | None, default=None
        Standard deviation of the observed values, drawn as error bars when given.
    title : str | None, default=None
        Figure title.
    credible_interval : float, default=68.0
        Width of the credible band, as a percentage.
    xlabel : str | None, default=None
        Label for the independent axis.
    ylabel : str | None, default=None
        Label for the dependent axis.
    figsize : tuple[float, float], default=(8.0, 5.0)
        Figure size in inches.

    Returns
    -------
    Figure
        The matplotlib Figure.

    Raises
    ------
    ValueError
        If ``predictions`` is not two-dimensional with one column per point in ``x``, or if
        ``credible_interval`` is not between 0 and 100.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[1] != len(x):
        raise ValueError(
            f'predictions must have shape (n_draws, {len(x)}). Got {predictions.shape}.'
        )
    if not 0 < credible_interval < 100:
        raise ValueError(f'credible_interval must be between 0 and 100. Got {credible_interval}.')

    tail = (100.0 - credible_interval) / 2.0
    lower, median, upper = np.percentile(predictions, [tail, 50.0, 100.0 - tail], axis=0)

    fig, axis = plt.subplots(figsize=figsize)
    if y_err is None:
        axis.plot(x, y, 'o', mfc='none', color='black', label='Data', markersize=4)
    else:
        axis.errorbar(
            x, y, np.asarray(y_err), fmt='o', mfc='none', color='black', label='Data', markersize=4
        )
    axis.fill_between(
        x,
        lower,
        upper,
        color='C3',
        alpha=0.3,
        label=f'{credible_interval:.0f}% credible band',
    )
    axis.plot(x, median, '-', color='C3', label='Posterior median')
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.legend()
    if title is not None:
        axis.set_title(title)
    fig.tight_layout()
    return fig


def _column_limits(draws: np.ndarray) -> list[tuple[float, float]]:
    """
    Compute one shared axis range per column of a corner plot.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``, all finite.

    Returns
    -------
    list[tuple[float, float]]
        A padded ``(low, high)`` range per column, widened to a usable span when a column is
        constant.
    """
    lows = draws.min(axis=0)
    highs = draws.max(axis=0)
    spans = highs - lows
    pads = np.where(spans > 0, 0.05 * spans, 0.05 * np.maximum(np.abs(highs), 1.0))
    return [(float(low), float(high)) for low, high in zip(lows - pads, highs + pads, strict=True)]


def _correlation_matrix(draws: np.ndarray) -> np.ndarray:
    """
    Compute the Pearson correlation matrix of a chain's columns.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.

    Returns
    -------
    np.ndarray
        The ``(n_parameters, n_parameters)`` correlation matrix, two-dimensional even for a
        single-parameter chain, with NaN wherever a column has zero variance. Numpy's
        division-by-zero warnings for those columns are suppressed, since the NaNs are handled by
        the caller rather than being a numerical accident.
    """
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        matrix = np.corrcoef(draws, rowvar=False)
    # np.corrcoef collapses a single-column input to a 0-d scalar; restore the 1x1 matrix.
    return np.atleast_2d(np.asarray(matrix, dtype=float))


def _unit_for(units: list[str] | None, column: int) -> str:
    """
    Get the unit to show for a column, if it is worth showing.

    Parameters
    ----------
    units : list[str] | None
        The units, one per column, or None.
    column : int
        The column to look up.

    Returns
    -------
    str
        The unit, or an empty string when there is none worth printing.
    """
    if units is None or column >= len(units):
        return ''
    unit = (units[column] or '').strip()
    return '' if unit.lower() in ('', 'dimensionless', 'none') else unit


def _with_unit(name: str, units: list[str] | None, column: int) -> str:
    """
    Append a column's unit to its label.

    Parameters
    ----------
    name : str
        The label to extend.
    units : list[str] | None
        The units, one per column, or None.
    column : int
        The column the label belongs to.

    Returns
    -------
    str
        The label, with the unit in parentheses when there is one.
    """
    unit = _unit_for(units, column)
    return f'{name} ({unit})' if unit else name


def _absorb_offset(
    axis_object: object,
    set_label: object,
    name: str,
    units: list[str] | None = None,
    column: int = 0,
) -> None:
    """
    Move an axis' shared exponent into its label, so the two stop overlapping.

    The exponent and the unit share one set of parentheses, since two adjacent parentheticals read
    badly: ``D (1e-8 m^2/s)`` rather than ``D (1e-8) (m^2/s)``.

    Parameters
    ----------
    axis_object : object
        The matplotlib ``XAxis`` or ``YAxis`` carrying the offset text.
    set_label : object
        The corresponding ``set_xlabel`` or ``set_ylabel`` callable.
    name : str
        The label the axis should carry, before the exponent and unit are appended.
    units : list[str] | None, default=None
        The units, one per column, or None.
    column : int, default=0
        The column the axis belongs to.
    """
    offset_text = axis_object.get_offset_text()
    offset = offset_text.get_text()
    unit = _unit_for(units, column)
    suffix = ' '.join(part for part in (offset, unit) if part)
    set_label(f'{name} ({suffix})' if suffix else name, fontsize=8)
    if offset:
        offset_text.set_visible(False)


def _verify_draws(draws: np.ndarray, names: list[str]) -> None:
    """
    Verify that a draws array is two-dimensional and matches its labels.

    Parameters
    ----------
    draws : np.ndarray
        The posterior draws to check.
    names : list[str]
        The labels to check against.

    Raises
    ------
    ValueError
        If ``draws`` is not two-dimensional or its column count differs from ``len(names)``.
    """
    if draws.ndim != 2:
        raise ValueError(f'draws must be two-dimensional. Got shape {draws.shape}.')
    if draws.shape[0] == 0:
        raise ValueError('draws is empty: there are no samples to plot.')
    if draws.shape[1] == 0:
        raise ValueError('draws has no columns: there are no parameters to plot.')
    if draws.shape[1] != len(names):
        raise ValueError(
            f'names must have one entry per column of draws. '
            f'Got {len(names)} names for {draws.shape[1]} columns.'
        )


def figures_with_slider(figures: dict[int, Figure], description: str = 'Q index') -> VBox:
    """
    Show one pre-rendered figure at a time, with a slider choosing which one.

    Every figure is rendered to PNG bytes once, up front, and the slider callback only swaps the
    stored bytes into an image widget. Moving the slider therefore costs no matplotlib work at
    all, which keeps it as responsive as the plopp slider on the data plots; re-rendering a
    figure on every move is what made the previous slider feel sluggish.

    The figures are closed after rendering, so no backend draws them a second time.

    Parameters
    ----------
    figures : dict[int, Figure]
        Mapping of slider position to the matplotlib Figure shown there. Only these positions are
        offered, so the slider cannot land on an index with nothing to show.
    description : str, default='Q index'
        Label shown next to the slider.

    Returns
    -------
    VBox
        An ipywidgets box holding the image and, under it, the slider.

    Raises
    ------
    ValueError
        If no figures are given.
    """
    import ipywidgets as widgets

    if not figures:
        raise ValueError('No figures to show.')

    indices = sorted(figures)
    rendered = {}
    for index in indices:
        figure = figures[index]
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', bbox_inches='tight')
        rendered[index] = buffer.getvalue()
        # Rendered to bytes already, so the figure is closed rather than left for a backend to
        # draw a second time.
        plt.close(figure)

    image = widgets.Image(value=rendered[indices[0]], format='png')
    image.layout.max_width = '100%'
    # Swapping stored bytes is instant, so the image can follow the slider continuously; there is
    # no need for the release-to-update behaviour an expensive redraw would force.
    slider = widgets.SelectionSlider(
        options=indices,
        value=indices[0],
        description=description,
        continuous_update=True,
    )
    slider.observe(lambda change: setattr(image, 'value', rendered[change['new']]), names='value')
    # Slider under the figure, matching where plopp puts its slicer controls.
    return widgets.VBox([image, slider])


def corner_with_slider(
    chains: dict[int, dict],
    title: str | None = None,
    **kwargs: dict[str, Any],
) -> VBox:
    """
    Show one corner plot at a time, with a slider choosing which chain to look at.

    Chains sampled separately share no draws, so there is no joint distribution across them to
    plot. Stepping through them one at a time shows the correlations that were actually sampled,
    which is what a single combined figure could not do honestly. The figures are pre-rendered
    through :func:`figures_with_slider`, so the slider moves without re-drawing anything.

    Parameters
    ----------
    chains : dict[int, dict]
        Mapping of index to a ``{'draws': ..., 'names': ..., 'units': ...}`` description of one
        chain. ``units`` is optional.
    title : str | None, default=None
        Title prefix, extended with the selected index.
    **kwargs : dict[str, Any]
        Forwarded to :func:`plot_corner`.

    Returns
    -------
    VBox
        An ipywidgets box holding the figure and the slider.

    Raises
    ------
    ValueError
        If no chains are given.
    """
    if not chains:
        raise ValueError('No chains to plot.')

    figures = {
        index: plot_corner(
            draws=chain['draws'],
            names=chain['names'],
            units=chain.get('units'),
            title=title if title is None else f'{title} (Q index {index})',
            **kwargs,
        )
        for index, chain in chains.items()
    }
    return figures_with_slider(figures)


def predictive_with_slider(
    energy: np.ndarray,
    q_values: np.ndarray,
    y: np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    y_variances: np.ndarray | None = None,
    energy_unit: str | None = None,
    q_unit: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    credible_interval: float = 68.0,
    **kwargs: dict[str, Any],
) -> InteractiveFigure:
    """
    Plot per-Q posterior-predictive bands behind a plopp Q slider.

    Built on ``plopp.slicer`` over a scipp DataGroup with a Q dimension, so the figure looks and
    handles exactly like ``Analysis.plot_data_and_model``: the data with its error bars, the model
    curves on top, and a Q slider underneath. Plopp draws no filled band for sliced data -- its
    only spread representation is variance-based error bars -- so the credible band is drawn as
    the posterior median with a dashed line along each band edge, labelled with the interval.

    Rows are laid out on one common energy grid; where a Q has no point (masked or never
    measured), NaN leaves a gap in the lines rather than inventing a value.

    Parameters
    ----------
    energy : np.ndarray
        The common energy grid, one column per point.
    q_values : np.ndarray
        The Q value of each row, shown on the slider.
    y : np.ndarray
        Observed values, shape ``(len(q_values), len(energy))``, NaN where a Q has no point.
    lower : np.ndarray
        Lower band edge per Q, same shape as ``y``.
    median : np.ndarray
        Posterior median prediction per Q, same shape as ``y``.
    upper : np.ndarray
        Upper band edge per Q, same shape as ``y``.
    y_variances : np.ndarray | None, default=None
        Variances of the observed values, drawn as error bars when given.
    energy_unit : str | None, default=None
        Unit of the energy grid, shown on the horizontal axis.
    q_unit : str | None, default=None
        Unit of the Q values, shown beside the slider.
    ylabel : str | None, default=None
        Label for the dependent axis.
    title : str | None, default=None
        Figure title.
    credible_interval : float, default=68.0
        Width of the credible band the edges enclose, as a percentage, used in their labels.
    **kwargs : dict[str, Any]
        Forwarded to ``plopp.slicer``, overriding the style defaults.

    Returns
    -------
    InteractiveFigure
        The plopp figure with its Q slider.

    Raises
    ------
    ValueError
        If the arrays do not share the shape ``(len(q_values), len(energy))``, or if
        ``credible_interval`` is not between 0 and 100.
    """
    import plopp as pp
    import scipp as sc

    if not 0 < credible_interval < 100:
        raise ValueError(f'credible_interval must be between 0 and 100. Got {credible_interval}.')
    expected = (len(q_values), len(energy))
    arrays = {'y': y, 'lower': lower, 'median': median, 'upper': upper}
    if y_variances is not None:
        arrays['y_variances'] = y_variances
    for name, array in arrays.items():
        if np.asarray(array).shape != expected:
            raise ValueError(f'{name} must have shape {expected}. Got {np.asarray(array).shape}.')

    coords = {
        'Q': sc.array(dims=['Q'], values=np.asarray(q_values, dtype=float), unit=q_unit),
        'energy': sc.array(
            dims=['energy'], values=np.asarray(energy, dtype=float), unit=energy_unit
        ),
    }

    def data_array(values: np.ndarray, variances: np.ndarray | None = None) -> sc.DataArray:
        return sc.DataArray(
            data=sc.array(
                dims=['Q', 'energy'],
                values=np.asarray(values, dtype=float),
                variances=None if variances is None else np.asarray(variances, dtype=float),
            ),
            coords=coords,
        )

    lower_key = f'{credible_interval:.0f}% band (lower)'
    upper_key = f'{credible_interval:.0f}% band (upper)'
    data_group = sc.DataGroup({
        'Data': data_array(y, y_variances),
        'Posterior median': data_array(median),
        lower_key: data_array(lower),
        upper_key: data_array(upper),
    })

    # The same styling plot_data_and_model gives its DataGroup: data as open black circles, the
    # model curves as lines, with the band edges dashed to read as edges rather than curves.
    style = {
        'keep': 'energy',
        'linestyle': {'Data': 'none', 'Posterior median': '-', lower_key: '--', upper_key: '--'},
        'marker': {'Data': 'o', 'Posterior median': None, lower_key: None, upper_key: None},
        'color': {'Data': 'black', 'Posterior median': 'C3', lower_key: 'C3', upper_key: 'C3'},
        'markerfacecolor': {'Data': 'none'},
    }
    if title is not None:
        style['title'] = title
    style.update(kwargs)

    fig = pp.slicer(data_group, **style)
    for widget in fig.bottom_bar[0].controls.values():
        widget.slider_toggler.value = '-o-'
    if ylabel is not None:
        fig.ax.set_ylabel(ylabel)
    fig.autoscale()
    return fig
