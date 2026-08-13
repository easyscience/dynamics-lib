# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Diagnostic plots for Bayesian posterior samples.

These take plain arrays rather than an Analysis, so they can be used on any chain, including one
loaded from disk. The Analysis classes wrap them in convenience methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_trace(
    draws: np.ndarray,
    names: list[str],
    logp: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot the chain trace of every sampled parameter.

    A converged chain looks like a "hairy caterpillar": noisy but stationary, with no drift or long
    excursions. A visible trend means the chain has not reached the typical set and needs a longer
    burn-in.

    A ``ValueError`` is raised if ``draws`` is not two-dimensional, or if ``names`` does not have
    one entry per column.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.
    names : list[str]
        One label per column of ``draws``.
    logp : np.ndarray | None, default=None
        Log-posterior values, plotted in an extra panel when given.
    title : str | None, default=None
        Figure title.
    figsize : tuple[float, float] | None, default=None
        Figure size in inches. Defaults to a height that scales with the number of panels.

    Returns
    -------
    Figure
        The matplotlib Figure.
    """
    draws = np.asarray(draws)
    _verify_draws(draws, names)

    n_panels = draws.shape[1] + (1 if logp is not None else 0)
    if figsize is None:
        figsize = (10.0, max(2.0, 1.6 * n_panels))

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes[:, 0]

    for axis, column, name in zip(axes, range(draws.shape[1]), names, strict=False):
        axis.plot(draws[:, column], lw=0.5)
        axis.set_ylabel(name, fontsize=8)
        axis.set_xlim(0, len(draws) - 1)

    if logp is not None:
        axes[-1].plot(np.asarray(logp), lw=0.5, color='C4')
        axes[-1].set_ylabel('log-posterior', fontsize=8)

    axes[-1].set_xlabel('sample index')
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_corner(
    draws: np.ndarray,
    names: list[str],
    title: str | None = None,
    bins: int = 40,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot marginal and pairwise posterior distributions.

    Diagonal panels show each parameter's marginal distribution. Off-diagonal panels show the joint
    distribution of a pair: a compact blob means the two are independent, while a narrow diagonal
    ridge means they are correlated and cannot be determined separately from this data.

    A ``ValueError`` is raised if ``draws`` is not two-dimensional, or if ``names`` does not have
    one entry per column.

    Parameters
    ----------
    draws : np.ndarray
        Posterior draws, shape ``(n_draws, n_parameters)``.
    names : list[str]
        One label per column of ``draws``.
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
    """
    draws = np.asarray(draws)
    _verify_draws(draws, names)

    n = draws.shape[1]
    if figsize is None:
        side = max(4.0, 2.0 * n)
        figsize = (side, side)

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
            if row == n - 1:
                axis.set_xlabel(names[col], fontsize=8)
            else:
                axis.set_xticklabels([])
            if col == 0 and row != 0:
                axis.set_ylabel(names[row], fontsize=8)
            else:
                axis.set_yticklabels([])
            axis.tick_params(labelsize=7)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_posterior_predictive(
    x: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray,
    y_err: np.ndarray | None = None,
    title: str | None = None,
    credible_interval: float = 68.0,
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
    axis.legend()
    if title is not None:
        axis.set_title(title)
    fig.tight_layout()
    return fig


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
    if draws.shape[1] != len(names):
        raise ValueError(
            f'names must have one entry per column of draws. '
            f'Got {len(names)} names for {draws.shape[1]} columns.'
        )
