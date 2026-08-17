# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Bayesian MCMC sampling for the Analysis classes, backed by the BUMPS DREAM sampler.

The sampler is composed into an Analysis rather than inherited by it: an Analysis exposes one
``bayesian`` property, and everything to do with sampling lives here instead of being mixed into
three classes. Labelling lives in :mod:`easydynamics.analysis.posterior_labels` and the figures in
:mod:`easydynamics.utils.posterior_plotting`; this module only runs chains.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import Sampler

from easydynamics.analysis.posterior import degenerate_parameters
from easydynamics.analysis.posterior import parameters_at_bounds
from easydynamics.analysis.posterior import suggest_bounds_for_parameters
from easydynamics.analysis.posterior import summarize_draws
from easydynamics.analysis.posterior import unbounded_parameters

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

    from easyscience.fitting.sampler import SamplingResults
    from easyscience.variable import Parameter
    from matplotlib.figure import Figure

    from easydynamics.analysis.posterior import BoundsSuggestions
    from easydynamics.analysis.posterior import PosteriorSummary
    from easydynamics.analysis.posterior_labels import ParameterLabels

# Suffix of the sidecar mapping chain columns to stable labels, written next to the BUMPS chain
# files by save().
_LABEL_MAP_SUFFIX = '.parameter-names.json'


class PosteriorSampler:
    """
    Draws samples from the posterior distribution of an Analysis' free parameters.

    Reached as ``analysis.bayesian``. Sampling explores the whole posterior rather than reporting a
    single best-fit point, which is worth doing when parameters are correlated or their
    uncertainties are strongly non-Gaussian, both common in QENS.

    Running a fit first is not required, but it helps: DREAM seeds its population in a small ball
    around the parameters' current values, so starting from fitted values shortens the burn-in.

    The Analysis passes in everything that differs between the Analysis classes, so this class
    needs no knowledge of how any of them is built.

    Parameters
    ----------
    analysis : object
        The Analysis being sampled, used for its ``display_name`` and its ``fitter``.
    sampling_data : Callable[[], tuple]
        Returns the ``(x, y, weights)`` to bind to the sampler. Each is an array, or a list of
        arrays for a multi-dataset fit.
    chain_parameters : Callable[[], list[Parameter]]
        Returns the free parameters that will form the chain's columns.
    parameter_labels : Callable[[], ParameterLabels]
        Returns labels for those parameters.
    prepare : Callable[[], None] | None, default=None
        Brings any cached computation on the Analysis up to date before a run.

    Notes
    -----
    Every free parameter must have finite bounds before sampling, because in DREAM the bounds are
    the prior. :meth:`suggest_bounds` proposes bounds for any parameter still missing one.

    Examples
    --------
    ```python
    analysis.fit()
    analysis.bayesian.suggest_bounds().apply()
    analysis.bayesian.sample(samples=10000, burn=2000, thin=10)
    analysis.bayesian.summary()
    ```
    """

    def __init__(
        self,
        analysis: object,
        sampling_data: Callable[[], tuple],
        chain_parameters: Callable[[], list[Parameter]],
        parameter_labels: Callable[[], ParameterLabels],
        prepare: Callable[[], None] | None = None,
    ) -> None:
        self._analysis = analysis
        self._sampling_data = sampling_data
        self._chain_parameters = chain_parameters
        self._parameter_labels = parameter_labels
        self._prepare_hook = prepare
        self._sampler: Sampler | None = None
        self._sampler_is_dirty = True
        self._results: SamplingResults | None = None
        # Maps a chain column's unique_name to the label it had when saved. Only populated by
        # load(), because unique names are per-session and do not survive a round trip.
        self._saved_labels: dict[str, str] = {}

    #############
    # State
    #############

    def invalidate(self) -> None:
        """
        Mark the underlying Sampler as needing a rebuild.

        Called by the Analysis when its data changes, since the Sampler binds its data at
        construction.
        """
        self._sampler_is_dirty = True

    @property
    def sampler(self) -> Sampler | None:
        """
        The EasyScience Sampler holding the chain, or None before the first run.

        Returns
        -------
        Sampler | None
            The cached Sampler.
        """
        return self._sampler

    @property
    def results(self) -> SamplingResults | None:
        """
        The results of the most recent run, or None if there has not been one.

        Returns
        -------
        SamplingResults | None
            The most recent sampling results.
        """
        return self._results

    #############
    # Bounds
    #############

    def suggest_bounds(
        self,
        n_sigma: float = 10.0,
        relative_pad: float = 0.2,
        absolute_floor: float | None = None,
    ) -> BoundsSuggestions:
        """
        Propose finite bounds for free parameters that still have an infinite one.

        Nothing changes until :meth:`BoundsSuggestions.apply` is called, so the proposal can be
        reviewed first. Bounds that are already finite are never widened or narrowed, so physical
        limits such as a non-negative area are left alone.

        Because the bounds act as a uniform prior in DREAM, a generous width is the safe choice:
        too tight a bound truncates the posterior and understates the uncertainty.

        Parameters
        ----------
        n_sigma : float, default=10.0
            How many standard deviations of the fitted uncertainty to allow on each side.
        relative_pad : float, default=0.2
            Extra half-width as a fraction of the absolute parameter value, guarding against
            minimizers that report a zero or absurdly small uncertainty.
        absolute_floor : float | None, default=None
            A minimum half-width in the parameter's own units, for when neither the uncertainty nor
            the value carries the natural scale.

        Returns
        -------
        BoundsSuggestions
            The proposed bounds, which must be applied explicitly.
        """
        labels = self._labels()
        return suggest_bounds_for_parameters(
            labels.parameters,
            labels=[labels.label(parameter) for parameter in labels.parameters],
            n_sigma=n_sigma,
            relative_pad=relative_pad,
            absolute_floor=absolute_floor,
        )

    def check_bounds(self) -> None:
        """
        Verify that every free parameter has finite bounds.

        Raises
        ------
        ValueError
            If any free parameter has an infinite lower or upper bound, or finite bounds that
            enclose no range (``min >= max``).
        """
        labels = self._labels()
        unbounded = unbounded_parameters(labels.parameters)
        if unbounded:
            names = ', '.join(labels.label(parameter) for parameter in unbounded)
            raise ValueError(
                f'Bayesian sampling requires finite bounds on every free parameter, because the '
                f'bounds act as the prior. These parameters are unbounded: {names}. '
                f'Set their min and max, or call suggest_bounds() to propose values.'
            )
        degenerate = degenerate_parameters(labels.parameters)
        if degenerate:
            names = ', '.join(labels.label(parameter) for parameter in degenerate)
            raise ValueError(
                f'Bayesian sampling requires min < max on every free parameter, because the '
                f'bounds act as the prior and a zero-width range leaves the sampler nothing to '
                f'explore. These parameters have degenerate bounds: {names}. '
                f'Widen their min and max, or fix them instead of sampling them.'
            )

    #############
    # Sampling
    #############

    def sample(
        self,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        population: int | None = None,
        parameters: list[Parameter] | list[str] | None = None,
        progress: bool = False,
        **sampler_options: dict[str, Any],
    ) -> SamplingResults:
        """
        Draw samples from the posterior distribution of the free parameters.

        Starts a fresh chain, replacing any existing one; use :meth:`extend` to continue one.
        Parameter values are restored afterwards, so sampling never silently moves the model off
        its fitted values; use :meth:`set_parameters_to_median` to adopt the posterior.

        Parameters
        ----------
        samples : int, default=10000
            Number of raw samples to draw across all chains, before thinning. A guaranteed minimum
            rather than an exact count.
        burn : int, default=2000
            Burn-in generations to discard before collecting samples.
        thin : int, default=10
            Thinning interval, which reduces autocorrelation between retained draws.
        population : int | None, default=None
            DREAM population scale factor: BUMPS runs ``ceil(population * n_parameters)`` chains.
        parameters : list[Parameter] | list[str] | None, default=None
            Restrict the chain to these parameters, given as Parameter objects or labels. All other
            free parameters are held fixed for the run. Holding a parameter fixed is not the same
            as marginalizing over it: the resulting intervals are conditional on those values and
            will be too narrow if the parameters are correlated. The default samples everything.
        progress : bool, default=False
            Print a progress line, redrawn in place as the sampler advances and closed with a done
            marker when the run finishes. Off by default so scripted runs stay quiet; a
            ``progress_callback`` given in ``sampler_options`` takes precedence over it.
        **sampler_options : dict[str, Any]
            Forwarded to the EasyScience Sampler, e.g. ``sampler_kwargs`` or ``progress_callback``.

        Returns
        -------
        SamplingResults
            The sampling results, also stored on :attr:`results`.

        Notes
        -----
        Runs are not reproducible. BUMPS' DREAM sampler draws from NumPy's global random state and
        the underlying EasyScience Sampler exposes no seed control, so two identical calls return
        two different chains. Their summaries should nevertheless agree to well within the reported
        credible intervals; if they do not, the chain is too short to have converged.
        """
        reporter = _install_progress_reporter(progress, sampler_options)
        completed = False
        try:
            results = self._run(
                parameters=parameters,
                run=lambda sampler: sampler.sample(
                    samples=samples, burn=burn, thin=thin, population=population, **sampler_options
                ),
            )
            completed = True
        finally:
            if reporter is not None:
                reporter.close(completed=completed)
        return results

    def extend(
        self,
        additional_samples: int = 5000,
        thin: int = 10,
        parameters: list[Parameter] | list[str] | None = None,
        progress: bool = False,
        **sampler_options: dict[str, Any],
    ) -> SamplingResults:
        """
        Continue the existing chain with additional samples.

        Parameters
        ----------
        additional_samples : int, default=5000
            Number of additional samples to draw, in the same units as ``samples``.
        thin : int, default=10
            Thinning interval for the retained draws.
        parameters : list[Parameter] | list[str] | None, default=None
            The same restriction as in :meth:`sample`. It must leave the chain the same width,
            since BUMPS resumes from a stored chain whose columns are fixed.
        progress : bool, default=False
            Print a progress line, redrawn in place as the sampler advances, as in :meth:`sample`.
        **sampler_options : dict[str, Any]
            Forwarded to the EasyScience Sampler.

        Returns
        -------
        SamplingResults
            The sampling results for the full extended chain.

        Raises
        ------
        RuntimeError
            If there is no chain to extend, or the previous run failed and left no results.

        Notes
        -----
        A ``ValueError`` propagates from the run guards if the model or data changed since the
        chain was started, or if this run's parameters differ from the ones the chain holds.

        Like :meth:`sample`, extensions are not reproducible: the sampler draws from NumPy's global
        random state and exposes no seed control.
        """
        if self._sampler is None:
            raise RuntimeError('No chain to extend. Call sample() or load() first.')
        reporter = _install_progress_reporter(progress, sampler_options)
        completed = False
        try:
            results = self._run(
                parameters=parameters,
                run=lambda sampler: sampler.extend(
                    additional_samples=additional_samples, thin=thin, **sampler_options
                ),
                reuse_sampler=True,
            )
            completed = True
        finally:
            if reporter is not None:
                reporter.close(completed=completed)
        return results

    def _run(
        self,
        parameters: list[Parameter] | list[str] | None,
        run: Callable[[Sampler], SamplingResults],
        reuse_sampler: bool = False,
    ) -> SamplingResults:
        """
        Run a sampling operation with the surrounding guards in place.

        Checks the bounds, switches the minimizer to BUMPS, optionally holds parameters fixed, and
        restores the parameter values, fixed flags and minimizer afterwards.

        Parameters
        ----------
        parameters : list[Parameter] | list[str] | None
            Parameters to restrict the chain to, or None for all free parameters.
        run : Callable[[Sampler], SamplingResults]
            The operation to perform on the prepared Sampler.
        reuse_sampler : bool, default=False
            Whether to reuse the cached Sampler, as an extension must.

        Returns
        -------
        SamplingResults
            The results of the run.

        Raises
        ------
        IndexError
            Re-raised untouched when it did not come from BUMPS, since that is a bug here rather
            than a modelling problem.
        RuntimeError
            If the BUMPS sampler fails while removing outlier chains.
        ValueError
            If there are no free parameters to sample.
        """
        held_fixed = self._resolve_parameters_to_hold_fixed(parameters)
        _warn_about_held_parameters(self._labels(), held_fixed)

        with _FixedParameters(held_fixed):
            self.check_bounds()
            self._prepare()

            chain_parameters = self._chain_parameters()
            if not chain_parameters:
                raise ValueError(
                    'There are no free parameters to sample: every parameter is fixed. '
                    'Free at least one parameter before sampling.'
                )
            saved_values = [(p, p.value) for p in chain_parameters]

            if reuse_sampler:
                self._verify_chain_shape_unchanged(chain_parameters)

            fitter = self._analysis.fitter
            original_minimizer = fitter.minimizer.enum
            fitter.switch_minimizer(AvailableMinimizers.Bumps)
            try:
                results = run(self._get_or_build_sampler(reuse_sampler=reuse_sampler))
            except IndexError as error:
                if not _raised_inside_bumps(error):
                    raise
                raise RuntimeError(
                    'The BUMPS sampler failed while removing outlier chains. This happens when '
                    'the chains scatter because two or more free parameters are degenerate, and '
                    'also on short chains, where BUMPS has too few generations to work with. '
                    'Check for degenerate parameters, raise samples, or switch the outlier '
                    "removal off with sampler_kwargs={'outliers': 'none'}."
                ) from error
            finally:
                fitter.switch_minimizer(original_minimizer)
                for parameter, value in saved_values:
                    parameter.value = value

        # Labelled outside the block above, so a subset run records the labels a full run would.
        # Inside it the other parameters are fixed, nothing looks ambiguous, and the sidecar would
        # be written with unqualified names that no longer match on reload.
        self._saved_labels = self._labels().name_map()
        self._results = results
        self._warn_about_bounds_occupancy(results)
        return results

    def _get_or_build_sampler(self, reuse_sampler: bool) -> Sampler:
        """
        Get the cached Sampler, rebuilding it if the data changed.

        Parameters
        ----------
        reuse_sampler : bool
            Whether to reuse the cached Sampler, as an extension must.

        Returns
        -------
        Sampler
            The Sampler to run.

        Raises
        ------
        ValueError
            If the cached Sampler must be reused but the model or data has changed since it was
            built, so continuing its chain would silently mix draws against different data.
        """
        if reuse_sampler and self._sampler is not None and self._sampler_is_dirty:
            raise ValueError(
                'Cannot extend the chain: the model or data has changed since the chain was '
                'started, and an extension would mix draws taken against different data. '
                'Start a fresh chain with sample() instead.'
            )
        if self._sampler is None or (self._sampler_is_dirty and not reuse_sampler):
            x, y, weights = self._sampling_data()
            self._sampler = Sampler(self._analysis.fitter, x, y, weights=weights)
            self._sampler_is_dirty = False
        return self._sampler

    def _verify_chain_shape_unchanged(self, chain_parameters: list[Parameter]) -> None:
        """
        Check that an extension keeps the chain's columns, both in count and in identity.

        Parameters
        ----------
        chain_parameters : list[Parameter]
            The parameters that would form the chain for this run.

        Raises
        ------
        RuntimeError
            If there are no stored results to continue from, as after a failed run.
        ValueError
            If the number or the identity of the parameters differs from the existing chain's.
        """
        if self._results is None:
            raise RuntimeError(
                'Cannot extend: the previous run failed and left no results to continue from. '
                'Start a fresh chain with sample() instead.'
            )
        existing = self._results.draws.shape[1]
        if len(chain_parameters) != existing:
            raise ValueError(
                f'Cannot extend a chain of {existing} parameters with a run of '
                f'{len(chain_parameters)}. An extension continues the stored chain, whose columns '
                f'are fixed, so it needs the same parameters the chain was started with. Start a '
                f'fresh chain with sample() instead.'
            )

        # An equal count is not enough: the columns must be draws of the same parameters. For a
        # chain from this session the stored column names are current unique names; for a loaded
        # chain they are foreign, so they are resolved through the saved labels instead.
        requested = {parameter.unique_name for parameter in chain_parameters}
        if set(self._results.param_names) == requested:
            return
        resolved = self._resolve(self._results)
        if (
            all(parameter is not None for parameter in resolved)
            and {parameter.unique_name for parameter in resolved} == requested
        ):
            return
        labels = self._labels()
        chain_names = ', '.join(self._display_names(self._results))
        run_names = ', '.join(labels.label(parameter) for parameter in chain_parameters)
        raise ValueError(
            f'Cannot extend the chain: it holds draws of [{chain_names}], but this run would '
            f'sample [{run_names}]. An extension continues the stored chain, whose columns are '
            f'fixed, so it needs the same parameters the chain was started with. Start a fresh '
            f'chain with sample() instead.'
        )

    def _resolve_parameters_to_hold_fixed(
        self,
        parameters: list[Parameter] | list[str] | None,
    ) -> list[Parameter]:
        """
        Work out which free parameters must be held fixed to honour a subset request.

        Parameters
        ----------
        parameters : list[Parameter] | list[str] | None
            The requested subset, as Parameter objects or labels, or None for everything.

        Returns
        -------
        list[Parameter]
            The free parameters that are not in the requested subset.

        Raises
        ------
        TypeError
            If parameters is not a list of Parameters or strings, or None.
        ValueError
            If a requested label or Parameter matches no free parameter of this analysis, or the
            subset is empty.
        """
        if parameters is None:
            return []
        if not isinstance(parameters, (list, tuple)):
            raise TypeError('parameters must be a list of Parameters, a list of labels, or None.')

        labels = self._labels()
        by_label = {labels.label(parameter): parameter for parameter in labels.parameters}
        by_unique_name = {parameter.unique_name: parameter for parameter in labels.parameters}
        requested = []
        for entry in parameters:
            if isinstance(entry, str):
                if entry not in by_label:
                    raise ValueError(
                        f'No free parameter named {entry!r}. '
                        f'Available: {", ".join(sorted(by_label))}.'
                    )
                requested.append(by_label[entry])
            elif hasattr(entry, 'unique_name'):
                # A Parameter object gets the same membership check a label does. Without it a
                # fixed or foreign parameter slips through, every free parameter ends up held
                # fixed, and the run dies with a cryptic zero-parameter failure deep in BUMPS.
                if entry.unique_name not in by_unique_name:
                    name = getattr(entry, 'name', entry.unique_name)
                    raise ValueError(
                        f'Parameter {name!r} is not a free parameter of this analysis, so it '
                        f'cannot be sampled. It is either fixed or not part of this analysis. '
                        f'Available: {", ".join(sorted(by_label))}.'
                    )
                requested.append(by_unique_name[entry.unique_name])
            else:
                raise TypeError('parameters must contain Parameter objects or labels (strings).')

        wanted = {parameter.unique_name for parameter in requested}
        if not wanted:
            raise ValueError('parameters must name at least one parameter to sample.')
        return [p for p in labels.parameters if p.unique_name not in wanted]

    def _warn_about_bounds_occupancy(self, results: SamplingResults) -> None:
        """
        Warn when the posterior has piled up against a bound.

        Parameters
        ----------
        results : SamplingResults
            The sampling results to inspect.
        """
        piled_up = parameters_at_bounds(results.draws, self._resolve(results))
        if not piled_up:
            return
        labels = self._labels()
        by_unique_name = {parameter.unique_name: parameter for parameter in labels.parameters}
        details = ', '.join(
            f'{labels.label(by_unique_name[unique_name])} ({fraction:.0%} of draws)'
            if unique_name in by_unique_name
            else f'{unique_name} ({fraction:.0%} of draws)'
            for unique_name, fraction in piled_up.items()
        )
        warnings.warn(
            (
                f'The posterior is piled up against the bounds for: {details}. '
                f'The bounds, rather than the data, are setting these credible intervals. '
                f'Widen the bounds, or check whether these parameters are degenerate with others.'
            ),
            UserWarning,
            stacklevel=4,
        )

    #############
    # Results
    #############

    def summary(self) -> PosteriorSummary:
        """
        Summarize the marginal posterior of each sampled parameter.

        Reports the median and the 68% credible interval under the parameter's own label and unit.

        Returns
        -------
        PosteriorSummary
            One entry per sampled parameter.
        """
        results = self._require_results()
        labels = self._labels()
        return summarize_draws(
            draws=results.draws,
            labels=labels.display_names(results.param_names, self._saved_labels),
            parameters_by_column=self._resolve(results),
        )

    def set_parameters_to_median(self) -> list[Parameter]:
        """
        Set every sampled parameter to the median of its marginal posterior.

        The vector of marginal medians is not in general the highest-posterior point, and for
        strongly correlated parameters need not even be a good fit.

        Returns
        -------
        list[Parameter]
            The parameters that were changed.
        """
        results = self._require_results()
        changed = []
        for column, parameter in enumerate(self._resolve(results)):
            if parameter is None:
                continue
            parameter.value = float(np.median(results.draws[:, column]))
            changed.append(parameter)
        return changed

    #############
    # Persistence
    #############

    def save(self, path: str | os.PathLike) -> None:
        """
        Save the MCMC chain to disk.

        Writes the BUMPS chain files plus a sidecar recording the column labels, because the unique
        names BUMPS stores are per-session and cannot be matched up again on their own.

        Parameters
        ----------
        path : str | os.PathLike
            Path prefix for the chain files.

        Raises
        ------
        RuntimeError
            If there is no chain to save.
        """
        if self._sampler is None:
            raise RuntimeError('No chain to save. Call sample() first.')
        self._sampler.save(path)
        if not self._saved_labels:
            # A chain loaded without a sidecar has no labels to record. Writing an empty sidecar
            # would be worse than none: the next load() would find a "valid" file, warn about
            # nothing, and report every column under its raw internal name.
            warnings.warn(
                (
                    f'No parameter labels are recorded for this chain, so no parameter-name '
                    f'sidecar was written next to {path}; the chain was probably loaded without '
                    f'one. A future load() will report the columns under their internal names.'
                ),
                UserWarning,
                stacklevel=2,
            )
            return
        Path(f'{path}{_LABEL_MAP_SUFFIX}').write_text(
            json.dumps(self._saved_labels, indent=2), encoding='utf-8'
        )

    def load(self, path: str | os.PathLike, skip: int = 0) -> SamplingResults:
        """
        Load a previously saved MCMC chain.

        The loaded chain can be summarized, plotted, or continued with :meth:`extend`.

        Parameters
        ----------
        path : str | os.PathLike
            The path prefix the chain was saved under.
        skip : int, default=0
            Number of initial samples to skip when reading the chain.

        Returns
        -------
        SamplingResults
            The loaded results, also stored on :attr:`results`.
        """
        self._prepare()
        sidecar = Path(f'{path}{_LABEL_MAP_SUFFIX}')
        self._saved_labels = (
            json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.is_file() else {}
        )
        if not self._saved_labels:
            # An empty sidecar is as unusable as a missing one, so both warn the same way.
            warnings.warn(
                (
                    f'No parameter-name sidecar with usable content found at {sidecar}. The '
                    f'chain will be reported under the internal names it was saved with, because '
                    f'those cannot be matched to this Analysis.'
                ),
                UserWarning,
                stacklevel=2,
            )

        fitter = self._analysis.fitter
        original_minimizer = fitter.minimizer.enum
        fitter.switch_minimizer(AvailableMinimizers.Bumps)
        try:
            self._results = self._get_or_build_sampler(reuse_sampler=False).load_state(
                path, skip=skip
            )
        finally:
            fitter.switch_minimizer(original_minimizer)
        return self._results

    #############
    # Figures, each one a call into posterior_plotting
    #############

    def plot_trace(self, **kwargs: dict[str, Any]) -> Figure:
        """
        Plot the chain trace of each sampled parameter.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Forwarded to :func:`easydynamics.utils.posterior_plotting.plot_trace`.

        Returns
        -------
        Figure
            The matplotlib Figure.
        """
        from easydynamics.utils.posterior_plotting import plot_trace

        results = self._require_results()
        return plot_trace(
            draws=results.draws,
            names=self._display_names(results),
            logp=results.logp,
            units=self._units(results),
            title=self._analysis.display_name,
            **kwargs,
        )

    def plot_corner(self, **kwargs: dict[str, Any]) -> Figure:
        """
        Plot the marginal and pairwise posterior distributions.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Forwarded to :func:`easydynamics.utils.posterior_plotting.plot_corner`.

        Returns
        -------
        Figure
            The matplotlib Figure.
        """
        from easydynamics.utils.posterior_plotting import plot_corner

        results = self._require_results()
        return plot_corner(
            draws=results.draws,
            names=self._display_names(results),
            units=self._units(results),
            title=self._analysis.display_name,
            **kwargs,
        )

    def plot_marginal(self, parameter: Parameter | str, **kwargs: dict[str, Any]) -> Figure:
        """
        Plot the marginal posterior distribution of a single sampled parameter.

        Shows a density-normalized histogram of the parameter's draws, with the median and the 16th
        and 84th percentiles marked -- the same 68% credible interval :meth:`summary` reports.

        Parameters
        ----------
        parameter : Parameter | str
            The parameter to plot, as a Parameter object or its label.
        **kwargs : dict[str, Any]
            Forwarded to :func:`easydynamics.utils.posterior_plotting.plot_marginal`.

        Returns
        -------
        Figure
            The matplotlib Figure.
        """
        from easydynamics.utils.posterior_plotting import plot_marginal

        results = self._require_results()
        column = self._resolve_column(results, parameter)
        return plot_marginal(
            values=results.draws[:, column],
            name=self._display_names(results)[column],
            unit=self._units(results)[column],
            title=self._analysis.display_name,
            **kwargs,
        )

    def plot_correlations(self, **kwargs: dict[str, Any]) -> Figure:
        """
        Plot the Pearson correlation matrix of the sampled parameters.

        A strongly correlated pair cannot be determined separately from this data. The matrix
        condenses what the off-diagonal panels of :meth:`plot_corner` show, one number per pair,
        which scales better to many parameters.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Forwarded to :func:`easydynamics.utils.posterior_plotting.plot_correlations`.

        Returns
        -------
        Figure
            The matplotlib Figure.
        """
        from easydynamics.utils.posterior_plotting import plot_correlations

        results = self._require_results()
        return plot_correlations(
            draws=results.draws,
            names=self._display_names(results),
            title=self._analysis.display_name,
            **kwargs,
        )

    def plot_posterior_predictive(
        self,
        n_draws: int = 200,
        credible_interval: float = 68.0,
        **kwargs: dict[str, Any],
    ) -> Figure:
        """
        Plot the data against the credible band implied by the posterior.

        Parameters
        ----------
        n_draws : int, default=200
            How many posterior draws to evaluate the model for. Each costs a full model evaluation.
        credible_interval : float, default=68.0
            Width of the credible band, as a percentage.
        **kwargs : dict[str, Any]
            Forwarded to :func:`easydynamics.utils.posterior_plotting.plot_posterior_predictive`.

        Returns
        -------
        Figure
            The matplotlib Figure.

        Raises
        ------
        NotImplementedError
            If this Analysis binds a list of datasets rather than a single one.
        ValueError
            If n_draws is not a positive integer.
        """
        from easydynamics.utils.posterior_plotting import plot_posterior_predictive

        if not isinstance(n_draws, int) or isinstance(n_draws, bool) or n_draws < 1:
            raise ValueError(f'n_draws must be a positive integer. Got {n_draws}.')

        self._require_results()
        x, y, weights = self._sampling_data()
        if isinstance(x, (list, tuple)):
            raise NotImplementedError(
                'plot_posterior_predictive supports a single dataset only. Plot each dataset '
                'from its own Analysis1d instead.'
            )

        energy = getattr(self._analysis, 'energy', None)
        sample_model = getattr(self._analysis, 'sample_model', None)
        y_unit = None if sample_model is None else getattr(sample_model, 'y_unit', None)
        kwargs.setdefault('xlabel', None if energy is None else f'Energy ({energy.unit})')
        kwargs.setdefault('ylabel', 'Intensity' if y_unit is None else f'Intensity ({y_unit})')

        # When the data carries no variances the weights are all-ones placeholders, and inverting
        # them would fabricate error bars of 1.0 that the data never had.
        experiment = getattr(self._analysis, 'experiment', None)
        has_variances = experiment is None or getattr(experiment, 'has_variances', True)

        return plot_posterior_predictive(
            x=np.asarray(x),
            y=np.asarray(y),
            predictions=self.predictions(n_draws),
            y_err=1.0 / np.asarray(weights) if weights is not None and has_variances else None,
            title=self._analysis.display_name,
            credible_interval=credible_interval,
            **kwargs,
        )

    def predictions(self, n_draws: int = 200) -> np.ndarray:
        """
        Evaluate the model once per posterior draw, restoring the parameters afterwards.

        Parameters
        ----------
        n_draws : int, default=200
            How many draws to evaluate, taken evenly across the chain.

        Returns
        -------
        np.ndarray
            Model evaluations, shape ``(n_selected, len(x))``.
        """
        results = self._require_results()
        self._prepare()

        x, _, _ = self._sampling_data()
        columns = [
            (parameter, column)
            for column, parameter in enumerate(self._resolve(results))
            if parameter is not None
        ]
        saved_values = [(parameter, parameter.value) for parameter, _ in columns]

        total = results.draws.shape[0]
        indices = np.unique(np.linspace(0, total - 1, min(n_draws, total)).astype(int))

        fit_function = self._analysis.fitter.fit_function
        predictions = []
        try:
            for index in indices:
                for parameter, column in columns:
                    parameter.value = float(results.draws[index, column])
                predictions.append(np.asarray(fit_function(x)))
        finally:
            for parameter, value in saved_values:
                parameter.value = value
        return np.vstack(predictions)

    #############
    # Talking to the Analysis
    #############

    def _labels(self) -> ParameterLabels:
        """
        Get the label helper for the current free parameters.

        Returns
        -------
        ParameterLabels
            Built fresh, because which parameters are free can change between calls.
        """
        return self._parameter_labels()

    def _prepare(self) -> None:
        """Bring any cached computation on the Analysis up to date before a run."""
        if self._prepare_hook is not None:
            self._prepare_hook()

    def _resolve(self, results: SamplingResults) -> list[Parameter | None]:
        """
        Match each column of a chain to a parameter.

        Parameters
        ----------
        results : SamplingResults
            The results whose columns should be matched.

        Returns
        -------
        list[Parameter | None]
            The parameter for each column, or None where none could be matched.
        """
        return self._labels().resolve(results.param_names, self._saved_labels)

    def _resolve_column(self, results: SamplingResults, parameter: Parameter | str) -> int:
        """
        Find the chain column holding a parameter's draws.

        Labels are matched against the columns' display names, so the same names the summary and
        the plots report under are the ones accepted here. Parameter objects are matched through
        the resolved columns, so a parameter reloaded from a saved chain is found too.

        Parameters
        ----------
        results : SamplingResults
            The results whose columns should be searched.
        parameter : Parameter | str
            The parameter to look for, as a Parameter object or its label.

        Returns
        -------
        int
            The index of the column holding the parameter's draws.

        Raises
        ------
        TypeError
            If parameter is neither a Parameter object nor a string.
        ValueError
            If the parameter matches no column of the chain.
        """
        names = self._display_names(results)
        if isinstance(parameter, str):
            matches = [column for column, name in enumerate(names) if name == parameter]
        elif hasattr(parameter, 'unique_name'):
            matches = [
                column
                for column, candidate in enumerate(self._resolve(results))
                if candidate is not None and candidate.unique_name == parameter.unique_name
            ]
        else:
            raise TypeError('parameter must be a Parameter object or a label (string).')
        if not matches:
            requested = (
                parameter if isinstance(parameter, str) else getattr(parameter, 'name', '?')
            )
            raise ValueError(
                f'No sampled parameter named {requested!r}. Available: {", ".join(sorted(names))}.'
            )
        return matches[0]

    def _display_names(self, results: SamplingResults) -> list[str]:
        """
        Get a readable label for each column of a chain.

        Parameters
        ----------
        results : SamplingResults
            The results whose columns should be named.

        Returns
        -------
        list[str]
            One label per column.
        """
        return self._labels().display_names(results.param_names, self._saved_labels)

    def _units(self, results: SamplingResults) -> list[str]:
        """
        Get the unit of each column of a chain.

        Parameters
        ----------
        results : SamplingResults
            The results whose columns should be described.

        Returns
        -------
        list[str]
            One unit per column.
        """
        return self._labels().units(results.param_names, self._saved_labels)

    def _require_results(self) -> SamplingResults:
        """
        Get the stored results, raising if there are none.

        Returns
        -------
        SamplingResults
            The most recent sampling results.

        Raises
        ------
        RuntimeError
            If no sampling has been run yet.
        """
        if self._results is None:
            raise RuntimeError('No posterior samples yet. Call sample() or load() first.')
        return self._results


def _warn_about_held_parameters(labels: object, held_fixed: list[Parameter]) -> None:
    """
    Warn that holding parameters fixed makes the credible intervals conditional.

    Parameters
    ----------
    labels : object
        The ParameterLabels used to name them.
    held_fixed : list[Parameter]
        The parameters being held fixed for the run.
    """
    if not held_fixed:
        return
    names = ', '.join(labels.label(parameter) for parameter in held_fixed)
    warnings.warn(
        (
            f'Holding these parameters fixed while sampling: {names}. '
            f'Fixing a parameter is not the same as marginalizing over it, so the resulting '
            f'credible intervals are conditional on these values and will be too narrow if the '
            f'parameters are correlated.'
        ),
        UserWarning,
        stacklevel=4,
    )


def _install_progress_reporter(
    progress: bool,
    sampler_options: dict[str, Any],
) -> _SamplingProgress | None:
    """
    Put a progress reporter into the sampler options when one is asked for.

    A ``progress_callback`` the caller supplied themselves is left untouched, since an explicit
    callback is more specific than the boolean convenience flag.

    Parameters
    ----------
    progress : bool
        Whether a progress line was requested.
    sampler_options : dict[str, Any]
        The options about to be forwarded to the EasyScience Sampler, modified in place.

    Returns
    -------
    _SamplingProgress | None
        The installed reporter, which the caller must close after the run, or None when nothing was
        installed.
    """
    if not progress or 'progress_callback' in sampler_options:
        return None
    reporter = _SamplingProgress()
    sampler_options['progress_callback'] = reporter
    return reporter


class _SamplingProgress:
    """
    Renders the sampler's per-generation callbacks as a single self-overwriting progress line.

    BUMPS invokes the callback once per DREAM generation, which for a long run is far too often to
    print, so the line is only redrawn when the percentage changes. Carriage-return output works in
    terminals and notebooks alike, and needs no extra dependency.

    The generation total in the payload is the backend's own estimate, and it overestimates when
    DREAM runs more chains than the estimate assumes, so a finished run can stop short of 100%. The
    line is therefore closed with an explicit done marker rather than trusting the estimate.
    """

    def __init__(self) -> None:
        self._last_percent = -1
        self._line_length = 0
        self._printed = False

    def __call__(self, payload: dict[str, Any]) -> None:
        """
        Handle one progress callback from the sampler.

        Parameters
        ----------
        payload : dict[str, Any]
            The sampler's progress payload. ``iteration`` carries the DREAM generation and
            ``total_steps``, when present, the estimated total number of generations.
        """
        iteration = payload.get('iteration')
        if iteration is None:
            return
        total = payload.get('total_steps')
        if total:
            # Clamped, so the line never reports more than 100% when the run outlives the
            # backend's estimate of its own length.
            percent = min(100, int(100 * iteration / total))
            if percent == self._last_percent:
                return
            self._last_percent = percent
            line = f'Sampling: {percent:3d}% ({iteration}/{total} generations)'
        else:
            line = f'Sampling: generation {iteration}'
        self._write(line)

    def close(self, completed: bool) -> None:
        """
        End the progress line, so any later output starts on a line of its own.

        Parameters
        ----------
        completed : bool
            Whether the run finished. A finished run gets a done marker; a failed one only has its
            line terminated, so the exception is not decorated with a claim of success.
        """
        if not self._printed:
            return
        if completed:
            self._write('Sampling: done')
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _write(self, line: str) -> None:
        """
        Redraw the progress line in place.

        Parameters
        ----------
        line : str
            The text to show, padded so it fully overwrites a longer previous line.
        """
        sys.stdout.write(f'\r{line.ljust(self._line_length)}')
        sys.stdout.flush()
        self._line_length = max(self._line_length, len(line))
        self._printed = True


def _raised_inside_bumps(error: BaseException) -> bool:
    """
    Check whether an exception came from inside BUMPS.

    Used so only BUMPS' own failures are relabelled, and a bug in this package is not reported as a
    modelling problem.

    Parameters
    ----------
    error : BaseException
        The exception to inspect.

    Returns
    -------
    bool
        True when any frame of the traceback lies in the bumps package.
    """
    traceback = error.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get('__name__', '')
        if module == 'bumps' or module.startswith('bumps.'):
            return True
        traceback = traceback.tb_next
    return False


class _FixedParameters:
    """Context manager that temporarily fixes parameters and restores their flags on exit."""

    def __init__(self, parameters: list[Parameter]) -> None:
        self._parameters = list(parameters)
        self._saved: list[tuple[Parameter, bool]] = []

    def __enter__(self) -> None:
        """Fix the parameters, remembering their previous state."""
        self._saved = [(parameter, parameter.fixed) for parameter in self._parameters]
        for parameter in self._parameters:
            parameter.fixed = True

    def __exit__(self, *_exc_info: object) -> None:
        """
        Restore the previous fixed state of every parameter.

        Parameters
        ----------
        *_exc_info : object
            Exception information, ignored.
        """
        for parameter, was_fixed in self._saved:
            parameter.fixed = was_fixed
