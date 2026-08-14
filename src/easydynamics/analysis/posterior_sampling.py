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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import Sampler

from easydynamics.analysis.posterior import PosteriorSummary
from easydynamics.analysis.posterior import parameters_at_bounds
from easydynamics.analysis.posterior import suggest_bounds_for_parameters
from easydynamics.analysis.posterior import summarize_draws
from easydynamics.analysis.posterior import unbounded_parameters
from easydynamics.utils.utils import _in_notebook

if TYPE_CHECKING:
    import os
    from collections.abc import Callable

    from easyscience.fitting.sampler import SamplingResults
    from easyscience.variable import Parameter
    from ipywidgets import VBox
    from matplotlib.figure import Figure

    from easydynamics.analysis.posterior import BoundsSuggestions
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
            If any free parameter has an infinite lower or upper bound.
        """
        labels = self._labels()
        unbounded = unbounded_parameters(labels.parameters)
        if not unbounded:
            return
        names = ', '.join(labels.label(parameter) for parameter in unbounded)
        raise ValueError(
            f'Bayesian sampling requires finite bounds on every free parameter, because the '
            f'bounds act as the prior. These parameters are unbounded: {names}. '
            f'Set their min and max, or call suggest_bounds() to propose values.'
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
        **sampler_options : dict[str, Any]
            Forwarded to the EasyScience Sampler, e.g. ``sampler_kwargs`` or ``progress_callback``.

        Returns
        -------
        SamplingResults
            The sampling results, also stored on :attr:`results`.
        """
        return self._run(
            parameters=parameters,
            run=lambda sampler: sampler.sample(
                samples=samples, burn=burn, thin=thin, population=population, **sampler_options
            ),
        )

    def extend(
        self,
        additional_samples: int = 5000,
        thin: int = 10,
        parameters: list[Parameter] | list[str] | None = None,
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
        **sampler_options : dict[str, Any]
            Forwarded to the EasyScience Sampler.

        Returns
        -------
        SamplingResults
            The sampling results for the full extended chain.

        Raises
        ------
        RuntimeError
            If there is no chain to extend.
        """
        if self._sampler is None:
            raise RuntimeError('No chain to extend. Call sample() or load() first.')
        return self._run(
            parameters=parameters,
            run=lambda sampler: sampler.extend(
                additional_samples=additional_samples, thin=thin, **sampler_options
            ),
            reuse_sampler=True,
        )

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
        """
        held_fixed = self._resolve_parameters_to_hold_fixed(parameters)
        _warn_about_held_parameters(self._labels(), held_fixed)

        with _FixedParameters(held_fixed):
            self.check_bounds()
            self._prepare()

            chain_parameters = self._chain_parameters()
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
            Whether to reuse the cached Sampler even if it is marked dirty.

        Returns
        -------
        Sampler
            The Sampler to run.
        """
        if self._sampler is None or (self._sampler_is_dirty and not reuse_sampler):
            x, y, weights = self._sampling_data()
            self._sampler = Sampler(self._analysis.fitter, x, y, weights=weights)
            self._sampler_is_dirty = False
        return self._sampler

    def _verify_chain_shape_unchanged(self, chain_parameters: list[Parameter]) -> None:
        """
        Check that an extension keeps the chain's columns.

        Parameters
        ----------
        chain_parameters : list[Parameter]
            The parameters that would form the chain for this run.

        Raises
        ------
        ValueError
            If the number of parameters differs from the existing chain's.
        """
        if self._results is None:
            return
        existing = self._results.draws.shape[1]
        if len(chain_parameters) != existing:
            raise ValueError(
                f'Cannot extend a chain of {existing} parameters with a run of '
                f'{len(chain_parameters)}. An extension continues the stored chain, whose columns '
                f'are fixed, so it needs the same parameters the chain was started with. Start a '
                f'fresh chain with sample() instead.'
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
            If a requested label matches no free parameter, or the subset is empty.
        """
        if parameters is None:
            return []
        if not isinstance(parameters, (list, tuple)):
            raise TypeError('parameters must be a list of Parameters, a list of labels, or None.')

        labels = self._labels()
        by_label = {labels.label(parameter): parameter for parameter in labels.parameters}
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
                requested.append(entry)
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
        details = ', '.join(
            f'{name} ({fraction:.0%} of draws)' for name, fraction in piled_up.items()
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
        if sidecar.is_file():
            self._saved_labels = json.loads(sidecar.read_text(encoding='utf-8'))
        else:
            self._saved_labels = {}
            warnings.warn(
                (
                    f'No parameter-name sidecar found at {sidecar}. The chain will be reported '
                    f'under the internal names it was saved with, because those cannot be matched '
                    f'to this Analysis.'
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

        return plot_posterior_predictive(
            x=np.asarray(x),
            y=np.asarray(y),
            predictions=self.predictions(n_draws),
            y_err=None if weights is None else 1.0 / np.asarray(weights),
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


class MultiQPosteriorSampler(PosteriorSampler):
    """
    Posterior sampling for an Analysis covering several Q values.

    Reached as ``analysis.bayesian``. Sampling can run either way round:

    - ``fit_method='independent'`` gives each Q index its own chain, which is cheaper and keeps the
      Q values from influencing one another.
    - ``fit_method='simultaneous'`` runs a single chain over every Q at once, which is what is
      needed when parameters are shared across Q, and costs considerably more: DREAM runs a number
      of chains proportional to the parameter count, and a simultaneous run has every Q's
      parameters in play together.

    Results from independent runs stay on the per-Q samplers. This class gathers them where
    gathering is sound, and declines where it is not; see :meth:`summary` and :meth:`plot_corner`.

    Parameters
    ----------
    per_q : Callable[[], list]
        Returns the per-Q Analysis objects, each exposing ``Q_index`` and its own ``bayesian``.
    **kwargs : dict[str, Any]
        Forwarded to :class:`PosteriorSampler`.
    """

    def __init__(self, per_q: Callable[[], list], **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self._per_q = per_q

    @property
    def results_per_q(self) -> list[SamplingResults | None] | None:
        """
        The per-Q chains from independent sampling, or None if there are none.

        A simultaneous run produces one chain covering every Q, which is on :attr:`results`.

        Returns
        -------
        list[SamplingResults | None] | None
            One entry per Q index, None where that Q has not been sampled, or None overall if no Q
            index has been sampled.
        """
        results = [analysis1d.bayesian.results for analysis1d in self._per_q()]
        return results if any(result is not None for result in results) else None

    def sample(
        self,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        fit_method: str = 'independent',
        Q_index: int | None = None,
        **sampler_options: dict[str, Any],
    ) -> SamplingResults | list[SamplingResults]:
        """
        Draw samples from the posterior, per Q index or over all of them at once.

        Parameters
        ----------
        samples : int, default=10000
            Number of raw samples to draw across all chains, before thinning.
        burn : int, default=2000
            Burn-in generations to discard before collecting samples.
        thin : int, default=10
            Thinning interval, which reduces autocorrelation between retained draws.
        fit_method : str, default='independent'
            Either "independent" (a separate chain per Q index) or "simultaneous" (one chain over
            all Q indices at once).
        Q_index : int | None, default=None
            With ``fit_method='independent'``, sample only this Q index. Ignored when sampling
            simultaneously.
        **sampler_options : dict[str, Any]
            Forwarded to the underlying sampler.

        Returns
        -------
        SamplingResults | list[SamplingResults]
            A single result when a specific Q index was sampled or when sampling simultaneously,
            and otherwise one result per Q index.

        Raises
        ------
        ValueError
            If fit_method is not "independent" or "simultaneous".
        """
        if fit_method not in ('independent', 'simultaneous'):
            raise ValueError("Invalid fit method. Choose 'independent' or 'simultaneous'.")
        per_q = self._per_q()
        if not per_q:
            raise ValueError(
                'No Q values available for sampling. Please check the experiment data.'
            )
        if fit_method == 'simultaneous':
            return super().sample(samples=samples, burn=burn, thin=thin, **sampler_options)
        if Q_index is not None:
            return per_q[Q_index].bayesian.sample(
                samples=samples, burn=burn, thin=thin, **sampler_options
            )
        # The per-Q chains live on their own samplers; this one then holds nothing of its own.
        self._results = None
        return [
            analysis1d.bayesian.sample(samples=samples, burn=burn, thin=thin, **sampler_options)
            for analysis1d in per_q
        ]

    def summary(self) -> PosteriorSummary:
        """
        Summarize the posterior, gathering the per-Q chains when sampling was independent.

        Every entry is a marginal distribution of one parameter, and a marginal is well defined
        within its own chain, so collecting them into one table is sound even though the chains are
        separate. Labels carry the Q index either way, so the table reads the same.

        Returns
        -------
        PosteriorSummary
            One entry per sampled parameter, across every Q index that has been sampled.
        """
        per_q = self.results_per_q
        if self._results is not None or per_q is None:
            return super().summary()

        labels = self._labels()
        entries = []
        for result in per_q:
            if result is None:
                continue
            entries.extend(
                summarize_draws(
                    draws=result.draws,
                    labels=labels.display_names(result.param_names),
                    parameters_by_column=labels.resolve(result.param_names),
                ).entries
            )
        return PosteriorSummary(entries)

    def set_parameters_to_median(self) -> list[Parameter]:
        """
        Set every sampled parameter to the median of its marginal posterior.

        Applies the per-Q chains to their own Q when sampling was independent.

        Returns
        -------
        list[Parameter]
            The parameters that were changed.
        """
        if self._results is not None or self.results_per_q is None:
            return super().set_parameters_to_median()
        changed = []
        for analysis1d in self._per_q():
            if analysis1d.bayesian.results is not None:
                changed.extend(analysis1d.bayesian.set_parameters_to_median())
        return changed

    def plot_corner(self, Q_index: int | None = None, **kwargs: dict[str, Any]) -> Figure | VBox:
        """
        Plot the marginal and pairwise posterior distributions.

        After independent sampling each Q has its own chain, and no draw pairs a parameter at one Q
        with a parameter at another, so there is no joint distribution across Q to plot. Rather
        than combine them into a figure showing correlations that came from how the sampling was
        run, this steps through the chains one at a time: pick one with ``Q_index``, or leave it
        out in a notebook to get a slider.

        Parameters
        ----------
        Q_index : int | None, default=None
            Which Q index to plot, when the chains are per-Q. If None, a slider is returned. Not
            used for a simultaneous chain, which already covers every Q.
        **kwargs : dict[str, Any]
            Forwarded to :func:`easydynamics.utils.posterior_plotting.plot_corner`.

        Returns
        -------
        Figure | VBox
            The matplotlib Figure, or an ipywidgets box with a Q slider.

        Raises
        ------
        RuntimeError
            If a slider is asked for outside a notebook.
        """
        from easydynamics.utils.posterior_plotting import corner_with_slider

        per_q = self.results_per_q
        if self._results is not None or per_q is None:
            return super().plot_corner(**kwargs)

        analyses = self._per_q()
        if Q_index is not None:
            return analyses[Q_index].bayesian.plot_corner(**kwargs)

        if not _in_notebook():
            sampled = [index for index, result in enumerate(per_q) if result is not None]
            raise RuntimeError(
                f'Each Q index has its own chain, and the slider needs a Jupyter notebook. '
                f'Pass Q_index to plot one of them; sampled Q indices are {sampled}.'
            )

        chains = {}
        for analysis1d, result in zip(analyses, per_q, strict=True):
            if result is None:
                continue
            # Named by the per-Q sampler, so the labels match that Q's own summary and stay short:
            # the Q index is on the slider, and repeating it in every axis label would only cost
            # width. The summary entries follow the draw columns, so the order lines up.
            entries = list(analysis1d.bayesian.summary())
            chains[analysis1d.Q_index] = {
                'draws': result.draws,
                'names': [entry.name for entry in entries],
                'units': [entry.unit for entry in entries],
            }
        return corner_with_slider(chains, title=self._analysis.display_name)

    def plot_trace(self, **kwargs: dict[str, Any]) -> Figure:
        """
        Plot the chain trace of each sampled parameter.

        Only available for a simultaneous chain, since the per-Q chains are separate runs of
        different lengths rather than one trace.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Forwarded to :func:`easydynamics.utils.posterior_plotting.plot_trace`.

        Returns
        -------
        Figure
            The matplotlib Figure.

        Raises
        ------
        RuntimeError
            If only independent per-Q chains exist.
        """
        if self._results is None and self.results_per_q is not None:
            raise RuntimeError(
                'Each Q index has its own chain, so there is no single trace to draw. Use '
                'analysis.analysis_list[Q_index].bayesian.plot_trace() for one of them, or sample '
                "with fit_method='simultaneous'."
            )
        return super().plot_trace(**kwargs)

    def _require_results(self) -> SamplingResults:
        """
        Get the stored results, pointing at the per-Q chains when those are what exist.

        Returns
        -------
        SamplingResults
            The results of the most recent simultaneous run.

        Raises
        ------
        RuntimeError
            If no simultaneous sampling has been run.
        """
        if self._results is None and self.results_per_q is not None:
            raise RuntimeError(
                'This Analysis holds no chain of its own, but its Q indices do: sampling with '
                "fit_method='independent' gives each Q its own chain. summary() and "
                'set_parameters_to_median() gather those up; for anything needing a single chain, '
                'use analysis.analysis_list[Q_index].bayesian, or sample with '
                "fit_method='simultaneous'."
            )
        return super()._require_results()


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
