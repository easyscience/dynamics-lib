# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Shared Bayesian MCMC sampling machinery for the Analysis classes.

Everything that does not depend on how a particular Analysis is wired up lives here: caching the
Fitter and the Sampler, guarding the parameter bounds, restoring parameter values afterwards, and
turning raw draws into a readable summary. A concrete Analysis supplies the three things that do
differ, via :meth:`BayesianSamplingMixin._build_bayesian_fitter`,
:meth:`BayesianSamplingMixin._get_sampling_data`, and
:meth:`BayesianSamplingMixin._get_chain_parameters`.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
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

if TYPE_CHECKING:
    import os
    from collections.abc import Callable
    from collections.abc import Iterator

    from easyscience.fitting.fitter import Fitter
    from easyscience.fitting.sampler import SamplingResults
    from easyscience.variable import Parameter
    from matplotlib.figure import Figure

    from easydynamics.analysis.posterior import BoundsSuggestions

# Suffix of the sidecar mapping chain columns to stable parameter names, written next to the BUMPS
# chain files by save_chain().
_NAME_MAP_SUFFIX = '.parameter-names.json'


class BayesianSamplingMixin:
    """
    Bayesian MCMC sampling on top of an Analysis, backed by the BUMPS DREAM sampler.

    Sampling explores the full posterior distribution of the free parameters rather than reporting
    a single best-fit point, which is worth doing when parameters are correlated or their
    uncertainties are strongly non-Gaussian -- both common in QENS.

    Running :meth:`fit` first is not required, but it helps: DREAM seeds its population in a small
    ball around the parameters' current values, so starting from fitted values shortens the burn-in
    needed to reach the typical set.

    Notes
    -----
    All free parameters must have finite bounds before sampling, because in DREAM the bounds are
    the prior. :meth:`suggest_bounds` proposes bounds for any parameter still missing one.

    Examples
    --------
    ```python
    analysis.fit()
    analysis.suggest_bounds().apply()
    results = analysis.sample_posterior(samples=10000, burn=2000, thin=10)
    analysis.posterior_summary()
    ```
    """

    #############
    # Setup
    #############

    def _init_bayesian_state(self) -> None:
        """
        Initialize the cached sampling state.

        Must be called by the concrete Analysis before any observer callback can fire, in the same
        way as the other cached objects on the class.
        """
        self._fitter = None
        self._fitter_is_dirty = True
        self._bayesian_sampler = None
        self._bayesian_sampler_is_dirty = True
        self._posterior_result = None
        # Set only while a bulk operation holds the parameter list, see _bulk_parameter_access.
        self._chain_parameters_cache = None
        # Maps a chain column's unique_name to the parameter name it had when saved. Only populated
        # by load_chain, because unique_names are per-session and do not survive a round trip.
        self._chain_name_map = {}

    def _invalidate_fitter(self) -> None:
        """
        Mark the cached Fitter and Sampler as needing a rebuild.

        The Sampler binds its data at construction, so anything that invalidates the Fitter
        invalidates the Sampler too.
        """
        self._fitter_is_dirty = True
        self._bayesian_sampler_is_dirty = True

    def _invalidate_bayesian_sampler(self) -> None:
        """
        Mark only the cached Sampler as needing a rebuild.

        Used when the data changed but the model did not.
        """
        self._bayesian_sampler_is_dirty = True

    #############
    # Hooks for concrete Analysis classes
    #############

    def _build_bayesian_fitter(self) -> Fitter:
        """
        Build the EasyScience Fitter (or MultiFitter) for this Analysis.

        Returns
        -------
        Fitter
            A configured Fitter or MultiFitter.

        Raises
        ------
        NotImplementedError
            If the concrete Analysis does not implement it.
        """
        raise NotImplementedError('Subclasses must implement _build_bayesian_fitter.')

    def _get_sampling_data(self) -> tuple:
        """
        Get the ``(x, y, weights)`` to bind to the Sampler.

        Each element is either an array (single dataset) or a list of arrays (MultiFitter).

        Returns
        -------
        tuple
            The ``(x, y, weights)`` triple.

        Raises
        ------
        NotImplementedError
            If the concrete Analysis does not implement it.
        """
        raise NotImplementedError('Subclasses must implement _get_sampling_data.')

    def _get_chain_parameters(self) -> list[Parameter]:
        """
        Get the free parameters that will appear as columns of the chain.

        Returns
        -------
        list[Parameter]
            The free parameters of the underlying model(s).

        Raises
        ------
        NotImplementedError
            If the concrete Analysis does not implement it.
        """
        raise NotImplementedError('Subclasses must implement _get_chain_parameters.')

    def _prepare_for_sampling(self) -> None:
        """
        Bring any cached computation up to date before a sampling run.

        The default does nothing; Analysis classes that cache a convolver override it.
        """

    #############
    # Properties
    #############

    @property
    def fitter(self) -> Fitter:
        """
        The EasyScience Fitter used for fitting and sampling, built on first use.

        Exposed so the minimizer, tolerance, and maximum evaluation count can be configured
        directly, e.g. ``analysis.fitter.switch_minimizer(AvailableMinimizers.Bumps)``.

        Returns
        -------
        Fitter
            The cached Fitter or MultiFitter.
        """
        if self._fitter_is_dirty or self._fitter is None:
            self._fitter = self._build_bayesian_fitter()
            self._fitter_is_dirty = False
        return self._fitter

    @property
    def bayesian_sampler(self) -> Sampler | None:
        """
        The EasyScience Sampler holding the MCMC chain, or None before the first run.

        Named to avoid confusion with the SampleModel: this samples the posterior, not the sample.

        Returns
        -------
        Sampler | None
            The cached Sampler, or None if no chain has been started.
        """
        return self._bayesian_sampler

    @property
    def posterior_result(self) -> SamplingResults | None:
        """
        The results of the most recent sampling run, or None if there has not been one.

        Returns
        -------
        SamplingResults | None
            The most recent sampling results.
        """
        return self._posterior_result

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

        Nothing is changed until :meth:`BoundsSuggestions.apply` is called, so the proposal can be
        reviewed first. Bounds that are already finite are never widened or narrowed, so physical
        limits such as a non-negative area are left alone.

        Because the bounds act as a uniform prior in DREAM, a generous width is the safe choice:
        too tight a bound truncates the posterior and understates the uncertainty.

        Parameters
        ----------
        n_sigma : float, default=10.0
            How many standard deviations of the fitted uncertainty to allow on each side.
        relative_pad : float, default=0.2
            Extra half-width as a fraction of the absolute parameter value. This guards against
            minimizers that report a zero or absurdly small uncertainty.
        absolute_floor : float | None, default=None
            A minimum half-width in the parameter's own units, for when neither the uncertainty nor
            the value carries the natural scale.

        Returns
        -------
        BoundsSuggestions
            The proposed bounds, which must be applied explicitly.
        """
        with self._bulk_parameter_access() as parameters:
            labels = [self.parameter_label(parameter) for parameter in parameters]
        return suggest_bounds_for_parameters(
            parameters,
            labels=labels,
            n_sigma=n_sigma,
            relative_pad=relative_pad,
            absolute_floor=absolute_floor,
        )

    def check_bounds_for_sampling(self) -> None:
        """
        Verify that every free parameter has finite bounds.

        Raises
        ------
        ValueError
            If any free parameter has an infinite lower or upper bound.
        """
        unbounded = unbounded_parameters(self._get_chain_parameters())
        if not unbounded:
            return
        names = ', '.join(self.parameter_label(parameter) for parameter in unbounded)
        raise ValueError(
            f'Bayesian sampling requires finite bounds on every free parameter, because the '
            f'bounds act as the prior. These parameters are unbounded: {names}. '
            f'Set their min and max, or call suggest_bounds() to propose values.'
        )

    #############
    # Sampling
    #############

    def sample_posterior(
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

        This starts a fresh chain, replacing any existing one; use :meth:`extend_sampling` to
        continue a chain instead. Parameter values are restored to what they were beforehand, so
        sampling never silently moves the model off its fitted values; use
        :meth:`set_parameters_to_posterior_median` to adopt the posterior.

        Parameters
        ----------
        samples : int, default=10000
            Number of raw samples to draw across all chains, before thinning. This is a guaranteed
            minimum rather than an exact count.
        burn : int, default=2000
            Burn-in generations to discard before collecting samples.
        thin : int, default=10
            Thinning interval, which reduces autocorrelation between retained draws.
        population : int | None, default=None
            DREAM population scale factor: BUMPS runs ``ceil(population * n_parameters)`` chains.
        parameters : list[Parameter] | list[str] | None, default=None
            Restrict the chain to these parameters, given as Parameter objects or names. All other
            free parameters are held fixed for the duration of the run. Note that holding a
            parameter fixed is not the same as marginalizing over it: the resulting credible
            intervals are conditional on the fixed values and will be too narrow if the parameters
            are correlated. The default samples every free parameter.
        **sampler_options : dict[str, Any]
            Forwarded to the EasyScience Sampler, e.g. ``sampler_kwargs``, ``progress_callback``,
            or ``abort_test``.

        Returns
        -------
        SamplingResults
            The sampling results, also stored on :attr:`posterior_result`.
        """
        return self._run_sampling(
            parameters=parameters,
            run=lambda sampler: sampler.sample(
                samples=samples,
                burn=burn,
                thin=thin,
                population=population,
                **sampler_options,
            ),
        )

    def extend_sampling(
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
            The same restriction as in :meth:`sample_posterior`. Pass the same value that started
            the chain, since the chain's columns cannot change on extension.
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
        if self._bayesian_sampler is None:
            raise RuntimeError(
                'No chain to extend. Call sample_posterior() or load_chain() first.'
            )
        return self._run_sampling(
            parameters=parameters,
            run=lambda sampler: sampler.extend(
                additional_samples=additional_samples,
                thin=thin,
                **sampler_options,
            ),
            reuse_sampler=True,
        )

    def _run_sampling(
        self,
        parameters: list[Parameter] | list[str] | None,
        run: Callable[[Sampler], SamplingResults],
        reuse_sampler: bool = False,
    ) -> SamplingResults:
        """
        Run a sampling operation with all the surrounding guards in place.

        Checks the bounds, switches the minimizer to BUMPS, optionally holds parameters fixed,
        runs, and then restores the parameter values, fixed flags, and minimizer.

        Parameters
        ----------
        parameters : list[Parameter] | list[str] | None
            Parameters to restrict the chain to, or None for all free parameters.
        run : Callable[[Sampler], SamplingResults]
            The operation to perform on the prepared Sampler.
        reuse_sampler : bool, default=False
            Whether to reuse the cached Sampler rather than rebuilding it. Required when extending
            a chain, since the chain lives on the Sampler.

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
        self._warn_about_held_parameters(held_fixed)

        with _FixedParameters(held_fixed):
            self.check_bounds_for_sampling()
            self._prepare_for_sampling()

            chain_parameters = self._get_chain_parameters()
            saved_values = [(p, p.value) for p in chain_parameters]

            if reuse_sampler:
                self._verify_chain_shape_unchanged(chain_parameters)

            fitter = self.fitter
            original_minimizer = fitter.minimizer.enum
            fitter.switch_minimizer(AvailableMinimizers.Bumps)
            try:
                sampler = self._get_or_build_sampler(reuse_sampler=reuse_sampler)
                results = run(sampler)
            except IndexError as error:
                if not _raised_inside_bumps(error):
                    raise
                # BUMPS' own outlier removal indexes past the end of its buffer. Seen both when
                # chains scatter because the model is not identifiable, and on short chains where
                # its buffer has too few generations to work with. The bare IndexError says
                # nothing useful, so name both causes and the way out.
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

        # Labelled outside the block above, so that a subset run records the same labels a full run
        # would. Inside it the other parameters are fixed, nothing looks ambiguous, and the sidecar
        # would be written with unqualified names that no longer match on reload.
        with self._bulk_parameter_access():
            self._chain_name_map = {
                parameter.unique_name: self.parameter_label(parameter)
                for parameter in chain_parameters
            }

        self._posterior_result = results
        self._warn_about_bounds_occupancy(results, self._resolve_chain_parameters(results))
        return results

    def _verify_chain_shape_unchanged(self, chain_parameters: list[Parameter]) -> None:
        """
        Check that an extension keeps the chain's columns.

        BUMPS resumes from a stored state whose width is fixed, so a run that would add or drop a
        parameter cannot continue that chain. Caught here rather than left to fail obscurely inside
        the sampler.

        Parameters
        ----------
        chain_parameters : list[Parameter]
            The parameters that would form the chain for this run.

        Raises
        ------
        ValueError
            If the number of parameters differs from the existing chain's.
        """
        if self._posterior_result is None:
            return
        existing = self._posterior_result.draws.shape[1]
        if len(chain_parameters) != existing:
            raise ValueError(
                f'Cannot extend a chain of {existing} parameters with a run of '
                f'{len(chain_parameters)}. An extension continues the stored chain, whose columns '
                f'are fixed, so it needs the same parameters the chain was started with. Start a '
                f'fresh chain with sample_posterior() instead.'
            )

    def _get_or_build_sampler(self, reuse_sampler: bool) -> Sampler:
        """
        Get the cached Sampler, rebuilding it if the data or model changed.

        Parameters
        ----------
        reuse_sampler : bool
            Whether to reuse the cached Sampler even if it is marked dirty.

        Returns
        -------
        Sampler
            The Sampler to run.
        """
        needs_rebuild = self._bayesian_sampler is None or (
            self._bayesian_sampler_is_dirty and not reuse_sampler
        )
        if needs_rebuild:
            x, y, weights = self._get_sampling_data()
            self._bayesian_sampler = Sampler(self.fitter, x, y, weights=weights)
            self._bayesian_sampler_is_dirty = False
        return self._bayesian_sampler

    def _resolve_parameters_to_hold_fixed(
        self,
        parameters: list[Parameter] | list[str] | None,
    ) -> list[Parameter]:
        """
        Work out which free parameters must be held fixed to honour a subset request.

        Parameters
        ----------
        parameters : list[Parameter] | list[str] | None
            The requested subset, as Parameter objects or names, or None for all free parameters.

        Returns
        -------
        list[Parameter]
            The free parameters that are not in the requested subset.

        Raises
        ------
        TypeError
            If parameters is not a list of Parameters or strings, or None.
        ValueError
            If a requested name does not match any free parameter, or the subset is empty.
        """
        if parameters is None:
            return []
        if not isinstance(parameters, (list, tuple)):
            raise TypeError('parameters must be a list of Parameters, a list of names, or None.')

        free = self._get_chain_parameters()
        by_name = {parameter.name: parameter for parameter in free}
        requested = []
        for entry in parameters:
            if isinstance(entry, str):
                if entry not in by_name:
                    available = ', '.join(sorted(by_name))
                    raise ValueError(f'No free parameter named {entry!r}. Available: {available}.')
                requested.append(by_name[entry])
            elif hasattr(entry, 'unique_name'):
                requested.append(entry)
            else:
                raise TypeError(
                    'parameters must contain Parameter objects or parameter names (strings).'
                )

        requested_unique_names = {parameter.unique_name for parameter in requested}
        if not requested_unique_names:
            raise ValueError('parameters must name at least one parameter to sample.')
        return [
            parameter for parameter in free if parameter.unique_name not in requested_unique_names
        ]

    @staticmethod
    def _warn_about_held_parameters(held_fixed: list[Parameter]) -> None:
        """
        Warn that holding parameters fixed makes the credible intervals conditional.

        Parameters
        ----------
        held_fixed : list[Parameter]
            The parameters being held fixed for the run.
        """
        if not held_fixed:
            return
        names = ', '.join(parameter.name for parameter in held_fixed)
        warnings.warn(
            (
                f'Holding these parameters fixed while sampling: {names}. '
                f'Fixing a parameter is not the same as marginalizing over it, so the resulting '
                f'credible intervals are conditional on these values and will be too narrow if '
                f'the parameters are correlated.'
            ),
            UserWarning,
            stacklevel=4,
        )

    @staticmethod
    def _warn_about_bounds_occupancy(
        results: SamplingResults,
        parameters_by_column: list[Parameter | None],
    ) -> None:
        """
        Warn when the posterior has piled up against a bound.

        Parameters
        ----------
        results : SamplingResults
            The sampling results to inspect.
        parameters_by_column : list[Parameter | None]
            The parameter for each column of the chain, or None where none could be matched.
        """
        piled_up = parameters_at_bounds(results.draws, parameters_by_column)
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

    def posterior_summary(self) -> PosteriorSummary:
        """
        Summarize the marginal posterior of each sampled parameter.

        Reports the median and the 68% credible interval under the parameter's own name and unit,
        rather than the opaque unique name the sampler uses internally. Requires a completed
        sampling run.

        Returns
        -------
        PosteriorSummary
            One entry per sampled parameter.
        """
        results = self._require_posterior_result()
        return summarize_draws(
            draws=results.draws,
            labels=self._chain_display_names(results),
            parameters_by_column=self._resolve_chain_parameters(results),
        )

    def set_parameters_to_posterior_median(self) -> list[Parameter]:
        """
        Set every sampled parameter to the median of its marginal posterior.

        Note that the vector of marginal medians is not in general the same as the
        highest-posterior point, and for strongly correlated parameters it need not even be a good
        fit. Requires a completed sampling run.

        Returns
        -------
        list[Parameter]
            The parameters that were changed.
        """
        results = self._require_posterior_result()
        changed = []
        for column, parameter in enumerate(self._resolve_chain_parameters(results)):
            if parameter is None:
                continue
            parameter.value = float(np.median(results.draws[:, column]))
            changed.append(parameter)
        return changed

    def _require_posterior_result(self) -> SamplingResults:
        """
        Get the stored sampling results, raising if there are none.

        Returns
        -------
        SamplingResults
            The most recent sampling results.

        Raises
        ------
        RuntimeError
            If no sampling has been run yet.
        """
        if self._posterior_result is None:
            raise RuntimeError(
                'No posterior samples yet. Call sample_posterior() or load_chain() first.'
            )
        return self._posterior_result

    #############
    # Persistence
    #############

    def save_chain(self, path: str | os.PathLike) -> None:
        """
        Save the MCMC chain to disk.

        Writes the BUMPS chain files alongside a sidecar recording the parameter names and a
        fingerprint of the data that was sampled.

        Parameters
        ----------
        path : str | os.PathLike
            Path prefix for the chain files.

        Raises
        ------
        RuntimeError
            If there is no chain to save.
        """
        if self._bayesian_sampler is None:
            raise RuntimeError('No chain to save. Call sample_posterior() first.')
        self._bayesian_sampler.save(path)
        # The BUMPS sidecar records unique names, which are handed out per session and so mean
        # nothing on reload. Record the parameter names alongside them, which are stable.
        Path(f'{path}{_NAME_MAP_SUFFIX}').write_text(
            json.dumps(self._chain_name_map, indent=2),
            encoding='utf-8',
        )

    def load_chain(self, path: str | os.PathLike, skip: int = 0) -> SamplingResults:
        """
        Load a previously saved MCMC chain.

        The loaded chain can be inspected, summarized, or continued with :meth:`extend_sampling`. A
        chain saved from different data loads with a warning.

        Parameters
        ----------
        path : str | os.PathLike
            The path prefix the chain was saved under.
        skip : int, default=0
            Number of initial samples to skip when reading the chain.

        Returns
        -------
        SamplingResults
            The loaded sampling results, also stored on :attr:`posterior_result`.
        """
        self._prepare_for_sampling()
        name_map_path = Path(f'{path}{_NAME_MAP_SUFFIX}')
        if name_map_path.is_file():
            self._chain_name_map = json.loads(name_map_path.read_text(encoding='utf-8'))
        else:
            self._chain_name_map = {}
            warnings.warn(
                (
                    f'No parameter-name sidecar found at {name_map_path}. The chain will be '
                    f'reported under the internal names it was saved with, because those cannot '
                    f'be matched to this Analysis.'
                ),
                UserWarning,
                stacklevel=2,
            )

        fitter = self.fitter
        original_minimizer = fitter.minimizer.enum
        fitter.switch_minimizer(AvailableMinimizers.Bumps)
        try:
            sampler = self._get_or_build_sampler(reuse_sampler=False)
            results = sampler.load_state(path, skip=skip)
        finally:
            fitter.switch_minimizer(original_minimizer)
        self._posterior_result = results
        return results

    #############
    # Plotting
    #############

    def plot_trace(self, **kwargs: dict[str, Any]) -> Figure:
        """
        Plot the chain trace of each sampled parameter.

        A well-mixed chain looks like a "hairy caterpillar" with no drift; visible trends mean the
        chain has not converged and needs a longer burn-in. Requires a completed sampling run.

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

        results = self._require_posterior_result()
        kwargs.setdefault('units', self._chain_units(results))
        return plot_trace(
            draws=results.draws,
            logp=results.logp,
            names=self._chain_display_names(results),
            title=self.display_name,
            **kwargs,
        )

    def plot_corner(self, **kwargs: dict[str, Any]) -> Figure:
        """
        Plot the marginal and pairwise posterior distributions.

        Diagonal panels show each parameter's marginal distribution; off-diagonal panels show the
        joint distribution of a pair, where a strong diagonal ridge means the two are correlated.
        Requires a completed sampling run.

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

        results = self._require_posterior_result()
        kwargs.setdefault('units', self._chain_units(results))
        return plot_corner(
            draws=results.draws,
            names=self._chain_display_names(results),
            title=self.display_name,
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

        The model is re-evaluated for a random subset of the posterior draws, and the spread of
        those curves becomes the band. Data straying outside the band systematically points at a
        model that is missing something, rather than at parameters that need tuning. Requires a
        completed sampling run.

        Parameters
        ----------
        n_draws : int, default=200
            How many posterior draws to evaluate the model for. Each draw costs one full model
            evaluation, so this trades smoothness of the band against time.
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

        results = self._require_posterior_result()
        x, y, weights = self._get_sampling_data()
        if isinstance(x, (list, tuple)):
            raise NotImplementedError(
                'plot_posterior_predictive supports a single dataset only. Plot each dataset '
                'from its own Analysis1d instead.'
            )

        predictions = self._evaluate_over_draws(results, x, n_draws)
        y_err = None if weights is None else 1.0 / np.asarray(weights)
        kwargs.setdefault('xlabel', self._predictive_axis_labels()[0])
        kwargs.setdefault('ylabel', self._predictive_axis_labels()[1])
        return plot_posterior_predictive(
            x=np.asarray(x),
            y=np.asarray(y),
            predictions=predictions,
            y_err=y_err,
            title=self.display_name,
            credible_interval=credible_interval,
            **kwargs,
        )

    def _predictive_axis_labels(self) -> tuple[str | None, str | None]:
        """
        Get default axis labels for the posterior predictive plot.

        The base implementation reads the energy and intensity units off the analysis when they are
        available, and falls back to no label rather than guessing.

        Returns
        -------
        tuple[str | None, str | None]
            The ``(xlabel, ylabel)`` pair.
        """
        energy = getattr(self, 'energy', None)
        xlabel = None if energy is None else f'Energy ({energy.unit})'
        sample_model = getattr(self, 'sample_model', None)
        y_unit = None if sample_model is None else getattr(sample_model, 'y_unit', None)
        ylabel = 'Intensity' if y_unit is None else f'Intensity ({y_unit})'
        return xlabel, ylabel

    def _evaluate_over_draws(
        self,
        results: SamplingResults,
        x: np.ndarray,
        n_draws: int,
    ) -> np.ndarray:
        """
        Evaluate the model once per posterior draw, restoring the parameters afterwards.

        Parameters
        ----------
        results : SamplingResults
            The sampling results supplying the draws.
        x : np.ndarray
            The independent variable to evaluate the model on.
        n_draws : int
            How many draws to evaluate. Draws are taken evenly across the chain.

        Returns
        -------
        np.ndarray
            Model evaluations, shape ``(n_selected, len(x))``.
        """
        self._prepare_for_sampling()

        columns = [
            (parameter, column)
            for column, parameter in enumerate(self._resolve_chain_parameters(results))
            if parameter is not None
        ]
        saved_values = [(parameter, parameter.value) for parameter, _ in columns]

        total = results.draws.shape[0]
        indices = np.unique(np.linspace(0, total - 1, min(n_draws, total)).astype(int))

        fit_function = self.fitter.fit_function
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

    def _resolve_chain_parameters(self, results: SamplingResults) -> list[Parameter | None]:
        """
        Match each column of the chain to one of this Analysis's parameters.

        Columns are matched on ``unique_name`` first. That fails for a chain loaded from disk,
        because unique names are handed out per session, so a saved chain also records the
        parameter names and those are used as a fallback.

        Parameters
        ----------
        results : SamplingResults
            The sampling results whose columns should be matched.

        Returns
        -------
        list[Parameter | None]
            The parameter for each column, or None where no match could be made.
        """
        parameters = self._chain_parameters()
        by_unique_name = {p.unique_name: p for p in parameters}
        by_label = {self.parameter_label(p): p for p in parameters}
        resolved = []
        for unique_name in results.param_names:
            parameter = by_unique_name.get(unique_name)
            if parameter is None:
                saved_label = self._chain_name_map.get(unique_name)
                parameter = None if saved_label is None else by_label.get(saved_label)
            resolved.append(parameter)
        return resolved

    def parameter_label(self, parameter: Parameter) -> str:
        """
        Get the label a parameter is reported under.

        The parameter's own name is enough when it identifies the parameter uniquely. Analysis
        classes that can hold several identically named parameters override this to qualify the
        name, but only when it is actually ambiguous -- see :meth:`_name_is_ambiguous`.

        Parameters
        ----------
        parameter : Parameter
            The parameter to label.

        Returns
        -------
        str
            The label to report the parameter under.
        """
        return parameter.name

    def _name_is_ambiguous(self, parameter: Parameter) -> bool:
        """
        Check whether another parameter in the chain shares this parameter's name.

        Qualifying a label is only worth the extra width when the bare name would be ambiguous, so
        a single-Q analysis, or a single binding, keeps its short names.

        Parameters
        ----------
        parameter : Parameter
            The parameter to check.

        Returns
        -------
        bool
            True when at least one other chain parameter has the same name.
        """
        names = [p.name for p in self._chain_parameters()]
        return names.count(parameter.name) > 1

    def _chain_parameters(self) -> list[Parameter]:
        """
        Get the chain parameters, reusing the list when a bulk operation is in progress.

        Collecting them walks every sub-model, so labelling a chain one parameter at a time is
        quadratic in the parameter count -- seconds, for a dataset with many Q values.

        Returns
        -------
        list[Parameter]
            The free parameters of the underlying model(s).
        """
        if self._chain_parameters_cache is not None:
            return self._chain_parameters_cache
        return self._get_chain_parameters()

    @contextmanager
    def _bulk_parameter_access(self) -> Iterator[list[Parameter]]:
        """
        Collect the chain parameters once for the duration of a block.

        Scoped rather than stored, so the cache cannot outlive the operation that wanted it and go
        stale against a changed model.

        Yields
        ------
        list[Parameter]
            The chain parameters, also served to :meth:`_chain_parameters` inside the block.
        """
        previous = self._chain_parameters_cache
        self._chain_parameters_cache = self._get_chain_parameters()
        try:
            yield self._chain_parameters_cache
        finally:
            self._chain_parameters_cache = previous

    def _chain_units(self, results: SamplingResults) -> list[str]:
        """
        Get the unit of each column of the chain.

        Parameters
        ----------
        results : SamplingResults
            The sampling results whose columns should be described.

        Returns
        -------
        list[str]
            One unit per column, empty where no parameter could be matched.
        """
        return [
            '' if parameter is None else str(parameter.unit)
            for parameter in self._resolve_chain_parameters(results)
        ]

    def _chain_display_names(self, results: SamplingResults) -> list[str]:
        """
        Translate the chain's column names into readable labels.

        Parameters
        ----------
        results : SamplingResults
            The sampling results whose columns should be named.

        Returns
        -------
        list[str]
            One label per column of the chain.
        """
        with self._bulk_parameter_access():
            resolved = self._resolve_chain_parameters(results)
            return [
                self._chain_name_map.get(unique_name, unique_name)
                if parameter is None
                else self.parameter_label(parameter)
                for unique_name, parameter in zip(results.param_names, resolved, strict=True)
            ]


def _raised_inside_bumps(error: BaseException) -> bool:
    """
    Check whether an exception came from inside BUMPS.

    Used to make sure only BUMPS' own failures are relabelled, so a bug in this package is not
    reported as a modelling problem.

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
    """
    Context manager that temporarily fixes parameters and restores their flags on exit.
    """

    def __init__(self, parameters: list[Parameter]) -> None:
        """
        Initialize the context manager.

        Parameters
        ----------
        parameters : list[Parameter]
            The parameters to hold fixed for the duration of the block.
        """
        self._parameters = list(parameters)
        self._saved: list[tuple[Parameter, bool]] = []

    def __enter__(self) -> None:
        """
        Fix the parameters, remembering their previous state.
        """
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
