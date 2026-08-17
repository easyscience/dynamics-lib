# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for the posterior sampler, driven through an Analysis1d, with the EasyScience Sampler
mocked out.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.fitting import AvailableMinimizers
from easyscience.variable import Parameter

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.gaussian import Gaussian

SAMPLER_PATH = 'easydynamics.analysis.posterior_sampling.Sampler'


def make_analysis(with_variances=True):
    energy_values = np.linspace(-5.0, 5.0, 20)
    intensity = 3.0 * np.exp(-0.5 * (energy_values / 1.2) ** 2)
    data = sc.array(
        dims=['Q', 'energy'],
        values=intensity[None, :],
        variances=np.full_like(intensity, 0.01)[None, :] if with_variances else None,
    )
    experiment = Experiment(
        data=sc.DataArray(
            data=data,
            coords={
                'Q': sc.array(dims=['Q'], values=[1.0], unit='1/Angstrom'),
                'energy': sc.array(dims=['energy'], values=energy_values, unit='meV'),
            },
        )
    )
    analysis = Analysis1d(
        display_name='TestBayesian',
        experiment=experiment,
        sample_model=SampleModel(components=Gaussian(area=3.0, width=1.2, center=0.0)),
        instrument_model=InstrumentModel(),
        Q_index=0,
    )
    analysis.instrument_model.fix_energy_offset(Q_index=0)
    return analysis


def bound_all(analysis, half_width=5.0):
    """Give every free parameter finite bounds so the pre-flight passes."""
    for parameter in analysis.get_free_parameters():
        parameter.min = float(parameter.value) - half_width
        parameter.max = float(parameter.value) + half_width


def fake_results(analysis, n_draws=100, values=None):
    """Build a SamplingResults-shaped object for the free parameters of an analysis."""
    parameters = analysis.get_free_parameters()
    if values is None:
        draws = np.tile([float(p.value) for p in parameters], (n_draws, 1))
    else:
        draws = np.asarray(values, dtype=float)
    return SimpleNamespace(
        draws=draws,
        param_names=[p.unique_name for p in parameters],
        logp=np.zeros(draws.shape[0]),
        state=MagicMock(Ngen=10, Npop=4),
    )


@pytest.fixture
def analysis():
    return make_analysis()


class TestPosteriorSampler:
    #############
    # Bounds pre-flight
    #############

    def test_sampling_refuses_unbounded_parameters(self, analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='finite bounds'):
            analysis.bayesian.sample(samples=10)

    def test_error_names_the_offending_parameters(self, analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='Gaussian area'):
            analysis.bayesian.check_bounds()

    def test_bounded_parameters_pass(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN EXPECT: does not raise
        analysis.bayesian.check_bounds()

    def test_suggest_bounds_covers_the_free_parameters(self, analysis):
        # THEN
        suggestions = analysis.bayesian.suggest_bounds()

        # EXPECT
        assert len(suggestions) == len(analysis.get_free_parameters())

    def test_degenerate_bounds_are_rejected(self, analysis):
        # WHEN one parameter's bounds collapse to a zero-width range, which internal state can
        # carry even though the setters refuse it
        bound_all(analysis)
        parameter = analysis.get_free_parameters()[0]
        parameter._min.value = float(parameter.max)

        # THEN EXPECT
        with pytest.raises(ValueError, match='degenerate bounds'):
            analysis.bayesian.check_bounds()

    #############
    # Sampling
    #############

    def test_restores_parameter_values_and_minimizer(self, analysis):
        # WHEN
        bound_all(analysis)
        before = [(p.unique_name, p.value) for p in analysis.get_free_parameters()]

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:

            def mutate_then_return(**_kwargs):
                # The real sampler leaves the parameters wherever the last evaluation put them.
                for parameter in analysis.get_free_parameters():
                    parameter.value = float(parameter.value) + 1.0
                return fake_results(analysis)

            sampler_class.return_value.sample.side_effect = mutate_then_return
            analysis.bayesian.sample(samples=10, burn=1, thin=1)

        # EXPECT
        after = [(p.unique_name, p.value) for p in analysis.get_free_parameters()]
        assert after == before
        assert analysis.fitter.minimizer.enum == AvailableMinimizers.LMFit_leastsq

    def test_switches_to_bumps_for_the_run(self, analysis):
        # WHEN
        bound_all(analysis)
        seen = []

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: (
                seen.append(analysis.fitter.minimizer.enum),
                fake_results(analysis),
            )[1]
            analysis.bayesian.sample(samples=10)

        # EXPECT
        assert seen == [AvailableMinimizers.Bumps]

    def test_restores_the_minimizer_even_when_sampling_raises(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = RuntimeError('boom')
            with pytest.raises(RuntimeError, match='boom'):
                analysis.bayesian.sample(samples=10)

        # EXPECT
        assert analysis.fitter.minimizer.enum == AvailableMinimizers.LMFit_leastsq

    def test_forwards_sampling_arguments(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=123, burn=7, thin=3, population=5)

        # EXPECT
        kwargs = sampler_class.return_value.sample.call_args.kwargs
        assert kwargs['samples'] == 123
        assert kwargs['burn'] == 7
        assert kwargs['thin'] == 3
        assert kwargs['population'] == 5

    def test_stores_the_result(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            expected = fake_results(analysis)
            sampler_class.return_value.sample.return_value = expected
            returned = analysis.bayesian.sample(samples=10)

        # EXPECT
        assert returned is expected
        assert analysis.bayesian.results is expected

    def test_warns_when_the_posterior_piles_up_against_a_bound(self, analysis):
        # WHEN a parameter's draws span its whole allowed range
        bound_all(analysis)
        parameters = analysis.get_free_parameters()
        draws = np.tile([float(p.value) for p in parameters], (500, 1))
        draws[:, 0] = np.linspace(parameters[0].min, parameters[0].max, 500)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis, values=draws)

            # THEN EXPECT
            with pytest.warns(UserWarning, match='piled up'):
                analysis.bayesian.sample(samples=10)

    def test_sampling_with_no_free_parameters_raises(self, analysis):
        # WHEN every parameter is fixed
        for parameter in analysis.get_free_parameters():
            parameter.fixed = True

        # THEN EXPECT a clear refusal, rather than a zero-parameter failure deep in BUMPS
        with pytest.raises(ValueError, match='no free parameters to sample'):
            analysis.bayesian.sample(samples=10)

    def test_does_not_warn_when_the_posterior_is_well_inside(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis)

            # THEN EXPECT
            with warnings_as_errors():
                analysis.bayesian.sample(samples=10)

    #############
    # Parameter subsets
    #############

    def test_holds_other_parameters_fixed_during_the_run(self, analysis):
        # WHEN
        bound_all(analysis)
        target = analysis.get_free_parameters()[0]
        seen = {}

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:

            def record(**_kwargs):
                seen['free'] = [p.unique_name for p in analysis.get_free_parameters()]
                return fake_results(analysis)

            sampler_class.return_value.sample.side_effect = record
            with pytest.warns(UserWarning, match='Holding these parameters fixed'):
                analysis.bayesian.sample(samples=10, parameters=[target.name])

        # EXPECT
        assert seen['free'] == [target.unique_name]

    def test_restores_the_fixed_flags_afterwards(self, analysis):
        # WHEN
        bound_all(analysis)
        before = [(p.unique_name, p.fixed) for p in analysis.get_all_parameters()]
        target = analysis.get_free_parameters()[0]

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            with pytest.warns(UserWarning):
                analysis.bayesian.sample(samples=10, parameters=[target])

        # EXPECT
        assert [(p.unique_name, p.fixed) for p in analysis.get_all_parameters()] == before

    def test_unknown_parameter_name_raises(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN EXPECT
        with pytest.raises(ValueError, match='No free parameter named'):
            analysis.bayesian.sample(samples=10, parameters=['not a parameter'])

    def test_fixed_parameter_object_is_rejected(self, analysis):
        # WHEN a Parameter object that is currently fixed is requested
        bound_all(analysis)
        target = analysis.get_free_parameters()[0]
        target.fixed = True

        # THEN EXPECT the same membership check a label gets, instead of every free parameter
        # ending up held fixed and BUMPS failing with zero parameters
        with pytest.raises(ValueError, match='not a free parameter'):
            analysis.bayesian.sample(samples=10, parameters=[target])

    def test_parameter_from_another_model_is_rejected(self, analysis):
        # WHEN
        bound_all(analysis)
        foreign = Parameter(name='foreign', value=1.0, unit='meV')

        # THEN EXPECT
        with pytest.raises(ValueError, match='not a free parameter'):
            analysis.bayesian.sample(samples=10, parameters=[foreign])

    def test_non_list_parameters_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(TypeError, match='must be a list'):
            analysis.bayesian.sample(samples=10, parameters='Gaussian area')

    def test_empty_parameter_list_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='at least one parameter'):
            analysis.bayesian.sample(samples=10, parameters=[])

    #############
    # Sampler caching
    #############

    def test_sampler_is_reused_between_runs(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)
            analysis.bayesian.sample(samples=10)

        # EXPECT the data is bound once, not per run
        assert sampler_class.call_count == 1

    def test_changing_the_q_index_rebuilds_the_sampler(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)
            analysis.Q_index = 0
            analysis.bayesian.sample(samples=10)

        # EXPECT the Sampler binds its data at construction, so it must be rebuilt
        assert sampler_class.call_count == 2

    def test_binds_the_same_data_the_fit_uses(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)

        # EXPECT
        expected_x, expected_y, expected_w = analysis._sampling_data()
        args, kwargs = sampler_class.call_args
        assert np.array_equal(args[1], expected_x)
        assert np.array_equal(args[2], expected_y)
        assert np.array_equal(kwargs['weights'], expected_w)

    #############
    # Extending and persistence
    #############

    def test_extend_without_a_chain_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(RuntimeError, match='No chain to extend'):
            analysis.bayesian.extend()

    def test_extend_delegates_to_the_sampler(self, analysis):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            sampler_class.return_value.extend.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)
            analysis.bayesian.extend(additional_samples=42, thin=2)

        # EXPECT
        kwargs = sampler_class.return_value.extend.call_args.kwargs
        assert kwargs['additional_samples'] == 42
        assert kwargs['thin'] == 2

    def test_extend_with_different_parameters_raises(self, analysis):
        # WHEN a chain was sampled over one parameter
        bound_all(analysis)
        first, second = analysis.get_free_parameters()[:2]

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            sampler_class.return_value.extend.side_effect = lambda **_k: fake_results(analysis)
            with pytest.warns(UserWarning, match='Holding these parameters fixed'):
                analysis.bayesian.sample(samples=10, parameters=[first.name])

            # THEN EXPECT extending with a different parameter, even at the same chain width,
            # is refused rather than silently merging draws of different quantities
            with (
                pytest.warns(UserWarning, match='Holding these parameters fixed'),
                pytest.raises(ValueError, match='holds draws of'),
            ):
                analysis.bayesian.extend(parameters=[second.name])

    def test_extend_after_a_data_change_raises(self, analysis):
        # WHEN the data changed after the chain was started
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)
            analysis.bayesian.invalidate()

            # THEN EXPECT the stale chain is refused rather than silently continued
            with pytest.raises(ValueError, match='model or data has changed'):
                analysis.bayesian.extend()

    def test_extend_after_a_failed_run_raises(self, analysis):
        # WHEN the previous run failed after building the sampler, leaving no results
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = RuntimeError('boom')
            with pytest.raises(RuntimeError, match='boom'):
                analysis.bayesian.sample(samples=10)

            # THEN EXPECT
            with pytest.raises(RuntimeError, match='left no results'):
                analysis.bayesian.extend()

    def test_save_without_a_chain_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(RuntimeError, match='No chain to save'):
            analysis.bayesian.save('somewhere')

    def test_save_writes_the_parameter_name_sidecar(self, analysis, tmp_path):
        # WHEN
        import json

        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)
            analysis.bayesian.save(str(tmp_path / 'chain'))

        # EXPECT the unique names are recorded against the stable parameter names
        sidecar = tmp_path / 'chain.parameter-names.json'
        assert sidecar.is_file()
        mapping = json.loads(sidecar.read_text(encoding='utf-8'))
        assert set(mapping.values()) == {p.name for p in analysis.get_free_parameters()}

    def test_load_without_a_sidecar_warns(self, analysis, tmp_path):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.load_state.return_value = fake_results(analysis)

            # THEN EXPECT
            with pytest.warns(UserWarning, match='No parameter-name sidecar'):
                analysis.bayesian.load(str(tmp_path / 'missing'))

    def test_load_with_an_empty_sidecar_warns_like_a_missing_one(self, analysis, tmp_path):
        # WHEN a sidecar file exists but records no labels
        bound_all(analysis)
        (tmp_path / 'chain.parameter-names.json').write_text('{}', encoding='utf-8')

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.load_state.return_value = fake_results(analysis)

            # THEN EXPECT
            with pytest.warns(UserWarning, match='No parameter-name sidecar'):
                analysis.bayesian.load(str(tmp_path / 'chain'))

    def test_load_passes_skip_through(self, analysis, tmp_path):
        # WHEN
        bound_all(analysis)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.load_state.return_value = fake_results(analysis)
            with pytest.warns(UserWarning, match='No parameter-name sidecar'):
                analysis.bayesian.load(str(tmp_path / 'chain'), skip=7)

        # EXPECT
        assert sampler_class.return_value.load_state.call_args.kwargs['skip'] == 7

    def test_save_after_a_sidecarless_load_writes_no_empty_sidecar(self, analysis, tmp_path):
        # WHEN a chain was loaded without a sidecar, so there are no labels to record
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.load_state.return_value = fake_results(analysis)
            with pytest.warns(UserWarning, match='No parameter-name sidecar'):
                analysis.bayesian.load(str(tmp_path / 'original'))

            # THEN EXPECT saving warns instead of writing an empty sidecar, which a later load()
            # would mistake for a valid one and resolve every column to raw names
            with pytest.warns(UserWarning, match='no parameter-name sidecar was written'):
                analysis.bayesian.save(str(tmp_path / 'resaved'))

        # EXPECT
        assert not (tmp_path / 'resaved.parameter-names.json').exists()

    #############
    # Results
    #############

    def test_summary_without_sampling_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(RuntimeError, match='No posterior samples yet'):
            analysis.bayesian.summary()

    def test_summary_uses_parameter_names_and_units(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)

        # THEN
        summary = analysis.bayesian.summary()

        # EXPECT
        names = {entry.name for entry in summary}
        assert names == {p.name for p in analysis.get_free_parameters()}
        assert all(entry.unit == 'meV' for entry in summary)

    def test_set_parameters_to_posterior_median(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis.get_free_parameters()
        draws = np.tile([float(p.value) + 2.0 for p in parameters], (50, 1))

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis, values=draws)
            expected = [float(p.value) + 2.0 for p in parameters]
            analysis.bayesian.sample(samples=10)

        # THEN
        changed = analysis.bayesian.set_parameters_to_median()

        # EXPECT
        assert len(changed) == len(parameters)
        assert [float(p.value) for p in parameters] == pytest.approx(expected)

    def test_median_without_sampling_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(RuntimeError, match='No posterior samples yet'):
            analysis.bayesian.set_parameters_to_median()

    #############
    # Plots
    #############

    def test_predictive_rejects_a_bad_draw_count(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)

        # THEN EXPECT
        with pytest.raises(ValueError, match='positive integer'):
            analysis.bayesian.plot_posterior_predictive(n_draws=0)

    def test_predictive_restores_parameter_values(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis.get_free_parameters()
        draws = np.tile([float(p.value) + 0.5 for p in parameters], (20, 1))

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis, values=draws)
            analysis.bayesian.sample(samples=10)

        before = [float(p.value) for p in parameters]

        # THEN
        analysis.bayesian.plot_posterior_predictive(n_draws=5)

        # EXPECT
        assert [float(p.value) for p in parameters] == pytest.approx(before)

    def test_plots_without_sampling_raise(self, analysis):
        # THEN EXPECT
        with pytest.raises(RuntimeError):
            analysis.bayesian.plot_trace()
        with pytest.raises(RuntimeError):
            analysis.bayesian.plot_corner()

    def test_predictive_forwards_the_measured_error_bars(self, analysis):
        # WHEN the data carries variances of 0.01, i.e. an uncertainty of 0.1
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)

        # THEN
        with patch('easydynamics.utils.posterior_plotting.plot_posterior_predictive') as plot:
            analysis.bayesian.plot_posterior_predictive(n_draws=2)

        # EXPECT
        assert plot.call_args.kwargs['y_err'] == pytest.approx(np.full(20, 0.1))

    def test_predictive_omits_error_bars_when_the_data_has_no_variances(self):
        # WHEN the data has no variances, so the weights are all-ones placeholders
        analysis = make_analysis(with_variances=False)
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)

        # THEN
        with patch('easydynamics.utils.posterior_plotting.plot_posterior_predictive') as plot:
            analysis.bayesian.plot_posterior_predictive(n_draws=2)

        # EXPECT no error bars fabricated from the placeholder weights
        assert plot.call_args.kwargs['y_err'] is None

    #############
    # Predictions
    #############

    def test_predictions_have_one_row_per_selected_draw(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                analysis, n_draws=100
            )
            analysis.bayesian.sample(samples=10)

        # THEN
        predictions = analysis.bayesian.predictions(n_draws=10)

        # EXPECT
        x, _, _ = analysis._sampling_data()
        assert predictions.shape == (10, len(x))

    def test_predictions_clamp_to_the_chain_length(self, analysis):
        # WHEN more draws are requested than the chain holds
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                analysis, n_draws=100
            )
            analysis.bayesian.sample(samples=10)

        # THEN
        predictions = analysis.bayesian.predictions(n_draws=500)

        # EXPECT one row per available draw, not 500
        x, _, _ = analysis._sampling_data()
        assert predictions.shape == (100, len(x))

    def test_predictions_take_draws_evenly_across_the_chain(self, analysis):
        # WHEN the area column identifies each draw, since the model scales linearly with it
        bound_all(analysis, half_width=500.0)
        parameters = analysis.get_free_parameters()
        column = [p.name for p in parameters].index('Gaussian area')
        draws = np.tile([float(p.value) for p in parameters], (100, 1))
        draws[:, column] = 1.0 + np.arange(100.0)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis, values=draws)
            analysis.bayesian.sample(samples=10)

        # THEN
        predictions = analysis.bayesian.predictions(n_draws=5)

        # EXPECT rows for draws 0, 24, 49, 74 and 99, read back through the model's linear
        # scaling with the area
        amplitudes = predictions.max(axis=1)
        expected = draws[[0, 24, 49, 74, 99], column]
        assert amplitudes / amplitudes[0] == pytest.approx(expected / expected[0])


class warnings_as_errors:
    """Context manager asserting that no UserWarning is emitted inside the block."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._caught = self._ctx.__enter__()
        warnings.simplefilter('always')
        return self

    def __exit__(self, *exc_info):
        caught = [w for w in self._caught if issubclass(w.category, UserWarning)]
        self._ctx.__exit__(*exc_info)
        if exc_info[0] is None:
            assert not caught, f'unexpected warnings: {[str(w.message) for w in caught]}'
        return False
