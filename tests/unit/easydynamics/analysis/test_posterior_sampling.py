# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for the posterior sampler, with the EasyScience Sampler mocked out.

The sampler is driven through the analyses that hold one: an Analysis1d and a ParameterAnalysis
for PosteriorSampler, and an Analysis for the multi-Q subclass.
"""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest
import scipp as sc
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting.multi_fitter import MultiFitter

mpl.use('Agg')

import easydynamics as edyn
import easydynamics.sample_model as sm
from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.gaussian import Gaussian

SAMPLER_PATH = 'easydynamics.analysis.posterior_sampling.Sampler'


def make_analysis():
    energy_values = np.linspace(-5.0, 5.0, 20)
    intensity = 3.0 * np.exp(-0.5 * (energy_values / 1.2) ** 2)
    data = sc.array(
        dims=['Q', 'energy'],
        values=intensity[None, :],
        variances=np.full_like(intensity, 0.01)[None, :],
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


def _bumps_style_index_error():
    """Build a callable that raises an IndexError from a frame that looks like it is in BUMPS."""

    def raise_index_error(**_kwargs):
        raise IndexError('index 71 is out of bounds for axis 0 with size 40')

    # The relabelling walks the traceback for a frame belonging to the bumps package, so the
    # function has to appear to live there.
    return types.FunctionType(
        raise_index_error.__code__,
        {'__name__': 'bumps.dream.state', '__builtins__': __builtins__},
    )


@pytest.fixture
def analysis():
    return make_analysis()


Q_VALUES = [0.5, 1.0, 1.5]


def make_multi_q_analysis():
    energy_values = np.linspace(-5.0, 5.0, 15)
    rows = [2.0 * np.exp(-0.5 * (energy_values / (0.8 + 0.4 * q**2)) ** 2) for q in Q_VALUES]
    observed = np.vstack(rows)
    experiment = edyn.Experiment(
        data=sc.DataArray(
            data=sc.array(
                dims=['Q', 'energy'],
                values=observed,
                variances=np.full_like(observed, 0.01),
            ),
            coords={
                'Q': sc.array(dims=['Q'], values=Q_VALUES, unit='1/Angstrom'),
                'energy': sc.array(dims=['energy'], values=energy_values, unit='meV'),
            },
        )
    )
    return edyn.Analysis(
        display_name='TestMultiQ',
        experiment=experiment,
        sample_model=sm.SampleModel(components=sm.Gaussian(area=2.0, width=1.0)),
        instrument_model=sm.InstrumentModel(),
    )


def bound_all_chain(multi_q_analysis, half_width=5.0):
    for parameter in multi_q_analysis._chain_parameters():
        parameter.min = float(parameter.value) - half_width
        parameter.max = float(parameter.value) + half_width


def fake_chain_results(parameters, n_draws=50):
    draws = np.tile([float(p.value) for p in parameters], (n_draws, 1))
    return SimpleNamespace(
        draws=draws,
        param_names=[p.unique_name for p in parameters],
        logp=np.zeros(n_draws),
        state=MagicMock(Ngen=10, Npop=4),
    )


@pytest.fixture
def multi_q_analysis():
    return make_multi_q_analysis()


Q = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0])


def make_dataset():
    widths = 0.10 + 0.35 * Q
    areas = 2.0 - 0.3 * Q
    return sc.Dataset({
        'Lorentzian width': sc.DataArray(
            data=sc.array(
                dims=['Q'], values=widths, variances=np.full_like(widths, 1e-4), unit='meV'
            ),
            coords={'Q': sc.array(dims=['Q'], values=Q, unit='1/angstrom')},
        ),
        'Lorentzian area': sc.DataArray(
            data=sc.array(
                dims=['Q'], values=areas, variances=np.full_like(areas, 4e-4), unit='meV'
            ),
            coords={'Q': sc.array(dims=['Q'], values=Q, unit='1/angstrom')},
        ),
    })


def make_parameter_analysis(two_bindings=True):
    bindings = [
        edyn.FitBinding(
            model=sm.Polynomial(
                coefficients=[0.1, 0.35], x_unit='1/angstrom', y_unit='meV', name='Width line'
            ),
            targets='Lorentzian width',
        )
    ]
    if two_bindings:
        bindings.append(
            edyn.FitBinding(
                model=sm.Polynomial(
                    coefficients=[2.0, -0.3], x_unit='1/angstrom', y_unit='meV', name='Area line'
                ),
                targets='Lorentzian area',
            )
        )
    return edyn.ParameterAnalysis(parameters=make_dataset(), bindings=bindings)


@pytest.fixture
def parameter_analysis():
    return make_parameter_analysis()


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

    #############
    # Error paths
    #############

    def test_bumps_outlier_crash_is_reported_helpfully(self, analysis):
        # WHEN BUMPS' own outlier removal indexes past the end of its buffer
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = _bumps_style_index_error()

            # THEN EXPECT the bare IndexError is replaced by something actionable, naming both
            # causes
            with pytest.raises(RuntimeError, match='degenerate') as raised:
                analysis.bayesian.sample(samples=10)
            assert 'short chains' in str(raised.value)
            assert isinstance(raised.value.__cause__, IndexError)

    def test_an_index_error_of_our_own_is_not_relabelled(self, analysis):
        # WHEN the IndexError comes from anywhere but BUMPS, it is a bug here and must not be
        # dressed up as a modelling problem
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = IndexError('list index out of range')

            # THEN EXPECT it propagates untouched
            with pytest.raises(IndexError, match='list index out of range'):
                analysis.bayesian.sample(samples=10)

    def test_parameters_entry_of_the_wrong_type_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(TypeError, match='Parameter objects or labels'):
            analysis.bayesian.sample(samples=10, parameters=[42])

    def test_median_skips_columns_with_no_matching_parameter(self, analysis):
        # WHEN a chain carries a column this analysis knows nothing about
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            results = fake_results(analysis)
            results.param_names = [*results.param_names, 'Parameter_does_not_exist']
            results.draws = np.column_stack([results.draws, np.zeros(results.draws.shape[0])])
            sampler_class.return_value.sample.return_value = results
            analysis.bayesian.sample(samples=10)

        # THEN
        changed = analysis.bayesian.set_parameters_to_median()

        # EXPECT the unknown column is skipped rather than crashing
        assert len(changed) == len(analysis.get_free_parameters())

    def test_load_chain_uses_the_sidecar_when_present(self, analysis, tmp_path):
        # WHEN a chain is saved and reloaded into a *different* analysis, whose unique names differ
        bound_all(analysis)
        with patch(SAMPLER_PATH) as sampler_class:
            saved = fake_results(analysis)
            sampler_class.return_value.sample.return_value = saved
            analysis.bayesian.sample(samples=10)
            analysis.bayesian.save(str(tmp_path / 'chain'))

        fresh = make_analysis()
        bound_all(fresh)

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.load_state.return_value = saved
            fresh.bayesian.load(str(tmp_path / 'chain'))

        # EXPECT the sidecar maps the old unique names onto the new analysis's parameters
        summary = fresh.bayesian.summary()
        assert {entry.name for entry in summary} == {p.name for p in fresh.get_free_parameters()}
        assert all(np.isfinite(entry.value) for entry in summary)

    #############
    # Plot rendering
    #############

    def test_trace_and_corner_render_from_a_chain(self, analysis):
        # WHEN
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.use('Agg')
        bound_all(analysis)
        n_parameters = len(analysis.get_free_parameters())

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis)
            analysis.bayesian.sample(samples=10)

        # THEN EXPECT
        assert len(analysis.bayesian.plot_trace().axes) == n_parameters + 1
        assert len(analysis.bayesian.plot_corner().axes) == n_parameters**2
        plt.close('all')

    #############
    # Extend guards
    #############

    def test_extending_with_a_different_subset_is_refused(self, analysis):
        # WHEN a chain is started over all parameters and then extended over one
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)

            target = analysis.get_free_parameters()[0]

            # THEN EXPECT refused up front, rather than failing obscurely inside BUMPS, which
            # resumes from a stored chain whose width is fixed
            with pytest.warns(UserWarning), pytest.raises(ValueError, match='Cannot extend'):
                analysis.bayesian.extend(additional_samples=10, parameters=[target.name])

    def test_extending_with_the_same_parameters_is_allowed(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            sampler_class.return_value.extend.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.sample(samples=10)

            # THEN EXPECT: does not raise
            analysis.bayesian.extend(additional_samples=10)

    #############
    # Sidecar labels
    #############

    def test_a_subset_run_records_the_same_labels_a_full_run_would(self, analysis):
        # WHEN only one parameter is sampled. Inside the run the others are fixed, so nothing looks
        # ambiguous; the recorded labels must still match what a full run would have written, or
        # the chain cannot be matched up again on reload.
        bound_all(analysis)
        target = analysis.get_free_parameters()[0]

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            with pytest.warns(UserWarning):
                analysis.bayesian.sample(samples=10, parameters=[target.name])

        # EXPECT
        assert analysis.bayesian._saved_labels[
            target.unique_name
        ] == analysis._parameter_labels().label(target)

    def test_extending_after_a_failed_run_is_allowed(self, analysis):
        # WHEN a run built the sampler but died before storing results, so there is a sampler to
        # extend but no chain shape to compare against
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = RuntimeError('died mid-run')
            with pytest.raises(RuntimeError, match='died mid-run'):
                analysis.bayesian.sample(samples=10)

            assert analysis.bayesian.sampler is not None
            assert analysis.bayesian.results is None

            # THEN EXPECT the shape guard steps aside rather than comparing against nothing
            sampler_class.return_value.extend.side_effect = lambda **_k: fake_results(analysis)
            analysis.bayesian.extend(additional_samples=10)

    #############
    # Driven through a ParameterAnalysis
    #############

    def test_refuses_unbounded_parameters(self, parameter_analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='finite bounds'):
            parameter_analysis.bayesian.sample(samples=10)

    def test_binds_one_dataset_per_target(self, parameter_analysis):
        # WHEN
        bound_all_chain(parameter_analysis)
        parameters = parameter_analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            parameter_analysis.bayesian.sample(samples=10)

        # EXPECT
        args, kwargs = sampler_class.call_args
        assert len(args[1]) == 2
        assert len(kwargs['weights']) == 2

    def test_summary_uses_model_qualified_labels(self, parameter_analysis):
        # WHEN
        bound_all_chain(parameter_analysis)
        parameters = parameter_analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            parameter_analysis.bayesian.sample(samples=10)

        # EXPECT
        names = [entry.name for entry in parameter_analysis.bayesian.summary()]
        assert len(set(names)) == len(names)
        assert 'Width line_c0' in names

    def test_restores_parameter_values(self, parameter_analysis):
        # WHEN
        bound_all_chain(parameter_analysis)
        parameters = parameter_analysis._chain_parameters()
        before = [float(p.value) for p in parameters]

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:

            def mutate(**_kwargs):
                for parameter in parameters:
                    parameter.value = float(parameter.value) + 1.0
                return fake_chain_results(parameters)

            sampler_class.return_value.sample.side_effect = mutate
            parameter_analysis.bayesian.sample(samples=10)

        # EXPECT
        assert [float(p.value) for p in parameters] == pytest.approx(before)

    def test_missing_parameters_dataset_raises(self):
        # WHEN
        parameter_analysis = edyn.ParameterAnalysis()

        # THEN EXPECT
        with pytest.raises(ValueError, match='No parameters Dataset'):
            parameter_analysis.bayesian.sample(samples=10)

    def test_missing_bindings_raises(self):
        # WHEN
        parameter_analysis = edyn.ParameterAnalysis(parameters=make_dataset())

        # THEN EXPECT
        with pytest.raises(ValueError, match='No fit bindings'):
            parameter_analysis.bayesian.sample(samples=10)


class TestMultiQPosteriorSampler:
    #############
    # Bounds pre-flight
    #############

    def test_sampling_refuses_unbounded_parameters(self, multi_q_analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='finite bounds'):
            multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

    def test_error_names_parameters_by_q_index(self, multi_q_analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match=r'Gaussian width \(Q_index=0\)'):
            multi_q_analysis.bayesian.check_bounds()

    def test_suggest_bounds_labels_every_q(self, multi_q_analysis):
        # THEN
        suggestions = multi_q_analysis.bayesian.suggest_bounds()

        # EXPECT
        labels = [s.label for s in suggestions]
        assert len(set(labels)) == len(labels)
        assert 'Gaussian area (Q_index=1)' in labels

    #############
    # Simultaneous sampling
    #############

    def test_binds_one_dataset_per_q_index(self, multi_q_analysis):
        # WHEN
        bound_all_chain(multi_q_analysis)
        parameters = multi_q_analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT
        args, kwargs = sampler_class.call_args
        assert len(args[1]) == len(Q_VALUES)
        assert len(args[2]) == len(Q_VALUES)
        assert len(kwargs['weights']) == len(Q_VALUES)

    def test_returns_a_single_result(self, multi_q_analysis):
        # WHEN
        bound_all_chain(multi_q_analysis)
        parameters = multi_q_analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            expected = fake_chain_results(parameters)
            sampler_class.return_value.sample.return_value = expected
            returned = multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT
        assert returned is expected
        assert multi_q_analysis.bayesian.results is expected

    def test_summary_is_labelled_by_q_index(self, multi_q_analysis):
        # WHEN
        bound_all_chain(multi_q_analysis)
        parameters = multi_q_analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT
        names = [entry.name for entry in multi_q_analysis.bayesian.summary()]
        assert len(set(names)) == len(names)
        assert all('Q_index=' in name for name in names)

    def test_refreshes_every_convolver_before_sampling(self, multi_q_analysis):
        # WHEN
        bound_all_chain(multi_q_analysis)
        parameters = multi_q_analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            for analysis1d in multi_q_analysis.analysis_list:
                analysis1d._convolver_is_dirty = True
            multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT the sampler sees the same prepared convolvers a simultaneous fit would
        assert all(not a._convolver_is_dirty for a in multi_q_analysis.analysis_list)

    def test_uses_a_multifitter(self, multi_q_analysis):
        # WHEN

        # EXPECT
        assert isinstance(multi_q_analysis.fitter, MultiFitter)
        assert len(multi_q_analysis.fitter.fit_object) == len(Q_VALUES)

    #############
    # Independent sampling
    #############

    def test_returns_one_result_per_q_index(self, multi_q_analysis):
        # WHEN
        for analysis1d in multi_q_analysis.analysis_list:
            for parameter in analysis1d.get_free_parameters():
                parameter.min = float(parameter.value) - 5.0
                parameter.max = float(parameter.value) + 5.0

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_chain_results(
                multi_q_analysis.analysis_list[0].get_free_parameters()
            )
            results = multi_q_analysis.bayesian.sample(fit_method='independent', samples=10)

        # EXPECT
        assert isinstance(results, list)
        assert len(results) == len(Q_VALUES)

    def test_single_q_index_returns_one_result(self, multi_q_analysis):
        # WHEN
        target = multi_q_analysis.analysis_list[1]
        for parameter in target.get_free_parameters():
            parameter.min = float(parameter.value) - 5.0
            parameter.max = float(parameter.value) + 5.0

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_chain_results(
                target.get_free_parameters()
            )
            result = multi_q_analysis.bayesian.sample(
                fit_method='independent', Q_index=1, samples=10
            )

        # EXPECT
        assert not isinstance(result, list)
        assert result is target.bayesian.results

    def test_invalid_q_index_raises(self, multi_q_analysis):
        # THEN EXPECT
        with pytest.raises((ValueError, IndexError)):
            multi_q_analysis.bayesian.sample(fit_method='independent', Q_index=99, samples=10)

    #############
    # Validation
    #############

    def test_invalid_fit_method_raises(self, multi_q_analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='Invalid fit method'):
            multi_q_analysis.bayesian.sample(fit_method='nonsense')

    def test_missing_q_values_raises(self):
        # WHEN
        multi_q_analysis = edyn.Analysis(display_name='Empty')

        # THEN EXPECT
        with pytest.raises(ValueError, match='No Q values available'):
            multi_q_analysis.bayesian.sample()

    #############
    # Predictive plot
    #############

    def test_predictive_is_not_supported_for_multiple_datasets(self, multi_q_analysis):
        # WHEN
        bound_all_chain(multi_q_analysis)
        parameters = multi_q_analysis._chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # THEN EXPECT
        with pytest.raises(NotImplementedError, match='single dataset only'):
            multi_q_analysis.bayesian.plot_posterior_predictive()

    #############
    # Discoverability
    #############

    def test_operations_needing_one_chain_point_at_the_per_q_chains(self, multi_q_analysis):
        # WHEN sampling independently, the chains live on the Analysis1d objects, not here
        remaining = iter(multi_q_analysis.analysis_list)
        for analysis1d in multi_q_analysis.analysis_list:
            for parameter in analysis1d.get_free_parameters():
                parameter.min = float(parameter.value) - 5.0
                parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_chain_results(
                next(remaining).get_free_parameters()
            )
            multi_q_analysis.bayesian.sample(fit_method='independent', samples=10)

        # THEN EXPECT anything that genuinely needs a single chain says where the chains
        # actually are, rather than claiming none exist
        with pytest.raises(RuntimeError, match='analysis_list'):
            multi_q_analysis.bayesian.plot_posterior_predictive()

    def test_untouched_analysis_still_reports_no_samples(self, multi_q_analysis):
        # THEN EXPECT the plain message when nothing has been sampled anywhere
        with pytest.raises(RuntimeError, match='No posterior samples yet'):
            multi_q_analysis.bayesian.summary()

    #############
    # Aggregating the per-Q chains
    #############

    def _sample_independently(self, multi_q_analysis):
        for analysis1d in multi_q_analysis.analysis_list:
            for parameter in analysis1d.get_free_parameters():
                parameter.min = float(parameter.value) - 5.0
                parameter.max = float(parameter.value) + 5.0

        # The Q indices sample in order, and each must get a chain over its own parameters.
        remaining = iter(multi_q_analysis.analysis_list)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_chain_results(
                next(remaining).get_free_parameters()
            )
            multi_q_analysis.bayesian.sample(fit_method='independent', samples=10)

    def test_posterior_results_holds_one_chain_per_q(self, multi_q_analysis):
        # WHEN
        self._sample_independently(multi_q_analysis)

        # EXPECT
        assert len(multi_q_analysis.bayesian.results_per_q) == len(Q_VALUES)
        assert all(result is not None for result in multi_q_analysis.bayesian.results_per_q)

    def test_posterior_results_is_none_before_sampling(self, multi_q_analysis):
        # EXPECT
        assert multi_q_analysis.bayesian.results_per_q is None

    def test_summary_gathers_every_q(self, multi_q_analysis):
        # WHEN
        self._sample_independently(multi_q_analysis)

        # THEN
        summary = multi_q_analysis.bayesian.summary()

        # EXPECT one entry per free parameter per Q, each labelled by its Q index
        expected = sum(len(a.get_free_parameters()) for a in multi_q_analysis.analysis_list)
        names = [entry.name for entry in summary]
        assert len(summary) == expected
        assert len(set(names)) == len(names)
        assert all('Q_index=' in name for name in names)

    def test_median_applies_each_chain_to_its_own_q(self, multi_q_analysis):
        # WHEN
        self._sample_independently(multi_q_analysis)

        # THEN
        changed = multi_q_analysis.bayesian.set_parameters_to_median()

        # EXPECT every Q's parameters are set, from that Q's own chain
        expected = sum(len(a.get_free_parameters()) for a in multi_q_analysis.analysis_list)
        assert len(changed) == expected

    def test_corner_plots_one_q_at_a_time(self, multi_q_analysis):
        # WHEN each Q was sampled separately, no draw pairs one Q with another, so a corner plot
        # can only show one chain at a time
        self._sample_independently(multi_q_analysis)

        # THEN
        figure = multi_q_analysis.bayesian.plot_corner(Q_index=1)

        # EXPECT that Q's own chain, not a combination across Q
        n_parameters = len(multi_q_analysis.analysis_list[1].get_free_parameters())
        assert len(figure.axes) == n_parameters**2

    def test_corner_offers_a_slider_in_a_notebook(self, multi_q_analysis):
        # WHEN
        self._sample_independently(multi_q_analysis)

        # THEN
        with patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=True):
            widget = multi_q_analysis.bayesian.plot_corner()

        # EXPECT a slider over the sampled Q indices, and a panel that actually holds a figure.
        # The obvious way to build this captures nothing and leaves the panel blank beside the
        # slider, so an empty panel is the regression worth guarding. Which mime type arrives
        # depends on the environment: a live kernel renders a PNG, plain pytest only the repr.
        # The figure comes first and the slider sits under it, where plopp puts its controls.
        panel, slider = widget.children
        assert list(slider.options) == list(range(len(Q_VALUES)))
        assert panel.outputs, 'the initial chain was not drawn'
        assert 'Figure' in str(panel.outputs[0]['data'])

        slider.value = 2
        assert panel.outputs, 'changing Q did not redraw'
        assert 'Figure' in str(panel.outputs[0]['data'])

    def test_corner_without_a_notebook_or_q_index_says_what_to_do(self, multi_q_analysis):
        # WHEN
        self._sample_independently(multi_q_analysis)

        # THEN EXPECT it names the sampled Q indices rather than just refusing
        with (
            patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=False),
            pytest.raises(RuntimeError, match=r'sampled Q indices are \[0, 1, 2\]'),
        ):
            multi_q_analysis.bayesian.plot_corner()

    def test_the_slider_only_offers_q_indices_that_were_sampled(self, multi_q_analysis):
        # WHEN only one Q index is sampled
        target = multi_q_analysis.analysis_list[2]
        for parameter in target.get_free_parameters():
            parameter.min = float(parameter.value) - 5.0
            parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_chain_results(
                target.get_free_parameters()
            )
            multi_q_analysis.bayesian.sample(fit_method='independent', Q_index=2, samples=10)

        # THEN
        with patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=True):
            widget = multi_q_analysis.bayesian.plot_corner()

        # EXPECT the slider cannot land on a Q with nothing to draw
        assert list(widget.children[1].options) == [2]

    def test_trace_points_at_the_individual_chains(self, multi_q_analysis):
        # WHEN
        self._sample_independently(multi_q_analysis)

        # THEN EXPECT
        with pytest.raises(RuntimeError, match='no single trace'):
            multi_q_analysis.bayesian.plot_trace()

    def test_a_simultaneous_chain_still_takes_precedence(self, multi_q_analysis):
        # WHEN a simultaneous run follows an independent one
        self._sample_independently(multi_q_analysis)
        bound_all_chain(multi_q_analysis)
        parameters = multi_q_analysis._chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT the single chain is summarized, not the stale per-Q ones
        assert len(multi_q_analysis.bayesian.summary()) == len(parameters)
        multi_q_analysis.bayesian.plot_corner()

    def test_only_the_sampled_q_indices_are_gathered(self, multi_q_analysis):
        # WHEN just one Q index is sampled
        target = multi_q_analysis.analysis_list[1]
        for parameter in target.get_free_parameters():
            parameter.min = float(parameter.value) - 5.0
            parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_chain_results(
                target.get_free_parameters()
            )
            multi_q_analysis.bayesian.sample(fit_method='independent', Q_index=1, samples=10)

        # THEN
        summary = multi_q_analysis.bayesian.summary()

        # EXPECT the unsampled Q indices are passed over rather than breaking the aggregation
        assert len(summary) == len(target.get_free_parameters())
        assert all('Q_index=1' in entry.name for entry in summary)
        assert len(multi_q_analysis.bayesian.set_parameters_to_median()) == len(
            target.get_free_parameters()
        )

    def test_a_simultaneous_chain_serves_the_median_and_the_trace(self, multi_q_analysis):
        # WHEN
        bound_all_chain(multi_q_analysis)
        parameters = multi_q_analysis._chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_chain_results(parameters)
            multi_q_analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT both come from the single chain, with no per-Q gathering involved
        assert len(multi_q_analysis.bayesian.set_parameters_to_median()) == len(parameters)
        assert len(multi_q_analysis.bayesian.plot_trace().axes) == len(parameters) + 1


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
