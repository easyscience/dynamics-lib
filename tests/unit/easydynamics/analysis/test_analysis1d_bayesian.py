# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Bayesian sampling on Analysis1d, with the EasyScience Sampler mocked out."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.fitting import AvailableMinimizers

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.gaussian import Gaussian

SAMPLER_PATH = 'easydynamics.analysis.bayesian_sampling.Sampler'


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


@pytest.fixture
def analysis():
    return make_analysis()


class TestFitterExposure:
    def test_fitter_is_built_lazily_and_cached(self, analysis):
        # WHEN
        fitter = analysis.fitter

        # EXPECT
        assert fitter is analysis.fitter
        assert fitter.fit_object is analysis

    def test_fitter_is_rebuilt_when_the_sample_model_changes(self, analysis):
        # WHEN
        original = analysis.fitter
        analysis.sample_model = SampleModel(components=Gaussian(area=1.0))

        # EXPECT
        assert analysis.fitter is not original

    def test_minimizer_can_be_switched_through_the_fitter(self, analysis):
        # WHEN
        analysis.fitter.switch_minimizer(AvailableMinimizers.Bumps)

        # EXPECT
        assert analysis.fitter.minimizer.enum == AvailableMinimizers.Bumps

    def test_fit_uses_the_persistent_fitter(self, analysis):
        # WHEN
        result = analysis.fit()

        # EXPECT
        assert result is analysis._fit_result
        assert np.isfinite(result.reduced_chi2)


class TestBoundsPreflight:
    def test_sampling_refuses_unbounded_parameters(self, analysis):
        # EXPECT
        with pytest.raises(ValueError, match='finite bounds'):
            analysis.sample_posterior(samples=10)

    def test_error_names_the_offending_parameters(self, analysis):
        # EXPECT
        with pytest.raises(ValueError, match='Gaussian area'):
            analysis.check_bounds_for_sampling()

    def test_bounded_parameters_pass(self, analysis):
        # WHEN
        bound_all(analysis)

        # EXPECT: does not raise
        analysis.check_bounds_for_sampling()

    def test_suggest_bounds_covers_the_free_parameters(self, analysis):
        # WHEN
        suggestions = analysis.suggest_bounds()

        # EXPECT
        assert len(suggestions) == len(analysis.get_free_parameters())


class TestSamplePosterior:
    def test_restores_parameter_values_and_minimizer(self, analysis):
        # WHEN
        bound_all(analysis)
        before = [(p.unique_name, p.value) for p in analysis.get_free_parameters()]

        with patch(SAMPLER_PATH) as sampler_class:

            def mutate_then_return(**_kwargs):
                # The real sampler leaves the parameters wherever the last evaluation put them.
                for parameter in analysis.get_free_parameters():
                    parameter.value = float(parameter.value) + 1.0
                return fake_results(analysis)

            sampler_class.return_value.sample.side_effect = mutate_then_return
            analysis.sample_posterior(samples=10, burn=1, thin=1)

        # EXPECT
        after = [(p.unique_name, p.value) for p in analysis.get_free_parameters()]
        assert after == before
        assert analysis.fitter.minimizer.enum == AvailableMinimizers.LMFit_leastsq

    def test_switches_to_bumps_for_the_run(self, analysis):
        # WHEN
        bound_all(analysis)
        seen = []

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: (
                seen.append(analysis.fitter.minimizer.enum),
                fake_results(analysis),
            )[1]
            analysis.sample_posterior(samples=10)

        # EXPECT
        assert seen == [AvailableMinimizers.Bumps]

    def test_restores_the_minimizer_even_when_sampling_raises(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = RuntimeError('boom')
            with pytest.raises(RuntimeError, match='boom'):
                analysis.sample_posterior(samples=10)

        # EXPECT
        assert analysis.fitter.minimizer.enum == AvailableMinimizers.LMFit_leastsq

    def test_forwards_sampling_arguments(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=123, burn=7, thin=3, population=5)

        # EXPECT
        kwargs = sampler_class.return_value.sample.call_args.kwargs
        assert kwargs['samples'] == 123
        assert kwargs['burn'] == 7
        assert kwargs['thin'] == 3
        assert kwargs['population'] == 5

    def test_stores_the_result(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            expected = fake_results(analysis)
            sampler_class.return_value.sample.return_value = expected
            returned = analysis.sample_posterior(samples=10)

        # EXPECT
        assert returned is expected
        assert analysis.posterior_result is expected

    def test_warns_when_the_posterior_piles_up_against_a_bound(self, analysis):
        # WHEN a parameter's draws span its whole allowed range
        bound_all(analysis)
        parameters = analysis.get_free_parameters()
        draws = np.tile([float(p.value) for p in parameters], (500, 1))
        draws[:, 0] = np.linspace(parameters[0].min, parameters[0].max, 500)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis, values=draws)

            # EXPECT
            with pytest.warns(UserWarning, match='piled up'):
                analysis.sample_posterior(samples=10)

    def test_does_not_warn_when_the_posterior_is_well_inside(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis)

            # EXPECT
            with warnings_as_errors():
                analysis.sample_posterior(samples=10)


class TestParameterSubset:
    def test_holds_other_parameters_fixed_during_the_run(self, analysis):
        # WHEN
        bound_all(analysis)
        target = analysis.get_free_parameters()[0]
        seen = {}

        with patch(SAMPLER_PATH) as sampler_class:

            def record(**_kwargs):
                seen['free'] = [p.unique_name for p in analysis.get_free_parameters()]
                return fake_results(analysis)

            sampler_class.return_value.sample.side_effect = record
            with pytest.warns(UserWarning, match='Holding these parameters fixed'):
                analysis.sample_posterior(samples=10, parameters=[target.name])

        # EXPECT
        assert seen['free'] == [target.unique_name]

    def test_restores_the_fixed_flags_afterwards(self, analysis):
        # WHEN
        bound_all(analysis)
        before = [(p.unique_name, p.fixed) for p in analysis.get_all_parameters()]
        target = analysis.get_free_parameters()[0]

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            with pytest.warns(UserWarning):
                analysis.sample_posterior(samples=10, parameters=[target])

        # EXPECT
        assert [(p.unique_name, p.fixed) for p in analysis.get_all_parameters()] == before

    def test_unknown_parameter_name_raises(self, analysis):
        # WHEN
        bound_all(analysis)

        # EXPECT
        with pytest.raises(ValueError, match='No free parameter named'):
            analysis.sample_posterior(samples=10, parameters=['not a parameter'])

    def test_non_list_parameters_raises(self, analysis):
        # EXPECT
        with pytest.raises(TypeError, match='must be a list'):
            analysis.sample_posterior(samples=10, parameters='Gaussian area')

    def test_empty_parameter_list_raises(self, analysis):
        # EXPECT
        with pytest.raises(ValueError, match='at least one parameter'):
            analysis.sample_posterior(samples=10, parameters=[])


class TestSamplerCaching:
    def test_sampler_is_reused_between_runs(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=10)
            analysis.sample_posterior(samples=10)

        # EXPECT the data is bound once, not per run
        assert sampler_class.call_count == 1

    def test_changing_the_q_index_rebuilds_the_sampler(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=10)
            analysis.Q_index = 0
            analysis.sample_posterior(samples=10)

        # EXPECT the Sampler binds its data at construction, so it must be rebuilt
        assert sampler_class.call_count == 2

    def test_binds_the_same_data_the_fit_uses(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=10)

        # EXPECT
        expected_x, expected_y, expected_w = analysis._get_sampling_data()
        args, kwargs = sampler_class.call_args
        assert np.array_equal(args[1], expected_x)
        assert np.array_equal(args[2], expected_y)
        assert np.array_equal(kwargs['weights'], expected_w)


class TestExtendAndPersistence:
    def test_extend_without_a_chain_raises(self, analysis):
        # EXPECT
        with pytest.raises(RuntimeError, match='No chain to extend'):
            analysis.extend_sampling()

    def test_extend_delegates_to_the_sampler(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            sampler_class.return_value.extend.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=10)
            analysis.extend_sampling(additional_samples=42, thin=2)

        # EXPECT
        kwargs = sampler_class.return_value.extend.call_args.kwargs
        assert kwargs['additional_samples'] == 42
        assert kwargs['thin'] == 2

    def test_save_without_a_chain_raises(self, analysis):
        # EXPECT
        with pytest.raises(RuntimeError, match='No chain to save'):
            analysis.save_chain('somewhere')

    def test_save_writes_the_parameter_name_sidecar(self, analysis, tmp_path):
        # WHEN
        import json

        bound_all(analysis)
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=10)
            analysis.save_chain(str(tmp_path / 'chain'))

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

            # EXPECT
            with pytest.warns(UserWarning, match='No parameter-name sidecar'):
                analysis.load_chain(str(tmp_path / 'missing'))


class TestResults:
    def test_summary_without_sampling_raises(self, analysis):
        # EXPECT
        with pytest.raises(RuntimeError, match='No posterior samples yet'):
            analysis.posterior_summary()

    def test_summary_uses_parameter_names_and_units(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=10)

        # EXPECT
        summary = analysis.posterior_summary()
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
            analysis.sample_posterior(samples=10)

        changed = analysis.set_parameters_to_posterior_median()

        # EXPECT
        assert len(changed) == len(parameters)
        assert [float(p.value) for p in parameters] == pytest.approx(expected)

    def test_median_without_sampling_raises(self, analysis):
        # EXPECT
        with pytest.raises(RuntimeError, match='No posterior samples yet'):
            analysis.set_parameters_to_posterior_median()


class TestPlots:
    def test_predictive_rejects_a_bad_draw_count(self, analysis):
        # WHEN
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(analysis)
            analysis.sample_posterior(samples=10)

        # EXPECT
        with pytest.raises(ValueError, match='positive integer'):
            analysis.plot_posterior_predictive(n_draws=0)

    def test_predictive_restores_parameter_values(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis.get_free_parameters()
        draws = np.tile([float(p.value) + 0.5 for p in parameters], (20, 1))

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis, values=draws)
            analysis.sample_posterior(samples=10)

        before = [float(p.value) for p in parameters]
        analysis.plot_posterior_predictive(n_draws=5)

        # EXPECT
        assert [float(p.value) for p in parameters] == pytest.approx(before)

    def test_plots_without_sampling_raise(self, analysis):
        # EXPECT
        with pytest.raises(RuntimeError):
            analysis.plot_trace()
        with pytest.raises(RuntimeError):
            analysis.plot_corner()


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


class TestErrorPaths:
    def test_bumps_outlier_crash_is_reported_as_a_degeneracy(self, analysis):
        # WHEN BUMPS' own outlier removal indexes past the end of its buffer, which happens when
        # chains scatter because the model is not identifiable
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = IndexError(
                'index 71 is out of bounds for axis 0 with size 40'
            )

            # EXPECT the bare IndexError is replaced by something actionable
            with pytest.raises(RuntimeError, match='degenerate'):
                analysis.sample_posterior(samples=10)

    def test_parameters_entry_of_the_wrong_type_raises(self, analysis):
        # EXPECT
        with pytest.raises(TypeError, match='Parameter objects or parameter names'):
            analysis.sample_posterior(samples=10, parameters=[42])

    def test_median_skips_columns_with_no_matching_parameter(self, analysis):
        # WHEN a chain carries a column this analysis knows nothing about
        bound_all(analysis)

        with patch(SAMPLER_PATH) as sampler_class:
            results = fake_results(analysis)
            results.param_names = [*results.param_names, 'Parameter_does_not_exist']
            results.draws = np.column_stack([results.draws, np.zeros(results.draws.shape[0])])
            sampler_class.return_value.sample.return_value = results
            analysis.sample_posterior(samples=10)

        # EXPECT the unknown column is skipped rather than crashing
        changed = analysis.set_parameters_to_posterior_median()
        assert len(changed) == len(analysis.get_free_parameters())

    def test_load_chain_uses_the_sidecar_when_present(self, analysis, tmp_path):
        # WHEN a chain is saved and reloaded into a *different* analysis, whose unique names differ
        bound_all(analysis)
        with patch(SAMPLER_PATH) as sampler_class:
            saved = fake_results(analysis)
            sampler_class.return_value.sample.return_value = saved
            analysis.sample_posterior(samples=10)
            analysis.save_chain(str(tmp_path / 'chain'))

        fresh = make_analysis()
        bound_all(fresh)
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.load_state.return_value = saved
            fresh.load_chain(str(tmp_path / 'chain'))

        # EXPECT the sidecar maps the old unique names onto the new analysis's parameters
        summary = fresh.posterior_summary()
        assert {entry.name for entry in summary} == {p.name for p in fresh.get_free_parameters()}
        assert all(np.isfinite(entry.value) for entry in summary)


class TestPlotRendering:
    def test_trace_and_corner_render_from_a_chain(self, analysis):
        # WHEN
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.use('Agg')
        bound_all(analysis)
        n_parameters = len(analysis.get_free_parameters())

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(analysis)
            analysis.sample_posterior(samples=10)

        # EXPECT
        assert len(analysis.plot_trace().axes) == n_parameters + 1
        assert len(analysis.plot_corner().axes) == n_parameters**2
        plt.close('all')
