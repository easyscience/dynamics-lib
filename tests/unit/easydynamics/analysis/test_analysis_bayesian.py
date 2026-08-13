# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Bayesian sampling on the 2D Analysis, with the EasyScience Sampler mocked out."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc

import easydynamics as edyn
import easydynamics.sample_model as sm

SAMPLER_PATH = 'easydynamics.analysis.bayesian_sampling.Sampler'
Q_VALUES = [0.5, 1.0, 1.5]


def make_analysis():
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


def bound_all(analysis, half_width=5.0):
    for parameter in analysis._get_chain_parameters():
        parameter.min = float(parameter.value) - half_width
        parameter.max = float(parameter.value) + half_width


def fake_results(parameters, n_draws=50):
    draws = np.tile([float(p.value) for p in parameters], (n_draws, 1))
    return SimpleNamespace(
        draws=draws,
        param_names=[p.unique_name for p in parameters],
        logp=np.zeros(n_draws),
        state=MagicMock(Ngen=10, Npop=4),
    )


@pytest.fixture
def analysis():
    return make_analysis()


class TestChainParameters:
    def test_union_covers_every_q_index(self, analysis):
        # WHEN
        parameters = analysis._get_chain_parameters()

        # EXPECT one copy of each per-Q parameter, with no duplicates
        assert len(parameters) == sum(len(a.get_free_parameters()) for a in analysis.analysis_list)
        assert len({p.unique_name for p in parameters}) == len(parameters)

    def test_labels_are_qualified_by_q_index(self, analysis):
        # WHEN
        labels = [analysis.parameter_label(p) for p in analysis._get_chain_parameters()]

        # EXPECT every per-Q copy is distinguishable, which the bare name would not be
        assert len(set(labels)) == len(labels)
        assert 'Gaussian width (Q_index=0)' in labels
        assert 'Gaussian width (Q_index=2)' in labels

    def test_bare_names_would_collide(self, analysis):
        # WHEN
        names = [p.name for p in analysis._get_chain_parameters()]

        # EXPECT the collision the Q-qualified label exists to solve
        assert len(set(names)) < len(names)


class TestBoundsPreflight:
    def test_sampling_refuses_unbounded_parameters(self, analysis):
        # EXPECT
        with pytest.raises(ValueError, match='finite bounds'):
            analysis.sample_posterior(fit_method='simultaneous', samples=10)

    def test_error_names_parameters_by_q_index(self, analysis):
        # EXPECT
        with pytest.raises(ValueError, match=r'Gaussian width \(Q_index=0\)'):
            analysis.check_bounds_for_sampling()

    def test_suggest_bounds_labels_every_q(self, analysis):
        # WHEN
        suggestions = analysis.suggest_bounds()

        # EXPECT
        labels = [s.label for s in suggestions]
        assert len(set(labels)) == len(labels)
        assert 'Gaussian area (Q_index=1)' in labels


class TestSimultaneousSampling:
    def test_binds_one_dataset_per_q_index(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.sample_posterior(fit_method='simultaneous', samples=10)

        # EXPECT
        args, kwargs = sampler_class.call_args
        assert len(args[1]) == len(Q_VALUES)
        assert len(args[2]) == len(Q_VALUES)
        assert len(kwargs['weights']) == len(Q_VALUES)

    def test_returns_a_single_result(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            expected = fake_results(parameters)
            sampler_class.return_value.sample.return_value = expected
            returned = analysis.sample_posterior(fit_method='simultaneous', samples=10)

        # EXPECT
        assert returned is expected
        assert analysis.posterior_result is expected

    def test_summary_is_labelled_by_q_index(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.sample_posterior(fit_method='simultaneous', samples=10)

        # EXPECT
        names = [entry.name for entry in analysis.posterior_summary()]
        assert len(set(names)) == len(names)
        assert all('Q_index=' in name for name in names)

    def test_refreshes_every_convolver_before_sampling(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            for analysis1d in analysis.analysis_list:
                analysis1d._convolver_is_dirty = True
            analysis.sample_posterior(fit_method='simultaneous', samples=10)

        # EXPECT the sampler sees the same prepared convolvers a simultaneous fit would
        assert all(not a._convolver_is_dirty for a in analysis.analysis_list)

    def test_uses_a_multifitter(self, analysis):
        # WHEN
        from easyscience.fitting.multi_fitter import MultiFitter

        # EXPECT
        assert isinstance(analysis.fitter, MultiFitter)
        assert len(analysis.fitter.fit_object) == len(Q_VALUES)


class TestIndependentSampling:
    def test_returns_one_result_per_q_index(self, analysis):
        # WHEN
        for analysis1d in analysis.analysis_list:
            for parameter in analysis1d.get_free_parameters():
                parameter.min = float(parameter.value) - 5.0
                parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                analysis.analysis_list[0].get_free_parameters()
            )
            results = analysis.sample_posterior(fit_method='independent', samples=10)

        # EXPECT
        assert isinstance(results, list)
        assert len(results) == len(Q_VALUES)

    def test_single_q_index_returns_one_result(self, analysis):
        # WHEN
        target = analysis.analysis_list[1]
        for parameter in target.get_free_parameters():
            parameter.min = float(parameter.value) - 5.0
            parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                target.get_free_parameters()
            )
            result = analysis.sample_posterior(fit_method='independent', Q_index=1, samples=10)

        # EXPECT
        assert not isinstance(result, list)
        assert result is target.posterior_result

    def test_invalid_q_index_raises(self, analysis):
        # EXPECT
        with pytest.raises((ValueError, IndexError)):
            analysis.sample_posterior(fit_method='independent', Q_index=99, samples=10)


class TestSamplePosteriorValidation:
    def test_invalid_fit_method_raises(self, analysis):
        # EXPECT
        with pytest.raises(ValueError, match='Invalid fit method'):
            analysis.sample_posterior(fit_method='nonsense')

    def test_missing_q_values_raises(self):
        # WHEN
        analysis = edyn.Analysis(display_name='Empty')

        # EXPECT
        with pytest.raises(ValueError, match='No Q values available'):
            analysis.sample_posterior()


class TestPredictivePlot:
    def test_predictive_is_not_supported_for_multiple_datasets(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.sample_posterior(fit_method='simultaneous', samples=10)

        # EXPECT
        with pytest.raises(NotImplementedError, match='single dataset only'):
            analysis.plot_posterior_predictive()


class TestParameterLabelEdgeCases:
    def test_single_q_analysis_keeps_plain_names(self):
        # WHEN there is only one Q index, nothing needs disambiguating
        energy_values = np.linspace(-5.0, 5.0, 15)
        intensity = 2.0 * np.exp(-0.5 * (energy_values / 1.2) ** 2)
        experiment = edyn.Experiment(
            data=sc.DataArray(
                data=sc.array(
                    dims=['Q', 'energy'],
                    values=intensity[None, :],
                    variances=np.full_like(intensity, 0.01)[None, :],
                ),
                coords={
                    'Q': sc.array(dims=['Q'], values=[1.0], unit='1/Angstrom'),
                    'energy': sc.array(dims=['energy'], values=energy_values, unit='meV'),
                },
            )
        )
        analysis = edyn.Analysis(
            display_name='SingleQ',
            experiment=experiment,
            sample_model=sm.SampleModel(components=sm.Gaussian(area=2.0, width=1.0)),
            instrument_model=sm.InstrumentModel(),
        )

        # THEN
        labels = [analysis.parameter_label(p) for p in analysis._get_chain_parameters()]

        # EXPECT the short form, not 'Gaussian width (Q_index=0)'
        assert 'Gaussian width' in labels
        assert not any('Q_index=' in label for label in labels)

    def test_parameter_from_outside_the_analysis_keeps_its_name(self, analysis):
        # WHEN a parameter belongs to no Q index of this analysis
        from easyscience.variable import Parameter

        stranger = Parameter(name='Gaussian width', value=1.0)

        # EXPECT it is returned unqualified rather than mislabelled
        assert analysis.parameter_label(stranger) == 'Gaussian width'
