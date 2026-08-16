# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Bayesian sampling on the 2D Analysis, with the EasyScience Sampler mocked out."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest
import scipp as sc

mpl.use('Agg')

import easydynamics as edyn
import easydynamics.sample_model as sm

SAMPLER_PATH = 'easydynamics.analysis.posterior_sampling.Sampler'
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
    for parameter in analysis._chain_parameters():
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
        # THEN
        parameters = analysis._chain_parameters()

        # EXPECT one copy of each per-Q parameter, with no duplicates
        assert len(parameters) == sum(len(a.get_free_parameters()) for a in analysis.analysis_list)
        assert len({p.unique_name for p in parameters}) == len(parameters)

    def test_labels_are_qualified_by_q_index(self, analysis):
        # THEN
        labels = [analysis._parameter_labels().label(p) for p in analysis._chain_parameters()]

        # EXPECT every per-Q copy is distinguishable, which the bare name would not be
        assert len(set(labels)) == len(labels)
        assert 'Gaussian width (Q_index=0)' in labels
        assert 'Gaussian width (Q_index=2)' in labels

    def test_bare_names_would_collide(self, analysis):
        # THEN
        names = [p.name for p in analysis._chain_parameters()]

        # EXPECT the collision the Q-qualified label exists to solve
        assert len(set(names)) < len(names)


class TestBoundsPreflight:
    def test_sampling_refuses_unbounded_parameters(self, analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='finite bounds'):
            analysis.bayesian.sample(fit_method='simultaneous', samples=10)

    def test_error_names_parameters_by_q_index(self, analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match=r'Gaussian width \(Q_index=0\)'):
            analysis.bayesian.check_bounds()

    def test_suggest_bounds_labels_every_q(self, analysis):
        # THEN
        suggestions = analysis.bayesian.suggest_bounds()

        # EXPECT
        labels = [s.label for s in suggestions]
        assert len(set(labels)) == len(labels)
        assert 'Gaussian area (Q_index=1)' in labels


class TestSimultaneousSampling:
    def test_binds_one_dataset_per_q_index(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT
        args, kwargs = sampler_class.call_args
        assert len(args[1]) == len(Q_VALUES)
        assert len(args[2]) == len(Q_VALUES)
        assert len(kwargs['weights']) == len(Q_VALUES)

    def test_returns_a_single_result(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            expected = fake_results(parameters)
            sampler_class.return_value.sample.return_value = expected
            returned = analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT
        assert returned is expected
        assert analysis.bayesian.results is expected

    def test_summary_is_labelled_by_q_index(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT
        names = [entry.name for entry in analysis.bayesian.summary()]
        assert len(set(names)) == len(names)
        assert all('Q_index=' in name for name in names)

    def test_refreshes_every_convolver_before_sampling(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._chain_parameters()

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            for analysis1d in analysis.analysis_list:
                analysis1d._convolver_is_dirty = True
            analysis.bayesian.sample(fit_method='simultaneous', samples=10)

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

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                analysis.analysis_list[0].get_free_parameters()
            )
            results = analysis.bayesian.sample(fit_method='independent', samples=10)

        # EXPECT
        assert isinstance(results, list)
        assert len(results) == len(Q_VALUES)

    def test_single_q_index_returns_one_result(self, analysis):
        # WHEN
        target = analysis.analysis_list[1]
        for parameter in target.get_free_parameters():
            parameter.min = float(parameter.value) - 5.0
            parameter.max = float(parameter.value) + 5.0

        # THEN
        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                target.get_free_parameters()
            )
            result = analysis.bayesian.sample(fit_method='independent', Q_index=1, samples=10)

        # EXPECT
        assert not isinstance(result, list)
        assert result is target.bayesian.results

    def test_invalid_q_index_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises((ValueError, IndexError)):
            analysis.bayesian.sample(fit_method='independent', Q_index=99, samples=10)


class TestSamplePosteriorValidation:
    def test_invalid_fit_method_raises(self, analysis):
        # THEN EXPECT
        with pytest.raises(ValueError, match='Invalid fit method'):
            analysis.bayesian.sample(fit_method='nonsense')

    def test_missing_q_values_raises(self):
        # WHEN
        analysis = edyn.Analysis(display_name='Empty')

        # THEN EXPECT
        with pytest.raises(ValueError, match='No Q values available'):
            analysis.bayesian.sample()


class TestPredictivePlot:
    def test_predictive_is_not_supported_for_multiple_datasets(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # THEN EXPECT
        with pytest.raises(NotImplementedError, match='single dataset only'):
            analysis.bayesian.plot_posterior_predictive()


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
        labels = [analysis._parameter_labels().label(p) for p in analysis._chain_parameters()]

        # EXPECT the short form, not 'Gaussian width (Q_index=0)'
        assert 'Gaussian width' in labels
        assert not any('Q_index=' in label for label in labels)

    def test_parameter_from_outside_the_analysis_keeps_its_name(self, analysis):
        # WHEN a parameter belongs to no Q index of this analysis
        from easyscience.variable import Parameter

        stranger = Parameter(name='Gaussian width', value=1.0)

        # EXPECT it is returned unqualified rather than mislabelled
        assert analysis._parameter_labels().label(stranger) == 'Gaussian width'


class TestIndependentSamplingDiscoverability:
    def test_operations_needing_one_chain_point_at_the_per_q_chains(self, analysis):
        # WHEN sampling independently, the chains live on the Analysis1d objects, not here
        remaining = iter(analysis.analysis_list)
        for analysis1d in analysis.analysis_list:
            for parameter in analysis1d.get_free_parameters():
                parameter.min = float(parameter.value) - 5.0
                parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                next(remaining).get_free_parameters()
            )
            analysis.bayesian.sample(fit_method='independent', samples=10)

        # THEN EXPECT anything that genuinely needs a single chain says where the chains
        # actually are, rather than claiming none exist
        with pytest.raises(RuntimeError, match='analysis_list'):
            analysis.bayesian.plot_posterior_predictive()

    def test_untouched_analysis_still_reports_no_samples(self, analysis):
        # THEN EXPECT the plain message when nothing has been sampled anywhere
        with pytest.raises(RuntimeError, match='No posterior samples yet'):
            analysis.bayesian.summary()


class TestSharedParameterLabels:
    def test_a_parameter_shared_across_q_is_not_tied_to_one_index(self):
        # WHEN a diffusion model contributes global parameters, the same objects appear at every Q
        energy_values = np.linspace(-5.0, 5.0, 15)
        rows = [2.0 * np.exp(-0.5 * (energy_values / 1.2) ** 2) for _ in Q_VALUES]
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
        analysis = edyn.Analysis(
            display_name='Shared',
            experiment=experiment,
            sample_model=sm.SampleModel(
                components=sm.ComponentCollection(components=[sm.DeltaFunction(area=0.2)]),
                diffusion_models=sm.BrownianTranslationalDiffusion(
                    name='Brownian', diffusion_coefficient=2.4e-9, scale=0.5
                ),
            ),
            instrument_model=sm.InstrumentModel(),
        )

        # THEN
        owners = analysis._parameter_owner_index()
        shared = [p for p in analysis._chain_parameters() if p.unique_name not in owners]

        # EXPECT the shared parameters are left out of the owner map, since no single Q owns them,
        # and so keep their plain names rather than being labelled with an arbitrary Q
        assert shared, 'expected the diffusion model to contribute parameters shared across Q'
        for parameter in shared:
            assert analysis._parameter_labels().label(parameter) == parameter.name


class TestAggregatingPerQChains:
    def _sample_independently(self, analysis):
        for analysis1d in analysis.analysis_list:
            for parameter in analysis1d.get_free_parameters():
                parameter.min = float(parameter.value) - 5.0
                parameter.max = float(parameter.value) + 5.0

        # The Q indices sample in order, and each must get a chain over its own parameters.
        remaining = iter(analysis.analysis_list)

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                next(remaining).get_free_parameters()
            )
            analysis.bayesian.sample(fit_method='independent', samples=10)

    def test_posterior_results_holds_one_chain_per_q(self, analysis):
        # WHEN
        self._sample_independently(analysis)

        # EXPECT
        assert len(analysis.bayesian.results_per_q) == len(Q_VALUES)
        assert all(result is not None for result in analysis.bayesian.results_per_q)

    def test_posterior_results_is_none_before_sampling(self, analysis):
        # EXPECT
        assert analysis.bayesian.results_per_q is None

    def test_summary_gathers_every_q(self, analysis):
        # WHEN
        self._sample_independently(analysis)

        # THEN
        summary = analysis.bayesian.summary()

        # EXPECT one entry per free parameter per Q, each labelled by its Q index
        expected = sum(len(a.get_free_parameters()) for a in analysis.analysis_list)
        names = [entry.name for entry in summary]
        assert len(summary) == expected
        assert len(set(names)) == len(names)
        assert all('Q_index=' in name for name in names)

    def test_median_applies_each_chain_to_its_own_q(self, analysis):
        # WHEN
        self._sample_independently(analysis)

        # THEN
        changed = analysis.bayesian.set_parameters_to_median()

        # EXPECT every Q's parameters are set, from that Q's own chain
        expected = sum(len(a.get_free_parameters()) for a in analysis.analysis_list)
        assert len(changed) == expected

    def test_corner_plots_one_q_at_a_time(self, analysis):
        # WHEN each Q was sampled separately, no draw pairs one Q with another, so a corner plot
        # can only show one chain at a time
        self._sample_independently(analysis)

        # THEN
        figure = analysis.bayesian.plot_corner(Q_index=1)

        # EXPECT that Q's own chain, not a combination across Q
        n_parameters = len(analysis.analysis_list[1].get_free_parameters())
        assert len(figure.axes) == n_parameters**2

    def test_corner_offers_a_slider_in_a_notebook(self, analysis):
        # WHEN
        self._sample_independently(analysis)

        # THEN
        with patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=True):
            widget = analysis.bayesian.plot_corner()

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

    def test_corner_without_a_notebook_or_q_index_says_what_to_do(self, analysis):
        # WHEN
        self._sample_independently(analysis)

        # THEN EXPECT it names the sampled Q indices rather than just refusing
        with (
            patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=False),
            pytest.raises(RuntimeError, match=r'sampled Q indices are \[0, 1, 2\]'),
        ):
            analysis.bayesian.plot_corner()

    def test_the_slider_only_offers_q_indices_that_were_sampled(self, analysis):
        # WHEN only one Q index is sampled
        target = analysis.analysis_list[2]
        for parameter in target.get_free_parameters():
            parameter.min = float(parameter.value) - 5.0
            parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                target.get_free_parameters()
            )
            analysis.bayesian.sample(fit_method='independent', Q_index=2, samples=10)

        # THEN
        with patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=True):
            widget = analysis.bayesian.plot_corner()

        # EXPECT the slider cannot land on a Q with nothing to draw
        assert list(widget.children[1].options) == [2]

    def test_trace_points_at_the_individual_chains(self, analysis):
        # WHEN
        self._sample_independently(analysis)

        # THEN EXPECT
        with pytest.raises(RuntimeError, match='no single trace'):
            analysis.bayesian.plot_trace()

    def test_a_simultaneous_chain_still_takes_precedence(self, analysis):
        # WHEN a simultaneous run follows an independent one
        self._sample_independently(analysis)
        bound_all(analysis)
        parameters = analysis._chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT the single chain is summarized, not the stale per-Q ones
        assert len(analysis.bayesian.summary()) == len(parameters)
        analysis.bayesian.plot_corner()

    def test_only_the_sampled_q_indices_are_gathered(self, analysis):
        # WHEN just one Q index is sampled
        target = analysis.analysis_list[1]
        for parameter in target.get_free_parameters():
            parameter.min = float(parameter.value) - 5.0
            parameter.max = float(parameter.value) + 5.0

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.side_effect = lambda **_k: fake_results(
                target.get_free_parameters()
            )
            analysis.bayesian.sample(fit_method='independent', Q_index=1, samples=10)

        # THEN
        summary = analysis.bayesian.summary()

        # EXPECT the unsampled Q indices are passed over rather than breaking the aggregation
        assert len(summary) == len(target.get_free_parameters())
        assert all('Q_index=1' in entry.name for entry in summary)
        assert len(analysis.bayesian.set_parameters_to_median()) == len(
            target.get_free_parameters()
        )

    def test_a_simultaneous_chain_serves_the_median_and_the_trace(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.bayesian.sample(fit_method='simultaneous', samples=10)

        # EXPECT both come from the single chain, with no per-Q gathering involved
        assert len(analysis.bayesian.set_parameters_to_median()) == len(parameters)
        assert len(analysis.bayesian.plot_trace().axes) == len(parameters) + 1
