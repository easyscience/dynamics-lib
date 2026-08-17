# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Integration tests running real BUMPS DREAM chains through Analysis and ParameterAnalysis.

Slow by nature, and with the same two BUMPS options switched off as the single-Q integration tests:
its burn-point trimming, which re-runs a convergence detector on every call, and its outlier
removal, which indexes past the end of its own buffer on chains as short as these.
"""

import warnings
from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest
import scipp as sc

mpl.use('Agg')

import easydynamics as edyn
import easydynamics.sample_model as sm

Q_VALUES = [0.5, 1.0, 1.5]
NOISE = 0.02
TRUE_AREA = 2.0

SAMPLE_KWARGS = {
    'samples': 2000,
    'burn': 100,
    'thin': 2,
    # 'trim': BUMPS' burn-point detector re-runs on every call and is not worth paying
    # for here. 'outliers': its outlier removal indexes past the end of its own buffer on
    # chains this short, which has failed in CI; the sampling itself is unaffected.
    'sampler_kwargs': {'trim': False, 'outliers': 'none'},
}


def true_width(q):
    return 0.8 + 0.4 * q**2


def build_analysis():
    energy_values = np.linspace(-5.0, 5.0, 40)
    rng = np.random.default_rng(0)
    rows = []
    for q in Q_VALUES:
        width = true_width(q)
        row = TRUE_AREA / (width * np.sqrt(2 * np.pi))
        row = row * np.exp(-0.5 * (energy_values / width) ** 2)
        rows.append(row + rng.normal(0.0, NOISE, size=row.shape))
    observed = np.vstack(rows)

    experiment = edyn.Experiment(
        data=sc.DataArray(
            data=sc.array(
                dims=['Q', 'energy'],
                values=observed,
                variances=np.full_like(observed, NOISE**2),
            ),
            coords={
                'Q': sc.array(dims=['Q'], values=Q_VALUES, unit='1/Angstrom'),
                'energy': sc.array(dims=['energy'], values=energy_values, unit='meV'),
            },
        )
    )
    return edyn.Analysis(
        display_name='MultiQIntegration',
        experiment=experiment,
        sample_model=sm.SampleModel(components=sm.Gaussian(area=TRUE_AREA, width=1.0)),
        instrument_model=sm.InstrumentModel(),
    )


@pytest.fixture(scope='module')
def simultaneously_sampled():
    analysis = build_analysis()
    analysis.fit(fit_method='simultaneous')
    analysis.bayesian.suggest_bounds().apply()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        analysis.bayesian.sample(fit_method='simultaneous', **SAMPLE_KWARGS)
    return analysis


@pytest.fixture(scope='module')
def independently_sampled():
    """One independent DREAM run shared by every test that only reads the per-Q chains."""
    analysis = build_analysis()
    analysis.fit(fit_method='independent')
    for analysis1d in analysis.analysis_list:
        analysis1d.bayesian.suggest_bounds().apply()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = analysis.bayesian.sample(fit_method='independent', **SAMPLE_KWARGS)
    return analysis, results


class TestSimultaneousChain:
    def test_chain_covers_every_q_index(self, simultaneously_sampled):
        # THEN
        results = simultaneously_sampled.bayesian.results

        # EXPECT one column per free parameter across all Q, in one chain
        assert results.draws.shape[1] == len(simultaneously_sampled._chain_parameters())
        assert results.draws.shape[1] == 3 * len(Q_VALUES)

    def test_summary_labels_are_unique_and_q_qualified(self, simultaneously_sampled):
        # THEN
        names = [entry.name for entry in simultaneously_sampled.bayesian.summary()]

        # EXPECT
        assert len(set(names)) == len(names)
        assert all('Q_index=' in name for name in names)

    @pytest.mark.parametrize('q_index', range(len(Q_VALUES)))
    def test_posterior_recovers_the_true_width_at_each_q(self, simultaneously_sampled, q_index):
        # THEN
        entry = simultaneously_sampled.bayesian.summary()[f'Gaussian width (Q_index={q_index})']

        # EXPECT the truth within a few posterior standard deviations. A 68% interval is not used
        # here: it excludes the truth about a third of the time for a single noise realization.
        spread = max(entry.minus, entry.plus)
        assert abs(entry.median - true_width(Q_VALUES[q_index])) < 4 * spread

    def test_sampling_leaves_the_fitted_values_untouched(self):
        # WHEN
        analysis = build_analysis()
        analysis.fit(fit_method='simultaneous')
        analysis.bayesian.suggest_bounds().apply()
        before = [float(p.value) for p in analysis._chain_parameters()]

        # THEN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis.bayesian.sample(fit_method='simultaneous', **SAMPLE_KWARGS)

        # EXPECT
        after = [float(p.value) for p in analysis._chain_parameters()]
        assert after == pytest.approx(before)

    def test_plots_render(self, simultaneously_sampled):
        # WHEN
        import matplotlib.pyplot as plt

        n_parameters = len(simultaneously_sampled._chain_parameters())

        # THEN
        trace = simultaneously_sampled.bayesian.plot_trace()
        corner = simultaneously_sampled.bayesian.plot_corner()

        # EXPECT
        assert len(trace.axes) == n_parameters + 1
        assert len(corner.axes) == n_parameters**2
        plt.close('all')


class TestIndependentChains:
    def test_one_chain_per_q_index(self, independently_sampled):
        # THEN
        analysis, results = independently_sampled

        # EXPECT
        assert len(results) == len(Q_VALUES)
        for analysis1d, result in zip(analysis.analysis_list, results, strict=True):
            assert result.draws.shape[1] == len(analysis1d.get_free_parameters())

    def test_independent_and_simultaneous_agree_on_the_widths(
        self, simultaneously_sampled, independently_sampled
    ):
        # THEN the same data sampled per-Q is compared with the single simultaneous chain
        analysis, _ = independently_sampled

        # EXPECT both routes land on the same widths, since nothing is shared across Q here
        for q_index, analysis1d in enumerate(analysis.analysis_list):
            independent = analysis1d.bayesian.summary()['Gaussian width']
            simultaneous = simultaneously_sampled.bayesian.summary()[
                f'Gaussian width (Q_index={q_index})'
            ]
            spread = max(independent.minus, independent.plus, simultaneous.plus)
            assert abs(independent.median - simultaneous.median) < 4 * spread


class TestIndependentChainWidgets:
    def test_corner_slider_renders_from_the_real_chains(self, independently_sampled):
        # WHEN
        analysis, _ = independently_sampled

        # THEN
        with patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=True):
            widget = analysis.bayesian.plot_corner()

        # EXPECT every real chain pre-rendered behind the slider, and moving the slider swapping
        # the stored renderings rather than drawing anything new
        image, slider = widget.children
        assert list(slider.options) == list(range(len(Q_VALUES)))
        assert bytes(image.value).startswith(b'\x89PNG')
        first_bytes = image.value
        slider.value = 1
        assert image.value != first_bytes
        slider.value = 0
        assert image.value == first_bytes

    def test_predictive_slider_renders_from_the_real_chains(self, independently_sampled):
        # WHEN the plopp slicer needs an interactive matplotlib backend, switched in for the test
        import matplotlib.pyplot as plt

        analysis, _ = independently_sampled
        plt.switch_backend('module://ipympl.backend_nbagg')
        try:
            # THEN
            with patch('easydynamics.analysis.posterior_sampling._in_notebook', return_value=True):
                fig = analysis.bayesian.plot_posterior_predictive(n_draws=10)

            # EXPECT a plopp figure whose one slider spans the sampled Q values, labelled like
            # the single-Q predictive plot
            controls = list(fig.bottom_bar[0].controls.values())
            assert len(controls) == 1
            assert controls[0].slider.min == 0
            assert controls[0].slider.max == len(Q_VALUES) - 1
            assert fig.ax.get_ylabel().startswith('Intensity')
        finally:
            plt.switch_backend('Agg')

    def test_predictive_q_index_plots_one_q_from_its_own_chain(self, independently_sampled):
        # WHEN
        import matplotlib.pyplot as plt

        analysis, _ = independently_sampled

        # THEN
        figure = analysis.bayesian.plot_posterior_predictive(Q_index=1, n_draws=10)

        # EXPECT the single-Q matplotlib figure, with its data and credible band
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert 'Data' in labels
        assert any('credible band' in label for label in labels)
        plt.close('all')


class TestParameterAnalysisChain:
    def test_recovers_a_straight_line_through_the_widths(self):
        # WHEN the fitted widths are themselves fitted against a model of their Q dependence
        q = np.array(Q_VALUES)
        widths = true_width(q)
        dataset = sc.Dataset({
            'Gaussian width': sc.DataArray(
                data=sc.array(
                    dims=['Q'],
                    values=widths,
                    variances=np.full_like(widths, 0.01**2),
                    unit='meV',
                ),
                coords={'Q': sc.array(dims=['Q'], values=q, unit='1/angstrom')},
            )
        })
        model = sm.Polynomial(
            coefficients=[0.8, 0.0, 0.4], x_unit='1/angstrom', y_unit='meV', name='Width model'
        )
        analysis = edyn.ParameterAnalysis(
            parameters=dataset,
            bindings=edyn.FitBinding(model=model, targets='Gaussian width'),
        )
        analysis.fit()
        # The linear coefficient sits at exactly zero with a vanishing uncertainty, so the sigma
        # rule has no scale to work from and flags it rather than inventing one. absolute_floor
        # supplies the scale the data cannot; the asserts guard that this setup really leaves
        # every coefficient bounded before sampling.
        flagged = analysis.bayesian.suggest_bounds().needing_attention
        assert [s.label for s in flagged] == ['Width model_c1']
        analysis.bayesian.suggest_bounds(absolute_floor=1.0).apply()
        assert not analysis.bayesian.suggest_bounds().needing_attention

        # THEN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = analysis.bayesian.sample(**SAMPLE_KWARGS)

        # EXPECT the posterior recovers the generating polynomial within a few posterior
        # standard deviations (a 68% interval would exclude the truth too often to be strict),
        # with a column per coefficient and a readable, collision-free summary
        summary = analysis.bayesian.summary()
        for name, truth in (
            ('Width model_c0', 0.8),
            ('Width model_c1', 0.0),
            ('Width model_c2', 0.4),
        ):
            entry = summary[name]
            spread = max(entry.minus, entry.plus)
            assert abs(entry.median - truth) < 4 * spread
        assert results.draws.shape[1] == len(analysis._chain_parameters())
        names = [entry.name for entry in summary]
        assert len(set(names)) == len(names)


class TestAggregatedIndependentChains:
    def test_summary_gathers_the_real_per_q_chains(self, independently_sampled):
        # THEN
        analysis, _ = independently_sampled
        summary = analysis.bayesian.summary()

        # EXPECT one table covering every Q, and the widths still recovered
        assert len(summary) == sum(len(a.get_free_parameters()) for a in analysis.analysis_list)
        for q_index in range(len(Q_VALUES)):
            entry = summary[f'Gaussian width (Q_index={q_index})']
            spread = max(entry.minus, entry.plus)
            assert abs(entry.median - true_width(Q_VALUES[q_index])) < 4 * spread

    def test_median_applies_each_chain_to_its_own_q(self, independently_sampled):
        # WHEN the fixture is module-scoped, so the values moved here are restored afterwards
        analysis, _ = independently_sampled
        parameters = [p for a in analysis.analysis_list for p in a.get_free_parameters()]
        saved_values = [(p, float(p.value)) for p in parameters]

        try:
            # THEN
            changed = analysis.bayesian.set_parameters_to_median()

            # EXPECT every Q's parameters land on that Q's own median
            assert len(changed) == sum(
                len(a.get_free_parameters()) for a in analysis.analysis_list
            )
            summary = analysis.bayesian.summary()
            for entry in summary:
                assert entry.value == pytest.approx(entry.median, rel=1e-6)
        finally:
            for parameter, value in saved_values:
                parameter.value = value
