# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Integration tests running real BUMPS DREAM chains through Analysis and ParameterAnalysis.

Slow by nature, and with the same two BUMPS options switched off as the single-Q integration tests:
its burn-point trimming, which re-runs a convergence detector on every call, and its outlier
removal, which indexes past the end of its own buffer on chains as short as these.
"""

import warnings

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
    analysis.suggest_bounds().apply()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        analysis.sample_posterior(fit_method='simultaneous', **SAMPLE_KWARGS)
    return analysis


class TestSimultaneousChain:
    def test_chain_covers_every_q_index(self, simultaneously_sampled):
        # EXPECT one column per free parameter across all Q, in one chain
        results = simultaneously_sampled.posterior_result
        assert results.draws.shape[1] == len(simultaneously_sampled._get_chain_parameters())
        assert results.draws.shape[1] == 3 * len(Q_VALUES)

    def test_summary_labels_are_unique_and_q_qualified(self, simultaneously_sampled):
        # WHEN
        names = [entry.name for entry in simultaneously_sampled.posterior_summary()]

        # EXPECT
        assert len(set(names)) == len(names)
        assert all('Q_index=' in name for name in names)

    @pytest.mark.parametrize('q_index', range(len(Q_VALUES)))
    def test_posterior_recovers_the_true_width_at_each_q(self, simultaneously_sampled, q_index):
        # WHEN
        entry = simultaneously_sampled.posterior_summary()[f'Gaussian width (Q_index={q_index})']

        # EXPECT the truth within a few posterior standard deviations. A 68% interval is not used
        # here: it excludes the truth about a third of the time for a single noise realization.
        spread = max(entry.minus, entry.plus)
        assert abs(entry.median - true_width(Q_VALUES[q_index])) < 4 * spread

    def test_sampling_leaves_the_fitted_values_untouched(self):
        # WHEN
        analysis = build_analysis()
        analysis.fit(fit_method='simultaneous')
        analysis.suggest_bounds().apply()
        before = [float(p.value) for p in analysis._get_chain_parameters()]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis.sample_posterior(fit_method='simultaneous', **SAMPLE_KWARGS)

        # EXPECT
        after = [float(p.value) for p in analysis._get_chain_parameters()]
        assert after == pytest.approx(before)

    def test_plots_render(self, simultaneously_sampled):
        # WHEN
        import matplotlib.pyplot as plt

        n_parameters = len(simultaneously_sampled._get_chain_parameters())
        trace = simultaneously_sampled.plot_trace()
        corner = simultaneously_sampled.plot_corner()

        # EXPECT
        assert len(trace.axes) == n_parameters + 1
        assert len(corner.axes) == n_parameters**2
        plt.close('all')


class TestIndependentChains:
    def test_one_chain_per_q_index(self):
        # WHEN
        analysis = build_analysis()
        analysis.fit(fit_method='independent')
        for analysis1d in analysis.analysis_list:
            analysis1d.suggest_bounds().apply()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = analysis.sample_posterior(fit_method='independent', **SAMPLE_KWARGS)

        # EXPECT
        assert len(results) == len(Q_VALUES)
        for analysis1d, result in zip(analysis.analysis_list, results, strict=True):
            assert result.draws.shape[1] == len(analysis1d.get_free_parameters())

    def test_independent_and_simultaneous_agree_on_the_widths(self, simultaneously_sampled):
        # WHEN the same data is sampled per-Q instead of all at once
        analysis = build_analysis()
        analysis.fit(fit_method='independent')
        for analysis1d in analysis.analysis_list:
            analysis1d.suggest_bounds().apply()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis.sample_posterior(fit_method='independent', **SAMPLE_KWARGS)

        # EXPECT both routes land on the same widths, since nothing is shared across Q here
        for q_index, analysis1d in enumerate(analysis.analysis_list):
            independent = analysis1d.posterior_summary()['Gaussian width']
            simultaneous = simultaneously_sampled.posterior_summary()[
                f'Gaussian width (Q_index={q_index})'
            ]
            spread = max(independent.minus, independent.plus, simultaneous.plus)
            assert abs(independent.median - simultaneous.median) < 4 * spread


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
        # supplies the scale the data cannot.
        flagged = analysis.suggest_bounds().needing_attention
        assert [s.label for s in flagged] == ['Width model_c1']

        analysis.suggest_bounds(absolute_floor=1.0).apply()
        assert not analysis.suggest_bounds().needing_attention

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = analysis.sample_posterior(**SAMPLE_KWARGS)

        # EXPECT a column per polynomial coefficient, and a readable summary
        assert results.draws.shape[1] == len(analysis._get_chain_parameters())
        names = [entry.name for entry in analysis.posterior_summary()]
        assert len(set(names)) == len(names)


class TestAggregatedIndependentChains:
    def test_summary_gathers_the_real_per_q_chains(self):
        # WHEN each Q is sampled on its own
        analysis = build_analysis()
        analysis.fit(fit_method='independent')
        for analysis1d in analysis.analysis_list:
            analysis1d.suggest_bounds().apply()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis.sample_posterior(fit_method='independent', **SAMPLE_KWARGS)

        # THEN
        summary = analysis.posterior_summary()

        # EXPECT one table covering every Q, and the widths still recovered
        assert len(summary) == sum(len(a.get_free_parameters()) for a in analysis.analysis_list)
        for q_index in range(len(Q_VALUES)):
            entry = summary[f'Gaussian width (Q_index={q_index})']
            spread = max(entry.minus, entry.plus)
            assert abs(entry.median - true_width(Q_VALUES[q_index])) < 4 * spread

    def test_median_applies_each_chain_to_its_own_q(self):
        # WHEN
        analysis = build_analysis()
        analysis.fit(fit_method='independent')
        for analysis1d in analysis.analysis_list:
            analysis1d.suggest_bounds().apply()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis.sample_posterior(fit_method='independent', **SAMPLE_KWARGS)

        changed = analysis.set_parameters_to_posterior_median()

        # EXPECT every Q's parameters land on that Q's own median
        assert len(changed) == sum(len(a.get_free_parameters()) for a in analysis.analysis_list)
        summary = analysis.posterior_summary()
        for entry in summary:
            assert entry.value == pytest.approx(entry.median, rel=1e-6)
