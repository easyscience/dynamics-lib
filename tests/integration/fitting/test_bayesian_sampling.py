# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Integration tests running real BUMPS DREAM chains through Analysis1d.

These are slow by nature. Two BUMPS options are switched off deliberately: its burn-point trimming,
which re-runs a convergence detector on every call, and its outlier removal, which indexes past the
end of its own buffer on chains as short as these. Neither affects the sampling itself.
"""

import warnings

import matplotlib as mpl
import numpy as np
import pytest
import scipp as sc

mpl.use('Agg')

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.gaussian import Gaussian

TRUE_AREA = 9.0
TRUE_WIDTH = 1.2
NOISE = 0.05

# Keep the chains short enough to stay usable in CI; long enough to locate the peak.
SAMPLE_KWARGS = {
    'samples': 2000,
    'burn': 100,
    'thin': 2,
    # 'trim': BUMPS' burn-point detector re-runs on every call and is not worth paying
    # for here. 'outliers': its outlier removal indexes past the end of its own buffer on
    # chains this short, which has failed in CI; the sampling itself is unaffected.
    'sampler_kwargs': {'trim': False, 'outliers': 'none'},
}


def build_analysis():
    energy_values = np.linspace(-5.0, 5.0, 60)
    truth = TRUE_AREA / (TRUE_WIDTH * np.sqrt(2 * np.pi))
    truth = truth * np.exp(-0.5 * (energy_values / TRUE_WIDTH) ** 2)
    observed = truth + np.random.default_rng(0).normal(0.0, NOISE, size=truth.shape)

    data = sc.array(
        dims=['Q', 'energy'],
        values=observed[None, :],
        variances=np.full_like(observed, NOISE**2)[None, :],
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
        display_name='BayesianIntegration',
        experiment=experiment,
        sample_model=SampleModel(
            components=Gaussian(area=TRUE_AREA, width=TRUE_WIDTH, center=0.0)
        ),
        instrument_model=InstrumentModel(),
        Q_index=0,
    )
    # The energy offset shifts the spectrum exactly as the Gaussian centre does. Leaving both free
    # makes the model unidentifiable, which no amount of sampling can repair.
    analysis.instrument_model.fix_energy_offset(Q_index=0)
    return analysis


@pytest.fixture(scope='module')
def sampled_analysis():
    analysis = build_analysis()
    analysis.fit()
    analysis.suggest_bounds().apply()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        analysis.sample_posterior(**SAMPLE_KWARGS)
    return analysis


class TestRealChain:
    def test_chain_has_one_column_per_free_parameter(self, sampled_analysis):
        # EXPECT
        results = sampled_analysis.posterior_result
        assert results.draws.shape[1] == len(sampled_analysis.get_free_parameters())
        assert results.draws.shape[0] > 0

    @pytest.mark.parametrize(
        ('name', 'truth'),
        [('Gaussian area', TRUE_AREA), ('Gaussian width', TRUE_WIDTH)],
    )
    def test_posterior_recovers_the_true_parameters(self, sampled_analysis, name, truth):
        # WHEN
        entry = sampled_analysis.posterior_summary()[name]

        # EXPECT the truth sits within a few posterior standard deviations of the median. A 68%
        # interval is deliberately not used: it excludes the truth about a third of the time for
        # any single noise realization, which would make this test flaky rather than strict.
        spread = max(entry.minus, entry.plus)
        assert abs(entry.median - truth) < 4 * spread

    def test_summary_is_reported_under_parameter_names_and_units(self, sampled_analysis):
        # WHEN
        summary = sampled_analysis.posterior_summary()

        # EXPECT
        assert {entry.name for entry in summary} == {
            p.name for p in sampled_analysis.get_free_parameters()
        }
        assert all(entry.unit == 'meV' for entry in summary)

    def test_sampling_leaves_the_fitted_values_untouched(self):
        # WHEN
        analysis = build_analysis()
        analysis.fit()
        analysis.suggest_bounds().apply()
        before = [float(p.value) for p in analysis.get_free_parameters()]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis.sample_posterior(**SAMPLE_KWARGS)

        # EXPECT
        after = [float(p.value) for p in analysis.get_free_parameters()]
        assert after == pytest.approx(before)

    def test_extend_grows_the_chain(self, sampled_analysis):
        # WHEN
        before = int(sampled_analysis.posterior_result.state.Ngen)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            extended = sampled_analysis.extend_sampling(
                additional_samples=500, thin=2, sampler_kwargs={'trim': False, 'outliers': 'none'}
            )

        # EXPECT
        assert int(extended.state.Ngen) > before

    def test_save_and_load_round_trip_keeps_parameter_identity(self, sampled_analysis, tmp_path):
        # WHEN
        prefix = str(tmp_path / 'chain')
        sampled_analysis.save_chain(prefix)

        fresh = build_analysis()
        fresh.fit()
        fresh.suggest_bounds().apply()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fresh.load_chain(prefix)

        # EXPECT the reloaded chain is reported under real names, not internal unique names
        summary = fresh.posterior_summary()
        assert {entry.name for entry in summary} == {p.name for p in fresh.get_free_parameters()}
        assert all(np.isfinite(entry.value) for entry in summary)

    def test_subset_sampling_produces_a_single_column(self):
        # WHEN
        analysis = build_analysis()
        analysis.fit()
        analysis.suggest_bounds().apply()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = analysis.sample_posterior(parameters=['Gaussian width'], **SAMPLE_KWARGS)

        # EXPECT
        assert results.draws.shape[1] == 1
        assert analysis.posterior_summary().entries[0].name == 'Gaussian width'

    def test_plots_render(self, sampled_analysis):
        # WHEN
        import matplotlib.pyplot as plt

        n_parameters = len(sampled_analysis.get_free_parameters())
        trace = sampled_analysis.plot_trace()
        corner = sampled_analysis.plot_corner()
        predictive = sampled_analysis.plot_posterior_predictive(n_draws=20)

        # EXPECT
        assert len(trace.axes) == n_parameters + 1
        assert len(corner.axes) == n_parameters**2
        assert len(predictive.axes) == 1
        plt.close('all')

    def test_posterior_median_is_close_to_the_least_squares_fit(self):
        # WHEN
        analysis = build_analysis()
        analysis.fit()
        analysis.suggest_bounds().apply()
        fitted = {p.name: float(p.value) for p in analysis.get_free_parameters()}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis.sample_posterior(**SAMPLE_KWARGS)
        summary = analysis.posterior_summary()

        # EXPECT the two agree within the posterior's own uncertainty, since with flat priors the
        # maximum-likelihood point sits inside the bulk of the posterior
        for entry in summary:
            spread = max(entry.minus, entry.plus)
            assert abs(entry.median - fitted[entry.name]) < 5 * spread
