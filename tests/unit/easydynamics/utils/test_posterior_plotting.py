# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl
import numpy as np
import pytest

mpl.use('Agg')

import matplotlib.pyplot as plt

from easydynamics.utils.posterior_plotting import plot_corner
from easydynamics.utils.posterior_plotting import plot_posterior_predictive
from easydynamics.utils.posterior_plotting import plot_trace


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def draws():
    return np.random.default_rng(0).normal(size=(200, 3))


class TestPlotTrace:
    def test_one_panel_per_parameter(self, draws):
        # WHEN
        fig = plot_trace(draws=draws, names=['a', 'b', 'c'])

        # EXPECT
        assert len(fig.axes) == 3

    def test_logp_adds_a_panel(self, draws):
        # WHEN
        fig = plot_trace(draws=draws, names=['a', 'b', 'c'], logp=np.zeros(len(draws)))

        # EXPECT
        assert len(fig.axes) == 4
        assert fig.axes[-1].get_ylabel() == 'log-posterior'

    def test_names_label_the_panels(self, draws):
        # WHEN
        fig = plot_trace(draws=draws, names=['alpha', 'beta', 'gamma'])

        # EXPECT
        assert [axis.get_ylabel() for axis in fig.axes] == ['alpha', 'beta', 'gamma']

    def test_single_parameter_works(self):
        # WHEN
        fig = plot_trace(draws=np.zeros((10, 1)), names=['only'])

        # EXPECT
        assert len(fig.axes) == 1

    def test_mismatched_names_raise(self, draws):
        # EXPECT
        with pytest.raises(ValueError, match='one entry per column'):
            plot_trace(draws=draws, names=['a', 'b'])

    def test_one_dimensional_draws_raise(self):
        # EXPECT
        with pytest.raises(ValueError, match='two-dimensional'):
            plot_trace(draws=np.zeros(10), names=['a'])


class TestPlotCorner:
    def test_grid_is_square_in_the_parameter_count(self, draws):
        # WHEN
        fig = plot_corner(draws=draws, names=['a', 'b', 'c'])

        # EXPECT
        assert len(fig.axes) == 9

    def test_upper_triangle_is_hidden(self, draws):
        # WHEN
        fig = plot_corner(draws=draws, names=['a', 'b', 'c'])

        # EXPECT: 3 hidden panels above the diagonal of a 3x3 grid
        assert sum(not axis.get_visible() for axis in fig.axes) == 3

    def test_mismatched_names_raise(self, draws):
        # EXPECT
        with pytest.raises(ValueError, match='one entry per column'):
            plot_corner(draws=draws, names=['a'])


class TestPlotPosteriorPredictive:
    def test_returns_a_figure_with_data_and_band(self):
        # WHEN
        x = np.linspace(0.0, 1.0, 25)
        predictions = np.random.default_rng(0).normal(size=(50, 25))

        fig = plot_posterior_predictive(x=x, y=np.zeros(25), predictions=predictions)

        # EXPECT
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert 'Data' in labels
        assert any('credible band' in label for label in labels)

    def test_error_bars_are_drawn_when_given(self):
        # WHEN
        x = np.linspace(0.0, 1.0, 10)

        fig = plot_posterior_predictive(
            x=x,
            y=np.zeros(10),
            predictions=np.zeros((5, 10)),
            y_err=np.full(10, 0.1),
        )

        # EXPECT
        assert len(fig.axes[0].containers) == 1

    def test_wrong_prediction_shape_raises(self):
        # EXPECT
        with pytest.raises(ValueError, match='predictions must have shape'):
            plot_posterior_predictive(x=np.zeros(10), y=np.zeros(10), predictions=np.zeros((5, 3)))

    @pytest.mark.parametrize('interval', [0.0, 100.0, -5.0])
    def test_invalid_credible_interval_raises(self, interval):
        # EXPECT
        with pytest.raises(ValueError, match='credible_interval'):
            plot_posterior_predictive(
                x=np.zeros(4),
                y=np.zeros(4),
                predictions=np.zeros((5, 4)),
                credible_interval=interval,
            )

    def test_band_widens_with_the_credible_interval(self):
        # WHEN
        x = np.linspace(0.0, 1.0, 8)
        predictions = np.random.default_rng(0).normal(size=(400, 8))

        narrow = plot_posterior_predictive(
            x=x, y=np.zeros(8), predictions=predictions, credible_interval=50.0
        )
        wide = plot_posterior_predictive(
            x=x, y=np.zeros(8), predictions=predictions, credible_interval=95.0
        )

        # EXPECT
        narrow_span = narrow.axes[0].collections[0].get_paths()[0].get_extents().height
        wide_span = wide.axes[0].collections[0].get_paths()[0].get_extents().height
        assert wide_span > narrow_span
