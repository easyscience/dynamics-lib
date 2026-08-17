# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl
import numpy as np
import pytest

mpl.use('Agg')

import matplotlib.pyplot as plt

from easydynamics.utils.posterior_plotting import corner_with_slider
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
        # THEN
        fig = plot_trace(draws=draws, names=['a', 'b', 'c'])

        # EXPECT
        assert len(fig.axes) == 3

    def test_logp_adds_a_panel(self, draws):
        # THEN
        fig = plot_trace(draws=draws, names=['a', 'b', 'c'], logp=np.zeros(len(draws)))

        # EXPECT
        assert len(fig.axes) == 4
        assert fig.axes[-1].get_ylabel() == 'log-posterior'

    def test_names_label_the_panels(self, draws):
        # THEN
        fig = plot_trace(draws=draws, names=['alpha', 'beta', 'gamma'])

        # EXPECT
        assert [axis.get_ylabel() for axis in fig.axes] == ['alpha', 'beta', 'gamma']

    def test_single_parameter_works(self):
        # THEN
        fig = plot_trace(draws=np.zeros((10, 1)), names=['only'])

        # EXPECT
        assert len(fig.axes) == 1

    def test_mismatched_names_raise(self, draws):
        # THEN EXPECT
        with pytest.raises(ValueError, match='one entry per column'):
            plot_trace(draws=draws, names=['a', 'b'])

    def test_one_dimensional_draws_raise(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='two-dimensional'):
            plot_trace(draws=np.zeros(10), names=['a'])

    def test_labels_carry_units(self, draws):
        # THEN
        fig = plot_trace(draws=draws, names=['a', 'b', 'c'], units=['meV', 'm^2/s', ''])

        # EXPECT the real units are shown, and an empty one is skipped
        assert [axis.get_ylabel() for axis in fig.axes] == ['a (meV)', 'b (m^2/s)', 'c']

    def test_zero_row_draws_raise(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='no samples'):
            plot_trace(draws=np.zeros((0, 2)), names=['a', 'b'])

    def test_zero_column_draws_raise(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='no parameters'):
            plot_trace(draws=np.zeros((5, 0)), names=[])

    def test_a_single_draw_keeps_a_usable_axis(self):
        # THEN
        fig = plot_trace(draws=np.ones((1, 2)), names=['a', 'b'])

        # EXPECT a non-inverted, non-degenerate x range
        left, right = fig.axes[0].get_xlim()
        assert left < right

    def test_mismatched_logp_length_raises(self, draws):
        # THEN EXPECT
        with pytest.raises(ValueError, match='one entry per draw'):
            plot_trace(draws=draws, names=['a', 'b', 'c'], logp=np.zeros(len(draws) - 1))


class TestPlotCorner:
    def test_grid_is_square_in_the_parameter_count(self, draws):
        # THEN
        fig = plot_corner(draws=draws, names=['a', 'b', 'c'])

        # EXPECT
        assert len(fig.axes) == 9

    def test_upper_triangle_is_hidden(self, draws):
        # THEN
        fig = plot_corner(draws=draws, names=['a', 'b', 'c'])

        # EXPECT: 3 hidden panels above the diagonal of a 3x3 grid
        assert sum(not axis.get_visible() for axis in fig.axes) == 3

    def test_mismatched_names_raise(self, draws):
        # THEN EXPECT
        with pytest.raises(ValueError, match='one entry per column'):
            plot_corner(draws=draws, names=['a'])

    def test_diagonal_panel_is_labelled_as_counts(self, draws):
        # THEN
        fig = plot_corner(draws=draws, names=['a', 'b', 'c'])

        # EXPECT the top-left panel says what its vertical axis actually is. It is a histogram, so
        # the parameter is on the x axis and labelling y with the parameter name would be wrong.
        assert fig.axes[0].get_ylabel() == 'counts'

    def test_units_are_appended_to_labels(self, draws):
        # THEN
        fig = plot_corner(draws=draws, names=['a', 'b', 'c'], units=['meV', '', 'dimensionless'])

        # EXPECT the real unit is shown, and empty or dimensionless ones are skipped
        bottom_row = fig.axes[-3:]
        assert bottom_row[0].get_xlabel() == 'a (meV)'
        assert bottom_row[1].get_xlabel() == 'b'
        assert bottom_row[2].get_xlabel() == 'c'

    def test_non_finite_draws_raise_naming_the_column(self, draws):
        # WHEN one column contains a NaN
        draws[5, 1] = np.nan

        # THEN EXPECT a clear error naming that column, not numpy's "range [nan, nan]"
        with pytest.raises(ValueError, match='non-finite') as excinfo:
            plot_corner(draws=draws, names=['a', 'b', 'c'])
        assert ': b.' in str(excinfo.value)

    def test_columns_share_limits_between_histogram_and_hexbin_panels(self, draws):
        # THEN
        fig = plot_corner(draws=draws, names=['a', 'b', 'c'])

        # EXPECT every panel of a column agrees with the diagonal histogram on x-limits, so the
        # ticks line up down the column
        grid = np.array(fig.axes, dtype=object).reshape(3, 3)
        for col in range(3):
            column_limits = [grid[row, col].get_xlim() for row in range(col, 3)]
            assert all(limits == pytest.approx(column_limits[0]) for limits in column_limits)

    #############
    # Scientific notation
    #############

    def test_shared_exponent_is_folded_into_the_label(self):
        # WHEN the values are small enough that matplotlib factors out an exponent, which it parks
        # on top of the axis label
        draws = np.random.default_rng(0).normal(size=(200, 2)) * 1e-8 + 1.15e-8

        # THEN
        fig = plot_corner(draws=draws, names=['D', 'scale'], units=['m^2/s', ''])

        # EXPECT the exponent and the unit share one parenthetical, and the overlapping offset
        # text is hidden
        xlabel = fig.axes[-2].get_xlabel()
        assert xlabel.startswith('D (1e')
        assert 'm^2/s' in xlabel
        assert not fig.axes[-2].xaxis.get_offset_text().get_visible()

    def test_shared_exponent_is_folded_into_the_y_label_too(self):
        # WHEN the values are small enough that the left column's y axes also factor out an
        # exponent
        draws = np.random.default_rng(0).normal(size=(200, 2)) * 1e-8 + 1.15e-8

        # THEN
        fig = plot_corner(draws=draws, names=['D', 'scale'], units=['m^2/s', ''])

        # EXPECT the hexbin panel in the left column folds the exponent into its y label and
        # hides the overlapping offset text
        axis = fig.axes[2]
        ylabel = axis.get_ylabel()
        assert ylabel.startswith('scale (1e')
        assert not axis.yaxis.get_offset_text().get_visible()


class TestCornerWithSlider:
    def test_empty_chains_raise(self):
        # THEN EXPECT a clear refusal rather than an IndexError from an empty selection
        with pytest.raises(ValueError, match='No chains to plot'):
            corner_with_slider({})


class TestPlotPosteriorPredictive:
    def test_returns_a_figure_with_data_and_band(self):
        # WHEN
        x = np.linspace(0.0, 1.0, 25)
        predictions = np.random.default_rng(0).normal(size=(50, 25))

        # THEN
        fig = plot_posterior_predictive(x=x, y=np.zeros(25), predictions=predictions)

        # EXPECT
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        assert 'Data' in labels
        assert any('credible band' in label for label in labels)

    def test_error_bars_are_drawn_when_given(self):
        # WHEN
        x = np.linspace(0.0, 1.0, 10)

        # THEN
        fig = plot_posterior_predictive(
            x=x,
            y=np.zeros(10),
            predictions=np.zeros((5, 10)),
            y_err=np.full(10, 0.1),
        )

        # EXPECT
        assert len(fig.axes[0].containers) == 1

    def test_wrong_prediction_shape_raises(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='predictions must have shape'):
            plot_posterior_predictive(x=np.zeros(10), y=np.zeros(10), predictions=np.zeros((5, 3)))

    @pytest.mark.parametrize('interval', [0.0, 100.0, -5.0])
    def test_invalid_credible_interval_raises(self, interval):
        # THEN EXPECT
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

        # THEN
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

    def test_axis_labels_are_set_when_given(self):
        # THEN
        fig = plot_posterior_predictive(
            x=np.zeros(4),
            y=np.zeros(4),
            predictions=np.zeros((5, 4)),
            xlabel='Energy (meV)',
            ylabel='Intensity',
        )

        # EXPECT
        assert fig.axes[0].get_xlabel() == 'Energy (meV)'
        assert fig.axes[0].get_ylabel() == 'Intensity'
