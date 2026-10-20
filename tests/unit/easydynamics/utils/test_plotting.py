# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import scipp as sc

from easydynamics.utils.plotting import add_x_axis_coord
from easydynamics.utils.plotting import slicerplot_with_residuals


class TestAdd_x_axis_coord:
    @pytest.fixture
    def dataset(self):
        return sc.Dataset(
            data={'width': sc.array(dims=['Q'], values=[0.1, 0.2], unit='meV')},
            coords={'Q': sc.array(dims=['Q'], values=[0.5, 2.0], unit='1/angstrom')},
        )

    @pytest.mark.parametrize(
        'x_axis, expected_values, expected_unit',
        [
            ('Q', [0.5, 2.0], '1/angstrom'),
            ('Q^2', [0.25, 4.0], '1/angstrom**2'),
        ],
        ids=['Q', 'Q_squared'],
    )
    def test_add_x_axis_coord(self, dataset, x_axis, expected_values, expected_unit):
        # THEN
        result = add_x_axis_coord(dataset, x_axis)

        # EXPECT the coordinate is added and the data is unchanged
        assert sc.identical(
            result.coords[x_axis],
            sc.array(dims=['Q'], values=expected_values, unit=expected_unit),
        )
        assert sc.identical(result['width'].data, dataset['width'].data)

    def test_add_x_axis_coord_does_not_modify_input(self, dataset):
        # THEN
        add_x_axis_coord(dataset, 'Q^2')

        # EXPECT
        assert 'Q^2' not in dataset.coords

    def test_add_x_axis_coord_invalid_x_axis_raises(self, dataset):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='x_axis must be one of'):
            add_x_axis_coord(dataset, 'Q^3')


class TestSlicerplot_with_residuals:
    @pytest.fixture
    def datagroup(self):
        return sc.DataGroup(
            Data=sc.DataArray(sc.array(dims=['x'], values=[1.0])),
            Residuals=sc.DataArray(sc.array(dims=['x'], values=[0.0])),
        )

    @pytest.fixture
    def datagroup_with_model(self):
        return sc.DataGroup(
            Data=sc.DataArray(sc.array(dims=['x'], values=[1])),
            Model=sc.DataArray(sc.array(dims=['x'], values=[2])),
            Residuals=sc.DataArray(sc.array(dims=['x'], values=[0])),
        )

    @pytest.mark.parametrize(
        'dg',
        [
            None,
            [],
            {},
            sc.scalar(1),
        ],
        ids=[
            'None',
            'List',
            'Dict',
            'Scalar',
        ],
    )
    def test_slicerplot_with_residuals_invalid_datagroup_raises(self, dg):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match=r'Expected a sc.DataGroup',
        ):
            slicerplot_with_residuals(dg)

    @pytest.mark.parametrize(
        'residuals_key',
        [
            None,
            123,
            [],
        ],
        ids=[
            'None',
            'Integer',
            'List',
        ],
    )
    def test_slicerplot_with_residuals_invalid_residuals_key_raises(
        self,
        residuals_key,
        datagroup,
    ):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='Expected residuals_key to be a string',
        ):
            slicerplot_with_residuals(
                datagroup,
                residuals_key=residuals_key,
            )

    def test_slicerplot_with_residuals_missing_residuals_key_raises(self, datagroup):
        # WHEN THEN EXPECT
        with pytest.raises(
            ValueError,
            match="Residuals key 'NonExistentKey' not found in DataGroup",
        ):
            slicerplot_with_residuals(datagroup, residuals_key='NonExistentKey')

    def test_slicerplot_with_residuals_excludes_residuals_from_main_plot(
        self, datagroup_with_model
    ):
        # WHEN
        mock_sp = MagicMock()
        mock_sp.slicer.keep = ['x']
        mock_sp.slicer.slider_node = MagicMock()
        mock_sp.figure = MagicMock()

        with (
            patch(
                'easydynamics.utils.plotting.SlicerPlot',
                return_value=mock_sp,
            ) as slicer_plot,
            patch('easydynamics.utils.plotting.pp.linefigure', return_value=MagicMock()),
            patch('easydynamics.utils.plotting.pp.widgets.VBar'),
            patch('easydynamics.utils.plotting.slice_dims'),
            patch('easydynamics.utils.plotting.pp.Node'),
        ):
            # THEN
            slicerplot_with_residuals(datagroup_with_model)

        # EXPECT
        plotted_group = slicer_plot.call_args.args[0]

        assert 'Residuals' not in plotted_group
        assert set(plotted_group.keys()) == {'Data', 'Model'}

    def test_slicerplot_with_residuals_hides_top_x_axis(self, datagroup):
        # WHEN
        mock_fig = MagicMock()
        mock_sp = MagicMock()
        mock_sp.slicer.keep = ['x']
        mock_sp.slicer.slider_node = MagicMock()
        mock_sp.figure = mock_fig

        with (
            patch(
                'easydynamics.utils.plotting.SlicerPlot',
                return_value=mock_sp,
            ),
            patch('easydynamics.utils.plotting.pp.linefigure', return_value=MagicMock()),
            patch('easydynamics.utils.plotting.pp.widgets.VBar'),
            patch('easydynamics.utils.plotting.slice_dims'),
            patch('easydynamics.utils.plotting.pp.Node'),
        ):
            # THEN
            slicerplot_with_residuals(datagroup)

        # EXPECT
        mock_fig.ax.set_xlabel.assert_called_once_with('')
        mock_fig.ax.xaxis.label.set_visible.assert_called_once_with(False)
        mock_fig.ax.tick_params.assert_called_once_with(labelbottom=False)
        mock_fig.autoscale.assert_called_once()

    @pytest.mark.parametrize(
        ('figsize', 'expected_figsize'),
        [
            (None, (6.0, 6.0)),
            ((8.0, 5.0), (8.0, 5.0)),
        ],
        ids=['Default', 'Custom'],
    )
    def test_slicerplot_with_residuals_draws_both_panels_in_one_figure(
        self, datagroup, figsize, expected_figsize
    ):
        # WHEN
        mock_sp = MagicMock()
        mock_sp.slicer.keep = ['x']
        kwargs = {} if figsize is None else {'figsize': figsize}

        with (
            patch(
                'easydynamics.utils.plotting.SlicerPlot',
                return_value=mock_sp,
            ) as slicer_plot,
            patch(
                'easydynamics.utils.plotting.pp.linefigure', return_value=MagicMock()
            ) as linefigure,
            patch('easydynamics.utils.plotting.slice_dims'),
            patch('easydynamics.utils.plotting.pp.Node'),
        ):
            # THEN
            slicerplot_with_residuals(datagroup, **kwargs)

        # EXPECT the data and residuals axes belong to one figure and share their x axis
        main_ax = slicer_plot.call_args.kwargs['ax']
        res_ax = linefigure.call_args.kwargs['ax']

        assert main_ax.get_figure() is res_ax.get_figure()
        assert main_ax.get_shared_x_axes().joined(main_ax, res_ax)
        assert tuple(main_ax.get_figure().get_size_inches()) == expected_figsize
        assert 'figsize' not in slicer_plot.call_args.kwargs
        assert linefigure.call_args.kwargs['hide_log_buttons'] is True

    def test_slicerplot_with_residuals_aligns_data_and_residuals_axes(self):
        # WHEN the plopp slicer needs an interactive matplotlib backend, switched in for the
        # test, and the two panels have y tick labels of different widths
        import matplotlib.pyplot as plt

        energy = sc.linspace('energy', -1.0, 1.0, 11, unit='meV')
        Q = sc.array(dims=['Q'], values=[0.5, 1.0], unit='1/angstrom')

        def data_array(values):
            return sc.DataArray(
                sc.array(dims=['Q', 'energy'], values=[values, values]),
                coords={'energy': energy, 'Q': Q},
            )

        datagroup = sc.DataGroup(
            Data=data_array([1000.0 * (i + 1) for i in range(11)]),
            Residuals=data_array([0.1 * (-1) ** i for i in range(11)]),
        )

        plt.switch_backend('module://ipympl.backend_nbagg')
        try:
            # THEN
            fig = slicerplot_with_residuals(datagroup, keep='energy')
            fig.canvas.to_widget()

            # EXPECT the axes span the same x range on screen and in data
            main_ax, res_ax = fig.canvas.fig.axes
            main_box = main_ax.get_window_extent()
            res_box = res_ax.get_window_extent()

            assert main_box.x0 == pytest.approx(res_box.x0)
            assert main_box.x1 == pytest.approx(res_box.x1)
            assert main_ax.get_xlim() == res_ax.get_xlim()
        finally:
            plt.switch_backend('Agg')
