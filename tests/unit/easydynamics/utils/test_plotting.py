# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import scipp as sc

from easydynamics.utils.plotting import slicerplot_with_residuals


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
