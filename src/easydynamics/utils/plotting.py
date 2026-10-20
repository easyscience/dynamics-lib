# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

import plopp as pp
import scipp as sc
from plopp.backends.matplotlib.figure import WidgetFigure
from plopp.backends.matplotlib.utils import make_figure
from plopp.plotting._slicer import SlicerPlot
from plopp.plotting._slicer import _maybe_reduce_dim
from plopp.widgets import slice_dims

# x-axes that Q-dependent parameters can be plotted against, each mapped to the function
# computing it from Q.
Q_AXIS_TRANSFORMS: dict[str, Callable[[sc.Variable], sc.Variable]] = {
    'Q': lambda Q: Q,
    'Q^2': lambda Q: Q**2,
}


def add_x_axis_coord(data: sc.Dataset, x_axis: str) -> sc.Dataset:
    """
    Add an x_axis coordinate computed from the Q coordinate, for plotting against it.

    Only the coordinate is transformed; the values at each Q are unchanged.

    Parameters
    ----------
    data : sc.Dataset
        Dataset with a 'Q' coordinate.
    x_axis : str
        Name of the axis to add, one of the keys of ``Q_AXIS_TRANSFORMS``.

    Returns
    -------
    sc.Dataset
        A copy of data with the x_axis coordinate added.

    Raises
    ------
    ValueError
        If x_axis is not one of the supported axes.
    """
    if x_axis not in Q_AXIS_TRANSFORMS:
        raise ValueError(f"x_axis must be one of {list(Q_AXIS_TRANSFORMS)}, got '{x_axis}'.")
    return data.assign_coords({x_axis: Q_AXIS_TRANSFORMS[x_axis](data.coords['Q'])})


def slicerplot_with_residuals(
    dg: sc.DataGroup,
    *,
    residuals_key: str = 'Residuals',
    keep: list[str] | str | None = None,
    operation: str = 'sum',
    **kwargs: object,
) -> WidgetFigure:
    """
    Create a SlicerPlot with an additional subplot for residuals.

    This function is called internally by ``Analysis.plot_data_and_model`` and
    ``Analysis1d.plot_data_and_model``. It can also be used directly with any ``sc.DataGroup`` that
    contains a residuals array.

    Examples
    --------
    **Plotting data, model, and residuals from a DataGroup**

    ```python
    import scipp as sc
    import easydynamics as edyn

    dg = sc.DataGroup({
        'Data': my_data,
        'Model': my_model,
        'Residuals': my_residuals,
    })
    fig = edyn.slicerplot_with_residuals(dg, residuals_key='Residuals', keep='energy')
    ```

    Parameters
    ----------
    dg : sc.DataGroup
        DataGroup containing the data to plot. Must include a key for residuals.
    residuals_key : str, default='Residuals'
        Key in the DataGroup that contains the residuals data.
    keep : list[str] | str | None, default=None
        Dimensions to keep in the SlicerPlot. Passed to SlicerPlot.
    operation : str, default='sum'
        Operation to apply when reducing the residuals data. Passed to SlicerPlot.
    **kwargs : object
        Additional keyword arguments passed to SlicerPlot.

    Returns
    -------
    WidgetFigure
        A figure containing the SlicerPlot and the residuals subplot.

    Raises
    ------
    TypeError
        If dg is not a sc.DataGroup or if residuals_key is not a string.
    ValueError
        If residuals_key is not found in the DataGroup.
    """

    if not isinstance(dg, sc.DataGroup):
        raise TypeError(f'Expected a sc.DataGroup, got {type(dg)}')

    if not isinstance(residuals_key, str):
        raise TypeError(f'Expected residuals_key to be a string, got {type(residuals_key)}')

    if residuals_key not in dg:
        raise ValueError(f"Residuals key '{residuals_key}' not found in DataGroup")

    # Draw the data and the residuals as two axes of one Matplotlib figure with a shared x axis.
    # Two separate figures are laid out independently, so their axes only line up when their y
    # tick labels and y labels happen to be equally wide.
    figsize = kwargs.pop('figsize', None)
    mpl_fig = make_figure(figsize=(6.0, 6.0) if figsize is None else figsize)
    main_ax, res_ax = mpl_fig.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )

    sp = SlicerPlot(
        {k: da for k, da in dg.items() if k != residuals_key},
        keep=keep,
        operation=operation,
        ax=main_ax,
        **kwargs,
    )

    other_dims = [dim for dim in dg.dims if dim not in sp.slicer.keep]

    slice_res = slice_dims(pp.Node(dg[residuals_key]), sp.slicer.slider_node)
    reduce_res = pp.Node(_maybe_reduce_dim, da=slice_res, dims=other_dims, op=operation)

    res_fig_kwargs = {}
    if 'autoscale' in kwargs:
        res_fig_kwargs['autoscale'] = kwargs['autoscale']

    # The log buttons of the main plot also apply to the shared x axis, and a log y axis makes
    # no sense for residuals, which can be negative.
    res_fig = pp.linefigure(reduce_res, ax=res_ax, hide_log_buttons=True, **res_fig_kwargs)

    fig1 = sp.figure
    for widget in fig1.bottom_bar[0].controls.values():
        widget.slider_toggler.value = '-o-'

    res_fig.autoscale()

    # Small tweaks
    fig1.ax.set_xlabel('')
    fig1.ax.xaxis.label.set_visible(False)
    fig1.ax.tick_params(labelbottom=False)

    fig1.autoscale()
    return fig1
