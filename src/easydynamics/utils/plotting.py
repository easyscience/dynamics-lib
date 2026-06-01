# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import plopp as pp
import scipp as sc
from plopp.backends.matplotlib.figure import InteractiveFigure
from plopp.plotting._slicer import SlicerPlot
from plopp.plotting._slicer import _maybe_reduce_dim
from plopp.widgets import slice_dims


def slicerplot_with_residuals(
    dg: sc.DataGroup,
    *,
    residuals_key: str = 'Residuals',
    keep: list[str] | str | None = None,
    operation: str = 'sum',
    **kwargs: object,
) -> InteractiveFigure:
    """
    Create a SlicerPlot with an additional subplot for residuals.

    Parameters
    ----------
    dg : sc.DataGroup
        DataGroup containing the data to plot. Must include a key for residuals.
    residuals_key : str, default="Residuals"
        Key in the DataGroup that contains the residuals data.
    keep : list[str] | str | None, default=None
        Dimensions to keep in the SlicerPlot. Passed to SlicerPlot.
    operation : str, default="sum"
        Operation to apply when reducing the residuals data. Passed to SlicerPlot.
    **kwargs : object
        Additional keyword arguments passed to SlicerPlot.

    Returns
    -------
    InteractiveFigure
        A figure containing the SlicerPlot and the residuals subplot.
    """

    sp = SlicerPlot(
        {k: da for k, da in dg.items() if k != residuals_key},
        keep=keep,
        operation=operation,
        **kwargs,
    )

    other_dims = [dim for dim in dg.dims if dim not in sp.slicer.keep]

    slice_res = slice_dims(pp.Node(dg[residuals_key]), sp.slicer.slider_node)
    reduce_res = pp.Node(_maybe_reduce_dim, da=slice_res, dims=other_dims, op=operation)

    res_fig_kwargs = {}
    if 'autoscale' in kwargs:
        res_fig_kwargs['autoscale'] = kwargs['autoscale']

    res_fig = pp.linefigure(reduce_res, figsize=(6, 2), **res_fig_kwargs)

    res_fig.layout.margin = '0px'
    res_fig.layout.padding = '0px'

    fig1 = sp.figure
    for widget in fig1.bottom_bar[0].controls.values():
        widget.slider_toggler.value = '-o-'
    fig1.bottom_bar.children = [pp.widgets.VBar([res_fig, fig1.bottom_bar.children[0]])]

    res_fig.autoscale()

    # Small tweaks
    fig1.ax.set_xlabel('')
    fig1.ax.xaxis.label.set_visible(False)
    fig1.ax.tick_params(labelbottom=False)

    fig1.autoscale()
    return fig1
