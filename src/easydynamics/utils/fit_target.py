# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FitTarget:
    """
    One fittable prediction of a model, bound to a key in a parameters Dataset.

    Models declare their predictions by returning FitTargets (see
    ``DiffusionModelBase.get_fit_targets``), and ``FitBinding`` maps them onto the dataset keys
    they should be fitted against. Instances are immutable snapshots created on demand, so the
    units always reflect the model state at the time the targets are built.

    Attributes
    ----------
    name : str
        The prediction's name (e.g. ``'width'``, ``'area'``, ``'delta_area'``, ``'value'``).
    dataset_key : str | None
        The key in the parameters Dataset holding the data this prediction is fitted against. None
        when the prediction has no default key (component models); ``FitBinding`` supplies the key
        in that case.
    function : Callable
        The fit function; called as ``function(x)`` with raw x values expressed in *x_unit* and
        returning raw values expressed in *y_unit*.
    label : str
        Display label used for plots and results (e.g. ``'DeltaLorentz width'``).
    x_unit : str | None
        The unit *function* expects its input in, or None if no unit conversion applies.
    y_unit : str | None
        The unit of *function*'s output, or None if no unit conversion applies.
    """

    name: str
    dataset_key: str | None
    function: Callable
    label: str
    x_unit: str | None
    y_unit: str | None
