# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by the file loaders."""

import math
from numbers import Real

import numpy as np
import scipp as sc


def apply_min_error(data: sc.DataArray, min_error: float | None) -> None:
    """
    Deal with bins whose error is too small to weight a fit by.

    Reduced data often has bins with zero counts and therefore zero error, typically far from the
    elastic line. A fit weights each bin by one over its error, so these bins cannot be fitted as
    they are.

    Parameters
    ----------
    data : sc.DataArray
        The loaded data, with variances. Changed in place.
    min_error : float | None
        None leaves the data as it is. A positive number raises every error below it to it, in the
        same units as the intensity. NaN instead masks the bins whose error is zero with a
        ``'zero_error'`` mask, so fits and rebinning leave them out. Those bins are usually ones in
        which nothing was counted, so leaving them out raises the average intensity wherever they
        are common, as when rebinning low-count data; a floor keeps them, with a small weight, and
        is then the better choice.

    Raises
    ------
    TypeError
        If ``min_error`` is not a number or None.
    ValueError
        If ``min_error`` is not positive and finite, or NaN.
    """
    if min_error is None:
        return
    if not isinstance(min_error, Real) or isinstance(min_error, bool):
        raise TypeError(f'min_error must be a number or None, not {type(min_error).__name__}')

    if math.isnan(min_error):
        data.masks['zero_error'] = sc.array(dims=data.dims, values=data.variances == 0)
    elif min_error > 0 and math.isfinite(min_error):
        data.variances = np.maximum(data.variances, min_error**2)
    else:
        raise ValueError(f'min_error must be positive and finite, or NaN, not {min_error}')
