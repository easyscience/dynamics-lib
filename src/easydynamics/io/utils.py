# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by the file loaders."""

import math
from numbers import Real

import numpy as np


def apply_min_error(
    values: np.ndarray,
    variances: np.ndarray,
    min_error: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deal with bins whose error is too small to weight a fit by.

    Reduced data often has bins with zero counts and therefore zero error, typically far from the
    elastic line. A fit weights each bin by one over its error, so these bins cannot be fitted as
    they are.

    Parameters
    ----------
    values : np.ndarray
        The intensities. Changed in place when ``min_error`` is NaN.
    variances : np.ndarray
        The variances, the squared errors. Changed in place unless ``min_error`` is None.
    min_error : float | None
        None leaves the data as it is. A positive number raises every error below it to it, in the
        same units as the intensity. NaN instead sets the bins whose error is zero to NaN, so fits
        leave them out.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The values and variances.

    Raises
    ------
    TypeError
        If ``min_error`` is not a number or None.
    ValueError
        If ``min_error`` is not positive and finite, or NaN.
    """
    if min_error is None:
        return values, variances
    if not isinstance(min_error, Real) or isinstance(min_error, bool):
        raise TypeError(f'min_error must be a number or None, not {type(min_error).__name__}')

    if math.isnan(min_error):
        zero = variances == 0
        values[zero] = np.nan
        variances[zero] = np.nan
    elif min_error > 0 and math.isfinite(min_error):
        np.maximum(variances, min_error**2, out=variances, where=~np.isnan(variances))
    else:
        raise ValueError(f'min_error must be positive and finite, or NaN, not {min_error}')
    return values, variances
