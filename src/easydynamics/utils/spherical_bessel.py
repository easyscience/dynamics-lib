# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
Spherical Bessel functions of the first kind, j_n(x), computed by scipy's compiled code called from
numba.

``scipy.special.spherical_jn`` has a high cost per call in scipy 1.18. The numba kernel calls the
same compiled C function directly (through ``scipy.special.cython_special``), so its results are
bit-for-bit those of scipy, about ten times faster for small arrays. The name of that C function is
a scipy implementation detail; if it cannot be found, or numba is not installed (it is optional:
``pip install easydynamics[fast]``), scipy's Python function is used instead, with the same values.
"""

import ctypes
import functools
from collections.abc import Callable

import numpy as np
import scipy.special

# The float64 version of spherical_jn in scipy.special.cython_special (__pyx_fuse_0 is complex)
_CYTHON_MODULE = 'scipy.special.cython_special'
_CYTHON_FUNCTION = '__pyx_fuse_1spherical_jn'


def spherical_jn(order: int | np.ndarray, x: float | np.ndarray) -> np.ndarray:
    """
    Calculate the spherical Bessel function of the first kind, j_order(x).

    ``order`` and ``x`` broadcast against each other, as in ``scipy.special.spherical_jn``, so all
    orders can be computed in one call, e.g. ``spherical_jn(np.arange(5)[:, None], x)``.

    Parameters
    ----------
    order : int | np.ndarray
        Non-negative integer order(s).
    x : float | np.ndarray
        Real argument(s).

    Returns
    -------
    np.ndarray
        The values, float64, with the broadcast shape of ``order`` and ``x``.
    """
    kernel = _load_kernel()
    if kernel is None:
        return np.asarray(scipy.special.spherical_jn(order, x), dtype=float)

    orders, arguments = np.broadcast_arrays(np.asarray(order), np.asarray(x, dtype=float))
    out = np.empty(arguments.shape, dtype=float)
    kernel(
        np.ascontiguousarray(orders, dtype=np.int64).ravel(),
        np.ascontiguousarray(arguments).ravel(),
        out.reshape(-1),
    )
    return out


@functools.cache
def _load_kernel() -> Callable | None:
    """
    Build the numba kernel that calls scipy's compiled spherical_jn, once.

    Returns
    -------
    Callable | None
        ``kernel(orders, xs, out)`` filling ``out`` with j_orders(xs), or None if numba is not
        installed or scipy's compiled function cannot be found.
    """
    try:
        import numba
        from numba.extending import get_cython_function_address

        address = get_cython_function_address(_CYTHON_MODULE, _CYTHON_FUNCTION)
    except (ImportError, ValueError, KeyError, AttributeError):
        return None

    # C signature: double spherical_jn(Py_ssize_t n, double z, int skip_dispatch, opt_args *).
    # The opt_args pointer is declared as a pointer-sized integer so numba can pass 0 (NULL).
    compiled_spherical_jn = ctypes.CFUNCTYPE(
        ctypes.c_double, ctypes.c_ssize_t, ctypes.c_double, ctypes.c_int, ctypes.c_ssize_t
    )(address)

    @numba.njit(cache=False)
    def kernel(orders: np.ndarray, xs: np.ndarray, out: np.ndarray) -> None:
        for i in range(xs.size):
            out[i] = compiled_spherical_jn(orders[i], xs[i], np.int32(0), 0)

    return kernel
