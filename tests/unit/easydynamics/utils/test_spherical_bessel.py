# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipy.special

from easydynamics.utils import spherical_bessel
from easydynamics.utils.spherical_bessel import spherical_jn


def test_matches_scipy_exactly():
    # WHEN
    order = np.arange(27)[:, None]
    x = np.linspace(0.0, 40.0, 2001)

    # THEN
    values = spherical_jn(order, x)

    # EXPECT bit for bit: the kernel calls scipy's own compiled function
    np.testing.assert_array_equal(values, scipy.special.spherical_jn(order, x))


@pytest.mark.parametrize(
    'order, x, shape',
    [
        (3, np.linspace(0.1, 5.0, 7), (7,)),
        (np.arange(4), 2.0, (4,)),
        (2, 1.5, ()),
        (np.arange(3)[:, None], np.linspace(0.1, 1.0, 5), (3, 5)),
    ],
    ids=['array_x', 'array_order', 'scalars', 'table'],
)
def test_broadcasts_order_and_x(order, x, shape):
    # WHEN
    values = spherical_jn(order, x)

    # THEN EXPECT
    assert np.shape(values) == shape
    np.testing.assert_array_equal(values, scipy.special.spherical_jn(order, x))


def test_falls_back_to_scipy(monkeypatch):
    # WHEN scipy's compiled function cannot be found (its name is a scipy implementation detail)
    monkeypatch.setattr(spherical_bessel, '_load_kernel', lambda: None)
    order = np.arange(5)[:, None]
    x = np.linspace(0.0, 20.0, 101)

    # THEN
    values = spherical_jn(order, x)

    # EXPECT
    np.testing.assert_array_equal(values, scipy.special.spherical_jn(order, x))


def test_uses_the_numba_kernel():
    # WHEN numba (an optional dependency) is installed
    pytest.importorskip('numba')

    # THEN EXPECT the fast path is available
    assert spherical_bessel._load_kernel() is not None
