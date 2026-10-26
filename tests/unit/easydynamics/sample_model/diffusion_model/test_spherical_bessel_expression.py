# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest
from asteval import Interpreter

from easydynamics.sample_model.diffusion_model.spherical_bessel_expression import (
    spherical_bessel_expression,
)
from easydynamics.utils.spherical_bessel import spherical_jn

ORDERS = range(27)  # DiffusionInSphere needs j_l and j_(l+1) for l up to 25


def evaluate(expression: str, x: np.ndarray) -> np.ndarray:
    """Evaluate the expression point by point with asteval, as EasyScience Core does."""
    interpreter = Interpreter()
    values = np.empty_like(x)
    for i, value in enumerate(x):
        interpreter.symtable['x'] = float(value)
        interpreter.symtable['y'] = float(value) ** 2
        values[i] = interpreter.eval(expression, raise_errors=True)
    return values


@pytest.mark.parametrize('order', ORDERS)
def test_matches_spherical_jn(order):
    # WHEN
    x = np.concatenate([np.linspace(1e-3, 1.0, 200), np.linspace(1.0, 40.0, 1800)])

    # THEN
    values = evaluate(spherical_bessel_expression(order), x)

    # EXPECT
    np.testing.assert_allclose(values, spherical_jn(order, x), rtol=0, atol=1e-12)


@pytest.mark.parametrize('order', range(1, 27))
def test_continuous_at_the_switch(order):
    # WHEN the expression switches from the power series to the finite sum at x = order
    x = np.array([order - 1e-9, order, order + 1e-9])

    # THEN
    values = evaluate(spherical_bessel_expression(order), x)

    # EXPECT
    np.testing.assert_allclose(values, spherical_jn(order, x), rtol=0, atol=1e-12)


@pytest.mark.parametrize('order', [0, 1, 5, 26])
def test_small_x(order):
    # WHEN
    x = np.array([1e-8])

    # THEN
    values = evaluate(spherical_bessel_expression(order), x)

    # EXPECT the leading term x^l / (2l + 1)!!
    double_factorial = math.prod(range(2 * order + 1, 0, -2))
    assert np.isfinite(values[0])
    assert values[0] == pytest.approx(1e-8**order / double_factorial, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    'order, expected_exception',
    [(1.5, TypeError), (True, TypeError), ('2', TypeError), (-1, ValueError)],
)
def test_order_must_be_a_non_negative_integer(order, expected_exception):
    # WHEN THEN EXPECT
    with pytest.raises(expected_exception, match='order must be'):
        spherical_bessel_expression(order)
