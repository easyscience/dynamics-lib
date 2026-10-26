# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
Spherical Bessel functions written as dependency expressions.

EasyScience Core evaluates dependency expressions with asteval and cannot call scipy or numba
there, so a model whose dependent Parameters need j_l(x) (for example ``DiffusionInSphere``) writes
it out in elementary functions with ``spherical_bessel_expression``. Python-side code uses
``easydynamics.utils.spherical_bessel.spherical_jn`` instead.
"""

import math

import numpy as np

SERIES_TOLERANCE = 1e-17  # Relative size of the first neglected power-series term at the switch


def spherical_bessel_expression(order: int) -> str:
    r"""
    Write the spherical Bessel function j_order(x) as an expression in elementary functions.

    The expression uses the names ``x`` and ``y`` (with ``y = x**2``), which must be assigned
    before it in the dependency expression. For $x \geq \max(l, 1)$ it is the finite sum for
    half-integer Bessel functions,

    $$ j_l(x) = \frac{1}{x}\left[\sin\left(x - \frac{l\pi}{2}\right) \sum_k \frac{(-1)^k
    (l+2k)!}{(2k)!\,(l-2k)!\,(2x)^{2k}} + \cos\left(x - \frac{l\pi}{2}\right) \sum_k \frac{(-1)^k
    (l+2k+1)!}{(2k+1)!\,(l-2k-1)!\,(2x)^{2k+1}}\right], $$

    and below it the power series

    $$ j_l(x) = \frac{x^l}{(2l+1)!!} \sum_m \frac{(-y/2)^m}{m!\,(2l+3)(2l+5)\cdots(2l+2m+1)}, $$

    written in nested (Horner) form, with as many terms as needed for SERIES_TOLERANCE at $x =
    \max(l, 1)$. The finite sum loses digits to cancellation at small x and the series at large x;
    the switch at $x = \max(l, 1)$ keeps both accurate.

    Parameters
    ----------
    order : int
        The order l, a non-negative integer.

    Raises
    ------
    TypeError
        If order is not an integer.
    ValueError
        If order is negative.

    Returns
    -------
    str
        An expression evaluating to j_order(x).
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise TypeError('order must be an integer.')
    if order < 0:
        raise ValueError('order must be non-negative.')
    order = int(order)

    switch = max(order, 1)
    return f'(({_finite_sum(order)}) if x >= {switch} else ({_power_series(order)}))'


def _finite_sum(order: int) -> str:
    """
    Write the finite sum for j_order(x), accurate for x of order and above.

    Parameters
    ----------
    order : int
        The order l.

    Returns
    -------
    str
        The expression, in the name ``x``.
    """
    sin_terms = []
    for k in range(order // 2 + 1):
        coefficient = (-1) ** k * math.factorial(order + 2 * k)
        coefficient /= math.factorial(2 * k) * math.factorial(order - 2 * k)
        sin_terms.append(f'{float(coefficient)!r} * (2 * x)**{-2 * k}')
    cos_terms = []
    for k in range((order - 1) // 2 + 1):
        coefficient = (-1) ** k * math.factorial(order + 2 * k + 1)
        coefficient /= math.factorial(2 * k + 1) * math.factorial(order - 2 * k - 1)
        cos_terms.append(f'{float(coefficient)!r} * (2 * x)**{-2 * k - 1}')

    phase = float(order * np.pi / 2)
    expression = f'sin(x - {phase!r}) * ({" + ".join(sin_terms)})'
    if cos_terms:
        expression += f' + cos(x - {phase!r}) * ({" + ".join(cos_terms)})'
    return f'({expression}) / x'


def _power_series(order: int) -> str:
    """
    Write the power series for j_order(x), accurate for x up to about order.

    Parameters
    ----------
    order : int
        The order l.

    Returns
    -------
    str
        The expression, in the names ``x`` and ``y = x**2``.
    """
    series = '1'
    for m in range(_series_terms(order) - 1, 0, -1):
        # ratio of term m to term m - 1: -y / (2 m (2l + 2m + 1))
        series = f'(1 - y / {float(2 * m * (2 * order + 2 * m + 1))!r} * {series})'
    double_factorial = float(math.prod(range(2 * order + 1, 0, -2)))
    return f'x**{order} / {double_factorial!r} * {series}'


def _series_terms(order: int) -> int:
    """
    Count the power-series terms needed at the switch point x = max(order, 1).

    Terms are added until the next one is smaller than SERIES_TOLERANCE times the first, and at
    least order + 1 terms are used, past the largest term.

    Parameters
    ----------
    order : int
        The order l.

    Returns
    -------
    int
        The number of terms, including the leading one.
    """
    y = float(max(order, 1)) ** 2
    term = 1.0
    m = 0
    while True:
        m += 1
        term *= y / (2 * m * (2 * order + 2 * m + 1))
        if m > order and term < SERIES_TOLERANCE:
            return m + 1
