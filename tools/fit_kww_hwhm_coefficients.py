# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
Refit ``_HWHM_POLY_COEFFS`` in ``easydynamics.sample_model.components.stretched_exponential``.

``StretchedExponential.width`` is a dependent easyscience ``Parameter``, and easyscience resolves
a dependency from a string expression, so the half width has to be written in closed form.  The
half width is not available in closed form: it is the ``w`` solving

    G_beta(w) = 0.5 * G_beta(0),     G_beta(w) = integral of exp(-u**beta) cos(w u) du,

which ``_reduced_hwhm`` brackets numerically.  It is very nearly closed form, though, because

    eps(beta) := beta * ln(HWHM(beta)) - ln(beta)

is small, smooth and O(0.06), so that

    HWHM(beta) = exp((ln(beta) + eps(beta)) / beta)

with ``eps`` a low-order polynomial.  This script fits that polynomial against ``_reduced_hwhm``
and prints the coefficient block to paste back into the module.

The fit degrades below beta = 0.1, but so does the root it is fitted to: by beta = 0.05 the
profile takes some 60 e-folds of ``w`` to fall by half, so ``G_beta`` is nearly flat in ``log w``
near the crossing.  Wuttke's reference implementation libkww (arXiv:0911.4796) likewise supports
only ``0.1 <= beta <= 1.9``.  Both are far below any energy grid step by then, so the accuracy
that matters is the one reported for ``beta >= 0.1``.

Usage
-----
```
pixi run python tools/fit_kww_hwhm_coefficients.py            # refit and print the block
pixi run python tools/fit_kww_hwhm_coefficients.py --check    # verify the committed values
pixi run python tools/fit_kww_hwhm_coefficients.py --degree 10 --samples 800
```
"""

import argparse
import sys

import numpy as np
from numpy.polynomial import polynomial as P

from easydynamics.sample_model.components.stretched_exponential import _HWHM_POLY_COEFFS
from easydynamics.sample_model.components.stretched_exponential import MAXIMUM_BETA
from easydynamics.sample_model.components.stretched_exponential import MINIMUM_BETA
from easydynamics.sample_model.components.stretched_exponential import _reduced_hwhm

# The committed coefficients were produced with these settings; changing them changes the fit.
DEFAULT_DEGREE = 8
DEFAULT_SAMPLES = 400
# Accuracy is quoted over the range the fit is meant to serve, and where libkww also stops.
REPORTING_FLOOR = 0.1
# Loose enough to absorb a scipy root-finder nudge, tight enough to catch a real regression.
CHECK_TOLERANCE = 5e-5


def sample_grid(samples: int) -> np.ndarray:
    """
    Build the beta grid the fit is evaluated on.

    Log spacing matches how the half width actually varies: it spans some 26 decades across the
    supported beta, almost all of that below beta = 0.5.

    Parameters
    ----------
    samples : int
        Number of grid points.

    Returns
    -------
    np.ndarray
        Log-spaced beta values over ``[MINIMUM_BETA, MAXIMUM_BETA]``.
    """
    return np.exp(np.linspace(np.log(MINIMUM_BETA), np.log(MAXIMUM_BETA), samples))


def fit_coefficients(beta: np.ndarray, degree: int) -> np.ndarray:
    """
    Least-squares fit the eps polynomial against the solved half width.

    Parameters
    ----------
    beta : np.ndarray
        Beta values to fit over.
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray
        Coefficients in ascending powers of beta.
    """
    hwhm = np.array([_reduced_hwhm(b) for b in beta])
    eps = beta * np.log(hwhm) - np.log(beta)
    return P.polyfit(beta, eps, degree)


def relative_error(beta: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Compare the closed form against the solved half width.

    Parameters
    ----------
    beta : np.ndarray
        Beta values to evaluate at.
    coefficients : np.ndarray
        Polynomial coefficients in ascending powers of beta.

    Returns
    -------
    np.ndarray
        Relative error of the closed form at each beta.
    """
    exact = np.array([_reduced_hwhm(b) for b in beta])
    approx = np.exp((np.log(beta) + P.polyval(beta, coefficients)) / beta)
    return np.abs(approx / exact - 1.0)


def report(beta: np.ndarray, coefficients: np.ndarray) -> float:
    """
    Print the accuracy of a coefficient set and return the error over the reporting range.

    Parameters
    ----------
    beta : np.ndarray
        Beta values to evaluate at.
    coefficients : np.ndarray
        Polynomial coefficients in ascending powers of beta.

    Returns
    -------
    float
        Maximum relative error for ``beta >= REPORTING_FLOOR``.
    """
    error = relative_error(beta, coefficients)
    above_floor = error[beta >= REPORTING_FLOOR].max()
    print(f'  max relative error, all beta        : {error.max():.2e}')
    print(f'  max relative error, beta >= {REPORTING_FLOOR}     : {above_floor:.2e}')
    for probe in (2.0, 1.0, 0.5, 0.2, 0.1, MINIMUM_BETA):
        single = relative_error(np.array([probe]), coefficients)[0]
        print(f'    beta = {probe:<5} -> {single:.2e}')
    return float(above_floor)


def main() -> int:
    """
    Refit the coefficients, or check the committed ones.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        '--check',
        action='store_true',
        help='verify the committed coefficients instead of refitting',
    )
    parser.add_argument('--degree', type=int, default=DEFAULT_DEGREE)
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()

    beta = sample_grid(args.samples)

    if args.check:
        print(f'Checking committed _HWHM_POLY_COEFFS ({len(_HWHM_POLY_COEFFS) - 1} degree):')
        above_floor = report(beta, np.array(_HWHM_POLY_COEFFS))
        if above_floor > CHECK_TOLERANCE:
            print(f'\nFAIL: {above_floor:.2e} exceeds the {CHECK_TOLERANCE:.0e} tolerance.')
            print('Rerun without --check and paste the new block into the module.')
            return 1
        print(f'\nOK: within the {CHECK_TOLERANCE:.0e} tolerance.')
        return 0

    coefficients = fit_coefficients(beta, args.degree)
    print(
        f'Fitted degree {args.degree} over {args.samples} log-spaced beta in '
        f'[{MINIMUM_BETA}, {MAXIMUM_BETA}]:'
    )
    report(beta, coefficients)

    print('\nPaste into stretched_exponential.py:\n')
    print('_HWHM_POLY_COEFFS = (')
    for coefficient in coefficients:
        print(f'    {float(coefficient)!r},')
    print(')')
    return 0


if __name__ == '__main__':
    sys.exit(main())
