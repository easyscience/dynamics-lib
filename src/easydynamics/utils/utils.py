# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from numpy.typing import ArrayLike
from scipp.constants import hbar as scipp_hbar

Numeric = float | int

Q_type = np.ndarray | Numeric | list | ArrayLike | sc.Variable
energy_type = np.ndarray | Numeric | list | ArrayLike | sc.Variable

hbar = DescriptorNumber.from_scipp('hbar', scipp_hbar)
angstrom = DescriptorNumber('angstrom', 1e-10, unit='m')


def _validate_and_convert_Q(
    Q: np.ndarray | Numeric | list | ArrayLike | sc.Variable | None,
) -> np.ndarray | None:
    """
    Validate and convert Q to a numpy array.

    Parameters
    ----------
    Q : np.ndarray | Numeric | list | ArrayLike | sc.Variable | None
        Scattering vector values in 1/angstrom.

    Raises
    ------
    TypeError
        If Q is not a number, list, numpy array, or scipp Variable.
    ValueError
        If Q is a numpy array with more than 1 dimension, or if Q is a scipp Variable that does not
        have a single dimension named 'Q'.

    Returns
    -------
    np.ndarray | None
        Q as a np.ndarray or None if Q is None.
    """
    if Q is None:
        return None
    if not isinstance(Q, (np.ndarray, Numeric, list, sc.Variable)):
        raise TypeError('Q must be a number, list, numpy array, or scipp Variable.')

    if isinstance(Q, Numeric):
        Q = np.array([Q])
    if isinstance(Q, list):
        Q = np.array(Q)
    if isinstance(Q, np.ndarray):
        if Q.ndim > 1:
            raise ValueError('Q must be a 1-dimensional array.')

        Q = sc.array(dims=['Q'], values=Q, unit='1/angstrom')

    if isinstance(Q, sc.Variable):
        if Q.dims != ('Q',):
            raise ValueError("Q must have a single dimension named 'Q'.")
        Q = Q.to(unit='1/angstrom')
    return Q.values


def _validate_unit(unit: str | sc.Unit | None) -> sc.Unit | None:
    """
    Validate that the unit is a string or scipp Unit.

    Parameters
    ----------
    unit : str | sc.Unit | None
        Unit to validate.

    Raises
    ------
    TypeError
        If unit is not None, a string, or a scipp Unit.

    Returns
    -------
    sc.Unit | None
        Validated unit or None.
    """

    if unit is not None and not isinstance(unit, (str, sc.Unit)):
        raise TypeError(f'unit must be None, a string, or a scipp Unit, got {type(unit).__name__}')
    if isinstance(unit, str):
        unit = sc.Unit(unit)
    return unit


def _in_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook.

    Returns
    -------
    bool
        True if in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or JupyterLab
        if shell == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        return False
    except (NameError, ImportError):
        return False  # Standard Python (no IPython)
