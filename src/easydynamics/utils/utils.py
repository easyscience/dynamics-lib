# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from numpy.typing import ArrayLike
from scipp.constants import hbar as scipp_hbar
from scipp.constants import k as scipp_k

Numeric = float | int

Q_type = np.ndarray | Numeric | list | ArrayLike | sc.Variable
energy_type = np.ndarray | Numeric | list | ArrayLike | sc.Variable

hbar = DescriptorNumber.from_scipp('hbar', scipp_hbar)
kb = DescriptorNumber.from_scipp('kb', scipp_k)
angstrom = DescriptorNumber('angstrom', 1e-10, unit='m')


def verify_Q_index(Q_index: int, Q: sc.Variable | None, allow_none: bool = False) -> None:
    """
    Verify that Q_index is a valid integer index into Q.

    Parameters
    ----------
    Q_index : int
        Index to validate.
    Q : sc.Variable | None
        The Q values (may be None if no data is loaded).
    allow_none : bool, default=False
        Whether or not to allow Q_index to be None

    Raises
    ------
    TypeError
        If Q_index is not an int (or not an int or None when allow_none=True).
    IndexError
        If Q_index is out of range.
    ValueError
        If Q is None and Q_index is not None.
    """
    if allow_none and Q_index is None:
        return

    if Q_index is None or not isinstance(Q_index, int):
        if allow_none:
            raise TypeError(f'Q_index must be an int or None, got {type(Q_index).__name__}')
        raise TypeError(f'Q_index must be an int, got {type(Q_index).__name__}')

    if Q is None:
        raise ValueError('Q is None, cannot validate Q_index.')

    if not (0 <= Q_index < len(Q)):
        raise IndexError(f'Q_index {Q_index} is out of bounds for Q of length {len(Q)}')


def convert_parameter_unit(parameter: Parameter, unit: str | sc.Unit) -> None:
    """
    Convert a parameter to a new unit, keeping dependent parameters consistent.

    Independent parameters are converted with ``convert_unit``. Dependent parameters are converted
    with ``set_desired_unit``, so the new unit survives later dependency-graph recomputations (a
    plain ``convert_unit`` would be reverted to the old desired unit the next time the dependency
    expression is re-evaluated).

    Parameters
    ----------
    parameter : Parameter
        The parameter to convert.
    unit : str | sc.Unit
        The unit to convert to.
    """
    if parameter.independent:
        parameter.convert_unit(str(unit))
    else:
        parameter.set_desired_unit(str(unit))


def energy_to_scipp(energy: np.ndarray, unit: str | sc.Unit) -> sc.Variable:
    """
    Convert a numpy energy array to a scipp Variable with dimension 'energy'.

    Parameters
    ----------
    energy : np.ndarray
        The energy array to be converted
    unit : str | sc.Unit
        The unit of the energy

    Returns
    -------
    sc.Variable
        Energy as sc.Variable.
    """
    return sc.array(dims=['energy'], values=energy, unit=unit)


def _validate_and_convert_Q(
    Q: np.ndarray | Numeric | list | ArrayLike | sc.Variable | None,
) -> sc.Variable | None:
    """
    Validate and convert Q to a scipp Variable in 1/angstrom.

    Numbers, lists, and numpy arrays are assumed to be in 1/angstrom. Scipp Variables may be in any
    unit convertible to 1/angstrom and are converted.

    Parameters
    ----------
    Q : np.ndarray | Numeric | list | ArrayLike | sc.Variable | None
        Scattering vector values.

    Raises
    ------
    TypeError
        If Q is not a number, list, numpy array, or scipp Variable.
    ValueError
        If Q is a numpy array with more than 1 dimension, or if Q is a scipp Variable that does not
        have a single dimension named 'Q'.

    Returns
    -------
    sc.Variable | None
        Q as a sc.Variable with dimension 'Q' and unit 1/angstrom, or None if Q is None.
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
    return Q


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
    # if isinstance(unit, str):
    #     unit = sc.Unit(unit)

    if isinstance(unit, sc.Unit):
        unit = str(unit)
    return unit


def _assert_valid_unit(unit: str | sc.Unit) -> None:
    """
    Assert that the given unit is recognised by scipp.

    Parameters
    ----------
    unit : str | sc.Unit
        The unit to validate.

    Raises
    ------
    TypeError
        If unit is not a string or scipp Unit.
    ValueError
        If the string is not a valid scipp unit.
    """
    if not isinstance(unit, (str, sc.Unit)):
        raise TypeError(f'unit must be a string or sc.Unit, got {type(unit).__name__}')
    try:
        sc.Unit(str(unit))
    except sc.UnitError as e:
        raise ValueError(f"'{unit}' is not a valid scipp unit.") from e


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
