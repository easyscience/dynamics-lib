# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Read the text files Mantid writes with ``SaveAscii``.

These hold one block of ``X, Y, E`` rows per spectrum, each block normally headed by a line with
the spectrum number. They store no Q, so the Q of each spectrum has to be given.
"""

import re
from pathlib import Path

import numpy as np
import scipp as sc
from numpy.typing import ArrayLike

from easydynamics.io.utils import apply_min_error

# SaveAscii can separate columns with commas, tabs, spaces, colons or semicolons.
_SEPARATORS = re.compile(r'[,;:\s]+')


def load_mantid_ascii(
    filename: str | Path,
    Q: sc.Variable | ArrayLike,
    energy_unit: str = 'meV',
    min_error: float | None = None,
) -> sc.DataArray:
    """
    Load a workspace saved by Mantid's ``SaveAscii``, with energy transfer as X.

    Every spectrum must have the same X values. An extra column after ``X, Y, E``, such as the X
    errors, is ignored.

    Examples
    --------
    ```python
    import easydynamics as edyn
    import scipp as sc

    Q = sc.linspace('Q', 0.5, 2.0, 13, unit='1/angstrom')
    data = edyn.load_mantid_ascii('sample_250K.dat', Q=Q, min_error=0.01)
    experiment = edyn.Experiment(display_name='250K', data=data)
    ```

    Parameters
    ----------
    filename : str | Path
        Path to the text file.
    Q : sc.Variable | ArrayLike
        Q of each spectrum, in file order, or Q bin edges, one more than there are spectra. A plain
        array is taken to be in 1/angstrom; a ``sc.Variable`` is converted to it.
    energy_unit : str, default='meV'
        Unit of the X column.
    min_error : float | None, default=None
        None keeps the errors as stored. A positive number raises every error below it to it, in
        the units of the intensity. NaN masks the bins with zero error with a ``'zero_error'``
        mask, so fits and rebinning leave them out.

    Returns
    -------
    sc.DataArray
        Dims ``('Q', 'energy')``, with ``Q`` in 1/angstrom and variances from the stored errors.

    Raises
    ------
    ValueError
        If the file holds no data, a row has fewer than three columns, the spectra do not share
        their X values, or ``Q`` is not one-dimensional with one value per spectrum or one more.
    """
    spectra = _read_spectra(Path(filename))
    if not spectra:
        raise ValueError(f'{Path(filename).name} holds no data rows')
    energy = spectra[0][:, 0]
    for spectrum in spectra[1:]:
        if spectrum.shape != spectra[0].shape or not np.allclose(spectrum[:, 0], energy):
            raise ValueError('Every spectrum must have the same X values')

    Q_values = _Q_in_inverse_angstrom(Q)
    if Q_values.ndim != 1 or Q_values.size not in (len(spectra), len(spectra) + 1):
        raise ValueError(
            f'Q must be one-dimensional with {len(spectra)} values, one per spectrum, or '
            f'{len(spectra) + 1} bin edges, not shape {Q_values.shape}'
        )

    data = sc.DataArray(
        data=sc.array(
            dims=['Q', 'energy'],
            values=np.array([spectrum[:, 1] for spectrum in spectra]),
            variances=np.array([spectrum[:, 2] for spectrum in spectra]) ** 2,
        ),
        coords={
            'Q': sc.array(dims=['Q'], values=Q_values, unit='1/angstrom'),
            'energy': sc.array(dims=['energy'], values=energy, unit=energy_unit),
        },
    )
    apply_min_error(data, min_error)
    return data


def _read_spectra(path: Path) -> list[np.ndarray]:
    """
    Split a ``SaveAscii`` file into its spectra.

    A line holding a single whole number starts a new spectrum. A file written without these
    spectrum numbers is split wherever X stops increasing.

    Parameters
    ----------
    path : Path
        Path to the text file.

    Returns
    -------
    list[np.ndarray]
        One array of ``X, Y, E`` rows per spectrum.

    Raises
    ------
    ValueError
        If a data row has fewer than three columns.
    """
    spectra: list[list[list[float]]] = []
    numbered = False
    for line_number, line in enumerate(path.read_text(errors='replace').splitlines(), start=1):
        fields = [field for field in _SEPARATORS.split(line.strip()) if field]
        if not fields or line.lstrip().startswith('#'):
            continue
        if len(fields) == 1 and fields[0].isdigit():
            spectra.append([])
            numbered = True
            continue
        if len(fields) < 3:
            raise ValueError(f'{path.name}, line {line_number}: expected X, Y and E columns')
        row = [float(field) for field in fields[:3]]
        if not spectra or (not numbered and spectra[-1] and row[0] <= spectra[-1][-1][0]):
            spectra.append([])
        spectra[-1].append(row)
    return [np.array(spectrum) for spectrum in spectra if spectrum]


def _Q_in_inverse_angstrom(Q: sc.Variable | ArrayLike) -> np.ndarray:
    """
    Get Q as plain values in 1/angstrom.

    Parameters
    ----------
    Q : sc.Variable | ArrayLike
        Q as a ``sc.Variable`` with a unit, or as plain values in 1/angstrom.

    Returns
    -------
    np.ndarray
        The Q values in 1/angstrom.
    """
    if isinstance(Q, sc.Variable):
        return sc.to_unit(Q, '1/angstrom', copy=True).values.astype(float)
    return np.asarray(Q, dtype=float)
