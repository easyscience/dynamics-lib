# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Read ASCII ``.inx`` files, the ILL's format for spectra reduced with LAMP, such as IN16B data.

An ``.inx`` file holds one block per spectrum. Each block starts with a line of whole numbers: the
number of lines that follow, the number of title lines, the number of information lines, and last
the number of data rows. Then come the title, the information lines and the ``X, Y, E`` rows. The
first information line starts with the scattering angle, or with Q, followed by the incident
energy.
"""

import math
from pathlib import Path

import numpy as np
import scipp as sc
from numpy.typing import ArrayLike

from easydynamics.io.utils import apply_min_error


def load_inx(
    filename: str | Path,
    Q: sc.Variable | ArrayLike | None = None,
    energy_unit: str = 'ueV',
    min_error: float | None = None,
) -> sc.DataArray:
    """
    Load an ``.inx`` file, one spectrum per Q.

    Files from IN16B's reduction store each spectrum's Q in the first field of its information
    line, with the incident energy left at zero, and that Q is used. A file whose information line
    gives an incident energy holds a scattering angle there instead, so its Q must be given.

    Every spectrum must have the same X values. Fields too wide for the file's fixed-width columns
    are written as ``*****``: a row whose X is such a field is left out, and an intensity or error
    written so becomes NaN, which fits and rebinning leave out.

    Examples
    --------
    ```python
    import easydynamics as edyn

    data = edyn.load_inx('sample_2K.inx', min_error=1e-5)
    experiment = edyn.Experiment(display_name='2K', data=data)
    ```

    Parameters
    ----------
    filename : str | Path
        Path to the ``.inx`` file.
    Q : sc.Variable | ArrayLike | None, default=None
        Q of each spectrum, in file order, or Q bin edges, one more than there are spectra. A plain
        array is taken to be in 1/angstrom; a ``sc.Variable`` is converted to it. If None, Q is
        read from the file.
    energy_unit : str, default='ueV'
        Unit of the X column; microelectronvolts, as IN16B writes it. The energy is returned in
        meV.
    min_error : float | None, default=None
        None keeps the errors as stored. A positive number raises every error below it to it, in
        the units of the intensity. NaN masks the bins with zero error with a ``'zero_error'``
        mask, so fits and rebinning leave them out; as these are usually empty bins, a floor is the
        better choice before rebinning low-count data.

    Returns
    -------
    sc.DataArray
        Dims ``('Q', 'energy')``, with ``Q`` in 1/angstrom, ``energy`` in meV and variances from
        the stored errors.

    Raises
    ------
    ValueError
        If the file holds no spectra, the spectra do not share their X values, ``Q`` is None and
        the file gives a scattering angle rather than Q, or ``Q`` does not have one value per
        spectrum or one more.
    """
    path = Path(filename)
    blocks = _read_blocks(path)
    if not blocks:
        raise ValueError(f'{path.name} holds no spectra')

    energy = blocks[0][1][:, 0]
    for _, rows in blocks[1:]:
        if rows.shape != blocks[0][1].shape or not np.allclose(rows[:, 0], energy, equal_nan=True):
            raise ValueError('Every spectrum must have the same X values')
    keep = np.isfinite(energy)

    if Q is None:
        if any(info[1] != 0 for info, _ in blocks):
            raise ValueError(
                f'{path.name} gives an incident energy, so its information lines start with a '
                'scattering angle rather than Q. Pass the Q of each spectrum with Q=...'
            )
        Q_values = np.array([info[0] for info, _ in blocks])
    elif isinstance(Q, sc.Variable):
        Q_values = sc.to_unit(Q, '1/angstrom', copy=True).values.astype(float)
    else:
        Q_values = np.asarray(Q, dtype=float)
    if Q_values.ndim != 1 or Q_values.size not in (len(blocks), len(blocks) + 1):
        raise ValueError(
            f'Q must be one-dimensional with {len(blocks)} values, one per spectrum, or '
            f'{len(blocks) + 1} bin edges, not shape {Q_values.shape}'
        )

    data = sc.DataArray(
        data=sc.array(
            dims=['Q', 'energy'],
            values=np.array([rows[keep, 1] for _, rows in blocks]),
            variances=np.array([rows[keep, 2] for _, rows in blocks]) ** 2,
        ),
        coords={
            'Q': sc.array(dims=['Q'], values=Q_values, unit='1/angstrom'),
            'energy': sc.to_unit(
                sc.array(dims=['energy'], values=energy[keep], unit=energy_unit), 'meV'
            ),
        },
    )
    apply_min_error(data, min_error)
    return data


def _read_blocks(path: Path) -> list[tuple[list[float], np.ndarray]]:
    """
    Split an ``.inx`` file into its blocks.

    Parameters
    ----------
    path : Path
        Path to the ``.inx`` file.

    Returns
    -------
    list[tuple[list[float], np.ndarray]]
        For each block, the numbers on its first information line, and its ``X, Y, E`` rows.

    Raises
    ------
    ValueError
        If a block's header line does not add up, or the file ends inside a block.
    """
    lines = path.read_text(errors='replace').splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        fields = lines[i].split()
        if not fields:
            i += 1
            continue
        try:
            header = [int(field) for field in fields]
        except ValueError as error:
            raise ValueError(
                f'{path.name}, line {i + 1}: expected the header line of a spectrum'
            ) from error
        if len(header) < 4 or header[1] + header[2] + header[-1] != header[0]:
            raise ValueError(f'{path.name}, line {i + 1}: the header line does not add up')
        n_lines, n_title, n_data = header[0], header[1], header[-1]
        if i + n_lines >= len(lines):
            raise ValueError(f'{path.name} ends inside the spectrum starting at line {i + 1}')
        info = [_number(field) for field in lines[i + 1 + n_title].split()]
        first_row = i + 1 + n_lines - n_data
        rows = [line.split()[:3] for line in lines[first_row : first_row + n_data]]
        if any(len(row) < 3 for row in rows):
            raise ValueError(
                f'{path.name}: the spectrum starting at line {i + 1} has a row without X, Y and E'
            )
        rows = [[_number(field) for field in row] for row in rows]
        blocks.append((info, np.array(rows, dtype=float)))
        i += 1 + n_lines
    return blocks


def _number(field: str) -> float:
    """
    Read one number, giving NaN for a field written as ``*****`` because it was too wide.

    Parameters
    ----------
    field : str
        The field.

    Returns
    -------
    float
        The number, which may be infinite or NaN.
    """
    if field and set(field) == {'*'}:
        return math.nan
    return float(field)
