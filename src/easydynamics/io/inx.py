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
import warnings
from pathlib import Path

import numpy as np
import scipp as sc
from numpy.typing import ArrayLike

from easydynamics.io.utils import apply_min_error

# No QENS instrument reaches this Q, in 1/angstrom; a larger first field is a scattering angle.
_MAX_Q = 10.0


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
    are written as ``*****``: a row whose X is such a field is left out, with a warning, and an
    intensity or error written so becomes NaN, which fits and rebinning leave out.

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
        If the file is not laid out as an ``.inx`` file, the spectra do not share their X values,
        ``Q`` is None and the file gives a scattering angle or no plausible Q, or ``Q`` does not
        have one value per spectrum or one more.
    """
    path = Path(filename)
    blocks = _read_blocks(path)
    if not blocks:
        raise ValueError(f'{path.name} holds no spectra')

    energy = blocks[0][1][:, 0]
    for _, rows in blocks[1:]:
        if rows.shape != blocks[0][1].shape or not np.allclose(rows[:, 0], energy, equal_nan=True):
            raise ValueError('Every spectrum must have the same X values')
    if Q is None:
        if any(info[1] != 0 for info, _ in blocks):
            raise ValueError(
                f'{path.name} gives an incident energy, so its information lines start with a '
                'scattering angle rather than Q. Pass the Q of each spectrum with Q=...'
            )
        Q_values = np.array([info[0] for info, _ in blocks])
        implausible = ~(np.isfinite(Q_values) & (Q_values > 0) & (Q_values < _MAX_Q))
        if implausible.any():
            raise ValueError(
                f'{path.name}: the information line of spectrum {int(np.argmax(implausible))} '
                f'starts with {Q_values[implausible][0]}, which is not a plausible Q in '
                '1/angstrom and may be a scattering angle. Pass the Q of each spectrum with Q=...'
            )
    elif isinstance(Q, sc.Variable):
        Q_values = sc.to_unit(Q, '1/angstrom', copy=True).values.astype(float)
    else:
        Q_values = np.asarray(Q, dtype=float)
    if Q_values.ndim != 1 or Q_values.size not in (len(blocks), len(blocks) + 1):
        raise ValueError(
            f'Q must be one-dimensional with {len(blocks)} values, one per spectrum, or '
            f'{len(blocks) + 1} bin edges, not shape {Q_values.shape}'
        )

    keep = np.isfinite(energy)
    if not keep.all():
        warnings.warn(
            f'{path.name}: {int((~keep).sum())} of {keep.size} rows have an X value too wide for '
            'its column (*****) and are left out.',
            UserWarning,
            stacklevel=2,
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
        If a block's header line does not add up, a block lacks an information line or data, a
        field is not a number, or the file ends inside a block.
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
        n_lines, n_title, n_info, n_data = header[0], header[1], header[2], header[-1]
        if n_info < 1 or n_data < 1:
            raise ValueError(
                f'{path.name}, line {i + 1}: a spectrum needs an information line and data rows'
            )
        if i + n_lines >= len(lines):
            raise ValueError(f'{path.name} ends inside the spectrum starting at line {i + 1}')
        info_line = i + 1 + n_title
        info = _numbers(lines[info_line].split(), path, info_line)
        if len(info) < 2:
            raise ValueError(
                f'{path.name}, line {info_line + 1}: the information line should start with Q or '
                'an angle, and the incident energy'
            )
        first_row = i + 1 + n_lines - n_data
        rows = []
        for line_index in range(first_row, first_row + n_data):
            fields = lines[line_index].split()[:3]
            if len(fields) < 3:
                raise ValueError(f'{path.name}, line {line_index + 1}: expected X, Y and E')
            rows.append(_numbers(fields, path, line_index))
        blocks.append((info, np.array(rows, dtype=float)))
        i += 1 + n_lines
    return blocks


def _numbers(fields: list[str], path: Path, line_index: int) -> list[float]:
    """
    Read the numbers of one line, giving NaN for a field written as ``*****`` because it was wide.

    Parameters
    ----------
    fields : list[str]
        The fields of the line.
    path : Path
        The file, for the error message.
    line_index : int
        The line, counted from zero, for the error message.

    Returns
    -------
    list[float]
        The numbers, which may be infinite or NaN.

    Raises
    ------
    ValueError
        If a field is not a number.
    """
    numbers = []
    for field in fields:
        if set(field) == {'*'}:
            numbers.append(math.nan)
            continue
        try:
            numbers.append(float(field))
        except ValueError as error:
            raise ValueError(
                f'{path.name}, line {line_index + 1}: {field!r} is not a number'
            ) from error
    return numbers
