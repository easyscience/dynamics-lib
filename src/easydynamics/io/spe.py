# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Read ASCII ``.spe`` files whose angle block holds the detectors' scattering angles.

An ``.spe`` file holds intensity on a grid of scattering angle by energy transfer. Each detector's
angle is turned into the momentum transfer it has at zero energy transfer,
``Q = 4 pi sin(phi / 2) / lambda``, which needs the incident wavelength. Some programs, such as
Mantid's ``SaveSPE``, write placeholder numbers in the angle block and the real angles in a
separate ``.par`` or ``.phx`` file; those files are not read yet.
"""

import math
import re
from numbers import Real
from pathlib import Path

import numpy as np
import scipp as sc

from easydynamics.io.utils import apply_min_error

# Mslice and Mantid write the intensity of a masked detector as -1e30.
_NULL_DATA = -1e29

# The number blocks are written in fixed-width fields, so a negative number runs into the one
# before it ("0.00e+000-6.40e-002"). Put a space wherever a sign follows a digit; the sign of an
# exponent follows an "e", not a digit, so it is left alone.
_RUN_TOGETHER = re.compile(r'(?<=[0-9])(?=[-+])')


def load_spe(
    filename: str | Path,
    wavelength: float | None = None,
    min_error: float | None = None,
) -> sc.DataArray:
    """
    Load an ``.spe`` file, converting each detector's scattering angle to Q.

    Each detector is given the Q it has at zero energy transfer. A detector at a fixed angle moves
    through Q as the energy transfer changes, so this is accurate close to the elastic line, which
    is where quasielastic data lives. The detectors are sorted by Q, one Q value each; to combine
    them into fewer Q bins, or to put several runs on the same Q axis, use ``Experiment.rebin``
    with the same Q bin edges for every run.

    Detectors that were masked in the reduction, written with an intensity of -1e30, are set to NaN
    and masked by a ``'null_data'`` mask.

    Examples
    --------
    ```python
    import easydynamics as edyn
    import scipp as sc

    data = edyn.load_spe('sample_2K.spe', min_error=0.01)
    experiment = edyn.Experiment(display_name='2K', data=data)
    experiment.rebin({'Q': sc.linspace('Q', 0.3, 2.0, 11, unit='1/angstrom')})
    ```

    Parameters
    ----------
    filename : str | Path
        Path to the ``.spe`` file.
    wavelength : float | None, default=None
        Incident wavelength in angstrom. If None, it is read from the ``Wavelength`` line of a
        ``.par`` file with the same name next to the ``.spe`` file, as the FOCUS reduction writes.
    min_error : float | None, default=None
        None keeps the errors as stored. A positive number raises every error below it to it, in
        the units of the intensity. NaN masks the bins with zero error with a ``'zero_error'``
        mask, so fits and rebinning leave them out; as these are usually empty bins, a floor is the
        better choice before rebinning low-count data.

    Returns
    -------
    sc.DataArray
        Dims ``('Q', 'energy')``, with ``Q`` in 1/angstrom, ``energy`` in meV, variances from the
        stored errors, and each detector's angle as a ``scattering_angle`` coordinate along Q.

    Raises
    ------
    ValueError
        If ``wavelength`` is not positive, or is None and no ``.par`` file gives it, the file is
        not laid out as an ``.spe`` file, or its angle block holds placeholder numbers rather than
        angles.
    """
    path = Path(filename)
    if wavelength is None:
        pars = [path.with_suffix(suffix) for suffix in ('.par', '.PAR')]
        par = next((par for par in pars if par.exists()), None)
        wavelength = _wavelength_from_par(par) if par is not None else None
        if wavelength is None:
            raise ValueError(
                f'No wavelength given, and no .par file with a Wavelength line next to '
                f'{path.name}. Pass wavelength=... in angstrom.'
            )
    if not isinstance(wavelength, Real) or not (wavelength > 0 and math.isfinite(wavelength)):
        raise ValueError(f'wavelength must be a positive number in angstrom, not {wavelength}')

    phi, energy, intensity, error = _read_spe(path)
    # Consecutive numbers counting up from 0, 0.5 or 1 are detector numbers standing in for angles.
    if phi.size > 2 and np.allclose(np.diff(phi), 1.0) and phi[0] in (0.0, 0.5, 1.0):
        raise ValueError(
            f'{path.name}: the angle block holds the placeholder numbers {phi[0]:g}, '
            f'{phi[1]:g}, ... rather than scattering angles; the angles are in a .par or .phx '
            'file, which load_spe does not read yet.'
        )
    if phi.size == intensity.shape[0] + 1:
        phi = 0.5 * (phi[:-1] + phi[1:])
    # Detectors on the far side of the beam may be written with negative angles.
    Q = 4.0 * np.pi * np.sin(np.radians(np.abs(phi)) / 2.0) / wavelength
    order = np.argsort(Q, kind='stable')

    null = intensity[order] <= _NULL_DATA
    data = sc.DataArray(
        data=sc.array(
            dims=['Q', 'energy'],
            values=np.where(null, np.nan, intensity[order]),
            variances=np.where(null, np.nan, error[order] ** 2),
        ),
        coords={
            'Q': sc.array(dims=['Q'], values=Q[order], unit='1/angstrom'),
            'energy': sc.array(dims=['energy'], values=energy, unit='meV'),
            'scattering_angle': sc.array(dims=['Q'], values=phi[order], unit='deg'),
        },
    )
    if null.any():
        data.masks['null_data'] = sc.array(dims=['Q', 'energy'], values=null)

    apply_min_error(data, min_error)
    return data


def _read_spe(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an ``.spe`` file into its angle and energy axes, intensities and errors.

    The first line holds the number of detectors and of energy channels. The rest is a series of
    blocks, each starting with a ``###`` line: the angles, the energies, and then the intensities
    and the errors of each detector in turn.

    Parameters
    ----------
    path : Path
        Path to the ``.spe`` file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The angles in degrees and the energies in meV, each as bin edges or centres as stored, and
        the intensities and errors, each with shape ``(n_detectors, n_energy)``.

    Raises
    ------
    ValueError
        If the counts on the first line do not match the blocks that follow.
    """
    lines = path.read_text(errors='replace').splitlines()
    try:
        n_phi, n_energy = (int(value) for value in lines[0].split()[:2])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f'{path.name}: the first line should hold the number of detectors and of energy '
            'channels'
        ) from error

    blocks = []
    for line in lines[1:]:
        if line.lstrip().startswith('###'):
            blocks.append([])
        elif blocks:
            blocks[-1].append(line)
    blocks = [_parse_numbers(block) for block in blocks]

    if len(blocks) != 2 + 2 * n_phi:
        raise ValueError(
            f'{path.name}: expected {2 + 2 * n_phi} blocks for {n_phi} detectors, found '
            f'{len(blocks)}'
        )
    phi, energy = blocks[0], blocks[1]
    for name, axis, n in (('angle', phi, n_phi), ('energy', energy, n_energy)):
        if axis.size not in (n, n + 1):
            raise ValueError(f'{path.name}: the {name} block has {axis.size} values, not {n}')
    data = blocks[2:]
    if any(block.size != n_energy for block in data):
        raise ValueError(f'{path.name}: every detector should have {n_energy} values')

    return phi, energy, np.array(data[0::2]), np.array(data[1::2])


def _parse_numbers(lines: list[str]) -> np.ndarray:
    """
    Parse the lines of one block into a flat array of numbers.

    Parameters
    ----------
    lines : list[str]
        The lines of the block, without its ``###`` line.

    Returns
    -------
    np.ndarray
        The numbers, in the order they appear.
    """
    # Fortran programs may write the exponent with a D, as in 1.0D-02.
    text = ' '.join(lines).replace('D', 'e').replace('d', 'e')
    return np.array(_RUN_TOGETHER.sub(' ', text).split(), dtype=float)


def _wavelength_from_par(path: Path) -> float | None:
    """
    Read the incident wavelength from the ``Wavelength`` line of a ``.par`` file.

    Parameters
    ----------
    path : Path
        Path to the ``.par`` file.

    Returns
    -------
    float | None
        The wavelength in angstrom, or None if the file has no such line, as a ``.par`` file of
        detector positions does not.
    """
    for line in path.read_text(errors='replace').splitlines():
        key, _, value = line.partition(':')
        if key.strip().lower().startswith('wavelength'):
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
            if match:
                return float(match.group())
    return None
