# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Write the fake ``.spe`` files used by the loader tests.

The files copy the layout of ``.spe`` and ``.par`` files written by the FOCUS reduction, but every
number and string in them is made up. Remake them with:

    pixi run python tests/data/make_fake_spe.py

``fake.spe`` and ``fake.par``
    Angles and energies as bin edges, numbers in fixed-width fields with three-digit exponents
    so that a negative number runs into the one before it, one masked detector (-1e30) and one
    channel with zero error. The ``.par`` file gives the wavelength.
``fake_centres.spe``
    Angles and energies as bin centres, detectors out of angle order, and no ``.par`` file.
"""

from pathlib import Path

import numpy as np

OUTPUT_DIRECTORY = Path(__file__).parent / 'spe'


def _number(value: float) -> str:
    """
    Format a number as the FOCUS reduction does, ten characters wide with a three-digit exponent.

    Parameters
    ----------
    value : float
        The number.

    Returns
    -------
    str
        The number, such as ``' 4.43e-002'``; a negative one fills all ten characters.
    """
    mantissa, exponent = f'{value:.2e}'.split('e')
    return f'{mantissa}e{int(exponent):+04d}'.rjust(10)


def _block(title: str, values: np.ndarray) -> list[str]:
    """
    Write a ``###`` block, eight numbers to a line.

    Parameters
    ----------
    title : str
        The text after ``###``.
    values : np.ndarray
        The numbers.

    Returns
    -------
    list[str]
        The lines of the block.
    """
    numbers = [_number(value) for value in values]
    return [f'###  {title}'] + [''.join(numbers[i : i + 8]) for i in range(0, len(numbers), 8)]


def write_spe(path: Path, phi: np.ndarray, energy: np.ndarray, intensity: np.ndarray) -> None:
    """
    Write an ``.spe`` file.

    Parameters
    ----------
    path : Path
        Where to write it.
    phi : np.ndarray
        Angles in degrees, edges or centres.
    energy : np.ndarray
        Energies in meV, edges or centres.
    intensity : np.ndarray
        Intensities, shape ``(n_detectors, n_energy)``; errors are made up from them.
    """
    error = 0.1 * np.sqrt(np.abs(intensity))
    error[np.abs(intensity) > 1e20] = 0.0
    lines = [f'{intensity.shape[0]:12d}{intensity.shape[1]:12d}']
    lines += _block('Phi  Phi:degrees', phi)
    lines += _block('Energy / meV', energy)
    for signal, uncertainty in zip(intensity, error, strict=True):
        lines += _block('S(phi,w)', signal)
        lines += _block('error', uncertainty)
    path.write_text('\n'.join(lines) + '\n')


def _intensity(n_detectors: int, energy: np.ndarray) -> np.ndarray:
    """
    Make up intensities: a peak at zero energy on a background, with some below zero.

    Parameters
    ----------
    n_detectors : int
        Number of detectors.
    energy : np.ndarray
        Energy of each channel, in meV.

    Returns
    -------
    np.ndarray
        Intensities, shape ``(n_detectors, len(energy))``.
    """
    peak = 1.0 / (1.0 + (energy / 0.2) ** 2)
    return np.outer(1.0 + 0.1 * np.arange(n_detectors), peak) - 0.05


if __name__ == '__main__':
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    # Five detectors; the last is out of angle order.
    phi_edges = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    energy_edges = np.linspace(-1.0, 1.0, 11)
    centres = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    intensity = _intensity(5, centres)
    intensity[2] = -1e30  # masked detector
    intensity[3, 4] = 0.0  # a channel with zero error
    write_spe(OUTPUT_DIRECTORY / 'fake.spe', phi_edges, energy_edges, intensity)
    (OUTPUT_DIRECTORY / 'fake.par').write_text(
        'Instrument: FAKE\n'
        'Sample: Fake sample\n'
        'Title: Made-up data for tests\n'
        'Number of Channels: 10\n'
        'Number of Groups: 5\n'
        'Wavelength / A: 4.0\n'
    )

    write_spe(
        OUTPUT_DIRECTORY / 'fake_centres.spe',
        np.array([60.0, 20.0, 40.0]),
        centres,
        _intensity(3, centres),
    )
