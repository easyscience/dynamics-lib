# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Write the fake ``SaveAscii`` files used by the loader tests.

The files copy the layout Mantid's ``SaveAscii`` gives a workspace of QENS spectra, but every
number in them is made up. Remake them with:

    pixi run python tests/data/make_fake_mantid_ascii.py

``fake.dat``
    Comma-separated, with a header and a spectrum number before each of three spectra, one
    point of which has zero error, as Mantid writes a bin with no counts.
``fake_no_spectrum_numbers.dat``
    Tab-separated, with an X error column and without spectrum numbers.
"""

from pathlib import Path

import numpy as np

OUTPUT_DIRECTORY = Path(__file__).parent / 'mantid_ascii'
ENERGY = np.linspace(-0.03, 0.03, 7)


def spectrum(index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Make up the intensity and error of one spectrum: a peak at zero energy on a background.

    Parameters
    ----------
    index : int
        The spectrum, counted from zero; the peak broadens with it.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The intensity and its error at each energy.
    """
    width = 0.005 * (1 + index)
    intensity = 10.0 / (1.0 + (ENERGY / width) ** 2) + 0.5
    return intensity, np.sqrt(intensity)


if __name__ == '__main__':
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    lines = ['# X , Y , E Distribution=true']
    for index in range(3):
        intensity, error = spectrum(index)
        if index == 1:
            intensity[0], error[0] = 0.0, 0.0
        lines.append(str(index + 1))
        lines += [f'{x:g},{y:g},{e:g}' for x, y, e in zip(ENERGY, intensity, error, strict=True)]
    (OUTPUT_DIRECTORY / 'fake.dat').write_text('\n'.join(lines) + '\n')

    lines = ['# X \t Y \t E \t DX']
    for index in range(2):
        intensity, error = spectrum(index)
        lines += [
            f'{x:g}\t{y:g}\t{e:g}\t0.001' for x, y, e in zip(ENERGY, intensity, error, strict=True)
        ]
    (OUTPUT_DIRECTORY / 'fake_no_spectrum_numbers.dat').write_text('\n'.join(lines) + '\n')
