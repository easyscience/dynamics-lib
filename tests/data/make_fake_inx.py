# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Write the fake ``.inx`` files used by the loader tests.

The files copy the layout of ``.inx`` files from the ILL, but every number and string in them is
made up. Remake them with:

    pixi run python tests/data/make_fake_inx.py

``fake.inx``
    Three spectra as IN16B's reduction writes them: Q first on the information line, no incident
    energy, energies in microelectronvolts, and one point with zero error.
``fake_angle.inx``
    One spectrum whose information line starts with a scattering angle and gives an incident
    energy, with an energy too wide for its column (``*****``) and an infinite intensity.
"""

from pathlib import Path

import numpy as np

OUTPUT_DIRECTORY = Path(__file__).parent / 'inx'
ENERGY = np.linspace(-20.0, 20.0, 9)


def intensity(index: int) -> np.ndarray:
    """
    Make up the intensity of one spectrum: a peak at zero energy that broadens with the index.

    Parameters
    ----------
    index : int
        The spectrum, counted from zero.

    Returns
    -------
    np.ndarray
        The intensity at each energy.
    """
    return 0.01 / (1.0 + (ENERGY / (2.0 * (1 + index))) ** 2) + 1e-4


def block(info: str, rows: list[str]) -> list[str]:
    """
    Write one spectrum: the header line, a title, two information lines and the data rows.

    Parameters
    ----------
    info : str
        The first information line.
    rows : list[str]
        The data rows.

    Returns
    -------
    list[str]
        The lines of the block.
    """
    header = f'{3 + len(rows):6d}    1    2    0    0    0    0{len(rows):5d}'
    second_info = '                  0.0000  8.0000  0.0000'
    return [header, 'Made-up sample for tests', info, second_info, *rows]


if __name__ == '__main__':
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    lines = []
    for index, Q in enumerate([0.5, 1.0, 1.5]):
        y = intensity(index)
        e = np.sqrt(y) * 0.01
        if index == 2:
            y[0], e[0] = 0.0, 0.0
        rows = [f'{x:15.5f} {v:12.5e} {s:11.4e}' for x, v, s in zip(ENERGY, y, e, strict=True)]
        lines += block(f'{Q:7.4f}   0.000  0.0000 1024.000   1.0 0', rows)
    (OUTPUT_DIRECTORY / 'fake.inx').write_text('\n'.join(lines) + '\n')

    y = intensity(0)
    rows = [f'{x:15.5f} {v:12.5e} {0.1 * v:11.4e}' for x, v in zip(ENERGY, y, strict=True)]
    rows[0] = '     **********  1.00000e-31  1.0000e-33'
    rows[1] = f'{ENERGY[1]:15.5f}          Inf  1.0000e-30'
    info = ' 105.00   3.000  1.2033  300.000   1.0 0'
    (OUTPUT_DIRECTORY / 'fake_angle.inx').write_text('\n'.join(block(info, rows)) + '\n')
