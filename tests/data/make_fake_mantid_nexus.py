# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Write the fake Mantid NeXus files used by the loader tests.

The files copy the layout ``SaveNexusProcessed`` gives to reduced QENS data, as found in files
from IN16B and from IRIS (after ``SofQW``), but every number and string in them is made up. They
are small, so they are kept in the repository; run this script to remake them:

    pixi run python tests/data/make_fake_mantid_nexus.py

``fake_in16b.nxs``
    Q as one point per spectrum, energy as bin edges, a handful of empty bins with zero error.
``fake_iris_sqw.nxs``
    Q and energy both as bin edges, with the ``frac_area`` of a ``SofQW`` rebinned workspace:
    bins no detector reached have zero area, zero intensity and zero error.
``fake_workspace_group.nxs``
    Two workspaces in one file, as a saved workspace group gives.
"""

from pathlib import Path

import h5py
import numpy as np

OUTPUT_DIRECTORY = Path(__file__).parent / 'mantid_nexus'


def _string(value: str) -> np.ndarray:
    """
    Return a string as the one-element fixed-length byte array Mantid writes.

    Parameters
    ----------
    value : str
        The string.

    Returns
    -------
    np.ndarray
        A one-element ``S<n>`` array.
    """
    return np.array([value.encode()])


def _group(parent: h5py.Group, name: str, nx_class: str, attrs: dict | None = None) -> h5py.Group:
    """
    Create a NeXus group.

    Parameters
    ----------
    parent : h5py.Group
        The group to create it in.
    name : str
        Its name.
    nx_class : str
        Its ``NX_class``.
    attrs : dict | None, default=None
        Further attributes.

    Returns
    -------
    h5py.Group
        The new group.
    """
    group = parent.create_group(name)
    group.attrs['NX_class'] = np.bytes_(nx_class.encode())
    for key, value in (attrs or {}).items():
        group.attrs[key] = value
    return group


def _log(logs: h5py.Group, name: str, value: np.ndarray, time: np.ndarray | None = None) -> None:
    """
    Add a sample log, the way Mantid stores run logs.

    Parameters
    ----------
    logs : h5py.Group
        The ``logs`` group.
    name : str
        Name of the log.
    value : np.ndarray
        Its values.
    time : np.ndarray | None, default=None
        Seconds since the start, for a time series log.
    """
    log = _group(logs, name, 'NXlog')
    log.create_dataset('value', data=value).attrs['units'] = np.bytes_(b' ')
    if time is not None:
        log.create_dataset('time', data=time).attrs['start'] = np.bytes_(b'2000-01-01T00:00:00')


def _sqw(Q: np.ndarray, energy: np.ndarray, width: float) -> np.ndarray:
    """
    Make up an intensity: a Lorentzian whose width grows with Q, on a flat background.

    Parameters
    ----------
    Q : np.ndarray
        Q of each spectrum, in 1/angstrom.
    energy : np.ndarray
        Energy of each channel, in meV.
    width : float
        Half width at half maximum at Q = 1/angstrom, in meV.

    Returns
    -------
    np.ndarray
        Intensity with shape ``(len(Q), len(energy))``.
    """
    gamma = width * Q[:, None] ** 2
    return gamma / np.pi / (energy[None, :] ** 2 + gamma**2) + 0.01


def _write_entry(
    file: h5py.File,
    entry_name: str,
    instrument: str,
    title: str,
    Q_axis: np.ndarray,
    energy_edges: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    frac_area: np.ndarray | None = None,
    temperature: float | None = None,
) -> None:
    """
    Write one workspace entry laid out as ``SaveNexusProcessed`` lays it out.

    Parameters
    ----------
    file : h5py.File
        The file to write to.
    entry_name : str
        Name of the entry, ``mantid_workspace_<n>``.
    instrument : str
        Instrument name.
    title : str
        Run title.
    Q_axis : np.ndarray
        The spectrum axis, Q points or Q bin edges.
    energy_edges : np.ndarray
        The energy bin edges.
    values : np.ndarray
        Intensity, shape ``(n_Q, n_energy)``.
    errors : np.ndarray
        Errors on the intensity.
    frac_area : np.ndarray | None, default=None
        Fractional bin areas of a rebinned workspace.
    temperature : float | None, default=None
        Sample temperature to store as a time series log.
    """
    entry = _group(file, entry_name, 'NXentry')
    entry.create_dataset('definition', data=_string('NXprocessed')).attrs['Version'] = np.bytes_(
        b'1.0'
    )
    entry.create_dataset('title', data=_string(title))
    entry.create_dataset('workspace_name', data=_string(f'fake_{instrument.lower()}'))

    instrument_group = _group(entry, 'instrument', 'NXinstrument', {'version': np.int32(1)})
    instrument_group.create_dataset('name', data=_string(instrument))
    detector = _group(instrument_group, 'detector', 'NXdetector')
    detector.create_dataset('spectra', data=np.arange(1, values.shape[0] + 1, dtype=np.int32))

    _group(entry, 'sample', 'NXsample', {'name': np.bytes_(b' '), 'version': np.int32(1)})
    process = _group(entry, 'process', 'NXprocess')
    note = _group(process, 'MantidAlgorithm_1', 'NXnote')
    note.create_dataset('data', data=_string('Algorithm: FakeReduction v1'))

    logs = _group(entry, 'logs', 'NXgroup', {'version': np.int32(1)})
    _log(logs, 'run_number', _string('000001'))
    _log(logs, 'run_title', _string(title))
    if temperature is not None:
        time = np.arange(5, dtype=float) * 60.0
        _log(logs, 'Sample', temperature + 0.1 * np.sin(time), time=time)

    workspace = _group(entry, 'workspace', 'NXdata')
    energy = workspace.create_dataset('axis1', data=energy_edges)
    energy.attrs['distribution'] = np.bytes_(b'1')
    energy.attrs['units'] = np.bytes_(b'DeltaE')
    workspace.create_dataset('axis2', data=Q_axis).attrs['units'] = np.bytes_(b'MomentumTransfer')
    signal = workspace.create_dataset('values', data=values)
    signal.attrs['axes'] = np.bytes_(b'axis2,axis1')
    signal.attrs['signal'] = np.int32(1)
    signal.attrs['unit_label'] = np.bytes_(b'Counts')
    signal.attrs['units'] = np.bytes_(b'')
    workspace.create_dataset('errors', data=errors)
    if frac_area is not None:
        area = workspace.create_dataset('frac_area', data=frac_area)
        area.attrs['finalized'] = np.bytes_(b'1')
        area.attrs['sqrd_errors'] = np.bytes_(b'0')


def _file(path: Path) -> h5py.File:
    """
    Open a new file with the root attributes Mantid writes.

    Parameters
    ----------
    path : Path
        Where to write it.

    Returns
    -------
    h5py.File
        The open file.
    """
    file = h5py.File(path, 'w')
    file.attrs['NeXus_version'] = np.bytes_(b'4.4.3')
    file.attrs['file_name'] = np.bytes_(path.name.encode())
    file.attrs['file_time'] = np.bytes_(b'2000-01-01T00:00:00')
    return file


def write_fake_in16b(path: Path) -> None:
    """
    Write the IN16B-like file: Q as points, a few empty bins with zero error.

    Parameters
    ----------
    path : Path
        Where to write it.
    """
    Q = np.array([0.2, 0.6, 1.0, 1.4])
    energy_edges = np.linspace(-0.03, 0.03, 13)
    energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    values = 100.0 * _sqw(Q, energy, width=0.002)
    errors = np.sqrt(values)
    values[[0, 3], [0, 11]] = 0.0
    errors[[0, 3], [0, 11]] = 0.0
    with _file(path) as file:
        _write_entry(file, 'mantid_workspace_1', 'IN16BF', ' ', Q, energy_edges, values, errors)


def write_fake_iris_sqw(path: Path) -> None:
    """
    Write the IRIS-like file: Q and energy as edges, with a rebinned workspace's fractional area.

    Parameters
    ----------
    path : Path
        Where to write it.
    """
    Q_edges = np.linspace(0.4, 1.6, 4)
    energy_edges = np.linspace(-0.4, 1.2, 17)
    Q = 0.5 * (Q_edges[:-1] + Q_edges[1:])
    energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    values = 0.01 * _sqw(Q, energy, width=0.05)
    errors = 0.1 * np.sqrt(values)
    frac_area = np.ones_like(values)
    # No detector reaches the high-energy corner at low Q.
    frac_area[0, -3:] = 0.0
    values[0, -3:] = 0.0
    errors[0, -3:] = 0.0
    # Covered bins in which nothing was counted.
    values[2, :2] = 0.0
    errors[2, :2] = 0.0
    with _file(path) as file:
        _write_entry(
            file,
            'mantid_workspace_1',
            'IRIS',
            'Fake sample 10K',
            Q_edges,
            energy_edges,
            values,
            errors,
            frac_area=frac_area,
            temperature=10.0,
        )


def write_fake_workspace_group(path: Path) -> None:
    """
    Write a file holding two workspaces, with 2 and 3 Q bins.

    Parameters
    ----------
    path : Path
        Where to write it.
    """
    energy_edges = np.linspace(-0.4, 0.4, 9)
    energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    with _file(path) as file:
        for index, n_Q in enumerate([2, 3], start=1):
            Q_edges = np.linspace(0.4, 1.6, n_Q + 1)
            Q = 0.5 * (Q_edges[:-1] + Q_edges[1:])
            values = _sqw(Q, energy, width=0.05)
            _write_entry(
                file,
                f'mantid_workspace_{index}',
                'IRIS',
                f'Fake group member {index}',
                Q_edges,
                energy_edges,
                values,
                0.1 * np.sqrt(values),
            )


if __name__ == '__main__':
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    write_fake_in16b(OUTPUT_DIRECTORY / 'fake_in16b.nxs')
    write_fake_iris_sqw(OUTPUT_DIRECTORY / 'fake_iris_sqw.nxs')
    write_fake_workspace_group(OUTPUT_DIRECTORY / 'fake_workspace_group.nxs')
