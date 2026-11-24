# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Read the NeXus files Mantid writes with ``SaveNexusProcessed``.

These hold one reduced workspace per ``mantid_workspace_<n>`` entry. For QENS data the workspace is
normally the output of ``SofQW`` (IRIS, OSIRIS) or of the ILL reduction (IN16B): the X axis is
energy transfer and the spectrum axis is momentum transfer.
"""

from pathlib import Path

import h5py
import numpy as np
import scipp as sc

from easydynamics.io.utils import apply_min_error

# Mantid unit IDs, and the scipp coordinate each one becomes.
_AXIS_UNITS = {
    'DeltaE': ('energy', 'meV'),
    'MomentumTransfer': ('Q', '1/angstrom'),
}


def load_mantid_nexus(
    filename: str | Path,
    entry: str | None = None,
    min_error: float | None = None,
) -> sc.DataArray:
    """
    Load an S(Q, energy) workspace saved by Mantid's ``SaveNexusProcessed``.

    The result can be given straight to an ``Experiment``. Coordinates are kept as Mantid wrote
    them, so they may be bin edges, which ``Experiment`` handles.

    Bins that no detector reached, marked by Mantid with a zero fractional area, are set to NaN so
    they are masked out of fits rather than read as zero intensity.

    Bins with zero counts are usually stored with zero error, which a fit cannot weight by. Use
    ``min_error`` to choose what happens to them.

    Examples
    --------
    ```python
    import easydynamics as edyn

    data = edyn.load_mantid_nexus('iris108490_graphite002_sqw.nxs')
    experiment = edyn.Experiment(display_name='NP+TCZ 2K', data=data)
    ```

    Parameters
    ----------
    filename : str | Path
        Path to the ``.nxs`` file.
    entry : str | None, default=None
        Name of the workspace entry to read, such as ``'mantid_workspace_2'``. Needed only when the
        file holds several workspaces.
    min_error : float | None, default=None
        None keeps the errors as stored. A positive number raises every error below it to it, in
        the units of the intensity. NaN sets the bins with zero error to NaN, so fits leave them
        out.

    Returns
    -------
    sc.DataArray
        Dims ``('Q', 'energy')``, with ``Q`` in 1/angstrom, ``energy`` in meV and variances from
        the stored errors.

    Raises
    ------
    ValueError
        If the file holds several workspaces and no ``entry`` is given, ``entry`` is not in the
        file, or the workspace axes are not energy transfer and momentum transfer.
    """
    with h5py.File(filename, 'r') as file:
        entries = sorted(name for name in file if name.startswith('mantid_workspace_'))
        if entry is None:
            if len(entries) != 1:
                raise ValueError(
                    f'The file holds {len(entries)} workspaces {entries}; pick one with entry=...'
                )
            entry = entries[0]
        elif entry not in entries:
            raise ValueError(f'No workspace {entry!r} in the file; it holds {entries}')
        workspace = file[entry]['workspace']

        # 'axes' lists the axes slowest first, so for a Workspace2D it is 'axis2,axis1'.
        axis_names = _decode(workspace['values'].attrs['axes']).split(',')
        dims = []
        coords = {}
        for axis_name in axis_names:
            unit_id = _decode(workspace[axis_name].attrs.get('units', b''))
            if unit_id not in _AXIS_UNITS:
                raise ValueError(
                    f'{axis_name} has Mantid unit {unit_id!r}; expected one of '
                    f'{list(_AXIS_UNITS)}. Convert the workspace to S(Q, energy) in Mantid '
                    'first, for example with SofQW.'
                )
            dim, unit = _AXIS_UNITS[unit_id]
            dims.append(dim)
            coords[dim] = sc.array(dims=[dim], values=workspace[axis_name][()], unit=unit)

        values = workspace['values'][()]
        variances = workspace['errors'][()] ** 2
        if 'frac_area' in workspace:
            uncovered = workspace['frac_area'][()] == 0
            values[uncovered] = np.nan
            variances[uncovered] = np.nan
    values, variances = apply_min_error(values, variances, min_error)

    data = sc.DataArray(
        data=sc.array(dims=dims, values=values, variances=variances),
        coords=coords,
    )
    return data.transpose(['Q', 'energy']).copy()


def _decode(value: bytes | str | np.ndarray) -> str:
    """
    Turn an HDF5 string attribute into a Python string.

    Parameters
    ----------
    value : bytes | str | np.ndarray
        The attribute as h5py returns it.

    Returns
    -------
    str
        The decoded string.
    """
    if isinstance(value, np.ndarray):
        value = value.flat[0]
    return value.decode() if isinstance(value, bytes) else str(value)
