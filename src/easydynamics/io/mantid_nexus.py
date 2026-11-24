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

    Bins that no detector reached, marked by Mantid with a zero fractional area, are masked by an
    ``'uncovered'`` mask, so fits and rebinning leave them out rather than read them as zero
    intensity.

    Bins with zero counts are usually stored with zero error, which a fit cannot weight by. Use
    ``min_error`` to choose what happens to them.

    Examples
    --------
    ```python
    import easydynamics as edyn

    data = edyn.load_mantid_nexus('iris108490_graphite002_sqw.nxs', min_error=0.001)
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
        the units of the intensity. NaN masks the bins with zero error with a ``'zero_error'``
        mask, so fits and rebinning leave them out; as these are usually empty bins, a floor is the
        better choice before rebinning low-count data.

    Returns
    -------
    sc.DataArray
        Dims ``('Q', 'energy')``, with ``Q`` in 1/angstrom, ``energy`` in meV and variances from
        the stored errors, and any masks described above.

    Raises
    ------
    ValueError
        If the file holds several workspaces and no ``entry`` is given, ``entry`` is not in the
        file, the workspace axes are not energy transfer and momentum transfer, the energy bins
        differ between spectra, or the workspace is a rebinned one that was saved before Mantid
        divided it by its fractional areas or with squared errors.
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
            values = workspace[axis_name][()]
            if values.ndim != 1:
                raise ValueError(
                    f'{axis_name} differs between spectra; rebin the workspace in Mantid so every '
                    'spectrum has the same energy bins.'
                )
            dims.append(dim)
            coords[dim] = sc.array(dims=[dim], values=values, unit=unit)

        data = sc.DataArray(
            data=sc.array(
                dims=dims, values=workspace['values'][()], variances=workspace['errors'][()] ** 2
            ),
            coords=coords,
        )
        if 'frac_area' in workspace:
            # A finalized RebinnedOutput already holds intensity divided by the fractional area.
            attrs = workspace['frac_area'].attrs
            if _decode(attrs.get('finalized', b'1')) != '1':
                raise ValueError(
                    'The workspace was saved before Mantid normalised it by its fractional '
                    'areas; finalize it in Mantid and save it again.'
                )
            if _decode(attrs.get('sqrd_errors', b'0')) != '0':
                raise ValueError(
                    'The workspace was saved with squared errors, which this loader does not '
                    'read; save it from Mantid with ordinary errors.'
                )
            data.masks['uncovered'] = sc.array(dims=dims, values=workspace['frac_area'][()] == 0)

    apply_min_error(data, min_error)
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
