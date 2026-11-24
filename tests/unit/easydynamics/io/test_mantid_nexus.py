# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import h5py
import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment
from easydynamics.io import load_mantid_nexus


def write_workspace(
    file: h5py.File,
    entry: str = 'mantid_workspace_1',
    n_Q: int = 3,
    n_energy: int = 4,
    Q_edges: bool = True,
    Q_unit: str = 'MomentumTransfer',
    frac_area: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Write a workspace laid out the way SaveNexusProcessed lays out an S(Q, energy) one."""
    workspace = file.create_group(f'{entry}/workspace')
    values = np.arange(n_Q * n_energy, dtype=float).reshape(n_Q, n_energy)
    errors = 0.5 * np.ones_like(values)
    workspace.create_dataset('values', data=values).attrs['axes'] = np.bytes_(b'axis2,axis1')
    workspace.create_dataset('errors', data=errors)
    energy = workspace.create_dataset('axis1', data=np.linspace(-1.0, 1.0, n_energy + 1))
    energy.attrs['units'] = np.bytes_(b'DeltaE')
    Q = workspace.create_dataset('axis2', data=np.linspace(0.5, 1.5, n_Q + Q_edges))
    Q.attrs['units'] = np.bytes_(Q_unit.encode())
    if frac_area is not None:
        workspace.create_dataset('frac_area', data=frac_area)
    return values, errors


class TestLoadMantidNexus:
    #############
    # Reading
    #############

    def test_reads_values_errors_and_axes(self, tmp_path):
        # WHEN
        path = tmp_path / 'sqw.nxs'
        with h5py.File(path, 'w') as file:
            values, errors = write_workspace(file)

        # THEN
        data = load_mantid_nexus(path)

        # EXPECT
        assert data.dims == ('Q', 'energy')
        assert data.coords['Q'].unit == sc.Unit('1/angstrom')
        assert data.coords['energy'].unit == sc.Unit('meV')
        np.testing.assert_array_equal(data.coords['Q'].values, np.linspace(0.5, 1.5, 4))
        np.testing.assert_array_equal(data.coords['energy'].values, np.linspace(-1.0, 1.0, 5))
        np.testing.assert_array_equal(data.values, values)
        np.testing.assert_array_equal(data.variances, errors**2)

    def test_reads_Q_points(self, tmp_path):
        # WHEN the spectrum axis holds one Q per spectrum, as IN16B writes it
        path = tmp_path / 'sqw.nxs'
        with h5py.File(path, 'w') as file:
            write_workspace(file, Q_edges=False)

        # THEN
        data = load_mantid_nexus(path)

        # EXPECT
        np.testing.assert_array_equal(data.coords['Q'].values, np.linspace(0.5, 1.5, 3))

    def test_uncovered_bins_become_nan(self, tmp_path):
        # WHEN
        path = tmp_path / 'sqw.nxs'
        frac_area = np.ones((3, 4))
        frac_area[1, 2] = 0.0
        with h5py.File(path, 'w') as file:
            write_workspace(file, frac_area=frac_area)

        # THEN
        data = load_mantid_nexus(path)

        # EXPECT only the bin no detector reached is NaN
        assert np.isnan(data.values[1, 2])
        assert np.isnan(data.variances[1, 2])
        assert np.isnan(data.values).sum() == 1

    def test_result_is_accepted_by_experiment(self, tmp_path):
        # WHEN
        path = tmp_path / 'sqw.nxs'
        with h5py.File(path, 'w') as file:
            write_workspace(file)

        # THEN
        experiment = Experiment(data=load_mantid_nexus(path))

        # EXPECT the bin edges are turned into centres for fitting
        np.testing.assert_allclose(experiment.binned_data.coords['Q'].values, [2 / 3, 1.0, 4 / 3])

    #############
    # Choosing a workspace
    #############

    def test_entry_picks_one_of_several_workspaces(self, tmp_path):
        # WHEN
        path = tmp_path / 'group.nxs'
        with h5py.File(path, 'w') as file:
            write_workspace(file, entry='mantid_workspace_1', n_Q=3)
            write_workspace(file, entry='mantid_workspace_2', n_Q=5)

        # THEN
        data = load_mantid_nexus(path, entry='mantid_workspace_2')

        # EXPECT
        assert data.sizes['Q'] == 5

    def test_several_workspaces_without_entry_raises(self, tmp_path):
        # WHEN
        path = tmp_path / 'group.nxs'
        with h5py.File(path, 'w') as file:
            write_workspace(file, entry='mantid_workspace_1')
            write_workspace(file, entry='mantid_workspace_2')

        # THEN EXPECT
        with pytest.raises(ValueError, match='holds 2 workspaces'):
            load_mantid_nexus(path)

    def test_unknown_entry_raises(self, tmp_path):
        # WHEN
        path = tmp_path / 'sqw.nxs'
        with h5py.File(path, 'w') as file:
            write_workspace(file)

        # THEN EXPECT
        with pytest.raises(ValueError, match="No workspace 'mantid_workspace_7'"):
            load_mantid_nexus(path, entry='mantid_workspace_7')

    #############
    # Axes that are not S(Q, energy)
    #############

    def test_axis_not_in_Q_raises(self, tmp_path):
        # WHEN the spectrum axis is still scattering angle, before SofQW
        path = tmp_path / 'red.nxs'
        with h5py.File(path, 'w') as file:
            write_workspace(file, Q_unit='Degrees')

        # THEN EXPECT
        with pytest.raises(ValueError, match=r"Mantid unit 'Degrees'.*SofQW"):
            load_mantid_nexus(path)
