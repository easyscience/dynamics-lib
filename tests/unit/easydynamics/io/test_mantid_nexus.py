# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment
from easydynamics.io import load_mantid_nexus

# Fake files laid out like real IN16B and IRIS ones; see tests/data/make_fake_mantid_nexus.py.
DATA = Path(__file__).resolve().parents[3] / 'data' / 'mantid_nexus'
IN16B = DATA / 'fake_in16b.nxs'
IRIS = DATA / 'fake_iris_sqw.nxs'
GROUP = DATA / 'fake_workspace_group.nxs'


def stored(path: Path, name: str) -> np.ndarray:
    """Read a dataset of the first workspace straight from the file."""
    with h5py.File(path, 'r') as file:
        return file['mantid_workspace_1/workspace'][name][()]


class TestLoadMantidNexus:
    #############
    # Reading
    #############

    def test_reads_Q_points_and_energy_edges(self):
        # THEN
        data = load_mantid_nexus(IN16B)

        # EXPECT
        assert data.dims == ('Q', 'energy')
        assert data.coords['Q'].unit == sc.Unit('1/angstrom')
        assert data.coords['energy'].unit == sc.Unit('meV')
        np.testing.assert_array_equal(data.coords['Q'].values, stored(IN16B, 'axis2'))
        np.testing.assert_array_equal(data.coords['energy'].values, stored(IN16B, 'axis1'))
        np.testing.assert_array_equal(data.values, stored(IN16B, 'values'))
        np.testing.assert_array_equal(data.variances, stored(IN16B, 'errors') ** 2)

    def test_reads_Q_edges(self):
        # THEN
        data = load_mantid_nexus(IRIS)

        # EXPECT
        assert data.sizes == {'Q': 3, 'energy': 16}
        np.testing.assert_array_equal(data.coords['Q'].values, stored(IRIS, 'axis2'))

    def test_accepts_a_string_path(self):
        # THEN
        data = load_mantid_nexus(str(IN16B))

        # EXPECT
        assert data.sizes == {'Q': 4, 'energy': 12}

    def test_uncovered_bins_become_nan(self):
        # THEN
        data = load_mantid_nexus(IRIS)

        # EXPECT exactly the bins with zero fractional area are NaN
        uncovered = stored(IRIS, 'frac_area') == 0
        assert uncovered.any()
        np.testing.assert_array_equal(np.isnan(data.values), uncovered)
        np.testing.assert_array_equal(np.isnan(data.variances), uncovered)

    def test_result_is_accepted_by_experiment(self):
        # THEN
        experiment = Experiment(data=load_mantid_nexus(IRIS))

        # EXPECT the bin edges are turned into centres for fitting
        np.testing.assert_allclose(experiment.binned_data.coords['Q'].values, [0.6, 1.0, 1.4])

    #############
    # Zero errors
    #############

    def test_zero_errors_are_kept_by_default(self):
        # THEN
        data = load_mantid_nexus(IN16B)

        # EXPECT
        assert (data.variances == 0).sum() == 2

    def test_min_error_raises_small_errors(self):
        # THEN
        data = load_mantid_nexus(IN16B, min_error=0.5)

        # EXPECT
        assert data.variances.min() == pytest.approx(0.25)
        np.testing.assert_array_equal(data.values, stored(IN16B, 'values'))

    def test_min_error_nan_masks_zero_error_bins(self):
        # THEN
        data = load_mantid_nexus(IRIS, min_error=np.nan)

        # EXPECT the covered empty bins are masked as well as the uncovered ones
        nan = np.isnan(data.values)
        assert nan.sum() == 5
        assert nan[2, :2].all()
        assert not (data.variances[~nan] == 0).any()

    def test_min_error_leaves_uncovered_bins_nan(self):
        # THEN
        data = load_mantid_nexus(IRIS, min_error=1.0)

        # EXPECT a bin no detector reached gets no error, and so stays out of fits
        assert np.isnan(data.variances[0, -3:]).all()

    def test_data_with_min_error_can_be_fitted(self):
        # WHEN
        experiment = Experiment(data=load_mantid_nexus(IN16B, min_error=1.0))

        # THEN
        _, _, weights, _ = experiment.extract_x_y_weights_only_finite(Q_index=0)

        # EXPECT
        assert np.isfinite(weights).all()

    def test_invalid_min_error_raises(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='min_error must be positive'):
            load_mantid_nexus(IN16B, min_error=0.0)

    #############
    # Choosing a workspace
    #############

    @pytest.mark.parametrize(
        ('entry', 'n_Q'), [('mantid_workspace_1', 2), ('mantid_workspace_2', 3)]
    )
    def test_entry_picks_one_of_several_workspaces(self, entry, n_Q):
        # THEN
        data = load_mantid_nexus(GROUP, entry=entry)

        # EXPECT
        assert data.sizes['Q'] == n_Q

    def test_several_workspaces_without_entry_raises(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='holds 2 workspaces'):
            load_mantid_nexus(GROUP)

    def test_unknown_entry_raises(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match="No workspace 'mantid_workspace_7'"):
            load_mantid_nexus(IN16B, entry='mantid_workspace_7')

    #############
    # Axes that are not S(Q, energy)
    #############

    def test_axis_not_in_Q_raises(self, tmp_path):
        # WHEN the spectrum axis is still scattering angle, as before SofQW
        path = tmp_path / 'not_sqw.nxs'
        with h5py.File(path, 'w') as file:
            workspace = file.create_group('mantid_workspace_1/workspace')
            workspace.create_dataset('values', data=np.ones((2, 3))).attrs['axes'] = np.bytes_(
                b'axis2,axis1'
            )
            workspace.create_dataset('errors', data=np.ones((2, 3)))
            energy = workspace.create_dataset('axis1', data=np.linspace(-1.0, 1.0, 4))
            energy.attrs['units'] = np.bytes_(b'DeltaE')
            angle = workspace.create_dataset('axis2', data=[20.0, 40.0])
            angle.attrs['units'] = np.bytes_(b'Degrees')

        # THEN EXPECT
        with pytest.raises(ValueError, match=r"Mantid unit 'Degrees'.*SofQW"):
            load_mantid_nexus(path)
