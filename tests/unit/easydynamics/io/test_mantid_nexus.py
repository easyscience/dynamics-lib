# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import shutil
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

    def test_uncovered_bins_are_masked(self):
        # THEN
        data = load_mantid_nexus(IRIS)

        # EXPECT exactly the bins with zero fractional area are masked, and nothing is NaN
        uncovered = stored(IRIS, 'frac_area') == 0
        assert uncovered.any()
        np.testing.assert_array_equal(data.masks['uncovered'].values, uncovered)
        assert np.isfinite(data.values).all()

    def test_no_mask_without_frac_area(self):
        # THEN
        data = load_mantid_nexus(IN16B)

        # EXPECT
        assert dict(data.masks) == {}

    def test_covered_values_are_not_divided_by_frac_area(self):
        # THEN
        data = load_mantid_nexus(IRIS)

        # EXPECT the stored values, which Mantid already divided by the fractional areas
        np.testing.assert_array_equal(data.values, stored(IRIS, 'values'))

    @pytest.mark.parametrize(
        ('attribute', 'value', 'message'),
        [('finalized', b'0', 'finalize it in Mantid'), ('sqrd_errors', b'1', 'squared errors')],
        ids=['unfinalized', 'squared_errors'],
    )
    def test_rebinned_workspace_not_read_as_stored_raises(
        self, tmp_path, attribute, value, message
    ):
        # WHEN the areas have not been divided out, or the errors are stored squared
        path = tmp_path / 'rebinned.nxs'
        shutil.copy(IRIS, path)
        with h5py.File(path, 'r+') as file:
            file['mantid_workspace_1/workspace/frac_area'].attrs[attribute] = np.bytes_(value)

        # THEN EXPECT
        with pytest.raises(ValueError, match=message):
            load_mantid_nexus(path)

    def test_energy_bins_differing_between_spectra_raises(self, tmp_path):
        # WHEN the energy axis is stored per spectrum
        path = tmp_path / 'ragged.nxs'
        shutil.copy(IN16B, path)
        with h5py.File(path, 'r+') as file:
            workspace = file['mantid_workspace_1/workspace']
            energy = workspace['axis1'][()]
            units = workspace['axis1'].attrs['units']
            del workspace['axis1']
            workspace.create_dataset('axis1', data=np.tile(energy, (4, 1)))
            workspace['axis1'].attrs['units'] = units

        # THEN EXPECT
        with pytest.raises(ValueError, match='differs between spectra'):
            load_mantid_nexus(path)

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

    def test_min_error_nan_masks_zero_error_bins(self):
        # THEN
        data = load_mantid_nexus(IRIS, min_error=np.nan)

        # EXPECT the covered empty bins are masked as well as the uncovered ones
        zero_error = data.masks['zero_error'].values
        assert zero_error.sum() == 5
        assert zero_error[2, :2].all()
        np.testing.assert_array_equal(zero_error, data.variances == 0)

    def test_min_error_nan_data_can_be_fitted(self):
        # WHEN
        experiment = Experiment(data=load_mantid_nexus(IRIS, min_error=np.nan))

        # THEN
        _, _, weights, kept = experiment.extract_x_y_weights_only_finite(Q_index=2)

        # EXPECT the two empty bins at this Q are left out
        assert kept.sum() == 14
        assert np.isfinite(weights).all()

    def test_data_with_min_error_can_be_fitted(self):
        # WHEN
        experiment = Experiment(data=load_mantid_nexus(IN16B, min_error=1.0))

        # THEN
        _, _, weights, _ = experiment.extract_x_y_weights_only_finite(Q_index=0)

        # EXPECT
        assert np.isfinite(weights).all()

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
