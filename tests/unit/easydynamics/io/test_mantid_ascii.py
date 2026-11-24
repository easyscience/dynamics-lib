# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment
from easydynamics.io import load_mantid_ascii

# Fake files laid out like those Mantid's SaveAscii writes; see
# tests/data/make_fake_mantid_ascii.py.
DATA = Path(__file__).resolve().parents[3] / 'data' / 'mantid_ascii'
NUMBERED = DATA / 'fake.dat'
UNNUMBERED = DATA / 'fake_no_spectrum_numbers.dat'
ENERGY = np.linspace(-0.03, 0.03, 7)
Q = [0.5, 1.0, 1.5]


def made_up_intensity(index):
    """The intensity make_fake_mantid_ascii.py writes for a spectrum."""
    return 10.0 / (1.0 + (ENERGY / (0.005 * (1 + index))) ** 2) + 0.5


class TestLoadMantidAscii:
    #############
    # Reading
    #############

    def test_reads_spectra_with_spectrum_numbers(self):
        # THEN
        data = load_mantid_ascii(NUMBERED, Q=Q)

        # EXPECT
        assert data.dims == ('Q', 'energy')
        assert data.coords['Q'].unit == sc.Unit('1/angstrom')
        assert data.coords['energy'].unit == sc.Unit('meV')
        np.testing.assert_array_equal(data.coords['Q'].values, Q)
        np.testing.assert_allclose(data.coords['energy'].values, ENERGY)
        np.testing.assert_allclose(data.values[2], made_up_intensity(2), rtol=1e-5)
        np.testing.assert_allclose(data.variances[2], made_up_intensity(2), rtol=1e-5)

    def test_reads_spectra_without_spectrum_numbers(self):
        # THEN
        data = load_mantid_ascii(UNNUMBERED, Q=[0.5, 1.0])

        # EXPECT the X error column is ignored and a new spectrum starts where X restarts
        assert data.sizes == {'Q': 2, 'energy': 7}
        np.testing.assert_allclose(data.values[1], made_up_intensity(1), rtol=1e-5)

    def test_Q_as_variable_is_converted(self):
        # WHEN
        Q_in_nm = sc.array(dims=['Q'], values=[5.0, 10.0, 15.0], unit='1/nm')

        # THEN
        data = load_mantid_ascii(NUMBERED, Q=Q_in_nm)

        # EXPECT
        np.testing.assert_allclose(data.coords['Q'].values, Q)

    def test_Q_as_bin_edges(self):
        # THEN
        data = load_mantid_ascii(NUMBERED, Q=[0.25, 0.75, 1.25, 1.75])

        # EXPECT Experiment fits at the centres
        np.testing.assert_allclose(Experiment(data=data).binned_data.coords['Q'].values, Q)

    def test_energy_unit(self):
        # THEN
        data = load_mantid_ascii(NUMBERED, Q=Q, energy_unit='ueV')

        # EXPECT
        assert data.coords['energy'].unit == sc.Unit('ueV')

    #############
    # Zero errors
    #############

    def test_zero_errors_are_kept_by_default(self):
        # THEN
        data = load_mantid_ascii(NUMBERED, Q=Q)

        # EXPECT
        assert (data.variances == 0).sum() == 1
        assert not data.variances[1, 0]

    def test_min_error_nan_masks_the_zero_error_point(self):
        # THEN
        data = load_mantid_ascii(NUMBERED, Q=Q, min_error=np.nan)

        # EXPECT
        zero_error = data.masks['zero_error'].values
        assert zero_error.sum() == 1
        assert zero_error[1, 0]

    def test_data_with_min_error_can_be_fitted(self):
        # WHEN
        experiment = Experiment(data=load_mantid_ascii(NUMBERED, Q=Q, min_error=0.1))

        # THEN
        _, _, weights, _ = experiment.extract_x_y_weights_only_finite(Q_index=1)

        # EXPECT
        assert np.isfinite(weights).all()

    #############
    # Errors
    #############

    @pytest.mark.parametrize('Q_values', [[0.5, 1.0], [[0.5, 1.0, 1.5]]], ids=['length', 'ndim'])
    def test_Q_not_matching_the_spectra_raises(self, Q_values):
        # THEN EXPECT
        with pytest.raises(ValueError, match='one per spectrum'):
            load_mantid_ascii(NUMBERED, Q=Q_values)

    @pytest.mark.parametrize(
        ('text', 'message'),
        [
            ('# header only\n', 'holds no data rows'),
            ('1\n0.1,2.0\n', 'expected X, Y and E columns'),
            ('1\n0.1,2,1\n0.2,2,1\n2\n0.1,2,1\n0.3,2,1\n', 'same X values'),
            ('1\n0.1,2,1\n0.2,2,1\n2\n0.1,2,1\n', 'same X values'),
        ],
        ids=['empty', 'two_columns', 'different_x', 'different_length'],
    )
    def test_malformed_file_raises(self, tmp_path, text, message):
        # WHEN
        path = tmp_path / 'bad.dat'
        path.write_text(text)

        # THEN EXPECT
        with pytest.raises(ValueError, match=message):
            load_mantid_ascii(path, Q=[1.0, 2.0])
