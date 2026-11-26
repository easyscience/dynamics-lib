# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment
from easydynamics.io import load_inx

# Fake files laid out like .inx files from the ILL; see tests/data/make_fake_inx.py.
DATA = Path(__file__).resolve().parents[3] / 'data' / 'inx'
INX = DATA / 'fake.inx'
ANGLE = DATA / 'fake_angle.inx'
ENERGY_UEV = np.linspace(-20.0, 20.0, 9)


def made_up_intensity(index):
    """The intensity make_fake_inx.py writes for a spectrum."""
    return 0.01 / (1.0 + (ENERGY_UEV / (2.0 * (1 + index))) ** 2) + 1e-4


class TestLoadInx:
    #############
    # Reading
    #############

    def test_reads_Q_from_the_file_and_energy_in_meV(self):
        # THEN
        data = load_inx(INX)

        # EXPECT
        assert data.dims == ('Q', 'energy')
        assert data.coords['Q'].unit == sc.Unit('1/angstrom')
        assert data.coords['energy'].unit == sc.Unit('meV')
        np.testing.assert_allclose(data.coords['Q'].values, [0.5, 1.0, 1.5])
        np.testing.assert_allclose(data.coords['energy'].values, ENERGY_UEV / 1000)
        np.testing.assert_allclose(data.values[1], made_up_intensity(1), rtol=1e-4)
        expected_errors = 0.01 * np.sqrt(made_up_intensity(1))
        np.testing.assert_allclose(data.variances[1], expected_errors**2, rtol=1e-3)

    def test_energy_unit(self):
        # THEN
        data = load_inx(INX, energy_unit='meV')

        # EXPECT
        np.testing.assert_allclose(data.coords['energy'].values, ENERGY_UEV)

    def test_Q_argument_overrides_the_file(self):
        # THEN
        data = load_inx(INX, Q=sc.array(dims=['Q'], values=[6.0, 7.0, 8.0], unit='1/nm'))

        # EXPECT
        np.testing.assert_allclose(data.coords['Q'].values, [0.6, 0.7, 0.8])

    def test_result_is_accepted_by_experiment(self):
        # WHEN
        experiment = Experiment(data=load_inx(INX, min_error=1e-5))

        # THEN
        _, _, weights, _ = experiment.extract_x_y_weights_only_finite(Q_index=2)

        # EXPECT
        assert np.isfinite(weights).all()

    #############
    # Scattering angles and fields too wide for their columns
    #############

    def test_scattering_angle_without_Q_raises(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='scattering angle rather than Q'):
            load_inx(ANGLE)

    def test_rows_with_overflowed_energy_are_left_out(self):
        # THEN
        data = load_inx(ANGLE, Q=[1.9])

        # EXPECT the first of the nine rows, whose energy is *****, is gone; the others keep
        # an overflowed intensity as NaN and an infinite one as infinite
        np.testing.assert_allclose(data.coords['energy'].values, ENERGY_UEV[1:] / 1000)
        assert np.isinf(data.values[0, 0])
        assert np.isnan(data.values[0, 3])
        assert np.isfinite(np.delete(data.values[0], [0, 3])).all()

    #############
    # Zero errors
    #############

    def test_min_error_nan_masks_the_zero_error_point(self):
        # THEN
        data = load_inx(INX, min_error=np.nan)

        # EXPECT
        zero_error = data.masks['zero_error'].values
        assert zero_error.sum() == 1
        assert zero_error[2, 0]

    #############
    # Errors
    #############

    @pytest.mark.parametrize('Q', [[0.5, 1.0], [[0.5, 1.0, 1.5]]], ids=['length', 'ndim'])
    def test_Q_not_matching_the_spectra_raises(self, Q):
        # THEN EXPECT
        with pytest.raises(ValueError, match='one per spectrum'):
            load_inx(INX, Q=Q)

    @pytest.mark.parametrize(
        ('text', 'message'),
        [
            ('', 'holds no spectra'),
            ('not a header\n', 'expected the header line'),
            ('5 1 2 0 0 0 0 3\ntitle\n', 'does not add up'),
            ('6 1 2 0 0 0 0 3\ntitle\n0.5 0 0\n0 8 0\n1 2 3\n', 'ends inside the spectrum'),
            ('4 1 2 0 0 0 0 1\ntitle\n0.5 0 0\n0 8 0\n1 2\n', 'a row without X, Y and E'),
            (
                (
                    '4 1 2 0 0 0 0 1\nt\n0.5 0 0\n0 8 0\n1 2 3\n'
                    '4 1 2 0 0 0 0 1\nt\n1.0 0 0\n0 8 0\n9 2 3\n'
                ),
                'same X values',
            ),
        ],
        ids=['empty', 'no_header', 'counts', 'truncated', 'two_columns', 'different_x'],
    )
    def test_malformed_file_raises(self, tmp_path, text, message):
        # WHEN
        path = tmp_path / 'bad.inx'
        path.write_text(text)

        # THEN EXPECT
        with pytest.raises(ValueError, match=message):
            load_inx(path)
