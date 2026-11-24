# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment
from easydynamics.io import load_spe

# Fake files laid out like those of the FOCUS reduction; see tests/data/make_fake_spe.py.
DATA = Path(__file__).resolve().parents[3] / 'data' / 'spe'
SPE = DATA / 'fake.spe'
CENTRES = DATA / 'fake_centres.spe'


def elastic_Q(phi_degrees, wavelength):
    return 4 * np.pi * np.sin(np.radians(np.asarray(phi_degrees)) / 2) / wavelength


def made_up_intensity(detector, energy):
    """The intensity make_fake_spe.py writes for a detector."""
    return (1.0 + 0.1 * detector) / (1.0 + (energy / 0.2) ** 2) - 0.05


class TestLoadSpe:
    #############
    # Reading
    #############

    def test_reads_axes_with_the_wavelength_from_the_par_file(self):
        # THEN
        data = load_spe(SPE)

        # EXPECT Q of each detector at its central angle, for the 4 angstrom of fake.par
        assert data.dims == ('Q', 'energy')
        assert data.coords['Q'].unit == sc.Unit('1/angstrom')
        assert data.coords['energy'].unit == sc.Unit('meV')
        np.testing.assert_allclose(
            data.coords['Q'].values, elastic_Q([15.0, 25.0, 35.0, 45.0, 55.0], 4.0)
        )
        np.testing.assert_allclose(data.coords['energy'].values, np.linspace(-1.0, 1.0, 11))
        np.testing.assert_allclose(
            data.coords['scattering_angle'].values, [15.0, 25.0, 35.0, 45.0, 55.0]
        )

    def test_reads_numbers_that_run_together(self):
        # THEN
        data = load_spe(SPE)

        # EXPECT the negative intensities written without a space before them are read whole
        energy = np.linspace(-0.9, 0.9, 10)
        expected = made_up_intensity(0, energy)
        assert (expected < 0).any()
        np.testing.assert_allclose(data.values[0], expected, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(
            data.variances[0], (0.1 * np.sqrt(np.abs(data.values[0]))) ** 2, rtol=3e-2
        )

    def test_wavelength_argument_overrides_the_par_file(self):
        # THEN
        data = load_spe(SPE, wavelength=2.0)

        # EXPECT
        np.testing.assert_allclose(data.coords['Q'].values[0], elastic_Q(15.0, 2.0))

    def test_reads_centres_and_sorts_detectors_by_Q(self):
        # THEN
        data = load_spe(CENTRES, wavelength=4.0)

        # EXPECT the detectors, written at 60, 20 and 40 degrees, come out in Q order
        np.testing.assert_allclose(data.coords['scattering_angle'].values, [20.0, 40.0, 60.0])
        np.testing.assert_allclose(data.coords['Q'].values, elastic_Q([20.0, 40.0, 60.0], 4.0))
        expected = made_up_intensity(1, data.coords['energy'].values)
        np.testing.assert_allclose(data.values[0], expected, rtol=1e-2, atol=1e-3)
        assert data.coords['energy'].sizes == {'energy': 10}

    def test_negative_angles_give_positive_Q(self, tmp_path):
        # WHEN the detectors are written with negative angles
        path = tmp_path / 'negative.spe'
        path.write_text(CENTRES.read_text().replace(' 6.00e+001', '-6.00e+001'))

        # THEN
        data = load_spe(path, wavelength=4.0)

        # EXPECT
        np.testing.assert_allclose(data.coords['Q'].values, elastic_Q([20.0, 40.0, 60.0], 4.0))

    def test_reads_fortran_exponents(self, tmp_path):
        # WHEN
        path = tmp_path / 'fortran.spe'
        path.write_text(
            '1 2\n### Phi\n 1.0D+001 2.0D+001\n### Energy\n-1.0D-001 0.0D+000 1.0D-001\n'
            '### S(Phi,w)\n 2.5D+000 3.5d+000\n### Errors\n 1.0D-001 1.0D-001\n'
        )

        # THEN
        data = load_spe(path, wavelength=4.0)

        # EXPECT
        np.testing.assert_allclose(data.values, [[2.5, 3.5]])
        np.testing.assert_allclose(data.coords['energy'].values, [-0.1, 0.0, 0.1])

    def test_placeholder_angles_raise(self, tmp_path):
        # WHEN the angle block holds detector numbers, as Mantid's SaveSPE writes
        path = tmp_path / 'placeholders.spe'
        path.write_text(
            '2 2\n### Phi Grid\n 0.5 1.5 2.5\n### Energy Grid\n 0 1 2\n'
            '### S(Phi,w)\n 1 1\n### Errors\n 1 1\n### S(Phi,w)\n 1 1\n### Errors\n 1 1\n'
        )

        # THEN EXPECT
        with pytest.raises(ValueError, match='placeholder numbers'):
            load_spe(path, wavelength=4.0)

    def test_masked_detector_is_nan_and_masked(self):
        # THEN
        data = load_spe(SPE)

        # EXPECT the detector written as -1e30 at 35 degrees
        null = data.masks['null_data'].values
        np.testing.assert_array_equal(null.all(axis=1), [False, False, True, False, False])
        assert np.isnan(data.values[2]).all()
        assert np.isnan(data.variances[2]).all()

    def test_no_mask_without_masked_detectors(self):
        # THEN
        data = load_spe(CENTRES, wavelength=4.0)

        # EXPECT
        assert dict(data.masks) == {}

    def test_result_can_be_rebinned_in_Q(self):
        # WHEN
        experiment = Experiment(data=load_spe(SPE, min_error=0.01))

        # THEN
        experiment.rebin({'Q': sc.linspace('Q', 0.2, 1.4, 3, unit='1/angstrom')})

        # EXPECT the masked detector is left out of its Q bin rather than making it NaN
        assert experiment.binned_data.sizes['Q'] == 2
        assert not experiment.binned_data.masks['no_data'].values.any()

    #############
    # Zero errors
    #############

    def test_min_error_nan_masks_the_zero_error_channel(self):
        # THEN
        data = load_spe(SPE, min_error=np.nan)

        # EXPECT only the channel with zero error; the masked detector has no error at all
        zero_error = data.masks['zero_error'].values
        assert zero_error.sum() == 1
        assert zero_error[3, 4]

    #############
    # Errors
    #############

    def test_no_wavelength_raises(self):
        # THEN EXPECT
        with pytest.raises(ValueError, match='No wavelength given'):
            load_spe(CENTRES)

    def test_par_file_without_wavelength_raises(self, tmp_path):
        # WHEN a .par file of detector positions, which has no wavelength
        path = tmp_path / 'run.spe'
        path.write_text(SPE.read_text())
        (tmp_path / 'run.par').write_text('5\n 4.0 15.0 0.0 1.0 1.0 1\n')

        # THEN EXPECT
        with pytest.raises(ValueError, match='No wavelength given'):
            load_spe(path)

    @pytest.mark.parametrize('wavelength', [0.0, -1.0, np.inf, '4'])
    def test_invalid_wavelength_raises(self, wavelength):
        # THEN EXPECT
        with pytest.raises(ValueError, match='wavelength must be a positive number'):
            load_spe(SPE, wavelength=wavelength)

    @pytest.mark.parametrize(
        ('text', 'message'),
        [
            ('not a header\n', 'the first line should hold'),
            ('2 3\n### Phi\n 1 2 3\n### Energy\n 1 2 3\n', 'expected 6 blocks'),
            (
                '1 3\n### Phi\n 1 2 3\n### Energy\n 1 2 3\n### S\n 1 2 3\n### E\n 1 2 3\n',
                'the angle block has 3 values',
            ),
            (
                '1 3\n### Phi\n 1 2\n### Energy\n 1 2 3\n### S\n 1 2\n### E\n 1 2 3\n',
                'every detector should have 3 values',
            ),
        ],
        ids=['header', 'block_count', 'axis_length', 'detector_length'],
    )
    def test_malformed_file_raises(self, tmp_path, text, message):
        # WHEN
        path = tmp_path / 'bad.spe'
        path.write_text(text)

        # THEN EXPECT
        with pytest.raises(ValueError, match=message):
            load_spe(path, wavelength=4.0)
