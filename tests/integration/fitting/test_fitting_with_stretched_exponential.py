# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Round-trip fit of a StretchedExponential through Analysis1d.

The transform of a stretched exponential has no closed form, so it is evaluated by quadrature.
That makes it worth checking not just that the numbers are right at fixed parameters — the unit
tests cover that — but that the profile stays smooth enough in the relaxation time and the
stretching exponent for a least-squares fitter to walk back to the truth from a poor start.
"""

import numpy as np
import pytest
import scipp as sc

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model import StretchedExponential

TRUE_AREA = 1.0
TRUE_RELAXATION_TIME = 8.0
TRUE_BETA = 0.6
NOISE_FRACTION = 0.01

TRUTHS = {
    'StretchedExponential area': TRUE_AREA,
    'StretchedExponential relaxation_time': TRUE_RELAXATION_TIME,
    'StretchedExponential beta': TRUE_BETA,
}


def build_analysis() -> Analysis1d:
    """Fit a badly-started StretchedExponential against noisy data drawn from the true one."""
    energy_values = np.linspace(-1.5, 1.5, 301)
    truth = StretchedExponential(
        area=TRUE_AREA,
        relaxation_time=TRUE_RELAXATION_TIME,
        beta=TRUE_BETA,
    )
    profile = truth.evaluate(energy_values)
    noise = NOISE_FRACTION * profile.max()
    observed = profile + np.random.default_rng(0).normal(0.0, noise, size=profile.shape)

    experiment = Experiment(
        data=sc.DataArray(
            data=sc.array(
                dims=['Q', 'energy'],
                values=observed[None, :],
                variances=np.full_like(observed, noise**2)[None, :],
            ),
            coords={
                'Q': sc.array(dims=['Q'], values=[1.0], unit='1/Angstrom'),
                'energy': sc.array(dims=['energy'], values=energy_values, unit='meV'),
            },
        )
    )

    # Start far from the truth: half the relaxation time, an almost unstretched exponent and half
    # again the area.  The centre is left at its default, fixed at zero.
    analysis = Analysis1d(
        display_name='StretchedExponentialIntegration',
        experiment=experiment,
        sample_model=SampleModel(
            components=StretchedExponential(area=1.5, relaxation_time=4.0, beta=0.9)
        ),
        instrument_model=InstrumentModel(),
        Q_index=0,
    )
    # The energy offset shifts the spectrum exactly as the component's centre does, so leaving
    # both free would make the model unidentifiable.
    analysis.instrument_model.fix_energy_offset(Q_index=0)
    return analysis


@pytest.fixture(scope='module')
def fitted_analysis():
    analysis = build_analysis()
    analysis.fit()
    return analysis


class TestFittingWithStretchedExponential:
    def test_fit_describes_the_data(self):
        # WHEN
        analysis = build_analysis()

        # THEN
        results = analysis.fit()

        # EXPECT
        assert results.success
        assert results.reduced_chi2 < 1.5

    def test_the_free_parameters_are_the_three_shape_parameters(self, fitted_analysis):
        # THEN
        names = {parameter.name for parameter in fitted_analysis.get_free_parameters()}

        # EXPECT the centre and the energy offset stay fixed
        assert names == set(TRUTHS)

    @pytest.mark.parametrize('name', list(TRUTHS))
    def test_fit_recovers_the_true_parameters(self, fitted_analysis, name):
        # WHEN
        parameter = next(p for p in fitted_analysis.get_free_parameters() if p.name == name)

        # THEN EXPECT the truth within a few standard errors, on a usable uncertainty
        assert np.isfinite(parameter.error)
        assert parameter.error > 0.0
        assert abs(parameter.value - TRUTHS[name]) < 4 * parameter.error
