# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

# Round trip for the delta-plus-Lorentzians diffusion models: synthetic data from the model,
# convolved with a Gaussian resolution, then a simultaneous fit over all Q values. The unit tests
# fit one Q value at a time without resolution; this exercises the shared Parameters across Q and
# the convolution through Experiment, Analysis and the EasyScience fitter.

import numpy as np
import pytest
import scipp as sc

import easydynamics as edyn

Q = sc.array(dims=['Q'], values=[0.5, 1.0, 1.5], unit='1/angstrom')
ENERGY = sc.linspace('energy', -2.0, 2.0, 401, unit='meV')
SHAPE = (Q.shape[0], ENERGY.shape[0])
NOISE = 0.003

# (model class, fixed arguments, true values of the fitted parameters, start values)
CASES = {
    'HallRossJumpDiffusion': (
        edyn.HallRossJumpDiffusion,
        {},
        {'scale': 1.0, 'residence_time': 2.0, 'jump_length': 1.5},
        {'scale': 0.8, 'residence_time': 1.5, 'jump_length': 1.2},
    ),
    'EquivalentSitesCircle': (
        edyn.EquivalentSitesCircle,
        {'radius': 1.2, 'number_of_sites': 3},
        {'scale': 1.0, 'residence_time': 2.0},
        {'scale': 0.8, 'residence_time': 1.5},
    ),
    'GaussianModel3D': (
        edyn.GaussianModel3D,
        {},
        {'scale': 1.0, 'diffusion_coefficient': 2e-9, 'mean_u_squared': 1.5},
        {'scale': 0.8, 'diffusion_coefficient': 1.5e-9, 'mean_u_squared': 1.2},
    ),
    'JumpSitesLogNormalDistribution': (
        edyn.JumpSitesLogNormalDistribution,
        {'radius': 1.2, 'number_of_sites': 3},
        {'scale': 1.0, 'residence_time': 2.0, 'sigma': 0.5},
        {'scale': 0.8, 'residence_time': 1.5, 'sigma': 0.7},
    ),
    # QR <= 1.5 keeps the default 10 terms below the truncation warning
    'DiffusionInSphere': (
        edyn.DiffusionInSphere,
        {},
        {'scale': 1.0, 'radius': 1.0, 'diffusion_coefficient': 0.1},
        {'scale': 0.8, 'radius': 0.8, 'diffusion_coefficient': 0.07},
    ),
    # QR <= 1.8 is well inside the default l_max = 6; the radius is fixed by default
    'IsotropicRotationalDiffusion': (
        edyn.IsotropicRotationalDiffusion,
        {'radius': 1.2},
        {'scale': 1.0, 'rotational_diffusion_coefficient': 0.1},
        {'scale': 0.8, 'rotational_diffusion_coefficient': 0.07},
    ),
}


def as_data(values: np.ndarray, variances: np.ndarray) -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['Q', 'energy'], values=values, variances=variances),
        coords={'Q': Q, 'energy': ENERGY},
    )


def make_analysis(experiment: edyn.Experiment, model: object) -> edyn.Analysis:
    resolution = edyn.ResolutionModel(components=edyn.Gaussian(width=0.05, area=1.0))
    resolution.fix_all_parameters()
    analysis = edyn.Analysis(
        experiment=experiment,
        sample_model=edyn.SampleModel(diffusion_models=model),
        instrument_model=edyn.InstrumentModel(resolution_model=resolution),
    )
    analysis.fix_energy_offset()
    return analysis


@pytest.mark.parametrize('case', CASES.values(), ids=CASES.keys())
def test_simultaneous_fit_recovers_parameters(case):
    # WHEN
    model_class, fixed, true_values, start_values = case
    true_model = model_class(**fixed, **true_values)
    experiment = edyn.Experiment('synthetic', data=as_data(np.ones(SHAPE), np.ones(SHAPE)))
    clean = np.array(make_analysis(experiment, true_model).calculate())
    rng = np.random.default_rng(seed=1)
    experiment.data = as_data(clean + rng.normal(0.0, NOISE, SHAPE), np.full(SHAPE, NOISE**2))
    fit_model = model_class(**fixed, **start_values)
    analysis = make_analysis(experiment, fit_model)

    # THEN
    results = analysis.fit(fit_method='simultaneous')

    # EXPECT
    assert all(result.success for result in results)
    residuals = (experiment.data.values - np.array(analysis.calculate())) / NOISE
    reduced_chi2 = float(np.sum(residuals**2) / residuals.size)
    assert 0.85 < reduced_chi2 < 1.15
    for name in true_values:
        fitted = getattr(fit_model, name)
        true_value = getattr(true_model, name).value
        assert fitted.variance > 0
        assert abs(fitted.value - true_value) < 5 * fitted.error, name
