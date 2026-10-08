# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import easydynamics as edyn


def test_smoke_build_and_evaluate_model():
    # WHEN a minimal sample model with a single Lorentzian component
    lorentzian = edyn.Lorentzian(area=1.0, width=0.1)
    model = edyn.SampleModel(components=lorentzian)

    # THEN evaluating the component on a small energy grid
    energy = np.linspace(-1.0, 1.0, 101)
    y = lorentzian.evaluate(energy)

    # EXPECT the package installs, the model builds, and the evaluation is finite and peaked
    assert model is not None
    assert y.shape == energy.shape
    assert np.all(np.isfinite(y))
    assert y.max() > 0.0
