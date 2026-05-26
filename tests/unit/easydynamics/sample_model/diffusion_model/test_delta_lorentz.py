# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easydynamics.sample_model.diffusion_model.delta_lorentz import DeltaLorentz


class TestDeltaLorentz:
    @pytest.fixture
    def delta_lorentz_model(self):
        return DeltaLorentz()

    def test_init_default(self, delta_lorentz_model):
        # WHEN THEN EXPECT
        assert delta_lorentz_model.display_name == 'DeltaLorentz'
        assert delta_lorentz_model.unit == 'meV'
        assert delta_lorentz_model.scale.value == pytest.approx(1.0)
        assert delta_lorentz_model.mean_u_squared.value == pytest.approx(0.0)
        assert delta_lorentz_model.A_0.value == pytest.approx(1.0)
        assert delta_lorentz_model.lorentzian_width.value == pytest.approx(1.0)
