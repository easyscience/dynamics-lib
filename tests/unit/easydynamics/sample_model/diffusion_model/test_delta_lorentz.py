# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easydynamics.sample_model.diffusion_model.delta_lorentz import DeltaLorentz


class TestDeltaLorentz:
    @pytest.fixture
    def delta_lorentz_model(self):
        return DeltaLorentz()

    @pytest.fixture
    def delta_lorentz_model_with_Q(self):
        Q = np.linspace(0.5, 2, 7)
        return DeltaLorentz(
            Q=Q,
            A_0=0.5,
            lorentzian_width=0.0015,
            allow_Q_variation={'A_0': True, 'lorentzian_width': True},
        )

    def test_init_default(self, delta_lorentz_model):
        # WHEN THEN EXPECT
        assert delta_lorentz_model.display_name == 'DeltaLorentz'
        assert delta_lorentz_model.unit == 'meV'
        assert delta_lorentz_model.scale.value == pytest.approx(1.0)
        assert delta_lorentz_model.mean_u_squared.value == pytest.approx(0.0)
        assert delta_lorentz_model.A_0.value == pytest.approx(1.0)
        assert delta_lorentz_model.lorentzian_width.value == pytest.approx(1.0)

    def test_init_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN THEN EXPECT
        assert delta_lorentz_model_with_Q.display_name == 'DeltaLorentz'
        assert delta_lorentz_model_with_Q.unit == 'meV'
        assert delta_lorentz_model_with_Q.scale.value == pytest.approx(1.0)
        assert delta_lorentz_model_with_Q.mean_u_squared.value == pytest.approx(0.0)
        assert delta_lorentz_model_with_Q.A_0.value == pytest.approx(0.5)
        assert delta_lorentz_model_with_Q.lorentzian_width.value == pytest.approx(0.0015)
        assert delta_lorentz_model_with_Q._allow_Q_variation == {
            'A_0': True,
            'lorentzian_width': True,
        }
        assert len(delta_lorentz_model_with_Q._A_0_list) == len(delta_lorentz_model_with_Q.Q)
        assert len(delta_lorentz_model_with_Q._lorentzian_width_list) == len(
            delta_lorentz_model_with_Q.Q
        )
        assert all(pytest.approx(a.value) == 0.5 for a in delta_lorentz_model_with_Q._A_0_list)
        assert all(
            pytest.approx(lw.value) == 0.0015
            for lw in delta_lorentz_model_with_Q._lorentzian_width_list
        )

    @pytest.mark.parametrize(
        'kwargs,expected_exception, expected_message',
        [
            (
                {
                    'mean_u_squared': -1.0,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                ValueError,
                'mean_u_squared must be non-negative',
            ),
            (
                {
                    'mean_u_squared': 'not a number',
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'mean_u_squared must be a number',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': -1.0,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                ValueError,
                'A_0 must be between 0 and 1',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 'not a number',
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'A_0 must be a number',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': -1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                ValueError,
                'lorentzian_width must be ',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 'not a number',
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'lorentzian_width must be a number',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': 'Not a dict',
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'allow_Q_variation must be a dict',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 123,
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'delta_name must be a string',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': None,
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'delta_name must be a string',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 123,
                },
                TypeError,
                'delta_display_name must be a string',
            ),
        ],
        ids=[
            'mean_u_squared negative',
            'mean_u_squared not a number',
            'A_0 negative',
            'A_0 not a number',
            'lorentzian_width negative',
            'lorentzian_width not a number',
            'allow_Q_variation not a dict',
            'delta_name not a string',
            'delta_name not a string (None)',
            'delta_display_name not a string',
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            DeltaLorentz(**kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def test_calculate_width_without_Q(self, delta_lorentz_model):
        # WHEN THEN
        width = delta_lorentz_model.calculate_width(Q=0.5)

        # EXPECT
        assert len(width) == 1
        assert width[0] == pytest.approx(1.0)

    def test_calculate_width_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN THEN
        width = delta_lorentz_model_with_Q.calculate_width()

        # EXPECT
        assert len(width) == len(delta_lorentz_model_with_Q.Q)
        assert all(width_i == pytest.approx(0.0015) for width_i in width)

    def test_calculate_EISF(self, delta_lorentz_model):
        # WHEN

        # THEN
        eisf = delta_lorentz_model.calculate_EISF(Q=0.5)

        # EXPECT
        assert len(eisf) == 1
        expected = delta_lorentz_model.A_0.value * np.exp(
            -delta_lorentz_model.mean_u_squared.value * 0.5**2
        )
        assert eisf[0] == pytest.approx(expected)

    def test_calculate_EISF_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN

        # THEN
        eisf = delta_lorentz_model_with_Q.calculate_EISF()

        # EXPECT
        assert len(eisf) == len(delta_lorentz_model_with_Q.Q)
        for i in range(len(eisf)):
            expected = delta_lorentz_model_with_Q._A_0_list[i].value * np.exp(
                -delta_lorentz_model_with_Q.mean_u_squared.value
                * delta_lorentz_model_with_Q.Q[i] ** 2
            )
            assert eisf[i] == pytest.approx(expected)

    def test_calculate_QISF(self, delta_lorentz_model):
        # WHEN THEN
        qisf = delta_lorentz_model.calculate_QISF(Q=0.5)

        # EXPECT
        assert len(qisf) == 1
        expected = delta_lorentz_model.A_1.value * np.exp(
            -delta_lorentz_model.mean_u_squared.value * 0.5**2
        )
        assert qisf[0] == pytest.approx(expected)

    def test_calculate_QISF_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN THEN
        qisf = delta_lorentz_model_with_Q.calculate_QISF()

        # EXPECT
        assert len(qisf) == len(delta_lorentz_model_with_Q.Q)
        for i in range(len(qisf)):
            expected = delta_lorentz_model_with_Q._A_1_list[i].value * np.exp(
                -delta_lorentz_model_with_Q.mean_u_squared.value
                * delta_lorentz_model_with_Q.Q[i] ** 2
            )

            assert qisf[i] == pytest.approx(expected)
