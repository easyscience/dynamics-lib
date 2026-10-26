# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
import scipp as sc
from scipp.constants import hbar as scipp_hbar
from scipy.special import spherical_jn

from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.isotropic_rotational_diffusion import MAXIMUM_L_MAX
from easydynamics.sample_model.diffusion_model.isotropic_rotational_diffusion import MINIMUM_RADIUS
from easydynamics.sample_model.diffusion_model.isotropic_rotational_diffusion import (
    MINIMUM_ROTATIONAL_DIFFUSION_COEFFICIENT,
)
from easydynamics.sample_model.diffusion_model.isotropic_rotational_diffusion import (
    IsotropicRotationalDiffusion,
)

# Rows of QENSmodels' reference data for isotropic_rotational_diffusion (tests/reference_data):
# energy transfer in meV and S(q, w) in 1/meV, for q = float32(0.7), scale = 1, center = 0,
# radius = 2 and rotational_diffusion_coefficient = 0.05 meV, with 5 Lorentzians (l = 1 to 5).
REFERENCE_ROWS = [
    (-1.0, 0.01993768579029),
    (-0.5, 0.07269305377548),
    (0.5, 0.07269305377548),
    (1.0, 0.01993768579029),
    (1.5, 0.009060005990534),
]


def hbar_rate(rate_per_ps: float) -> float:
    """hbar * rate in meV, for a rate in 1/ps, computed with Scipp's constants."""
    return (scipp_hbar * sc.scalar(rate_per_ps, unit='1/ps')).to(unit='meV').value


def model_without_warning(**kwargs) -> IsotropicRotationalDiffusion:
    """Create a model and fail if it warns about truncation."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return IsotropicRotationalDiffusion(**kwargs)


class TestIsotropicRotationalDiffusion:
    @pytest.fixture
    def model(self):
        return IsotropicRotationalDiffusion()

    def test_create_weight_requires_float_Q(self, model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be a float'):
            model._create_weight(1, 1)

    # ------------------------------------------------------------------
    # Construction and parameters
    # ------------------------------------------------------------------

    def test_init_default(self, model):
        # WHEN THEN EXPECT
        assert model.display_name == 'IsotropicRotationalDiffusion'
        assert model.x_unit == 'meV'
        assert model.y_unit == 'dimensionless'
        assert model.scale.value == pytest.approx(1.0)
        assert model.l_max == 6
        assert model.number_of_lorentzians == 6
        assert model.radius.value == pytest.approx(1.0)
        assert model.radius.unit == 'Å'
        assert model.radius.fixed is True
        assert model.radius.min == MINIMUM_RADIUS
        assert model.rotational_diffusion_coefficient.value == pytest.approx(1.0)
        assert sc.Unit(model.rotational_diffusion_coefficient.unit) == sc.Unit('1/ps')
        assert model.rotational_diffusion_coefficient.fixed is False
        assert (
            model.rotational_diffusion_coefficient.min == MINIMUM_ROTATIONAL_DIFFUSION_COEFFICIENT
        )

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            ({'radius': 'x'}, TypeError, 'radius must be a number'),
            ({'radius': 0.0}, ValueError, 'radius must be at least'),
            (
                {'rotational_diffusion_coefficient': 'x'},
                TypeError,
                'rotational_diffusion_coefficient must be a number',
            ),
            (
                {'rotational_diffusion_coefficient': 0.0},
                ValueError,
                'rotational_diffusion_coefficient must be at least',
            ),
            ({'l_max': 2.0}, TypeError, 'l_max must be an integer'),
            ({'l_max': True}, TypeError, 'l_max must be an integer'),
            ({'l_max': 0}, ValueError, 'l_max must be between'),
            ({'l_max': MAXIMUM_L_MAX + 1}, ValueError, 'l_max must be between'),
            ({'delta_name': 1}, TypeError, 'delta_name must be a string'),
        ],
        ids=[
            'radius_type',
            'radius_zero',
            'D_R_type',
            'D_R_zero',
            'l_max_float',
            'l_max_bool',
            'l_max_zero',
            'l_max_too_large',
            'delta_name_type',
        ],
    )
    def test_init_rejects_invalid_input(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            IsotropicRotationalDiffusion(**kwargs)

    def test_l_max_is_read_only(self, model):
        with pytest.raises(AttributeError):
            model.l_max = 5

    def test_setters(self, model):
        # WHEN
        model.radius = 2.5
        model.rotational_diffusion_coefficient = 0.3

        # THEN EXPECT
        assert model.radius.value == pytest.approx(2.5)
        assert model.rotational_diffusion_coefficient.value == pytest.approx(0.3)

    @pytest.mark.parametrize(
        'name, value, expected_exception',
        [
            ('radius', 'x', TypeError),
            ('radius', 0.0, ValueError),
            ('rotational_diffusion_coefficient', 'x', TypeError),
            ('rotational_diffusion_coefficient', 0.0, ValueError),
        ],
        ids=['radius_type', 'radius_zero', 'D_R_type', 'D_R_zero'],
    )
    def test_setters_raise(self, model, name, value, expected_exception):
        with pytest.raises(expected_exception, match=f'{name} must be'):
            setattr(model, name, value)

    # ------------------------------------------------------------------
    # Physics of the structure factors and widths
    # ------------------------------------------------------------------

    def test_EISF_is_j0_squared(self):
        # WHEN
        model = IsotropicRotationalDiffusion(radius=1.7)
        Q = np.array([0.2, 1.0, 2.5])

        # THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_EISF(Q), spherical_jn(0, Q * 1.7) ** 2, rtol=1e-13
        )

    @pytest.mark.parametrize('order', [1, 2, 6])
    def test_QISF_order_is_2l_plus_1_times_jl_squared(self, order):
        # WHEN
        model = IsotropicRotationalDiffusion(radius=1.7)
        Q = np.array([0.2, 1.0, 2.5])

        # THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_QISF_order(Q, order=order),
            (2 * order + 1) * spherical_jn(order, Q * 1.7) ** 2,
            rtol=1e-12,
            atol=1e-300,
        )

    def test_full_sum_has_no_missing_weight(self):
        """sum_l (2l+1) j_l(x)^2 = 1, so with enough orders nothing is missing."""
        # WHEN
        model = IsotropicRotationalDiffusion(radius=1.0, l_max=MAXIMUM_L_MAX)
        Q = np.array([0.1, 1.0, 3.0])

        # THEN EXPECT
        np.testing.assert_allclose(model.calculate_missing_weight(Q), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            model.calculate_EISF(Q) + model.calculate_QISF(Q), 1.0, rtol=1e-12
        )

    @pytest.mark.parametrize('order', [1, 2, 5])
    def test_width_is_l_times_l_plus_1_hbar_D_R(self, order):
        # WHEN
        model = IsotropicRotationalDiffusion(rotational_diffusion_coefficient=0.4)

        # THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_width(Q=np.array([0.5, 2.0]), order=order),
            order * (order + 1) * hbar_rate(0.4),
            rtol=1e-12,
        )

    def test_matches_QENSmodels_reference_spectrum(self):
        """QENSmodels takes D_R in meV, so D_R = 0.05 meV / hbar here."""
        # WHEN
        model = IsotropicRotationalDiffusion(
            scale=1.0,
            radius=2.0,
            rotational_diffusion_coefficient=0.05 / hbar_rate(1.0),
            l_max=5,
            Q=np.array([float(np.float32(0.7))]),
        )
        energy = np.array([row[0] for row in REFERENCE_ROWS])

        # THEN
        # Lorentzians only: the reference puts the delta function in the bin at w = 0, not listed.
        lorentzians = model.get_component_collections()[0][1:]
        spectrum = sum(component.evaluate(energy) for component in lorentzians)

        # EXPECT
        np.testing.assert_allclose(spectrum, [row[1] for row in REFERENCE_ROWS], rtol=1e-9)

    def test_Q_zero_is_purely_elastic(self, model):
        # WHEN
        Q = np.array([0.0])

        # THEN EXPECT
        np.testing.assert_allclose(model.calculate_EISF(Q), 1.0)
        np.testing.assert_allclose(model.calculate_QISF(Q), 0.0)

    def test_weights_follow_radius_unit(self):
        # WHEN
        model = IsotropicRotationalDiffusion(radius=2.0)
        Q = np.array([0.4, 1.3])
        before = model.calculate_QISF_order(Q, order=2)
        model.radius.convert_unit('nm')

        # THEN EXPECT
        np.testing.assert_allclose(model.calculate_QISF_order(Q, order=2), before, rtol=1e-12)

    @pytest.mark.parametrize('order', [0, 7, 1.5, True])
    def test_order_validation(self, model, order):
        with pytest.raises((TypeError, ValueError), match='order must be'):
            model.calculate_width(order=order)

    # ------------------------------------------------------------------
    # Truncation warning
    # ------------------------------------------------------------------

    def test_warns_when_truncation_drops_weight(self):
        with pytest.warns(UserWarning, match='l_max=2'):
            IsotropicRotationalDiffusion(radius=2.0, l_max=2, Q=np.array([0.5, 2.0]))

    def test_no_warning_when_truncation_is_negligible(self):
        model = model_without_warning(radius=1.0, Q=np.array([0.3, 1.0]))
        assert np.max(model.calculate_missing_weight()) < 1e-3

    def test_warns_again_when_Q_changes(self):
        # WHEN
        model = model_without_warning(radius=1.0, Q=np.array([0.5]))
        model.clear_Q(confirm=True)

        # THEN EXPECT QR = 6 is beyond what the default l_max = 6 covers
        with pytest.warns(UserWarning, match='omits more than'):
            model.Q = np.array([6.0])

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def test_no_Q_gives_no_collections(self, model):
        assert model.get_component_collections() == []

    def test_components_per_Q(self):
        # WHEN
        model = IsotropicRotationalDiffusion(l_max=3, Q=np.array([0.5, 1.0]))
        collections = model.get_component_collections()

        # THEN EXPECT
        assert len(collections) == 2
        components = collections[0][:]
        assert isinstance(components[0], DeltaFunction)
        assert all(isinstance(component, Lorentzian) for component in components[1:])
        assert [component.name for component in components] == [
            'Delta function',
            'IsotropicRotationalDiffusion_l1',
            'IsotropicRotationalDiffusion_l2',
            'IsotropicRotationalDiffusion_l3',
        ]

    def test_component_areas_and_widths_match_calculations(self):
        # WHEN
        Q = np.array([0.6, 1.4])
        model = IsotropicRotationalDiffusion(
            scale=2.0, radius=1.5, rotational_diffusion_coefficient=0.2, l_max=4, Q=Q
        )

        # THEN EXPECT
        for i, collection in enumerate(model.get_component_collections()):
            components = collection[:]
            assert components[0].area.value == pytest.approx(
                2.0 * model.calculate_EISF(Q[i])[0], rel=1e-10
            )
            for order in range(1, 5):
                assert components[order].area.value == pytest.approx(
                    2.0 * model.calculate_QISF_order(Q[i], order=order)[0], rel=1e-10
                )
                assert components[order].width.value == pytest.approx(
                    model.calculate_width(Q[i], order=order)[0], rel=1e-12
                )

    def test_components_follow_parameter_changes(self):
        # WHEN
        model = IsotropicRotationalDiffusion(l_max=3, Q=np.array([1.0]))
        model.radius = 1.6
        model.rotational_diffusion_coefficient = 0.7

        # THEN EXPECT
        components = model.get_component_collections()[0][:]
        assert components[0].area.value == pytest.approx(model.calculate_EISF(1.0)[0], rel=1e-10)
        assert components[2].area.value == pytest.approx(
            model.calculate_QISF_order(1.0, order=2)[0], rel=1e-10
        )
        assert components[2].width.value == pytest.approx(6 * hbar_rate(0.7), rel=1e-12)

    def test_parameter_updates_raise_no_runtime_warning(self):
        # WHEN
        model = IsotropicRotationalDiffusion(l_max=3, Q=np.array([1.0]))

        # THEN EXPECT
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            model.radius = 1.2
            model.rotational_diffusion_coefficient = 0.5
            model.scale = 3.0

    # ------------------------------------------------------------------
    # Fit targets, variables and representation
    # ------------------------------------------------------------------

    def test_fit_targets(self):
        # WHEN
        model = IsotropicRotationalDiffusion(l_max=2)

        # THEN EXPECT
        assert [target.name for target in model.get_fit_targets()] == [
            'elastic_area',
            'area_1',
            'width_1',
            'area_2',
            'width_2',
        ]

    def test_get_global_variables(self, model):
        assert model.get_global_variables() == [
            model.scale,
            model.radius,
            model.rotational_diffusion_coefficient,
        ]

    def test_repr(self, model):
        # WHEN
        text = repr(model)

        # THEN EXPECT
        assert text.startswith('IsotropicRotationalDiffusion(')
        assert 'radius=' in text
        assert 'rotational_diffusion_coefficient=' in text
        assert 'l_max=6' in text
