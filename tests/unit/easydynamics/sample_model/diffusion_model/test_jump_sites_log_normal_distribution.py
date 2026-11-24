# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
import scipp as sc
from scipp.constants import hbar as scipp_hbar

import easydynamics as edyn
from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.equivalent_sites_circle import EquivalentSitesCircle
from easydynamics.sample_model.diffusion_model.jump_sites_log_normal_distribution import (
    MINIMUM_SIGMA,
)
from easydynamics.sample_model.diffusion_model.jump_sites_log_normal_distribution import (
    NUMBER_OF_SAMPLES,
)
from easydynamics.sample_model.diffusion_model.jump_sites_log_normal_distribution import (
    JumpSitesLogNormalDistribution,
)

# A weight is a sum of N terms of size up to 1/N that can nearly cancel, so its rounding error is
# absolute, a few N * machine epsilon, not relative to the (possibly tiny) result.
WEIGHT_ATOL = 1e-14

# Rows of tests/manual/jump_sites_log_norm_dist_ref_data.dat (QENSmodels tests/reference_data):
# energy transfer w in 1/ps and S(q, w), for Nsites = 7, radius = 5, resTime = 2, sigma = 0.6,
# scale = 2, center = 0.8 and q = float32(0.7).
REFERENCE_ROWS = [
    (-1.0, 0.08158287388105),
    (-0.5, 0.1336353341603),
    (0.5, 0.6730922522463),
    (1.0, 0.8950268953231),
    (1.5, 0.2983075456912),
]


def hbar_over(residence_time_ps: float) -> float:
    """hbar / tau in meV, computed with Scipp's constants."""
    return (scipp_hbar / sc.scalar(residence_time_ps, unit='ps')).to(unit='meV').value


def order_of(circle_order: int, s: int) -> int:
    """The Lorentzian order j of circle order l and sample s."""
    return (circle_order - 1) * NUMBER_OF_SAMPLES + s + 1


class TestJumpSitesLogNormalDistribution:
    @pytest.fixture
    def model(self):
        return JumpSitesLogNormalDistribution()

    def test_create_weight_requires_float_Q(self, model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be a float'):
            model._create_weight(1, 1)

    # ------------------------------------------------------------------
    # Construction and parameters
    # ------------------------------------------------------------------

    def test_init_default(self, model):
        # WHEN THEN EXPECT
        assert model.display_name == 'JumpSitesLogNormalDistribution'
        assert model.number_of_sites == 3
        assert model.number_of_lorentzians == 2 * NUMBER_OF_SAMPLES
        assert model.radius.value == pytest.approx(1.0)
        assert model.radius.unit == 'Å'
        assert model.radius.fixed is True
        assert model.residence_time.unit == 'ps'
        assert model.residence_time.fixed is False
        assert model.sigma.value == pytest.approx(1.0)
        assert model.sigma.unit == 'dimensionless'
        assert model.sigma.fixed is False
        assert model.sigma.min == MINIMUM_SIGMA

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            ({'radius': 'invalid'}, TypeError, 'radius must be a number'),
            ({'radius': 0.0}, ValueError, 'radius must be at least'),
            ({'residence_time': 'invalid'}, TypeError, 'residence_time must be a number'),
            ({'residence_time': 0.0}, ValueError, 'residence_time must be at least'),
            ({'sigma': 'invalid'}, TypeError, 'sigma must be a number'),
            ({'sigma': 0.0}, ValueError, 'sigma must be at least'),
            ({'number_of_sites': 2.5}, TypeError, 'number_of_sites must be an integer'),
            ({'number_of_sites': 1}, ValueError, 'number_of_sites must be between 2 and'),
            ({'delta_name': 1}, TypeError, 'delta_name must be a string'),
        ],
    )
    def test_init_rejects_invalid_input(self, kwargs, expected_exception, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            JumpSitesLogNormalDistribution(**kwargs)

    def test_sigma_setter(self, model):
        # WHEN
        model.sigma = 0.4

        # THEN EXPECT
        assert model.sigma.value == pytest.approx(0.4)
        with pytest.raises(ValueError, match='sigma must be at least'):
            model.sigma = 0.0

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def test_matches_QENSmodels_doctest_values(self):
        """
        Doctest values of QENSmodels hwhmJumpSitesLogNormDist([1., 2.], 4, 0.5, 1.5, 1.0), rounded
        there to the digits given. Its widths are in 1/ps, so they are divided by hbar here.
        """
        # WHEN
        model = JumpSitesLogNormalDistribution(
            number_of_sites=4, radius=0.5, residence_time=1.5, sigma=1.0
        )
        Q = np.array([1.0, 2.0])
        hbar_per_ps = hbar_over(1.0)

        # THEN EXPECT
        np.testing.assert_allclose(model.calculate_EISF(Q), [0.92, 0.713], atol=5e-4)
        assert model.calculate_width(Q, order_of(3, 20))[1] / hbar_per_ps == pytest.approx(
            5.7, abs=5e-2
        )
        assert model.calculate_width(Q, order_of(1, 5))[1] / hbar_per_ps == pytest.approx(
            0.228, abs=5e-4
        )
        assert model.calculate_QISF_order(Q, order_of(3, 19))[0] == pytest.approx(
            0.000538, abs=5e-7
        )
        assert model.calculate_QISF_order(Q, order_of(2, 5))[1] == pytest.approx(
            0.000712, abs=5e-7
        )

    def test_matches_QENSmodels_reference_spectrum(self):
        """The QENSmodels reference spectrum, with w in 1/ps and hbar = 1 there."""
        # WHEN
        model = JumpSitesLogNormalDistribution(
            scale=2.0,
            radius=5.0,
            residence_time=2.0,
            sigma=0.6,
            number_of_sites=7,
            Q=np.array([float(np.float32(0.7))]),
        )
        hbar_meV_ps = scipp_hbar.to(unit='meV*ps').value
        w = np.array([row[0] for row in REFERENCE_ROWS])

        # THEN
        # Lorentzians only: on these few points the delta function would land in the point
        # nearest its centre (w = 1.0), while the reference has it at w = 0.8, not listed here.
        x = hbar_meV_ps * (w - 0.8)
        lorentzians = model.get_component_collections()[0][1:]
        spectrum = hbar_meV_ps * sum(component.evaluate(x) for component in lorentzians)

        # EXPECT
        np.testing.assert_allclose(spectrum, [row[1] for row in REFERENCE_ROWS], rtol=1e-9)

    @pytest.mark.parametrize('number_of_sites', [2, 3, 7])
    def test_weights_sum_to_one(self, number_of_sites):
        # WHEN
        model = JumpSitesLogNormalDistribution(number_of_sites=number_of_sites, radius=1.3)
        Q = np.array([0.1, 0.9, 2.5])

        # THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_EISF(Q) + model.calculate_QISF(Q), 1.0, rtol=0, atol=WEIGHT_ATOL
        )

    def test_sample_weights_do_not_depend_on_sigma(self):
        # WHEN
        narrow = JumpSitesLogNormalDistribution(sigma=0.2).sample_weights()
        wide = JumpSitesLogNormalDistribution(sigma=1.5).sample_weights()

        # THEN EXPECT
        assert narrow.shape == (NUMBER_OF_SAMPLES,)
        np.testing.assert_allclose(narrow, wide, rtol=1e-15, atol=0)
        assert narrow.sum() == pytest.approx(1.0, rel=1e-15)
        # gaussian in log(ratio) with the half: g_0 / g_10 = exp(-0.5 * 2 ln 10) = 1/10
        assert narrow[0] / narrow[10] == pytest.approx(0.1, rel=1e-12)

    def test_minimum_sigma_is_the_circle(self):
        # WHEN
        Q = np.array([0.5, 1.7])
        model = JumpSitesLogNormalDistribution(
            number_of_sites=5, radius=1.2, residence_time=0.7, sigma=MINIMUM_SIGMA
        )
        circle = EquivalentSitesCircle(number_of_sites=5, radius=1.2, residence_time=0.7)

        # THEN EXPECT
        for circle_order in range(1, 5):
            for s in range(NUMBER_OF_SAMPLES):
                np.testing.assert_allclose(
                    model.calculate_width(Q, order_of(circle_order, s)),
                    circle.calculate_width(Q, circle_order),
                    rtol=1e-9,
                )
            summed = sum(
                model.calculate_QISF_order(Q, order_of(circle_order, s))
                for s in range(NUMBER_OF_SAMPLES)
            )
            np.testing.assert_allclose(
                summed, circle.calculate_QISF_order(Q, circle_order), rtol=0, atol=WEIGHT_ATOL
            )
        np.testing.assert_allclose(
            model.calculate_EISF(Q), circle.calculate_EISF(Q), rtol=0, atol=WEIGHT_ATOL
        )

    def test_weights_follow_radius_unit(self):
        # WHEN
        Q = np.array([0.8, 1.6])
        model = JumpSitesLogNormalDistribution(number_of_sites=4, radius=1.5, Q=Q)
        EISF_angstrom = model.calculate_EISF(Q)
        areas_angstrom = [c[3].area.value for c in model.get_component_collections()]

        # THEN
        model.radius.convert_unit('nm')

        # EXPECT the same physical radius gives the same weights
        np.testing.assert_allclose(model.calculate_EISF(Q), EISF_angstrom, rtol=1e-12)
        areas_nm = [c[3].area.value for c in model.get_component_collections()]
        np.testing.assert_allclose(areas_nm, areas_angstrom, rtol=0, atol=WEIGHT_ATOL)

    @pytest.mark.parametrize('order', [0, 2 * NUMBER_OF_SAMPLES + 1, 1.5])
    def test_order_validation(self, model, order):
        # WHEN THEN EXPECT
        with pytest.raises((TypeError, ValueError), match='order must be'):
            model.calculate_QISF_order(np.array([1.0]), order)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    @pytest.mark.parametrize('number_of_sites', [3, 7])
    def test_components_per_Q(self, number_of_sites):
        # WHEN
        model = JumpSitesLogNormalDistribution(
            number_of_sites=number_of_sites, Q=np.array([0.5, 1.0])
        )
        n = NUMBER_OF_SAMPLES * (number_of_sites - 1)

        # THEN
        collections = model.get_component_collections()

        # EXPECT
        assert len(collections) == 2
        for collection in collections:
            assert len(collection) == n + 1
            assert isinstance(collection[0], DeltaFunction)
            assert all(isinstance(component, Lorentzian) for component in collection[1:])
            assert collection[n].name == model.lorentzian_component_name(n)
        assert model.lorentzian_component_name(5) == 'JumpSitesLogNormalDistribution_j5'

    def test_component_areas_and_widths_match_calculations(self):
        """The dependency strings must give what the Scipp calculations give."""
        # WHEN
        Q = np.array([0.4, 2.3])
        model = JumpSitesLogNormalDistribution(
            scale=2.5, radius=1.4, residence_time=0.8, sigma=0.7, number_of_sites=4, Q=Q
        )

        # THEN
        collections = model.get_component_collections()

        # EXPECT
        EISF = model.calculate_EISF(Q)
        for i, collection in enumerate(collections):
            assert collection[0].area.value == pytest.approx(2.5 * EISF[i], rel=0, abs=WEIGHT_ATOL)
            for order in range(1, model.number_of_lorentzians + 1):
                lorentzian = collection[order]
                weight = model.calculate_QISF_order(Q, order)[i]
                assert lorentzian.area.value == pytest.approx(2.5 * weight, rel=0, abs=WEIGHT_ATOL)
                assert lorentzian.width.value == pytest.approx(
                    model.calculate_width(Q, order)[i], rel=1e-12
                )
                assert lorentzian.width.unit == 'meV'

    def test_components_follow_parameter_changes(self):
        # WHEN
        Q = np.array([0.6, 1.8])
        model = JumpSitesLogNormalDistribution(number_of_sites=3, Q=Q)

        # THEN
        model.radius = 2.2
        model.residence_time = 3.5
        model.sigma = 0.3
        model.scale = 4.0

        # EXPECT
        collections = model.get_component_collections()
        for order in (1, 7, 21, 30, 42):
            for i, collection in enumerate(collections):
                assert collection[order].area.value == pytest.approx(
                    4.0 * model.calculate_QISF_order(Q, order)[i], rel=0, abs=WEIGHT_ATOL
                )
                assert collection[order].width.value == pytest.approx(
                    model.calculate_width(Q, order)[i], rel=1e-12
                )

    def test_component_parameters_are_dependent(self):
        # WHEN
        model = JumpSitesLogNormalDistribution(Q=np.array([1.0]))
        collection = model.get_component_collections()[0]

        # THEN
        fit_parameters = model.get_fit_parameters()

        # EXPECT
        for component in collection:
            assert component.area.independent is False
            assert all(component.area is not p for p in fit_parameters)
        for component in collection[1:]:
            assert component.width.independent is False
            assert all(component.width is not p for p in fit_parameters)

    def test_parameter_updates_raise_no_runtime_warning(self):
        """A fit changes tau and sigma on every step; the dependent values must not warn."""
        # WHEN
        model = JumpSitesLogNormalDistribution(Q=np.array([0.5, 1.0]))

        # THEN EXPECT
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            model.residence_time = 3.0
            model.sigma = 0.4

    def test_fit_targets(self, model):
        # WHEN THEN
        names = [target.name for target in model.get_fit_targets()]

        # EXPECT
        assert names[0] == 'elastic_area'
        assert len(names) == 1 + 2 * model.number_of_lorentzians

    def test_get_global_variables(self, model):
        # WHEN THEN EXPECT
        assert model.get_global_variables() == [
            model.scale,
            model.radius,
            model.residence_time,
            model.sigma,
        ]

    def test_repr(self, model):
        # WHEN THEN EXPECT
        assert 'sigma=' in repr(model)
        assert 'number_of_sites=3' in repr(model)

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------

    def test_fit_recovers_parameters(self):
        """One Q, synthetic data from the model; tau and sigma are fitted, as in a real fit."""
        # WHEN
        Q = sc.array(dims=['Q'], values=[1.2], unit='1/angstrom')
        energy = sc.linspace('energy', -2.0, 2.0, 201, unit='meV')
        shape = (1, energy.shape[0])

        def as_data(values, variances):
            return sc.DataArray(
                sc.array(dims=['Q', 'energy'], values=values, variances=variances),
                coords={'Q': Q, 'energy': energy},
            )

        true_model = JumpSitesLogNormalDistribution(
            radius=1.5, residence_time=2.0, sigma=0.5, number_of_sites=2
        )
        experiment = edyn.Experiment('synthetic', data=as_data(np.ones(shape), np.ones(shape)))
        clean = np.array(
            edyn.Analysis(
                experiment=experiment, sample_model=edyn.SampleModel(diffusion_models=true_model)
            ).calculate()
        )
        noise = 0.003 * clean.max()
        rng = np.random.default_rng(seed=1)
        experiment.data = as_data(clean + rng.normal(0.0, noise, shape), np.full(shape, noise**2))
        fit_model = JumpSitesLogNormalDistribution(
            radius=1.5, residence_time=1.5, sigma=0.8, number_of_sites=2
        )
        analysis = edyn.Analysis(
            experiment=experiment, sample_model=edyn.SampleModel(diffusion_models=fit_model)
        )
        analysis.fix_energy_offset()

        # THEN
        analysis.fit(fit_method='independent')

        # EXPECT
        for name, true_value in {'residence_time': 2.0, 'sigma': 0.5}.items():
            parameter = getattr(fit_model, name)
            assert parameter.variance > 0
            assert abs(parameter.value - true_value) < 5 * parameter.error
        assert fit_model.radius.value == pytest.approx(1.5)  # fixed, so unchanged
