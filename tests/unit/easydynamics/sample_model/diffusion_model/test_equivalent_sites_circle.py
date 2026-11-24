# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from scipp.constants import hbar as scipp_hbar

from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.equivalent_sites_circle import (
    MAXIMUM_NUMBER_OF_SITES,
)
from easydynamics.sample_model.diffusion_model.equivalent_sites_circle import MINIMUM_RADIUS
from easydynamics.sample_model.diffusion_model.equivalent_sites_circle import MINIMUM_WIDTH
from easydynamics.sample_model.diffusion_model.equivalent_sites_circle import EquivalentSitesCircle

# A weight is a sum of N terms of size up to 1/N that can nearly cancel, so its rounding error is
# absolute, a few N * machine epsilon, not relative to the (possibly tiny) result.
WEIGHT_ATOL = 1e-14


def hbar_over(residence_time_ps: float) -> float:
    """hbar / tau in meV, computed with Scipp's constants."""
    return (scipp_hbar / sc.scalar(residence_time_ps, unit='ps')).to(unit='meV').value


class TestEquivalentSitesCircle:
    @pytest.fixture
    def model(self):
        return EquivalentSitesCircle()

    def test_create_weight_requires_float_Q(self, model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be a float'):
            model._create_weight(1, 1)

    # ------------------------------------------------------------------
    # Construction and parameters
    # ------------------------------------------------------------------

    def test_init_default(self, model):
        # WHEN THEN EXPECT
        assert model.display_name == 'EquivalentSitesCircle'
        assert model.x_unit == 'meV'
        assert model.y_unit == 'dimensionless'
        assert model.scale.value == pytest.approx(1.0)
        assert model.number_of_sites == 3
        assert model.radius.value == pytest.approx(1.0)
        assert model.radius.unit == 'Å'
        assert model.radius.fixed is True
        assert model.radius.min == MINIMUM_RADIUS
        assert model.residence_time.value == pytest.approx(1.0)
        assert model.residence_time.unit == 'ps'
        assert model.residence_time.fixed is False
        assert model.residence_time.min == MINIMUM_WIDTH

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            ({'radius': 'invalid'}, TypeError, 'radius must be a number'),
            ({'radius': 0.0}, ValueError, 'radius must be at least'),
            ({'residence_time': 'invalid'}, TypeError, 'residence_time must be a number'),
            ({'residence_time': 0.0}, ValueError, 'residence_time must be at least'),
            ({'number_of_sites': 2.5}, TypeError, 'number_of_sites must be an integer'),
            ({'number_of_sites': True}, TypeError, 'number_of_sites must be an integer'),
            ({'number_of_sites': 1}, ValueError, 'number_of_sites must be between 2 and'),
            (
                {'number_of_sites': MAXIMUM_NUMBER_OF_SITES + 1},
                ValueError,
                'number_of_sites must be between 2 and',
            ),
            ({'delta_name': 1}, TypeError, 'delta_name must be a string'),
        ],
        ids=[
            'radius_type',
            'radius_zero',
            'residence_time_type',
            'residence_time_zero',
            'sites_float',
            'sites_bool',
            'sites_too_few',
            'sites_too_many',
            'delta_name_type',
        ],
    )
    def test_init_rejects_invalid_input(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            EquivalentSitesCircle(**kwargs)

    def test_number_of_sites_accepts_numpy_integer(self):
        # WHEN THEN EXPECT
        assert EquivalentSitesCircle(number_of_sites=np.int64(4)).number_of_sites == 4

    def test_number_of_sites_is_read_only(self, model):
        with pytest.raises(AttributeError):
            model.number_of_sites = 4

    def test_radius_setter(self, model):
        # WHEN
        model.radius = 2.5

        # THEN EXPECT
        assert model.radius.value == pytest.approx(2.5)

    @pytest.mark.parametrize(
        'value, expected_exception',
        [('invalid', TypeError), (0.0, ValueError)],
        ids=['type', 'zero'],
    )
    def test_radius_setter_raises(self, model, value, expected_exception):
        with pytest.raises(expected_exception, match='radius must be'):
            model.radius = value

    def test_residence_time_setter(self, model):
        # WHEN
        model.residence_time = 3.0

        # THEN EXPECT
        assert model.residence_time.value == pytest.approx(3.0)

    @pytest.mark.parametrize(
        'value, expected_exception',
        [('invalid', TypeError), (0.0, ValueError), (MINIMUM_WIDTH / 10, ValueError)],
        ids=['type', 'zero', 'below_minimum'],
    )
    def test_residence_time_setter_raises(self, model, value, expected_exception):
        with pytest.raises(expected_exception, match='residence_time must be'):
            model.residence_time = value

    # ------------------------------------------------------------------
    # Physics of the structure factors and widths
    # ------------------------------------------------------------------

    @pytest.mark.parametrize('number_of_sites', [2, 3, 6, 11])
    def test_weights_sum_to_one(self, number_of_sites):
        """EISF + sum of all quasi-elastic weights is exactly 1 (no truncation)."""
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=number_of_sites, radius=1.3)
        Q = np.array([1e-3, 0.3, 1.0, 2.5, 7.0])

        # THEN
        total = model.calculate_EISF(Q) + model.calculate_QISF(Q)

        # EXPECT
        np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-12)

    def test_QISF_is_sum_of_orders(self):
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=5, radius=1.3)
        Q = np.array([0.5, 1.5])

        # THEN
        orders = [model.calculate_QISF_order(Q, order) for order in range(1, 5)]

        # EXPECT
        np.testing.assert_allclose(
            model.calculate_QISF(Q), np.sum(orders, axis=0), rtol=0, atol=WEIGHT_ATOL
        )

    def test_two_sites_is_the_two_site_jump_model(self):
        """N = 2: EISF = (1 + j0(2QR)) / 2 and Gamma = 2 hbar / tau (Bee)."""
        # WHEN
        radius = 1.2
        model = EquivalentSitesCircle(number_of_sites=2, radius=radius, residence_time=1.5)
        Q = np.array([0.3, 1.0, 2.0])

        # THEN
        EISF = model.calculate_EISF(Q)
        width = model.calculate_width(Q, order=1)

        # EXPECT
        jump_distance = 2 * radius
        expected_EISF = (1 + np.sin(Q * jump_distance) / (Q * jump_distance)) / 2
        np.testing.assert_allclose(EISF, expected_EISF, rtol=0, atol=WEIGHT_ATOL)
        np.testing.assert_allclose(
            model.calculate_QISF_order(Q, 1), 1 - expected_EISF, rtol=0, atol=WEIGHT_ATOL
        )
        np.testing.assert_allclose(width, 2 * hbar_over(1.5), rtol=1e-12)

    def test_EISF_tends_to_one_at_small_Q(self):
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=4, radius=2.0)

        # THEN EXPECT
        assert model.calculate_EISF(np.array([1e-8]))[0] == pytest.approx(1.0, rel=1e-12)

    def test_orders_l_and_N_minus_l_are_equal(self):
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=6, radius=1.1, residence_time=2.0)
        Q = np.array([0.4, 1.7])

        # THEN EXPECT
        for order in range(1, 6):
            np.testing.assert_allclose(
                model.calculate_QISF_order(Q, order),
                model.calculate_QISF_order(Q, 6 - order),
                rtol=0,
                atol=WEIGHT_ATOL,
            )
            np.testing.assert_allclose(
                model.calculate_width(Q, order), model.calculate_width(Q, 6 - order), rtol=1e-12
            )

    def test_width_follows_sine_squared(self):
        """Gamma_l = (2 hbar / tau) sin^2(pi l / N), the residence-time convention."""
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=5, residence_time=2.0)
        Q = np.array([0.5, 1.5])

        # THEN EXPECT
        for order in range(1, 5):
            expected = 2 * hbar_over(2.0) * np.sin(np.pi * order / 5) ** 2
            np.testing.assert_allclose(model.calculate_width(Q, order), expected, rtol=1e-12)

    def test_matches_QENSmodels_reference_values(self):
        """
        Doctest values of QENSmodels hwhmEquivalentSitesCircle([1., 2.], 6, 0.5, 1.5), rounded
        there to the digits given. Its widths are in 1/ps, so they are multiplied by hbar here.
        """
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=6, radius=0.5, residence_time=1.5)
        Q = np.array([1.0, 2.0])

        # THEN
        EISF = model.calculate_EISF(Q)

        # EXPECT
        np.testing.assert_allclose(EISF, [0.92, 0.713], atol=5e-4)
        assert model.calculate_QISF_order(Q, 2)[0] == pytest.approx(0.000503, abs=5e-7)
        assert model.calculate_QISF_order(Q, 5)[1] == pytest.approx(0.13616, abs=5e-6)
        hbar_per_ps = hbar_over(1.0)
        assert model.calculate_width(Q, 1)[0] / hbar_per_ps == pytest.approx(0.333, abs=5e-4)
        assert model.calculate_width(Q, 3)[0] / hbar_per_ps == pytest.approx(1.333, abs=5e-4)

    def test_weights_follow_radius_unit(self):
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=4, radius=1.5)
        Q = np.array([0.8, 1.6])
        EISF_angstrom = model.calculate_EISF(Q)

        # THEN
        model.radius.convert_unit('nm')

        # EXPECT the same physical radius gives the same weights
        np.testing.assert_allclose(
            model.calculate_EISF(Q), EISF_angstrom, rtol=0, atol=WEIGHT_ATOL
        )

    @pytest.mark.parametrize('order', [0, 3, 1.0, True], ids=['zero', 'N', 'float', 'bool'])
    def test_order_validation(self, model, order):
        with pytest.raises((TypeError, ValueError), match='order must be'):
            model.calculate_QISF_order(np.array([1.0]), order)
        with pytest.raises((TypeError, ValueError), match='order must be'):
            model.calculate_width(np.array([1.0]), order)

    def test_calculate_EISF_type_error(self, model):
        with pytest.raises(TypeError, match='Q must be '):
            model.calculate_EISF(Q='invalid')

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def test_no_Q_gives_no_collections(self, model):
        # WHEN THEN EXPECT
        assert model.get_component_collections() == []

    def test_component_names_follow_the_base_convention(self):
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=3, Q=np.array([1.0]))

        # THEN EXPECT
        assert model.number_of_lorentzians == 2
        assert model.lorentzian_component_name(2) == 'EquivalentSitesCircle_l2'

    def test_components_per_Q(self):
        # WHEN
        model = EquivalentSitesCircle(number_of_sites=4, Q=np.array([0.5, 1.0]))

        # THEN
        collections = model.get_component_collections()

        # EXPECT one delta function and N - 1 Lorentzians per Q
        assert len(collections) == 2
        for collection in collections:
            assert len(collection) == 4
            assert isinstance(collection[0], DeltaFunction)
            assert all(isinstance(component, Lorentzian) for component in collection[1:])
            assert [component.name for component in collection[1:]] == [
                model.lorentzian_component_name(order) for order in range(1, 4)
            ]

    def test_component_areas_and_widths_match_calculations(self):
        """The dependency strings must give what the Scipp calculations give."""
        # WHEN
        Q = np.array([0.4, 1.1, 2.3])
        model = EquivalentSitesCircle(
            scale=2.5, radius=1.4, residence_time=0.8, number_of_sites=5, Q=Q
        )

        # THEN
        collections = model.get_component_collections()

        # EXPECT
        EISF = model.calculate_EISF(Q)
        for i, collection in enumerate(collections):
            assert collection[0].area.value == pytest.approx(2.5 * EISF[i], rel=0, abs=WEIGHT_ATOL)
            for order in range(1, 5):
                lorentzian = collection[order]
                weight = model.calculate_QISF_order(Q, order)[i]
                width = model.calculate_width(Q, order)[i]
                assert lorentzian.area.value == pytest.approx(2.5 * weight, rel=0, abs=WEIGHT_ATOL)
                assert lorentzian.width.value == pytest.approx(width, rel=1e-12)
                assert lorentzian.width.unit == 'meV'
                assert lorentzian.area.independent is False

    def test_components_follow_parameter_changes(self):
        # WHEN
        Q = np.array([0.6, 1.8])
        model = EquivalentSitesCircle(number_of_sites=3, Q=Q)

        # THEN
        model.radius = 2.2
        model.residence_time = 3.5
        model.scale = 4.0

        # EXPECT
        collections = model.get_component_collections()
        EISF = model.calculate_EISF(Q)
        for i, collection in enumerate(collections):
            assert collection[0].area.value == pytest.approx(4.0 * EISF[i], rel=0, abs=WEIGHT_ATOL)
            assert collection[1].area.value == pytest.approx(
                4.0 * model.calculate_QISF_order(Q, 1)[i], rel=0, abs=WEIGHT_ATOL
            )
            assert collection[1].width.value == pytest.approx(
                model.calculate_width(Q, 1)[i], rel=1e-12
            )

    def test_component_weights_follow_radius_unit(self):
        # WHEN
        Q = np.array([0.8, 1.6])
        model = EquivalentSitesCircle(number_of_sites=4, radius=1.5, Q=Q)
        areas_angstrom = [c[1].area.value for c in model.get_component_collections()]

        # THEN
        model.radius.convert_unit('nm')

        # EXPECT
        areas_nm = [c[1].area.value for c in model.get_component_collections()]
        np.testing.assert_allclose(areas_nm, areas_angstrom, rtol=0, atol=WEIGHT_ATOL)

    def test_create_component_collections_installs_collections(self):
        # WHEN
        model = EquivalentSitesCircle(Q=np.array([1.0, 2.0]))

        # THEN
        collections = model.create_component_collections()

        # EXPECT
        assert collections is model.get_component_collections()

    # ------------------------------------------------------------------
    # Fit targets, variables and representation
    # ------------------------------------------------------------------

    def test_fit_targets(self):
        # WHEN
        Q = np.array([0.5, 1.5])
        model = EquivalentSitesCircle(scale=2.0, number_of_sites=3, Q=Q)

        # THEN
        targets = {target.name: target for target in model.get_fit_targets()}

        # EXPECT
        assert set(targets) == {'elastic_area', 'area_1', 'width_1', 'area_2', 'width_2'}
        np.testing.assert_allclose(
            targets['elastic_area'].function(Q), 2.0 * model.calculate_EISF(Q), rtol=1e-12
        )
        np.testing.assert_allclose(
            targets['area_2'].function(Q), 2.0 * model.calculate_QISF_order(Q, 2), rtol=1e-12
        )
        np.testing.assert_allclose(
            targets['width_1'].function(Q), model.calculate_width(Q, 1), rtol=1e-12
        )

    def test_get_global_variables(self, model):
        # WHEN THEN EXPECT
        assert model.get_global_variables() == [
            model.scale,
            model.radius,
            model.residence_time,
        ]

    def test_repr(self, model):
        # WHEN THEN
        repr_str = repr(model)

        # EXPECT
        assert 'EquivalentSitesCircle' in repr_str
        assert 'number_of_sites=3' in repr_str
        assert 'radius' in repr_str
        assert 'residence_time' in repr_str
        assert 'x_unit=meV, y_unit=dimensionless' in repr_str
