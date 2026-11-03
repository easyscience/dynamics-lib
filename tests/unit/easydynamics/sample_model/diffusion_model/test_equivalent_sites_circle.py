# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from scipp.constants import hbar as scipp_hbar

from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import DeltaLorentzian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.diffusion_model.equivalent_sites_circle import EquivalentSitesCircle

# hbar in meV ps, from scipp's constants (independent of the model)
HBAR_MEV_PS = sc.to_unit(scipp_hbar, 'meV*ps').value


def j0(x: float | np.ndarray) -> np.ndarray:
    """Spherical Bessel function j0(x) = sin(x) / x (np.sinc(x) is sin(pi x) / (pi x))."""
    return np.sinc(np.asarray(x) / np.pi)


def direct_structure_factor(Q: np.ndarray, radius: float, sites: int, order: int) -> np.ndarray:
    """A_l(Q) = (1/N) sum_{k=1}^{N} j0(Q d_k) cos(2 pi l k / N), d_k = 2 R sin(pi k / N)."""
    k = np.arange(1, sites + 1)
    distance = 2 * radius * np.sin(np.pi * k / sites)
    return (j0(np.outer(Q, distance)) * np.cos(2 * np.pi * order * k / sites)).sum(axis=1) / sites


class TestEquivalentSitesCircle:
    @pytest.fixture
    def model(self):
        return EquivalentSitesCircle(radius=1.5, residence_time=2.0, number_of_sites=3)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_init_default(self):
        # WHEN THEN
        model = EquivalentSitesCircle()

        # EXPECT
        assert model.display_name == 'EquivalentSitesCircle'
        assert model.x_unit == 'meV'
        assert model.y_unit == 'dimensionless'
        assert model.scale.value == pytest.approx(1.0)
        assert model.radius.value == pytest.approx(1.0)
        assert model.residence_time.value == pytest.approx(1.0)
        assert model.number_of_sites == 3

    def test_parameters_units_and_freedom(self, model: EquivalentSitesCircle):
        # EXPECT: the radius is known from the geometry, so it is fixed by default
        assert sc.Unit(str(model.radius.unit)) == sc.Unit('angstrom')
        assert model.radius.fixed is True
        assert sc.Unit(str(model.residence_time.unit)) == sc.Unit('ps')
        assert model.residence_time.fixed is False

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            ({'radius': 'a'}, TypeError, 'radius must be a number'),
            ({'radius': 0.0}, ValueError, 'radius must be at least'),
            ({'radius': -1.0}, ValueError, 'radius must be at least'),
            ({'residence_time': 'a'}, TypeError, 'residence_time must be a number'),
            ({'residence_time': 0.0}, ValueError, 'residence_time must be at least'),
            ({'number_of_sites': 2.5}, TypeError, 'number_of_sites must be an integer'),
            ({'number_of_sites': True}, TypeError, 'number_of_sites must be an integer'),
            ({'number_of_sites': 1}, ValueError, 'number_of_sites must be between 2 and 50'),
            ({'number_of_sites': 51}, ValueError, 'number_of_sites must be between 2 and 50'),
            ({'scale': -1.0}, ValueError, 'scale must be non-negative'),
        ],
    )
    def test_init_invalid_inputs(self, kwargs, expected_exception, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            EquivalentSitesCircle(**kwargs)

    def test_radius_and_residence_time_setters(self, model: EquivalentSitesCircle):
        # WHEN
        model.radius = 2.0
        model.residence_time = 5.0

        # EXPECT
        assert model.radius.value == pytest.approx(2.0)
        assert model.residence_time.value == pytest.approx(5.0)
        with pytest.raises(ValueError, match='radius must be at least'):
            model.radius = 0.0
        with pytest.raises(TypeError, match='residence_time must be a number'):
            model.residence_time = 'a'

    def test_number_of_sites_is_read_only(self, model: EquivalentSitesCircle):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            model.number_of_sites = 4

    # ------------------------------------------------------------------
    # Physics, against closed forms
    # ------------------------------------------------------------------

    def test_two_sites_closed_forms(self):
        # GIVEN: N = 2 is the two-site jump model with jump distance d = 2R:
        # A0 = (1 + j0(Qd)) / 2, A1 = (1 - j0(Qd)) / 2, Gamma_1 = 2 hbar / tau
        radius, tau = 1.2, 3.0
        model = EquivalentSitesCircle(radius=radius, residence_time=tau, number_of_sites=2)
        Q = np.array([0.3, 1.0, 2.5])

        # WHEN THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_EISF(Q), (1 + j0(Q * 2 * radius)) / 2, rtol=1e-10
        )
        np.testing.assert_allclose(
            model.calculate_QISF_order(Q, 1), (1 - j0(Q * 2 * radius)) / 2, rtol=1e-10
        )
        np.testing.assert_allclose(model.calculate_width(Q, 1), 2 * HBAR_MEV_PS / tau, rtol=1e-6)

    def test_eisf_plus_qisf_is_one(self):
        # GIVEN: the elastic and quasi-elastic parts share all of the scattering
        model = EquivalentSitesCircle(radius=1.7, number_of_sites=5)
        Q = np.linspace(0.1, 4.0, 15)

        # WHEN
        total = model.calculate_EISF(Q) + model.calculate_QISF(Q)

        # EXPECT
        np.testing.assert_allclose(total, 1.0, rtol=1e-12)

    def test_qisf_is_sum_of_orders(self):
        # GIVEN
        model = EquivalentSitesCircle(radius=1.0, number_of_sites=4)
        Q = np.linspace(0.2, 3.0, 7)

        # WHEN
        by_order = sum(model.calculate_QISF_order(Q, order) for order in (1, 2, 3))

        # EXPECT
        np.testing.assert_allclose(model.calculate_QISF(Q), by_order, rtol=1e-12)

    def test_eisf_is_one_at_zero_q(self, model: EquivalentSitesCircle):
        # WHEN THEN EXPECT: no spatial resolution at Q = 0, so all scattering is elastic
        assert model.calculate_EISF(np.array([0.0]))[0] == pytest.approx(1.0)
        assert model.calculate_QISF(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize('sites', [2, 3, 4, 5, 6, 7, 12])
    def test_structure_factors_match_the_direct_sum(self, sites):
        # GIVEN: odd and even N; the oracle is the plain sum over all N sites of the derivation
        radius = 1.3
        model = EquivalentSitesCircle(radius=radius, number_of_sites=sites)
        Q = np.array([0.0, 0.2, 0.9, 1.7, 3.1, 6.0])

        # WHEN THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_EISF(Q),
            direct_structure_factor(Q, radius, sites, 0),
            rtol=1e-10,
            atol=1e-12,
        )
        for order in range(1, sites):
            np.testing.assert_allclose(
                model.calculate_QISF_order(Q, order),
                direct_structure_factor(Q, radius, sites, order),
                rtol=1e-10,
                atol=1e-12,
            )

    @pytest.mark.parametrize('sites', [2, 5, 6])
    def test_eisf_tends_to_one_over_n_at_large_q(self, sites):
        # GIVEN: for Q -> infinity only the term with d = 0 survives, j0(Q d) -> 0 otherwise
        model = EquivalentSitesCircle(radius=1.0, number_of_sites=sites)

        # WHEN THEN EXPECT
        assert model.calculate_EISF(np.array([500.0]))[0] == pytest.approx(1 / sites, abs=1e-2)

    def test_largest_number_of_sites_works(self):
        # GIVEN
        model = EquivalentSitesCircle(number_of_sites=50, Q=np.array([0.7]))

        # WHEN THEN EXPECT: the sum rule still holds and the component has 49 Lorentzians
        np.testing.assert_allclose(
            model.calculate_EISF() + model.calculate_QISF(), 1.0, rtol=1e-10
        )
        assert model.get_component_collections(0)[0].number_of_lorentzians == 49

    def test_widths_of_three_sites(self):
        # GIVEN: Gamma_l = (2 hbar / tau) sin^2(pi l / 3) = 1.5 hbar / tau for l = 1, 2
        model = EquivalentSitesCircle(residence_time=2.0, number_of_sites=3)
        Q = np.array([0.5, 1.5])

        # WHEN THEN EXPECT: independent of Q
        for order in (1, 2):
            np.testing.assert_allclose(
                model.calculate_width(Q, order), 1.5 * HBAR_MEV_PS / 2.0, rtol=1e-6
            )

    def test_width_follows_x_unit(self):
        # GIVEN
        model = EquivalentSitesCircle(residence_time=2.0, number_of_sites=2, x_unit='microeV')

        # WHEN THEN EXPECT: 2 hbar / tau in micro-eV
        np.testing.assert_allclose(
            model.calculate_width(np.array([1.0]), 1), 2000 * HBAR_MEV_PS / 2.0, rtol=1e-6
        )

    @pytest.mark.parametrize('order', [0, 3, 1.5, True, 'a'])
    def test_invalid_order_raises(self, model: EquivalentSitesCircle, order):
        # WHEN THEN EXPECT: N = 3 has Lorentzian orders 1 and 2
        error = TypeError if not isinstance(order, int) or isinstance(order, bool) else ValueError
        with pytest.raises(error, match='order'):
            model.calculate_width(np.array([1.0]), order)
        with pytest.raises(error, match='order'):
            model.calculate_QISF_order(np.array([1.0]), order)

    def test_calculate_without_q_raises(self, model: EquivalentSitesCircle):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='Q'):
            model.calculate_EISF()

    def test_calculate_uses_stored_q(self):
        # GIVEN
        Q = np.array([0.5, 1.0])
        model = EquivalentSitesCircle(Q=Q)

        # WHEN THEN EXPECT
        np.testing.assert_allclose(model.calculate_EISF(), model.calculate_EISF(Q))

    def test_calculate_scipp_q_converts_unit(self, model: EquivalentSitesCircle):
        # GIVEN: the same Q in 1/angstrom (numpy, assumed) and in 1/nm (scipp)
        Q = np.array([0.5, 1.0])
        Q_scipp = sc.array(dims=['Q'], values=Q * 10, unit='1/nm')

        # WHEN THEN EXPECT
        np.testing.assert_allclose(model.calculate_EISF(Q_scipp), model.calculate_EISF(Q))

    def test_radius_unit_conversion_keeps_the_physics(self):
        # GIVEN: a radius of 1.5 angstrom, converted to 0.15 nm
        model = EquivalentSitesCircle(radius=1.5, number_of_sites=3)
        Q = np.array([0.5, 1.0, 2.0])
        before = model.calculate_EISF(Q)

        # WHEN
        model.radius.convert_unit('nm')

        # EXPECT: the structure factor still describes the same circle
        np.testing.assert_allclose(model.calculate_EISF(Q), before, rtol=1e-10)

    # ------------------------------------------------------------------
    # Component collections
    # ------------------------------------------------------------------

    def test_no_collections_without_q(self, model: EquivalentSitesCircle):
        # WHEN THEN EXPECT
        assert model.get_component_collections() == []

    def test_one_delta_lorentzian_per_q(self):
        # GIVEN: N = 4 sites -> three Lorentzians
        Q = np.array([0.5, 1.0, 1.5])
        model = EquivalentSitesCircle(number_of_sites=4, Q=Q)

        # WHEN
        collections = model.get_component_collections()

        # EXPECT
        assert len(collections) == 3
        for collection in collections:
            assert len(collection) == 1
            component = collection[0]
            assert isinstance(component, DeltaLorentzian)
            assert component.number_of_lorentzians == 3
            assert component.name == 'EquivalentSitesCircle'

    def test_component_name_follows_lorentzian_name(self):
        # WHEN
        model = EquivalentSitesCircle(Q=np.array([1.0]), lorentzian_name='Rotor')

        # EXPECT
        assert model.get_component_collections(0)[0].name == 'Rotor'

    def test_collection_names_follow_q(self):
        # WHEN
        model = EquivalentSitesCircle(Q=np.array([0.5, 1.25]))

        # EXPECT
        assert model.get_component_collections(0).name == 'EquivalentSitesCircle_Q0.50'
        assert model.get_component_collections(1).name == 'EquivalentSitesCircle_Q1.25'

    def test_component_parameters_are_named_like_the_component_names_them(self):
        # GIVEN: a component made directly would call its parameters '<name> weight_<l>' ...
        model = EquivalentSitesCircle(
            number_of_sites=4, Q=np.array([1.0, 2.0]), lorentzian_name='Rotor'
        )

        # WHEN THEN EXPECT
        for i in range(2):
            component = model.get_component_collections(i)[0]
            assert [w.name for w in component.weights] == [
                f'Rotor weight_{order}' for order in range(4)
            ]
            assert [w.name for w in component.widths] == [
                f'Rotor width_{order}' for order in (1, 2, 3)
            ]

    def test_weights_of_the_component_match_the_direct_sum(self):
        # GIVEN
        Q = np.array([0.6, 2.2])
        model = EquivalentSitesCircle(radius=1.4, number_of_sites=5, Q=Q)

        for i in range(2):
            component = model.get_component_collections(i)[0]

            # WHEN THEN EXPECT: every weight, delta function first
            for order, weight in enumerate(component.weights):
                assert weight.value == pytest.approx(
                    direct_structure_factor(Q[i : i + 1], 1.4, 5, order)[0], abs=1e-10
                )

    def test_create_component_collections_installs_collections(self):
        # WHEN
        model = EquivalentSitesCircle(Q=np.array([1.0, 2.0]))

        # THEN
        collections = model.create_component_collections()

        # EXPECT: the returned list is the live one, as for the other diffusion models
        assert collections is model.get_component_collections()

    def test_component_values_match_the_model(self):
        # GIVEN
        Q = np.array([0.4, 1.3])
        model = EquivalentSitesCircle(scale=2.0, radius=1.1, residence_time=1.5, Q=Q)

        for i in range(len(Q)):
            component = model.get_component_collections(i)[0]

            # EXPECT: weights and widths are the model's structure factors and widths
            assert component.weights[0].value == pytest.approx(model.calculate_EISF(Q)[i])
            assert component.weights[1].value == pytest.approx(model.calculate_QISF_order(Q, 1)[i])
            assert component.weights[2].value == pytest.approx(model.calculate_QISF_order(Q, 2)[i])
            assert component.widths[0].value == pytest.approx(model.calculate_width(Q, 1)[0])
            assert component.area.value == pytest.approx(2.0)

    def test_evaluate_equals_separate_components(self):
        # GIVEN: the collection must be a delta plus Lorentzians with the model's areas and widths
        Q = np.array([0.8])
        model = EquivalentSitesCircle(scale=2.0, radius=1.1, residence_time=1.5, Q=Q)
        x = np.linspace(-3, 3, 601)

        reference = DeltaFunction(area=2.0 * model.calculate_EISF(Q)[0]).evaluate(x)
        for order in (1, 2):
            reference += Lorentzian(
                area=2.0 * model.calculate_QISF_order(Q, order)[0],
                width=model.calculate_width(Q, order)[0],
            ).evaluate(x)

        # WHEN
        result = model.get_component_collections(0).evaluate(x)

        # EXPECT
        np.testing.assert_allclose(result, reference, rtol=1e-9)

    def test_dependent_parameters(self):
        # GIVEN
        model = EquivalentSitesCircle(Q=np.array([1.0]))
        component = model.get_component_collections(0)[0]

        # EXPECT: area, weights and widths are driven by the model, not fitted themselves
        assert component.area.independent is False
        assert all(weight.independent is False for weight in component.weights)
        assert all(width.independent is False for width in component.widths)

    def test_widths_follow_residence_time(self):
        # GIVEN
        model = EquivalentSitesCircle(residence_time=2.0, number_of_sites=2, Q=np.array([1.0]))
        component = model.get_component_collections(0)[0]
        assert component.widths[0].value == pytest.approx(2 * HBAR_MEV_PS / 2.0, rel=1e-6)

        # WHEN
        model.residence_time.value = 4.0

        # EXPECT
        assert component.widths[0].value == pytest.approx(2 * HBAR_MEV_PS / 4.0, rel=1e-6)

    def test_weights_follow_radius(self):
        # GIVEN
        Q = np.array([0.9])
        model = EquivalentSitesCircle(radius=1.0, number_of_sites=2, Q=Q)
        component = model.get_component_collections(0)[0]

        # WHEN
        model.radius.value = 2.0

        # EXPECT: N = 2 closed forms with the new radius
        assert component.weights[0].value == pytest.approx((1 + j0(0.9 * 4.0)) / 2, rel=1e-9)
        assert component.weights[1].value == pytest.approx((1 - j0(0.9 * 4.0)) / 2, rel=1e-9)

    def test_area_follows_scale(self):
        # GIVEN
        model = EquivalentSitesCircle(scale=1.0, Q=np.array([1.0]))
        component = model.get_component_collections(0)[0]

        # WHEN
        model.scale.value = 3.0

        # EXPECT
        assert component.area.value == pytest.approx(3.0)

    def test_collections_follow_q_changes(self, model: EquivalentSitesCircle):
        # WHEN
        model.Q = np.array([0.5, 1.0])

        # THEN
        assert len(model.get_component_collections()) == 2

        # WHEN: Q is cleared, and then set to other values
        model.clear_Q(confirm=True)

        # THEN
        assert model.get_component_collections() == []

        # WHEN
        model.Q = np.array([0.5, 1.0, 1.5, 2.0])

        # EXPECT
        assert len(model.get_component_collections()) == 4

    def test_changing_q_without_clearing_raises(self):
        # GIVEN
        model = EquivalentSitesCircle(Q=np.array([0.5, 1.0]))

        # WHEN THEN EXPECT: as for the other diffusion models
        with pytest.raises(ValueError, match='clear_Q'):
            model.Q = np.array([0.5, 1.0, 1.5])

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------

    def test_convert_x_unit(self):
        # GIVEN
        model = EquivalentSitesCircle(residence_time=2.0, number_of_sites=2, Q=np.array([1.0]))
        component = model.get_component_collections(0)[0]
        width_mev = component.widths[0].value

        # WHEN
        model.convert_x_unit('microeV')

        # EXPECT: the width, the scale and the calculated width follow the unit; the weights do not
        assert model.x_unit == 'microeV'
        assert component.widths[0].value == pytest.approx(width_mev * 1000)
        assert sc.Unit(str(model.scale.unit)) == sc.Unit('microeV')
        assert model.calculate_width(np.array([1.0]), 1)[0] == pytest.approx(width_mev * 1000)
        assert component.weights[0].value == pytest.approx(
            model.calculate_EISF(np.array([1.0]))[0]
        )

    def test_convert_x_unit_persists_after_dependency_update(self):
        # GIVEN
        model = EquivalentSitesCircle(residence_time=2.0, number_of_sites=2, Q=np.array([1.0]))
        component = model.get_component_collections(0)[0]
        model.convert_x_unit('microeV')

        # WHEN: a parameter change triggers the dependency to be re-evaluated
        model.residence_time.value = 4.0

        # EXPECT: the width is still in micro-eV
        assert component.widths[0].value == pytest.approx(2000 * HBAR_MEV_PS / 4.0, rel=1e-6)

    def test_results_do_not_depend_on_the_unit_of_the_radius(self):
        # GIVEN: the same circle, 1.5 angstrom = 0.15 nm
        Q = np.array([0.4, 1.3, 2.6])
        reference = EquivalentSitesCircle(radius=1.5, number_of_sites=4, Q=Q)
        converted = EquivalentSitesCircle(radius=1.0, number_of_sites=4, Q=Q)
        converted.radius.convert_unit('nm')

        # WHEN: the radius is set in nanometre
        converted.radius.value = 0.15

        # EXPECT: the methods and the linked weights of the component agree with the reference
        np.testing.assert_allclose(
            converted.calculate_EISF(), reference.calculate_EISF(), rtol=1e-10
        )
        for order in (1, 2, 3):
            np.testing.assert_allclose(
                converted.calculate_QISF_order(order=order),
                reference.calculate_QISF_order(order=order),
                rtol=1e-10,
            )
        for i in range(3):
            converted_weights = converted.get_component_collections(i)[0].weights
            reference_weights = reference.get_component_collections(i)[0].weights
            for got, expected in zip(converted_weights, reference_weights, strict=True):
                assert got.value == pytest.approx(expected.value, abs=1e-10)

    def test_all_links_work_after_x_unit_conversion(self):
        # GIVEN
        Q = np.array([1.1])
        model = EquivalentSitesCircle(
            scale=2.0, radius=1.0, residence_time=2.0, number_of_sites=2, Q=Q
        )
        component = model.get_component_collections(0)[0]
        model.convert_x_unit('microeV')

        # WHEN: each model parameter is changed after the conversion
        model.residence_time.value = 4.0
        model.radius.value = 2.0
        model.scale.value = 3000.0

        # EXPECT: widths in microeV, weights (N = 2 closed forms) and area follow
        assert component.widths[0].value == pytest.approx(2000 * HBAR_MEV_PS / 4.0, rel=1e-6)
        assert component.weights[0].value == pytest.approx((1 + j0(1.1 * 4.0)) / 2, abs=1e-10)
        assert component.weights[1].value == pytest.approx((1 - j0(1.1 * 4.0)) / 2, abs=1e-10)
        assert component.area.value == pytest.approx(3000.0)

    def test_y_unit_setter_raises(self, model: EquivalentSitesCircle):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            model.y_unit = '1/meV'

    # ------------------------------------------------------------------
    # Parameters and fit targets
    # ------------------------------------------------------------------

    def test_global_variables(self, model: EquivalentSitesCircle):
        # WHEN
        variables = model.get_global_variables()

        # EXPECT
        assert model.scale in variables
        assert model.radius in variables
        assert model.residence_time in variables

    def test_free_parameters_exclude_the_fixed_radius(self, model: EquivalentSitesCircle):
        # WHEN
        free = model.get_free_parameters()

        # EXPECT
        assert model.radius not in free
        assert model.residence_time in free
        assert model.scale in free

    def test_fit_targets(self):
        # GIVEN: N = 3 -> elastic area plus area and width of orders 1 and 2
        Q = np.array([0.5, 1.0])
        model = EquivalentSitesCircle(
            scale=2.0, radius=1.0, residence_time=2.0, number_of_sites=3, Q=Q
        )

        # WHEN
        targets = {target.name: target for target in model.get_fit_targets()}

        # EXPECT
        assert set(targets) == {'elastic_area', 'area_1', 'width_1', 'area_2', 'width_2'}
        np.testing.assert_allclose(
            targets['elastic_area'].function(Q), 2.0 * model.calculate_EISF(Q)
        )
        np.testing.assert_allclose(
            targets['area_2'].function(Q), 2.0 * model.calculate_QISF_order(Q, 2)
        )
        np.testing.assert_allclose(targets['width_1'].function(Q), model.calculate_width(Q, 1))
        assert sc.Unit(str(targets['width_1'].y_unit)) == sc.Unit('meV')
        assert all(target.x_unit == '1/angstrom' for target in targets.values())
        assert sc.Unit(str(targets['area_1'].y_unit)) == sc.Unit(str(model.scale.unit))

    def test_fit_target_dataset_keys(self):
        # GIVEN
        model = EquivalentSitesCircle(number_of_sites=3, lorentzian_name='Rotor')

        # WHEN
        targets = {target.name: target for target in model.get_fit_targets()}

        # EXPECT: widths are parameters of the component, so they have a default key; the
        # areas are scale * weight and the component has no parameter for them
        assert targets['width_1'].dataset_key == 'Rotor width_1'
        assert targets['width_2'].dataset_key == 'Rotor width_2'
        assert targets['elastic_area'].dataset_key is None
        assert targets['area_1'].dataset_key is None

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def test_is_exported(self):
        # WHEN THEN EXPECT
        import easydynamics as edyn
        from easydynamics.sample_model import EquivalentSitesCircle as FromSampleModel
        from easydynamics.sample_model.diffusion_model import EquivalentSitesCircle as FromPackage

        assert edyn.EquivalentSitesCircle is EquivalentSitesCircle
        assert FromSampleModel is EquivalentSitesCircle
        assert FromPackage is EquivalentSitesCircle

    def test_repr(self, model: EquivalentSitesCircle):
        # WHEN
        text = repr(model)

        # EXPECT
        assert 'EquivalentSitesCircle' in text
        assert 'number_of_sites=3' in text
        assert 'radius=' in text
        assert 'residence_time=' in text
        assert 'scale=' in text
