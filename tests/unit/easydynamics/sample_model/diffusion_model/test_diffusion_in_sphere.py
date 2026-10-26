# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
import scipp as sc
from scipp.constants import hbar as scipp_hbar
from scipy.integrate import quad
from scipy.special import spherical_jn

import easydynamics as edyn
from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.diffusion_in_sphere import MAXIMUM_ROOT
from easydynamics.sample_model.diffusion_model.diffusion_in_sphere import DiffusionInSphere
from easydynamics.sample_model.diffusion_model.diffusion_in_sphere import sphere_roots

# Mantid InelasticDiffSphere.cpp (initXnlCoeff): the first 12 roots x_nl of j_l'(x) = 0, with
# their (l, n), and the last of its 98 roots. Mantid prints them with 6 decimals.
MANTID_FIRST_ROOTS = [
    2.081576, 3.342094, 4.493409, 4.514100, 5.646704, 5.940370,
    6.756456, 7.289932, 7.725252, 7.851078, 8.583755, 8.934839,
]  # fmt: skip
MANTID_FIRST_ORDERS = [
    (1, 0), (2, 0), (0, 1), (3, 0), (4, 0), (1, 1),
    (5, 0), (2, 1), (0, 2), (6, 0), (3, 1), (7, 0),
]  # fmt: skip
MANTID_LAST_ROOT = 26.837518

# Weights A near the 0/0 points QR = x_nl, computed once with mpmath at 50 digits from the root
# found to 50 digits (script make_sphere_limit_references.py in the 2026-10-23 session scratchpad).
LIMIT_REFERENCES = [  # (l, n, QR - x, weight A)
    (1, 0, 0.0, 0.46096729855722723),
    (1, 0, 1e-08, 0.4609673001282459),
    (1, 0, -1e-08, 0.46096729696343897),
    (1, 0, 1e-06, 0.460967456786031),
    (1, 0, -1e-06, 0.46096714030533814),
    (1, 0, 0.0001, 0.4609831210020438),
    (1, 0, -0.0001, 0.46095147293282407),
    (0, 1, 0.0, 0.07078567383644102),
    (0, 1, 1e-08, 0.0707856736811847),
    (0, 1, -1e-08, 0.07078567399624913),
    (0, 1, 1e-06, 0.07078565808547693),
    (0, 1, -1e-06, 0.07078568959191965),
    (0, 1, 0.0001, 0.070784098330299),
    (0, 1, -0.0001, 0.07078724897456277),
    (7, 2, 0.0, 0.06864322107763328),
    (7, 2, 1e-08, 0.0686432210551458),
    (7, 2, -1e-08, 0.06864322109744807),
    (7, 2, 1e-06, 0.06864321896116536),
    (7, 2, -1e-06, 0.06864322319139181),
    (7, 2, 0.0001, 0.06864300938149931),
    (7, 2, -0.0001, 0.0686434324041429),
    (12, 3, 0.0, 0.04595736490553809),
    (12, 3, 1e-08, 0.045957364897496314),
    (12, 3, -1e-08, 0.04595736491280735),
    (12, 3, 1e-06, 0.04595736413958797),
    (12, 3, -1e-06, 0.04595736567069177),
    (12, 3, 0.0001, 0.0459572882303148),
    (12, 3, -0.0001, 0.04595744134069539),
]


def order_of(model: DiffusionInSphere, angular_order: int, n: int) -> tuple[int, float]:
    """The Lorentzian order j and root x of the (l, n) term."""
    for j, (x, root_l, root_n) in enumerate(model.roots, start=1):
        if (root_l, root_n) == (angular_order, n):
            return j, x
    raise KeyError((angular_order, n))


class TestDiffusionInSphere:
    @pytest.fixture
    def model(self):
        # The physics tests check the full sum, all 98 terms
        return DiffusionInSphere(number_of_lorentzians=98)

    def test_create_weight_requires_float_Q(self, model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be a float'):
            model._create_weight(1, 1)

    # ------------------------------------------------------------------
    # Roots
    # ------------------------------------------------------------------

    def test_roots_match_mantid(self):
        # WHEN
        roots = sphere_roots()

        # THEN EXPECT
        assert len(roots) == 98
        assert all(0 < x < MAXIMUM_ROOT for x, _, _ in roots)
        assert [x for x, _, _ in roots] == sorted(x for x, _, _ in roots)
        np.testing.assert_allclose(
            [x for x, _, _ in roots[:12]], MANTID_FIRST_ROOTS, rtol=0, atol=1e-6
        )
        assert [(order, n) for _, order, n in roots[:12]] == MANTID_FIRST_ORDERS
        assert roots[-1][0] == pytest.approx(MANTID_LAST_ROOT, abs=1e-6)

    def test_roots_are_zeros_of_the_derivative(self):
        # WHEN THEN EXPECT
        for x, angular_order, _ in sphere_roots():
            assert abs(spherical_jn(angular_order, x, derivative=True)) < 1e-12

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def test_EISF(self, model):
        # WHEN
        Q = np.array([0.05, 0.7, 2.3, 4.0])

        # THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_EISF(Q), (3 * spherical_jn(1, Q) / Q) ** 2, rtol=1e-13
        )

    def test_weights_match_bicout_integral(self, model):
        """
        Bicout, PRE 62, 261 (2000), Eq. 2.12, single sphere of radius 1: A = (2l+1) |integral of
        r^2 sqrt(p_eq) j_l(Q r) psi(r) dr|^2 with p_eq = 3 and psi(r) = r j_l(x r) / sqrt(Z).
        """
        # WHEN THEN EXPECT
        for j, (x, order, _) in enumerate(model.roots[:10], start=1):
            Z = quad(
                lambda r, x=x, order=order: r * r * spherical_jn(order, x * r) ** 2,
                0,
                1,
                epsabs=1e-15,
                epsrel=1e-12,
            )[0]
            for QR in (0.3, 1.7, 4.2):
                overlap = quad(
                    lambda r, x=x, order=order, QR=QR: (
                        r
                        * np.sqrt(3.0)
                        * spherical_jn(order, QR * r)
                        * r
                        * spherical_jn(order, x * r)
                    ),
                    0,
                    1,
                    epsabs=1e-14,  # reachable for small overlaps (l = 0 at QR = 0.3: 5.6e-4)
                    epsrel=1e-12,
                    limit=200,
                )[0]
                expected = (2 * order + 1) * overlap**2 / Z
                assert model.calculate_QISF_order(np.array([QR]), j)[0] == pytest.approx(
                    expected, rel=1e-10, abs=1e-15
                )

    @pytest.mark.parametrize(
        'angular_order, n, distance, expected',
        LIMIT_REFERENCES,
        ids=[f'l{order}_n{n}_{d:+.0e}' for order, n, d, _ in LIMIT_REFERENCES],
    )
    def test_limit_region_matches_mpmath(self, model, angular_order, n, distance, expected):
        # WHEN radius 1, so QR = Q
        j, x = order_of(model, angular_order, n)

        # THEN
        weight = model.calculate_QISF_order(np.array([x + distance]), j)[0]

        # EXPECT
        assert weight == pytest.approx(expected, rel=1e-9, abs=0)

    def test_weights_sum_to_one_up_to_the_missing_weight(self, model):
        # WHEN
        Q = np.array([0.1, 1.0, 3.0, 5.0])

        # THEN
        missing = model.calculate_missing_weight(Q)

        # EXPECT
        np.testing.assert_allclose(
            model.calculate_EISF(Q) + model.calculate_QISF(Q) + missing, 1.0, rtol=1e-14
        )
        assert np.all(missing >= 0)
        assert np.all(missing < 3e-4)

    def test_small_QR_is_finite_and_elastic(self, model):
        # WHEN
        Q = np.array([1e-6])

        # THEN
        weights = [model.calculate_QISF_order(Q, j)[0] for j in range(1, 99)]

        # EXPECT
        assert model.calculate_EISF(Q)[0] == pytest.approx(1.0, abs=1e-10)
        assert all(np.isfinite(weights))
        assert all(w >= 0 for w in weights)
        assert sum(weights) < 1e-10

    def test_width(self):
        # WHEN radius 2 angstrom, D 0.1 angstrom^2/ps
        model = DiffusionInSphere(radius=2.0, diffusion_coefficient=0.1, number_of_lorentzians=98)
        hbar_meV_ps = scipp_hbar.to(unit='meV*ps').value

        # THEN
        widths = [model.calculate_width(np.array([1.0]), j)[0] for j in (1, 3, 98)]

        # EXPECT
        expected = [hbar_meV_ps * model.roots[j - 1][0] ** 2 * 0.1 / 4.0 for j in (1, 3, 98)]
        np.testing.assert_allclose(widths, expected, rtol=1e-12)

    def test_width_follows_diffusion_coefficient_unit(self):
        # WHEN
        model = DiffusionInSphere(radius=2.0, diffusion_coefficient=0.1)
        width_angstrom = model.calculate_width(np.array([1.0]), 5)

        # THEN 0.1 angstrom^2/ps = 1e-9 m^2/s
        model.diffusion_coefficient.convert_unit('m**2/s')

        # EXPECT
        assert model.diffusion_coefficient.value == pytest.approx(1e-9, rel=1e-12)
        np.testing.assert_allclose(
            model.calculate_width(np.array([1.0]), 5), width_angstrom, rtol=1e-12
        )

    def test_weights_follow_radius_unit(self):
        # WHEN
        model = DiffusionInSphere(radius=3.0)
        Q = np.array([0.5, 1.5])
        weights_angstrom = model.calculate_QISF_order(Q, 4)

        # THEN
        model.radius.convert_unit('nm')

        # EXPECT the same physical radius gives the same weights
        np.testing.assert_allclose(model.calculate_QISF_order(Q, 4), weights_angstrom, rtol=1e-12)
        np.testing.assert_allclose(
            model.calculate_EISF(Q), (3 * spherical_jn(1, 3.0 * Q) / (3.0 * Q)) ** 2, rtol=1e-13
        )

    def test_init_default(self):
        # WHEN
        model = DiffusionInSphere()

        # THEN EXPECT a short sum by default, as dependent-Parameter updates are slow in Core 2.5.1
        assert model.display_name == 'DiffusionInSphere'
        assert model.number_of_lorentzians == 10
        assert model.radius.unit == 'Å'
        assert model.radius.fixed is False
        # Scipp has no named unit for angstrom^2/ps; it is stored as the equivalent 1e-8 m^2/s
        assert model.diffusion_coefficient.full_value.to(unit='angstrom**2/ps').value == (
            pytest.approx(1.0, rel=1e-12)
        )
        assert model.diffusion_coefficient.fixed is False


# The dependency strings write j_l in elementary functions, accurate to about 3e-13 absolute;
# the weight multiplies that by up to (2l+1) 6x^2 / (x^2 - l(l+1)) and divides by a^2 - x^2.
STRING_ATOL = 1e-10


class TestDiffusionInSphereComponents:
    def test_components_per_Q(self):
        # WHEN
        model = DiffusionInSphere(Q=np.array([0.5, 1.0]))

        # THEN
        collections = model.get_component_collections()

        # EXPECT
        assert len(collections) == 2
        for collection in collections:
            assert len(collection) == 11
            assert isinstance(collection[0], DeltaFunction)
            assert all(isinstance(component, Lorentzian) for component in collection[1:])
        assert collections[0][10].name == 'DiffusionInSphere_j10'

    @pytest.mark.parametrize('radius', [0.5, 2.0, 4.5])
    def test_dependent_weights_follow_radius(self, radius):
        # WHEN
        Q = np.array([0.4, 1.3])
        model = DiffusionInSphere(scale=2.0, radius=1.0, Q=Q)

        # THEN
        model.radius = radius

        # EXPECT
        collections = model.get_component_collections()
        EISF = model.calculate_EISF(Q)
        for i, collection in enumerate(collections):
            assert collection[0].area.value == pytest.approx(2.0 * EISF[i], rel=0, abs=STRING_ATOL)
            for j in range(1, model.number_of_lorentzians + 1):
                assert collection[j].area.value == pytest.approx(
                    2.0 * model.calculate_QISF_order(Q, j)[i], rel=0, abs=STRING_ATOL
                )

    @pytest.mark.parametrize('distance', [0.0, 1e-7, -1e-7, 1e-3, -1e-3])
    def test_dependent_weights_at_and_near_a_root(self, distance):
        """A fit step can move QR onto a root, where the weight is a 0/0."""
        # WHEN
        model = DiffusionInSphere(Q=np.array([1.0]))
        j, x = order_of(model, 1, 0)

        # THEN radius = x + distance, so QR = x + distance at Q = 1
        model.radius = x + distance

        # EXPECT
        dependent = model.get_component_collections()[0][j].area.value
        assert dependent == pytest.approx(
            model.calculate_QISF_order(np.array([1.0]), j)[0], rel=1e-8, abs=0
        )

    def test_component_widths(self):
        # WHEN
        Q = np.array([0.7])
        model = DiffusionInSphere(radius=2.0, diffusion_coefficient=0.1, Q=Q)

        # THEN
        model.diffusion_coefficient = 0.3
        model.radius = 2.5

        # EXPECT
        collection = model.get_component_collections()[0]
        for j in (1, 5, 10):
            assert collection[j].width.value == pytest.approx(
                model.calculate_width(Q, j)[0], rel=1e-12
            )
            assert collection[j].width.unit == 'meV'

    def test_component_parameters_are_dependent(self):
        # WHEN
        model = DiffusionInSphere(Q=np.array([1.0]))
        collection = model.get_component_collections()[0]

        # THEN
        fit_parameters = model.get_fit_parameters()

        # EXPECT
        for component in collection:
            assert component.area.independent is False
            assert all(component.area is not p for p in fit_parameters)
        for component in collection[1:]:
            assert component.width.independent is False

    def test_weights_follow_radius_unit(self):
        # WHEN
        Q = np.array([0.8])
        # QR = 2.4: 30 terms keep the truncation below the warning threshold
        model = DiffusionInSphere(radius=3.0, Q=Q, number_of_lorentzians=30)
        areas_angstrom = [c.area.value for c in model.get_component_collections()[0]]
        widths_angstrom = [c.width.value for c in model.get_component_collections()[0][1:]]

        # THEN
        model.radius.convert_unit('nm')

        # EXPECT
        areas_nm = [c.area.value for c in model.get_component_collections()[0]]
        widths_nm = [c.width.value for c in model.get_component_collections()[0][1:]]
        np.testing.assert_allclose(areas_nm, areas_angstrom, rtol=0, atol=STRING_ATOL)
        np.testing.assert_allclose(widths_nm, widths_angstrom, rtol=1e-12)

    def test_large_QR_warns_and_stays_finite(self):
        # WHEN THEN QR = 30, beyond the 98 roots below 26.84
        with pytest.warns(UserWarning, match='number_of_lorentzians'):
            model = DiffusionInSphere(radius=10.0, Q=np.array([3.0]))

        # EXPECT
        collection = model.get_component_collections()[0]
        assert all(np.isfinite(component.area.value) for component in collection)
        assert all(np.isfinite(component.width.value) for component in collection[1:])

    @pytest.mark.parametrize('number_of_lorentzians', [30, 98])
    def test_number_of_lorentzians_keeps_the_lowest_roots(self, number_of_lorentzians):
        # WHEN
        Q = np.array([0.6, 1.1])
        full = DiffusionInSphere(radius=3.0, number_of_lorentzians=98)

        # THEN
        model = DiffusionInSphere(radius=3.0, Q=Q, number_of_lorentzians=number_of_lorentzians)

        # EXPECT the first number_of_lorentzians terms of the full sum, unchanged
        assert model.number_of_lorentzians == number_of_lorentzians
        assert model.roots == full.roots[:number_of_lorentzians]
        kept = sum(full.calculate_QISF_order(Q, j) for j in range(1, number_of_lorentzians + 1))
        np.testing.assert_allclose(model.calculate_QISF(Q), kept, rtol=1e-13)
        # 1 minus nearly 1: compare absolutely, at the rounding of 1
        np.testing.assert_allclose(
            model.calculate_missing_weight(Q),
            1.0 - full.calculate_EISF(Q) - kept,
            rtol=0,
            atol=1e-15,
        )
        for i, collection in enumerate(model.get_component_collections()):
            assert len(collection) == number_of_lorentzians + 1
            for j in range(1, number_of_lorentzians + 1):
                assert collection[j].area.value == pytest.approx(
                    full.calculate_QISF_order(Q, j)[i], rel=0, abs=STRING_ATOL
                )
                assert collection[j].width.value == pytest.approx(
                    full.calculate_width(Q, j)[i], rel=1e-12
                )

    def test_short_sum_warns_when_truncation_is_large(self):
        # WHEN THEN EXPECT QR = 3: the first 10 terms miss 4.3e-3 of the weight
        with pytest.warns(UserWarning, match='number_of_lorentzians=10'):
            DiffusionInSphere(radius=3.0, Q=np.array([1.0]), number_of_lorentzians=10)

    def test_no_warning_when_truncation_is_negligible(self):
        # WHEN THEN EXPECT QR = 1: the default 10 terms miss 3.7e-4 of the weight
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            DiffusionInSphere(radius=1.0, Q=np.array([1.0]))

    def test_parameter_updates_raise_no_runtime_warning(self):
        """A fit changes R and D on every step; the dependent values must not warn."""
        # WHEN
        model = DiffusionInSphere(Q=np.array([0.5]))

        # THEN EXPECT
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            model.radius = 1.7
            model.diffusion_coefficient = 0.4

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            ({'radius': 'invalid'}, TypeError, 'radius must be a number'),
            ({'radius': 0.0}, ValueError, 'radius must be at least'),
            ({'diffusion_coefficient': 'x'}, TypeError, 'diffusion_coefficient must be a number'),
            ({'diffusion_coefficient': 0.0}, ValueError, 'diffusion_coefficient must be at least'),
            ({'delta_name': 1}, TypeError, 'delta_name must be a string'),
            ({'number_of_lorentzians': 2.5}, TypeError, 'number_of_lorentzians must be an'),
            ({'number_of_lorentzians': 0}, ValueError, 'number_of_lorentzians must be at least'),
            (
                {'number_of_lorentzians': 99},
                ValueError,
                'number_of_lorentzians must be at most 98',
            ),
        ],
    )
    def test_init_rejects_invalid_input(self, kwargs, expected_exception, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            DiffusionInSphere(**kwargs)

    def test_get_global_variables_and_repr(self):
        # WHEN
        model = DiffusionInSphere()

        # THEN EXPECT
        assert model.get_global_variables() == [
            model.scale,
            model.radius,
            model.diffusion_coefficient,
        ]
        assert 'diffusion_coefficient=' in repr(model)
        assert 'number_of_lorentzians=10' in repr(model)

    def test_fit_recovers_parameters(self):
        """One Q, synthetic data from the model; R and D are fitted, as in a real fit."""
        # WHEN
        Q = sc.array(dims=['Q'], values=[1.0], unit='1/angstrom')
        energy = sc.linspace('energy', -2.0, 2.0, 201, unit='meV')
        shape = (1, energy.shape[0])

        def as_data(values, variances):
            return sc.DataArray(
                sc.array(dims=['Q', 'energy'], values=values, variances=variances),
                coords={'Q': Q, 'energy': energy},
            )

        # QR = 3: 30 terms keep the truncation below the warning threshold
        true_model = DiffusionInSphere(
            radius=3.0, diffusion_coefficient=0.1, number_of_lorentzians=30
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
        fit_model = DiffusionInSphere(
            radius=2.5, diffusion_coefficient=0.07, number_of_lorentzians=30
        )
        analysis = edyn.Analysis(
            experiment=experiment, sample_model=edyn.SampleModel(diffusion_models=fit_model)
        )
        analysis.fix_energy_offset()

        # THEN
        analysis.fit(fit_method='independent')

        # EXPECT
        for name, true_value in {'radius': 3.0, 'diffusion_coefficient': 0.1}.items():
            parameter = getattr(fit_model, name)
            true_in_unit = sc.scalar(
                true_value, unit='angstrom' if name == 'radius' else 'angstrom**2/ps'
            )
            true_value = true_in_unit.to(unit=parameter.unit).value
            assert parameter.variance > 0
            assert abs(parameter.value - true_value) < 5 * parameter.error
