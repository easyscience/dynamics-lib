# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
import scipp as sc
from scipp.constants import hbar as scipp_hbar
from scipy.stats import poisson

from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.delta_lorentz import DeltaLorentz
from easydynamics.sample_model.diffusion_model.gaussian_model_3d import (
    MAXIMUM_NUMBER_OF_LORENTZIANS,
)
from easydynamics.sample_model.diffusion_model.gaussian_model_3d import (
    MINIMUM_DIFFUSION_COEFFICIENT,
)
from easydynamics.sample_model.diffusion_model.gaussian_model_3d import MINIMUM_MEAN_U_SQUARED
from easydynamics.sample_model.diffusion_model.gaussian_model_3d import GaussianModel3D

# Volino's example below in this model's units: D = 0.5 angstrom^2/ps = 5e-9 m^2/s, and
# <u^2> = 3 <u_x^2> = 4.5 angstrom^2.
VOLINO_D = 5e-9
VOLINO_MEAN_U_SQUARED = 4.5


def hbar_rate(rate_per_ps: float) -> float:
    """hbar * rate in meV, for a rate in 1/ps, computed with Scipp's constants."""
    return (scipp_hbar * sc.scalar(rate_per_ps, unit='1/ps')).to(unit='meV').value


def model_without_warning(**kwargs) -> GaussianModel3D:
    """Create a model and fail if it warns about truncation."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return GaussianModel3D(**kwargs)


class TestGaussianModel3D:
    @pytest.fixture
    def model(self):
        return GaussianModel3D()

    def test_create_weight_requires_float_Q(self, model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be a float'):
            model._create_weight(1, 1)

    # ------------------------------------------------------------------
    # Construction and parameters
    # ------------------------------------------------------------------

    def test_init_default(self, model):
        # WHEN THEN EXPECT
        assert model.display_name == 'GaussianModel3D'
        assert model.x_unit == 'meV'
        assert model.y_unit == 'dimensionless'
        assert model.scale.value == pytest.approx(1.0)
        assert model.number_of_lorentzians == 20
        assert model.diffusion_coefficient.value == pytest.approx(1.0)
        assert model.diffusion_coefficient.unit == 'm^2/s'
        assert model.diffusion_coefficient.min == MINIMUM_DIFFUSION_COEFFICIENT
        assert model.mean_u_squared.value == pytest.approx(1.0)
        assert model.mean_u_squared.unit == 'Å^2'
        assert model.mean_u_squared.min == MINIMUM_MEAN_U_SQUARED

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            ({'diffusion_coefficient': 'x'}, TypeError, 'diffusion_coefficient must be a number'),
            ({'diffusion_coefficient': 0.0}, ValueError, 'diffusion_coefficient must be at least'),
            ({'mean_u_squared': 'x'}, TypeError, 'mean_u_squared must be a number'),
            ({'mean_u_squared': 0.0}, ValueError, 'mean_u_squared must be at least'),
            ({'number_of_lorentzians': 2.0}, TypeError, 'number_of_lorentzians must be an'),
            ({'number_of_lorentzians': True}, TypeError, 'number_of_lorentzians must be an'),
            ({'number_of_lorentzians': 0}, ValueError, 'number_of_lorentzians must be between'),
            (
                {'number_of_lorentzians': MAXIMUM_NUMBER_OF_LORENTZIANS + 1},
                ValueError,
                'number_of_lorentzians must be between',
            ),
            ({'delta_name': 1}, TypeError, 'delta_name must be a string'),
        ],
        ids=[
            'D_type',
            'D_zero',
            'u2_type',
            'u2_zero',
            'n_float',
            'n_bool',
            'n_zero',
            'n_too_many',
            'delta_name_type',
        ],
    )
    def test_init_rejects_invalid_input(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            GaussianModel3D(**kwargs)

    def test_number_of_lorentzians_is_read_only(self, model):
        with pytest.raises(AttributeError):
            model.number_of_lorentzians = 5

    def test_setters(self, model):
        # WHEN
        model.diffusion_coefficient = 2e-9
        model.mean_u_squared = 3.0

        # THEN EXPECT
        assert model.diffusion_coefficient.value == pytest.approx(2e-9)
        assert model.mean_u_squared.value == pytest.approx(3.0)

    @pytest.mark.parametrize(
        'name, value, expected_exception',
        [
            ('diffusion_coefficient', 'x', TypeError),
            ('diffusion_coefficient', 0.0, ValueError),
            ('mean_u_squared', 'x', TypeError),
            ('mean_u_squared', 0.0, ValueError),
        ],
        ids=['D_type', 'D_zero', 'u2_type', 'u2_zero'],
    )
    def test_setters_raise(self, model, name, value, expected_exception):
        with pytest.raises(expected_exception, match=f'{name} must be'):
            setattr(model, name, value)

    # ------------------------------------------------------------------
    # Physics of the structure factors and widths
    # ------------------------------------------------------------------

    def test_weights_are_poisson(self):
        """A_i(Q) = exp(-y) y^i / i! with y = Q^2 <u^2> / 3."""
        # WHEN
        model = GaussianModel3D(mean_u_squared=2.4, number_of_lorentzians=30)
        Q = np.array([0.3, 1.0, 2.2])
        y = Q**2 * 2.4 / 3

        # THEN EXPECT
        np.testing.assert_allclose(model.calculate_EISF(Q), poisson.pmf(0, y), rtol=1e-12)
        for order in (1, 2, 7, 30):
            np.testing.assert_allclose(
                model.calculate_QISF_order(Q, order), poisson.pmf(order, y), rtol=1e-12
            )

    def test_EISF_equals_DeltaLorentz_debye_waller_factor(self):
        """The same mean_u_squared gives the same exp(-<u^2> Q^2 / 3) as DeltaLorentz."""
        # WHEN
        Q = np.array([0.5, 1.0, 1.5])
        model = GaussianModel3D(mean_u_squared=1.7)
        delta_lorentz = DeltaLorentz(mean_u_squared=1.7, A_0=1.0)

        # THEN EXPECT
        np.testing.assert_allclose(
            model.calculate_EISF(Q), delta_lorentz.calculate_EISF(Q), rtol=1e-12
        )

    def test_weights_and_missing_weight_sum_to_one(self):
        # WHEN
        model = GaussianModel3D(mean_u_squared=3.0, number_of_lorentzians=5)
        Q = np.array([0.5, 1.5, 3.0])

        # THEN
        total = (
            model.calculate_EISF(Q) + model.calculate_QISF(Q) + model.calculate_missing_weight(Q)
        )

        # EXPECT
        np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            model.calculate_missing_weight(Q), poisson.sf(5, Q**2), rtol=1e-9
        )

    def test_width_is_order_times_3_hbar_D_over_u2(self):
        # WHEN
        model = GaussianModel3D(diffusion_coefficient=2e-9, mean_u_squared=1.2)
        Q = np.array([0.5, 2.0])

        # THEN EXPECT Gamma_i = 3 i hbar D / <u^2>, with D / <u^2> = 2e-9 m^2/s / 1.2 angstrom^2
        rate_per_ps = 3 * 2e-9 * 1e20 / 1e12 / 1.2
        for order in (1, 4, 20):
            np.testing.assert_allclose(
                model.calculate_width(Q, order), order * hbar_rate(rate_per_ps), rtol=1e-12
            )

    def test_matches_QENSmodels_reference_values(self):
        """
        Doctest values of QENSmodels hwhmGaussianModel3D([1., 2.], 0.5, 1.5), with D in
        angstrom^2/ps and <u_x^2> in angstrom^2, rounded there to the digits given.
        """
        # WHEN
        model = GaussianModel3D(
            diffusion_coefficient=VOLINO_D,
            mean_u_squared=VOLINO_MEAN_U_SQUARED,
            number_of_lorentzians=99,
        )
        Q = np.array([1.0, 2.0])

        # THEN EXPECT
        np.testing.assert_allclose(model.calculate_EISF(Q), [0.223, 0.002], atol=5e-4)
        np.testing.assert_allclose(model.calculate_QISF_order(Q, 1), [0.3347, 0.0149], atol=5e-5)
        width_10 = model.calculate_width(Q, 10) / hbar_rate(1.0)
        width_99 = model.calculate_width(Q, 99) / hbar_rate(1.0)
        np.testing.assert_allclose(width_10, 3.333, atol=5e-4)
        np.testing.assert_allclose(width_99, 33.0, atol=5e-2)

    def test_Q_zero_is_purely_elastic(self):
        # WHEN
        model = GaussianModel3D()

        # THEN
        Q = np.array([0.0])

        # EXPECT exactly: exp(0) = 1 and exp(-inf) = 0, with no NaN from 0 * log(0)
        np.testing.assert_array_equal(model.calculate_EISF(Q), [1.0])
        np.testing.assert_array_equal(model.calculate_QISF(Q), [0.0])
        np.testing.assert_array_equal(model.calculate_QISF_order(Q, 1), [0.0])

    def test_large_Q_gives_no_nan(self):
        """y = Q^2 <u^2> / 3 up to about 1000: the weights underflow to 0, never NaN."""
        # WHEN
        model = GaussianModel3D(mean_u_squared=30.0, number_of_lorentzians=100)
        Q = np.array([5.0, 10.0])

        # THEN
        weights = [model.calculate_QISF_order(Q, order) for order in (1, 50, 100)]

        # EXPECT
        assert np.all(np.isfinite(weights))
        assert np.all(np.isfinite(model.calculate_EISF(Q)))

    def test_weights_follow_mean_u_squared_unit(self):
        # WHEN
        model = GaussianModel3D(mean_u_squared=1.5)
        Q = np.array([0.8, 1.6])
        EISF_angstrom = model.calculate_EISF(Q)
        width_angstrom = model.calculate_width(Q, 1)

        # THEN
        model.mean_u_squared.convert_unit('nm**2')

        # EXPECT
        np.testing.assert_allclose(model.calculate_EISF(Q), EISF_angstrom, rtol=1e-12)
        np.testing.assert_allclose(model.calculate_width(Q, 1), width_angstrom, rtol=1e-12)

    @pytest.mark.parametrize(
        'order', [0, 21, 1.0, True], ids=['zero', 'n_plus_1', 'float', 'bool']
    )
    def test_order_validation(self, model, order):
        with pytest.raises((TypeError, ValueError), match='order must be'):
            model.calculate_QISF_order(np.array([1.0]), order)
        with pytest.raises((TypeError, ValueError), match='order must be'):
            model.calculate_width(np.array([1.0]), order)

    def test_calculate_EISF_type_error(self, model):
        with pytest.raises(TypeError, match='Q must be '):
            model.calculate_EISF(Q='invalid')

    # ------------------------------------------------------------------
    # Truncation warning
    # ------------------------------------------------------------------

    def test_warns_when_truncation_drops_weight(self):
        # WHEN THEN EXPECT y = 3 at Q = 3 needs more than 3 Lorentzians
        with pytest.warns(UserWarning, match='number_of_lorentzians'):
            GaussianModel3D(mean_u_squared=1.0, number_of_lorentzians=3, Q=np.array([3.0]))

    def test_no_warning_when_truncation_is_negligible(self):
        # WHEN THEN EXPECT y = 0.03 at Q = 0.3: 3 Lorentzians leave about 3e-8
        model_without_warning(mean_u_squared=1.0, number_of_lorentzians=3, Q=np.array([0.3]))

    def test_warns_again_when_Q_changes(self):
        """Guards the refactor: new Q values rebuild the components and check the truncation."""
        # WHEN
        model = model_without_warning(
            mean_u_squared=1.0, number_of_lorentzians=3, Q=np.array([0.3])
        )
        model.clear_Q(confirm=True)

        # THEN EXPECT y = 25/3 at Q = 5: 3 Lorentzians leave about 0.97 of the weight
        with pytest.warns(UserWarning, match='number_of_lorentzians'):
            model.Q = np.array([5.0])

    def test_component_names_follow_the_base_convention(self):
        # WHEN
        model = model_without_warning(number_of_lorentzians=3, Q=np.array([0.5]))

        # THEN EXPECT
        assert model.lorentzian_component_name(3) == 'GaussianModel3D_i3'

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def test_no_Q_gives_no_collections(self, model):
        # WHEN THEN EXPECT
        assert model.get_component_collections() == []

    def test_components_per_Q(self):
        # WHEN
        model = model_without_warning(number_of_lorentzians=4, Q=np.array([0.5, 1.0]))

        # THEN
        collections = model.get_component_collections()

        # EXPECT one delta function and n Lorentzians per Q
        assert len(collections) == 2
        for collection in collections:
            assert len(collection) == 5
            assert isinstance(collection[0], DeltaFunction)
            assert all(isinstance(component, Lorentzian) for component in collection[1:])
            assert [component.name for component in collection[1:]] == [
                model.lorentzian_component_name(order) for order in range(1, 5)
            ]

    def test_component_areas_and_widths_match_calculations(self):
        """The dependency strings must give what the Scipp calculations give."""
        # WHEN
        Q = np.array([0.4, 1.1, 2.3])
        model = model_without_warning(
            scale=2.5,
            diffusion_coefficient=3e-9,
            mean_u_squared=0.9,
            number_of_lorentzians=12,
            Q=Q,
        )

        # THEN
        collections = model.get_component_collections()

        # EXPECT
        EISF = model.calculate_EISF(Q)
        for i, collection in enumerate(collections):
            assert collection[0].area.value == pytest.approx(2.5 * EISF[i], rel=1e-12)
            for order in range(1, 13):
                lorentzian = collection[order]
                weight = model.calculate_QISF_order(Q, order)[i]
                width = model.calculate_width(Q, order)[i]
                assert lorentzian.area.value == pytest.approx(2.5 * weight, rel=1e-12, abs=0)
                assert lorentzian.width.value == pytest.approx(width, rel=1e-12)
                assert lorentzian.width.unit == 'meV'
                assert lorentzian.area.independent is False

    def test_components_follow_parameter_changes(self):
        # WHEN
        Q = np.array([0.6, 1.8])
        model = model_without_warning(number_of_lorentzians=15, mean_u_squared=0.5, Q=Q)

        # THEN
        model.mean_u_squared = 0.8
        model.diffusion_coefficient = 4e-9
        model.scale = 4.0

        # EXPECT
        collections = model.get_component_collections()
        EISF = model.calculate_EISF(Q)
        for i, collection in enumerate(collections):
            assert collection[0].area.value == pytest.approx(4.0 * EISF[i], rel=1e-12)
            assert collection[3].area.value == pytest.approx(
                4.0 * model.calculate_QISF_order(Q, 3)[i], rel=1e-12
            )
            assert collection[3].width.value == pytest.approx(
                model.calculate_width(Q, 3)[i], rel=1e-12
            )

    def test_parameter_updates_raise_no_runtime_warning(self):
        """A fit changes D and <u^2> on every step; the dependent widths must not warn."""
        # WHEN
        model = model_without_warning(number_of_lorentzians=3, Q=np.array([0.5, 1.0]))

        # THEN EXPECT
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            model.diffusion_coefficient = 3e-9
            model.mean_u_squared = 0.7

    def test_component_weights_follow_mean_u_squared_unit(self):
        # WHEN
        Q = np.array([0.8, 1.6])
        model = model_without_warning(mean_u_squared=1.5, Q=Q)
        areas_angstrom = [c[2].area.value for c in model.get_component_collections()]
        widths_angstrom = [c[2].width.value for c in model.get_component_collections()]

        # THEN
        model.mean_u_squared.convert_unit('nm**2')

        # EXPECT
        collections = model.get_component_collections()
        np.testing.assert_allclose(
            [c[2].area.value for c in collections], areas_angstrom, rtol=1e-12
        )
        np.testing.assert_allclose(
            [c[2].width.value for c in collections], widths_angstrom, rtol=1e-12
        )

    def test_create_component_collections_installs_collections(self):
        # WHEN
        model = model_without_warning(Q=np.array([1.0, 2.0]))

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
        model = model_without_warning(scale=2.0, number_of_lorentzians=6, Q=Q)

        # THEN
        targets = {target.name: target for target in model.get_fit_targets()}

        # EXPECT
        expected_names = {'elastic_area'} | {
            f'{kind}_{order}' for kind in ('area', 'width') for order in range(1, 7)
        }
        assert set(targets) == expected_names
        np.testing.assert_allclose(
            targets['elastic_area'].function(Q), 2.0 * model.calculate_EISF(Q), rtol=1e-12
        )
        np.testing.assert_allclose(
            targets['area_2'].function(Q), 2.0 * model.calculate_QISF_order(Q, 2), rtol=1e-12
        )
        np.testing.assert_allclose(
            targets['width_3'].function(Q), model.calculate_width(Q, 3), rtol=1e-12
        )

    def test_get_global_variables(self, model):
        # WHEN THEN EXPECT
        assert model.get_global_variables() == [
            model.scale,
            model.diffusion_coefficient,
            model.mean_u_squared,
        ]

    def test_repr(self, model):
        # WHEN THEN
        repr_str = repr(model)

        # EXPECT
        assert 'GaussianModel3D' in repr_str
        assert 'number_of_lorentzians=20' in repr_str
        assert 'diffusion_coefficient' in repr_str
        assert 'mean_u_squared' in repr_str
        assert 'x_unit=meV, y_unit=dimensionless' in repr_str
