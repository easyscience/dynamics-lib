# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model import DiffusionDampedMittagLeffler
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import MittagLefflerDiffusion
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import hbar

ALL_Q_VARIATION = {'A_0': True, 'relaxation_rate': True, 'alpha': True}


class TestMittagLefflerDiffusion:
    @pytest.fixture
    def model(self):
        return MittagLefflerDiffusion(
            scale=2.0,
            diffusion_coefficient=3.3e-11,
            A_0=0.05,
            relaxation_rate=0.02,
            alpha=0.85,
        )

    @pytest.fixture
    def model_with_Q(self):
        return MittagLefflerDiffusion(
            scale=2.0,
            diffusion_coefficient=3.3e-11,
            A_0=0.05,
            relaxation_rate=0.02,
            alpha=0.85,
            allow_Q_variation=ALL_Q_VARIATION,
            Q=np.linspace(0.8, 1.8, 5),
        )

    @pytest.fixture
    def model_with_Q_no_variation(self):
        return MittagLefflerDiffusion(
            scale=2.0,
            diffusion_coefficient=3.3e-11,
            A_0=0.05,
            relaxation_rate=0.02,
            alpha=0.85,
            Q=np.linspace(0.8, 1.8, 5),
        )

    #############
    # Construction
    #############

    def test_init_default(self):
        # WHEN THEN
        model = MittagLefflerDiffusion()

        # EXPECT
        assert model.scale.value == pytest.approx(1.0)
        assert model.diffusion_coefficient.value == pytest.approx(1.0)
        assert model.A_0.value == pytest.approx(0.0)
        assert model.A_1.value == pytest.approx(1.0)
        assert model.relaxation_rate.value == pytest.approx(1.0)
        assert model.alpha.value == pytest.approx(1.0)
        assert model.x_unit == 'meV'
        assert model.get_component_collections() == []

    def test_init_with_Q_builds_two_components_per_Q(self, model_with_Q):
        # WHEN THEN
        collections = model_with_Q.get_component_collections()

        # EXPECT one elastic Lorentzian and one Mittag-Leffler component at every Q
        assert len(collections) == 5
        for collection in collections:
            assert collection.list_component_names() == ['Elastic Lorentzian', 'Mittag-Leffler']
            assert isinstance(collection[0], Lorentzian)
            assert isinstance(collection[1], DiffusionDampedMittagLeffler)

    def test_parameter_units(self, model):
        # WHEN THEN EXPECT
        assert model.scale.unit == 'meV'
        assert model.diffusion_coefficient.unit == 'm^2/s'
        assert model.relaxation_rate.unit == 'meV'
        assert model.alpha.unit == 'dimensionless'

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            ({'diffusion_coefficient': 'invalid'}, TypeError, 'must be a number'),
            ({'diffusion_coefficient': -1.0}, ValueError, 'must be non-negative'),
            ({'A_0': 'invalid'}, TypeError, 'A_0 must be a number'),
            ({'A_0': 1.5}, ValueError, 'A_0 must be between 0 and 1'),
            ({'relaxation_rate': 'invalid'}, TypeError, 'must be a number'),
            ({'relaxation_rate': -1.0}, ValueError, 'relaxation_rate must be at least'),
            ({'alpha': 'invalid'}, TypeError, 'alpha must be a number'),
            ({'alpha': 1.5}, ValueError, 'greater than zero and at most one'),
            ({'alpha': 0.0}, ValueError, 'greater than zero and at most one'),
            ({'mittag_leffler_name': 123}, TypeError, 'mittag_leffler_name must be a string'),
            ({'allow_Q_variation': 'invalid'}, TypeError, 'must be a dict or None'),
            ({'allow_Q_variation': {'nope': True}}, ValueError, 'Unknown keys'),
        ],
    )
    def test_input_validation_raises(self, kwargs, expected_exception, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            MittagLefflerDiffusion(**kwargs)

    @pytest.mark.parametrize(
        'prop, valid_value',
        [
            ('diffusion_coefficient', 5e-11),
            ('A_0', 0.3),
            ('relaxation_rate', 0.05),
            ('alpha', 0.6),
        ],
    )
    def test_setters(self, model, prop, valid_value):
        # WHEN THEN
        setattr(model, prop, valid_value)

        # EXPECT
        assert getattr(model, prop).value == pytest.approx(valid_value)

        # WHEN: an invalid value — THEN EXPECT
        with pytest.raises(TypeError, match='must be a number'):
            setattr(model, prop, 'invalid')

    def test_A_1_setter_raises(self, model):
        # WHEN THEN EXPECT A_1 is derived from A_0
        with pytest.raises(AttributeError, match='derived from A_0'):
            model.A_1 = 0.5

    def test_A_1_follows_A_0(self, model):
        # WHEN THEN
        model.A_0 = 0.25

        # EXPECT
        assert model.A_1.value == pytest.approx(0.75)

    #############
    # Q-dependent quantities
    #############

    def test_calculate_width_is_hbar_D_Q_squared(self, model_with_Q):
        # WHEN THEN
        Q = model_with_Q.Q.values
        widths = model_with_Q.calculate_width()

        # EXPECT epsilon = hbar * D * Q**2, matching Eq. (39) of Hassani et al.
        factor = hbar * model_with_Q.diffusion_coefficient / angstrom**2
        factor.convert_unit('meV')
        np.testing.assert_allclose(widths, Q**2 * factor.value, rtol=1e-10)

    def test_calculate_EISF_and_QISF_sum_to_one(self, model_with_Q):
        # WHEN THEN
        eisf = model_with_Q.calculate_EISF()
        qisf = model_with_Q.calculate_QISF()

        # EXPECT
        np.testing.assert_allclose(eisf + qisf, np.ones(5), rtol=1e-12)

    def test_calculate_relaxation_rate_and_alpha_shared(self, model_with_Q_no_variation):
        # WHEN THEN EXPECT the shared values are returned at every Q
        np.testing.assert_allclose(
            model_with_Q_no_variation.calculate_relaxation_rate(), np.full(5, 0.02)
        )
        np.testing.assert_allclose(model_with_Q_no_variation.calculate_alpha(), np.full(5, 0.85))

    def test_per_Q_parameters_are_independent(self, model_with_Q):
        # WHEN a single Q gets its own alpha, relaxation rate and elastic fraction
        model_with_Q._alpha_list[2].value = 0.6
        model_with_Q._relaxation_rate_list[2].value = 0.05
        model_with_Q._A_0_list[2].value = 0.2

        # THEN
        alphas = model_with_Q.calculate_alpha()
        rates = model_with_Q.calculate_relaxation_rate()
        eisf = model_with_Q.calculate_EISF()

        # EXPECT only that Q changed
        np.testing.assert_allclose(alphas, [0.85, 0.85, 0.6, 0.85, 0.85])
        np.testing.assert_allclose(rates, [0.02, 0.02, 0.05, 0.02, 0.02])
        np.testing.assert_allclose(eisf, [0.05, 0.05, 0.2, 0.05, 0.05])

    def test_per_Q_parameters_back_the_components(self, model_with_Q):
        # WHEN
        model_with_Q._alpha_list[2].value = 0.6
        model_with_Q._relaxation_rate_list[2].value = 0.05
        model_with_Q._A_0_list[2].value = 0.2

        # THEN
        collection = model_with_Q.get_component_collections()[2]

        # EXPECT the component parameters are the very ones in the per-Q lists
        assert collection[1].alpha.value == pytest.approx(0.6)
        assert collection[1].width.value == pytest.approx(0.05)
        assert collection[0].area.value == pytest.approx(2.0 * 0.2)
        assert collection[1].scale.value == pytest.approx(2.0 * 0.8)

    def test_calculate_raises_when_Q_variation_enabled_but_Q_unset(self):
        # WHEN
        model = MittagLefflerDiffusion(allow_Q_variation=ALL_Q_VARIATION)

        # THEN EXPECT
        with pytest.raises(ValueError, match='Q must be provided'):
            model.calculate_relaxation_rate()
        with pytest.raises(ValueError, match='Q must be provided'):
            model.calculate_alpha()

    #############
    # Parameter wiring
    #############

    def test_damping_is_shared_by_both_components(self, model_with_Q):
        # WHEN THEN
        collections = model_with_Q.get_component_collections()
        widths = model_with_Q.calculate_width()

        # EXPECT the elastic Lorentzian's HWHM and the ML component's damping are both epsilon
        for i, collection in enumerate(collections):
            assert collection[0].width.value == pytest.approx(widths[i])
            assert collection[1].damping.value == pytest.approx(widths[i])

    def test_global_diffusion_coefficient_drives_every_Q(self, model_with_Q):
        # WHEN
        collections = model_with_Q.get_component_collections()
        before = [collection[1].damping.value for collection in collections]

        # THEN
        model_with_Q.diffusion_coefficient = 2 * model_with_Q.diffusion_coefficient.value

        # EXPECT
        after = [collection[1].damping.value for collection in collections]
        np.testing.assert_allclose(after, np.array(before) * 2, rtol=1e-10)

    def test_shared_alpha_and_rate_drive_every_Q(self, model_with_Q_no_variation):
        # WHEN
        collections = model_with_Q_no_variation.get_component_collections()

        # THEN
        model_with_Q_no_variation.alpha = 0.5
        model_with_Q_no_variation.relaxation_rate = 0.03

        # EXPECT
        for collection in collections:
            assert collection[1].alpha.value == pytest.approx(0.5)
            assert collection[1].width.value == pytest.approx(0.03)

    def test_scale_and_A_0_drive_the_component_amplitudes(self, model_with_Q_no_variation):
        # WHEN
        collections = model_with_Q_no_variation.get_component_collections()

        # THEN
        model_with_Q_no_variation.scale = 4.0
        model_with_Q_no_variation.A_0 = 0.25

        # EXPECT area = scale * EISF for the elastic line, scale * (1 - EISF) for the ML term
        for collection in collections:
            assert collection[0].area.value == pytest.approx(4.0 * 0.25)
            assert collection[1].scale.value == pytest.approx(4.0 * 0.75)

    def test_matches_equation_41_by_hand(self, model_with_Q_no_variation):
        # WHEN the collection is Eq. (41): EISF * Lorentzian(eps) + (1-EISF) * ML(eps)
        collection = model_with_Q_no_variation.get_component_collections()[0]
        epsilon = model_with_Q_no_variation.calculate_width()[0]
        x = np.linspace(-0.3, 0.3, 201)

        # THEN
        result = collection.evaluate(x)

        # EXPECT
        elastic = Lorentzian(area=2.0 * 0.05, width=epsilon)
        quasi_elastic = DiffusionDampedMittagLeffler(
            scale=2.0 * 0.95, alpha=0.85, width=0.02, damping=epsilon
        )
        expected = elastic.evaluate(x) + quasi_elastic.evaluate(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    #############
    # Variables and fit targets
    #############

    def test_get_global_variables_with_Q_variation(self, model_with_Q):
        # WHEN THEN
        names = [variable.name for variable in model_with_Q.get_global_variables()]

        # EXPECT only scale and D are global once everything else varies with Q
        assert names == ['scale', 'diffusion_coefficient']

    def test_get_global_variables_without_Q_variation(self, model_with_Q_no_variation):
        # WHEN THEN
        names = [variable.name for variable in model_with_Q_no_variation.get_global_variables()]

        # EXPECT
        assert names == [
            'scale',
            'diffusion_coefficient',
            'A_0',
            'A_1',
            'relaxation_rate',
            'alpha',
        ]

    def test_get_independent_variables_holds_the_per_Q_amplitudes(self, model_with_Q):
        # WHEN THEN
        for_all_Q = model_with_Q.get_independent_variables()
        for_one_Q = model_with_Q.get_independent_variables(Q_index=1)

        # EXPECT one A_0/A_1 pair per Q
        assert len(for_all_Q) == 10
        assert len(for_one_Q) == 2
        assert for_one_Q[0] is model_with_Q._A_0_list[1]

    def test_free_parameters_match_the_papers_fit(self, model_with_Q):
        # WHEN the paper fits tau, alpha and the EISF per Q against a single global D
        free = model_with_Q.get_free_parameters()

        # THEN
        names = [parameter.name for parameter in free]

        # EXPECT scale + D + 5 * (A_0, alpha, width)
        assert names.count('scale') == 1
        assert names.count('diffusion_coefficient') == 1
        assert names.count('MittagLefflerDiffusion A_0') == 5
        assert names.count('Mittag-Leffler alpha') == 5
        assert names.count('Mittag-Leffler width') == 5
        assert all(isinstance(parameter, Parameter) for parameter in free)

    def test_get_fit_targets(self, model_with_Q):
        # WHEN THEN
        targets = {target.name: target for target in model_with_Q.get_fit_targets()}

        # EXPECT the quasi-elastic weight points at the Mittag-Leffler component, and the
        # Lorentzian keys at the elastic line
        assert set(targets) == {'area', 'width', 'elastic_area'}
        assert targets['area'].dataset_key == 'Mittag-Leffler scale'
        assert targets['width'].dataset_key == 'Elastic Lorentzian width'
        assert targets['elastic_area'].dataset_key == 'Elastic Lorentzian area'

        Q = model_with_Q.Q.values
        np.testing.assert_allclose(
            targets['area'].function(Q), model_with_Q.calculate_QISF(Q) * 2.0
        )
        np.testing.assert_allclose(
            targets['elastic_area'].function(Q), model_with_Q.calculate_EISF(Q) * 2.0
        )

    #############
    # Formulas from the paper
    #############

    def test_relaxation_rate_spectrum_is_normalised(self, model_with_Q_no_variation):
        # WHEN Eq. (37) is a distribution over relaxation rates, so it integrates to 1
        rate = np.logspace(-8, 4, 400_001)

        # THEN
        spectrum = model_with_Q_no_variation.calculate_relaxation_rate_spectrum(rate)

        # EXPECT
        assert spectrum.shape == (5, rate.size)
        assert np.trapezoid(spectrum[0], rate) == pytest.approx(1.0, rel=1e-3)

    def test_relaxation_rate_spectrum_alpha_one_is_peaked_at_the_rate(self):
        # WHEN alpha=1 the relaxation is exponential, so p(lambda) collapses onto a delta at 1/tau
        model = MittagLefflerDiffusion(alpha=1.0, relaxation_rate=0.02, Q=np.array([1.0]))
        rate = np.linspace(0.001, 0.1, 2001)

        # THEN
        spectrum = model.calculate_relaxation_rate_spectrum(rate)

        # EXPECT
        assert rate[np.argmax(spectrum[0])] == pytest.approx(0.02, abs=1e-3)

    def test_relaxation_rate_spectrum_rejects_non_positive_rate(self, model_with_Q):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='rate must be strictly positive'):
            model_with_Q.calculate_relaxation_rate_spectrum(np.array([0.0, 1.0]))

    @pytest.mark.parametrize('alpha', [0.99, 0.7, 0.3])
    def test_energy_barrier_distribution_carries_half_the_rate_spectrum(self, alpha):
        # WHEN h >= 0 maps onto lambda * tau_R <= 1 only, which by the lambda -> 1/lambda symmetry
        # of Eq. (37) is half of p(lambda)
        model = MittagLefflerDiffusion(alpha=alpha, Q=np.array([1.0]))
        h = np.linspace(0.0, 30.0, 60_001)

        # THEN
        distribution = model.calculate_energy_barrier_distribution(h)[0]

        # EXPECT
        assert np.all(np.isfinite(distribution))
        assert np.trapezoid(distribution, h) == pytest.approx(0.5, rel=1e-3)

    def test_energy_barrier_distribution_narrows_as_alpha_goes_to_one(self):
        # WHEN Eq. (48) tends to delta(h) as alpha -> 1 and broadens as alpha -> 0
        h = np.linspace(0.0, 10.0, 2001)
        smooth = MittagLefflerDiffusion(alpha=0.99, Q=np.array([1.0]))
        rough = MittagLefflerDiffusion(alpha=0.3, Q=np.array([1.0]))

        # THEN
        smooth_distribution = smooth.calculate_energy_barrier_distribution(h)[0]
        rough_distribution = rough.calculate_energy_barrier_distribution(h)[0]

        # EXPECT the rough landscape peaks at, and reaches, much higher barriers
        assert h[np.argmax(smooth_distribution)] < h[np.argmax(rough_distribution)]
        assert smooth_distribution[-1] < rough_distribution[-1]

    def test_energy_barrier_distribution_vanishes_at_zero_barrier(self, model_with_Q):
        # WHEN THEN
        distribution = model_with_Q.calculate_energy_barrier_distribution(np.array([0.0]))

        # EXPECT
        np.testing.assert_allclose(distribution, np.zeros((5, 1)), atol=1e-15)

    #############
    # Units and Q handling
    #############

    def test_convert_x_unit(self, model_with_Q):
        # WHEN
        widths_before = model_with_Q.calculate_width()

        # THEN
        model_with_Q.convert_x_unit('microeV')

        # EXPECT the shared template, the per-Q rates and the components all follow
        assert model_with_Q.x_unit == 'microeV'
        assert sc.Unit(model_with_Q.relaxation_rate.unit) == sc.Unit('microeV')
        np.testing.assert_allclose(model_with_Q.calculate_width(), widths_before * 1e3, rtol=1e-10)
        for collection in model_with_Q.get_component_collections():
            assert collection[1].width.value == pytest.approx(20.0)
            assert collection[1].alpha.value == pytest.approx(0.85)

    def test_convert_y_unit_rescales_the_scale(self):
        # WHEN scale_unit = x_unit * y_unit = meV/s
        model = MittagLefflerDiffusion(scale=2.0, y_unit='1/s', Q=np.linspace(0.8, 1.8, 3))

        # THEN
        model.convert_y_unit('1/ms')

        # EXPECT the scale follows the y-unit, 1/s -> 1/ms being a factor 1e-3
        assert model.y_unit == '1/ms'
        assert model.scale.value == pytest.approx(2.0e-3)
        assert sc.Unit(model.scale.unit) == sc.Unit('meV/ms')

    def test_on_Q_change_rebuilds_the_per_Q_lists(self, model):
        # WHEN
        model._allow_Q_variation = dict(ALL_Q_VARIATION)
        model.Q = np.linspace(0.5, 1.5, 3)

        # THEN
        collections = model.get_component_collections()

        # EXPECT
        assert len(collections) == 3
        assert len(model._alpha_list) == 3
        assert len(model._relaxation_rate_list) == 3
        assert collections[1][1].alpha is model._alpha_list[1]

    def test_clear_Q_empties_the_model(self, model_with_Q):
        # WHEN THEN
        model_with_Q.clear_Q(confirm=True)

        # EXPECT
        assert model_with_Q.get_component_collections() == []
        assert model_with_Q._alpha_list == []
        assert model_with_Q._relaxation_rate_list == []
        assert model_with_Q._A_0_list == []

    def test_repr(self, model):
        # WHEN THEN
        repr_str = repr(model)

        # EXPECT
        assert 'MittagLefflerDiffusion' in repr_str
        assert 'diffusion_coefficient=' in repr_str
        assert 'relaxation_rate=' in repr_str
        assert 'alpha=' in repr_str
