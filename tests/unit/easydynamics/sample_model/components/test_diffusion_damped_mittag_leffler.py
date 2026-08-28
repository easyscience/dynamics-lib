# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import quad

from easydynamics.sample_model import DiffusionDampedMittagLeffler


def laplace_reference(x, scale, alpha, width, damping):
    r"""
    Reference lineshape built from the Laplace transform of the Mittag-Leffler function.

    Eqs. (32) and (40) of Hassani et al. give the spectrum as ``(1/pi) Re{phi_hat(damping +
    i|x|)}``, with ``phi_hat(s) = s**(alpha-1)/(s**alpha + width**alpha)``. Evaluating that with
    complex arithmetic is independent of the real-valued modulus/argument form of Eq. (42) that the
    component implements.
    """
    s = damping + 1j * np.abs(np.asarray(x, dtype=float))
    return scale * np.real(s ** (alpha - 1) / (s**alpha + width**alpha)) / np.pi


class TestDiffusionDampedMittagLeffler:
    @pytest.fixture
    def mittag_leffler(self):
        return DiffusionDampedMittagLeffler(
            name='TestMLName',
            display_name='TestML',
            scale=2.0,
            alpha=0.7,
            width=0.3,
            damping=0.05,
            x_unit='meV',
        )

    #############
    # Construction
    #############

    def test_init_no_inputs(self):
        # WHEN THEN
        ml = DiffusionDampedMittagLeffler()

        # EXPECT
        assert ml.display_name == 'DiffusionDampedMittagLeffler'
        assert ml.scale.value == pytest.approx(1.0)
        assert ml.alpha.value == pytest.approx(1.0)
        assert ml.width.value == pytest.approx(1.0)
        assert ml.damping.value == pytest.approx(1.0)
        assert ml.x_unit == 'meV'
        assert ml.y_unit == 'dimensionless'

    def test_initialization(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN THEN EXPECT
        assert mittag_leffler.display_name == 'TestML'
        assert mittag_leffler.scale.value == pytest.approx(2.0)
        assert mittag_leffler.alpha.value == pytest.approx(0.7)
        assert mittag_leffler.width.value == pytest.approx(0.3)
        assert mittag_leffler.damping.value == pytest.approx(0.05)
        assert mittag_leffler.x_unit == 'meV'

    def test_parameter_units(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN THEN EXPECT scale = x_unit * y_unit, width/damping = x_unit, alpha dimensionless
        assert mittag_leffler.scale.unit == 'meV'
        assert mittag_leffler.alpha.unit == 'dimensionless'
        assert mittag_leffler.width.unit == 'meV'
        assert mittag_leffler.damping.unit == 'meV'

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            ({'scale': 'invalid'}, 'scale must be a number'),
            ({'alpha': 'invalid'}, 'alpha must be a number'),
            ({'width': 'invalid'}, 'width must be a number'),
            ({'damping': 'invalid'}, 'damping must be a number'),
            ({'x_unit': 123}, 'unit must be None, a string'),
            ({'y_unit': 123}, 'unit must be None, a string'),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=expected_message):
            DiffusionDampedMittagLeffler(**kwargs)

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            ({'scale': -1.0}, 'scale must be non-negative'),
            ({'scale': np.nan}, 'scale must be a finite number'),
            ({'alpha': 0.0}, 'alpha must be greater than zero and at most one'),
            ({'alpha': 1.5}, 'alpha must be greater than zero and at most one'),
            ({'alpha': np.nan}, 'alpha must be a finite number'),
            ({'width': -0.6}, 'must be greater than zero'),
            ({'damping': -0.6}, 'must be greater than zero'),
        ],
    )
    def test_input_value_validation_raises(self, kwargs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=expected_message):
            DiffusionDampedMittagLeffler(**kwargs)

    #############
    # Property setters
    #############

    @pytest.mark.parametrize(
        'prop, valid_value',
        [
            ('scale', 3.0),
            ('alpha', 0.5),
            ('width', 0.7),
            ('damping', 0.2),
        ],
    )
    def test_property_setters(
        self,
        mittag_leffler: DiffusionDampedMittagLeffler,
        prop,
        valid_value,
    ):
        # WHEN THEN: set a valid value
        setattr(mittag_leffler, prop, valid_value)

        # EXPECT
        assert getattr(mittag_leffler, prop).value == valid_value

        # WHEN: set an invalid value — THEN EXPECT
        with pytest.raises(TypeError, match='must be a number'):
            setattr(mittag_leffler, prop, 'invalid')

    @pytest.mark.parametrize(
        'prop, invalid_value, expected_message',
        [
            ('scale', -1.0, 'violates the parameter bounds'),
            ('alpha', 1.5, 'violates the parameter bounds'),
            ('alpha', -0.1, 'violates the parameter bounds'),
            ('width', -0.5, 'width must be positive'),
            ('width', 1e-12, 'violates the parameter bounds'),
            ('damping', -0.5, 'damping must be positive'),
            ('damping', 1e-12, 'violates the parameter bounds'),
        ],
    )
    def test_setters_out_of_bounds_raise(
        self,
        mittag_leffler: DiffusionDampedMittagLeffler,
        prop,
        invalid_value,
        expected_message,
    ):
        # WHEN
        original = getattr(mittag_leffler, prop).value

        # THEN EXPECT the assignment raises instead of being silently clamped
        with pytest.raises(ValueError, match=expected_message):
            setattr(mittag_leffler, prop, invalid_value)
        assert getattr(mittag_leffler, prop).value == pytest.approx(original)

    def test_get_all_parameters(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN THEN
        params = mittag_leffler.get_all_parameters()

        # EXPECT
        assert len(params) == 4
        assert all(isinstance(param, Parameter) for param in params)
        expected_names = {
            'TestMLName scale',
            'TestMLName alpha',
            'TestMLName width',
            'TestMLName damping',
        }
        assert {param.name for param in params} == expected_names

    #############
    # Evaluation
    #############

    @pytest.mark.parametrize('alpha', [0.3, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize('damping', [0.01, 0.5])
    def test_evaluate_matches_laplace_reference(self, alpha, damping):
        # WHEN Eq. (42) is the real-valued form of (1/pi) Re{phi_hat(damping + i|omega|)}
        ml = DiffusionDampedMittagLeffler(scale=2.0, alpha=alpha, width=0.4, damping=damping)
        x = np.linspace(-5.0, 5.0, 401)

        # THEN
        result = ml.evaluate(x)

        # EXPECT
        expected = laplace_reference(x, scale=2.0, alpha=alpha, width=0.4, damping=damping)
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-14)

    def test_evaluate_reproduces_printed_equation_42(self):
        # WHEN width=1 reduces the model to Eq. (42) as printed (tau_R = 1), up to the 1/pi that
        # the printed equation omits but Eq. (41) requires
        alpha, epsilon = 0.6, 0.2
        ml = DiffusionDampedMittagLeffler(scale=1.0, alpha=alpha, width=1.0, damping=epsilon)
        omega = np.linspace(-4.0, 4.0, 201)

        # THEN
        result = ml.evaluate(omega)

        # EXPECT: Eq. (42) transcribed term by term
        abs_omega = np.abs(omega)
        square = abs_omega**2 + epsilon**2
        arg = alpha * np.angle(epsilon + 1j * abs_omega)
        numerator = (
            epsilon * square ** (alpha / 2) + abs_omega * np.sin(arg) + epsilon * np.cos(arg)
        )
        denominator = square * ((square**alpha + 1) * square ** (-alpha / 2) + 2 * np.cos(arg))
        np.testing.assert_allclose(result, numerator / denominator / np.pi, rtol=1e-10)

    def test_evaluate_alpha_one_is_lorentzian(self):
        # WHEN alpha=1 the Mittag-Leffler function reduces to a simple exponential, so the
        # spectrum is a Lorentzian of HWHM width+damping (Eq. 31 of the paper)
        ml = DiffusionDampedMittagLeffler(scale=2.0, alpha=1.0, width=0.3, damping=0.1)
        x = np.linspace(-4.0, 4.0, 301)

        # THEN
        result = ml.evaluate(x)

        # EXPECT
        hwhm = 0.3 + 0.1
        expected = 2.0 * hwhm / np.pi / (x**2 + hwhm**2)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_evaluate_is_symmetric(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN
        x = np.linspace(0.0, 3.0, 101)

        # THEN
        positive = mittag_leffler.evaluate(x)
        negative = mittag_leffler.evaluate(-x)

        # EXPECT
        np.testing.assert_allclose(positive, negative, rtol=1e-12)

    def test_evaluate_finite_and_maximal_at_zero(
        self, mittag_leffler: DiffusionDampedMittagLeffler
    ):
        # WHEN a strictly positive damping keeps the otherwise singular spectrum regular at x=0
        x = np.linspace(-1.0, 1.0, 201)

        # THEN
        values = mittag_leffler.evaluate(x)

        # EXPECT
        assert np.all(np.isfinite(values))
        assert np.argmax(values) == 100
        assert mittag_leffler.evaluate(0.0)[0] == pytest.approx(np.max(values))

    @pytest.mark.parametrize('alpha', [0.4, 0.7, 1.0])
    def test_scale_is_the_integrated_area(self, alpha):
        # WHEN THEN
        ml = DiffusionDampedMittagLeffler(scale=2.5, alpha=alpha, width=0.3, damping=0.05)
        integral, _ = quad(lambda x: ml.evaluate(x)[0], -np.inf, np.inf, limit=400)

        # EXPECT
        assert integral == pytest.approx(2.5, rel=1e-4)

    def test_smaller_alpha_broadens_the_wings(self):
        # WHEN two profiles differ only in alpha, at fixed scale
        narrow = DiffusionDampedMittagLeffler(alpha=1.0, width=0.3, damping=0.05)
        broad = DiffusionDampedMittagLeffler(alpha=0.5, width=0.3, damping=0.05)

        # THEN
        far_out = np.array([20.0])

        # EXPECT the sub-exponential relaxation leaves more intensity in the wings
        assert broad.evaluate(far_out)[0] > narrow.evaluate(far_out)[0]

    def test_evaluate_scipp_input_converts_units(self):
        # WHEN
        ml = DiffusionDampedMittagLeffler(scale=1.0, alpha=0.7, width=0.3, damping=0.05)
        x_mev = np.linspace(-2.0, 2.0, 51)
        x_microev = sc.array(dims=['x'], values=x_mev * 1e3, unit='microeV')

        # THEN
        from_mev = ml.evaluate(x_mev)
        from_microev = ml.evaluate(x_microev)

        # EXPECT the same dimensionless intensities: scale, width and damping are all resolved to
        # the unit of x, so the shape only depends on the ratios between them
        np.testing.assert_allclose(from_microev, from_mev, rtol=1e-10)

    def test_evaluate_scipp_output(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN
        x = np.linspace(-2.0, 2.0, 50)

        # THEN
        result = mittag_leffler.evaluate(x, output='scipp')

        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        np.testing.assert_allclose(result.values, mittag_leffler.evaluate(x, output='numpy'))

    #############
    # Unit conversion
    #############

    def test_convert_x_unit(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN THEN
        mittag_leffler.convert_x_unit('microeV')

        # EXPECT
        assert mittag_leffler.x_unit == 'microeV'
        assert mittag_leffler.scale.value == pytest.approx(2.0 * 1e3)
        assert mittag_leffler.width.value == pytest.approx(0.3 * 1e3)
        assert mittag_leffler.damping.value == pytest.approx(0.05 * 1e3)
        # EXPECT the dimensionless form parameter is untouched
        assert mittag_leffler.alpha.value == pytest.approx(0.7)
        assert mittag_leffler.alpha.unit == 'dimensionless'

    def test_convert_x_unit_invalid_type_raises(
        self, mittag_leffler: DiffusionDampedMittagLeffler
    ):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'x_unit must be a string or sc\.Unit'):
            mittag_leffler.convert_x_unit(123)

    def test_convert_x_unit_rollback_on_failure(
        self, mittag_leffler: DiffusionDampedMittagLeffler
    ):
        # WHEN THEN
        with pytest.raises(UnitError):
            mittag_leffler.convert_x_unit('m')

        # EXPECT: state rolled back
        assert mittag_leffler.x_unit == 'meV'
        assert mittag_leffler.scale.value == pytest.approx(2.0)
        assert mittag_leffler.width.value == pytest.approx(0.3)
        assert mittag_leffler.damping.value == pytest.approx(0.05)

    def test_convert_y_unit(self):
        # WHEN: x_unit='meV', y_unit='1/meV' → scale_unit='dimensionless'
        ml = DiffusionDampedMittagLeffler(scale=1.0, width=0.3, damping=0.05, y_unit='1/meV')

        # THEN
        ml.convert_y_unit('1/eV')

        # EXPECT
        assert ml.y_unit == '1/eV'
        assert ml.scale.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(
        self, mittag_leffler: DiffusionDampedMittagLeffler
    ):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            mittag_leffler.convert_y_unit(123)

    def test_convert_y_unit_rollback_on_failure(
        self, mittag_leffler: DiffusionDampedMittagLeffler
    ):
        # WHEN THEN
        with pytest.raises(UnitError):
            mittag_leffler.convert_y_unit('K')

        # EXPECT: state rolled back
        assert mittag_leffler.y_unit == 'dimensionless'
        assert mittag_leffler.scale.value == pytest.approx(2.0)

    def test_y_unit_setter_raises(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            mittag_leffler.y_unit = '1/meV'

    #############
    # Copy and repr
    #############

    def test_copy(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN THEN
        ml_copy = copy(mittag_leffler)

        # EXPECT
        assert ml_copy is not mittag_leffler
        assert ml_copy.display_name == mittag_leffler.display_name
        for prop in ('scale', 'alpha', 'width', 'damping'):
            assert getattr(ml_copy, prop).value == getattr(mittag_leffler, prop).value
            assert getattr(ml_copy, prop).fixed == getattr(mittag_leffler, prop).fixed
        assert ml_copy.x_unit == mittag_leffler.x_unit

    def test_repr(self, mittag_leffler: DiffusionDampedMittagLeffler):
        # WHEN THEN
        repr_str = repr(mittag_leffler)

        # EXPECT
        assert 'DiffusionDampedMittagLeffler' in repr_str
        assert 'name = TestMLName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'scale =' in repr_str
        assert 'alpha =' in repr_str
        assert 'width =' in repr_str
        assert 'damping =' in repr_str
