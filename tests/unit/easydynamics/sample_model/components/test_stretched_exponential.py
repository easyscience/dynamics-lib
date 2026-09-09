# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy
from itertools import pairwise

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.integrate import simpson
from scipy.special import gamma

from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import StretchedExponential
from easydynamics.sample_model.components.stretched_exponential import _kww_shape

# hbar in meV*ps (CODATA), so the tests derive the energy scale independently of the library.
HBAR_MEV_PS = 0.6582119569509066


def sinh_grid(reach: float, n_points: int = 20001) -> np.ndarray:
    """A symmetric grid that is dense near zero and reaches *reach*, for the heavy KWW tails."""
    t = np.linspace(-np.arcsinh(reach), np.arcsinh(reach), n_points)
    return np.sinh(t)


def naive_fft_transform(
    area: float, tau: float, beta: float, n_time: int, t_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform ``I(t) = area * exp(-(t / tau)**beta)`` to energy the obvious way, with an FFT.

    This is the textbook route the implementation deliberately does not take: sample the
    relaxation on a uniform time grid and let ``np.fft`` do the cosine transform.  It is an
    independent check because it shares no code and no idea with the rotated-contour quadrature
    in ``_kww_shape`` -- only the physics.

    The profile is even in *t*, so the two-sided transform is twice the one-sided one and

        S(x) = (1 / (pi hbar)) Re [ integral of I(t) exp(-i x t / hbar) dt over t >= 0 ],

    which on the uniform grid is a trapezoid sum (hence the half-weighted t = 0 sample) evaluated
    at the FFT frequencies ``x_n = 2 pi hbar n / (n_time dt)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The energy axis in meV and the spectrum on it.
    """
    dt = t_max / n_time
    time = np.arange(n_time) * dt
    intensity = area * np.exp(-((time / tau) ** beta))
    spectrum = (np.real(np.fft.rfft(intensity)) - 0.5 * intensity[0]) * dt / (np.pi * HBAR_MEV_PS)
    energy = 2.0 * np.pi * HBAR_MEV_PS * np.arange(spectrum.size) / (n_time * dt)
    return energy, spectrum


def fft_comparison(beta: float, n_time: int, tau: float = 5.0, area: float = 1.0) -> float:
    """Largest relative gap between the component and the FFT, over the resolvable region."""
    # Reach far enough in time that exp(-(t / tau)**beta) has fallen by exp(-40).
    energy, reference = naive_fft_transform(area, tau, beta, n_time, tau * 40.0 ** (1.0 / beta))
    inside = (energy > 0.0) & (energy < 2.0)
    energy, reference = energy[inside], reference[inside]

    stretched = StretchedExponential(area=area, relaxation_time=tau, beta=beta)
    values = stretched.evaluate(energy)

    # Below a thousandth of the peak the FFT reference is dominated by its own truncation error,
    # so comparing there would measure the reference rather than the implementation.
    resolvable = values > 1e-3 * stretched.evaluate(np.array([0.0]))[0]
    return float(np.max(np.abs(values[resolvable] - reference[resolvable]) / values[resolvable]))


#####################################
# The Fourier transform: _kww_shape
#####################################


@pytest.mark.parametrize(
    'beta, rel',
    # The smallest supported beta spreads exp(-u**beta) over so many decades that the grid only
    # just reaches its tail, which costs a few digits; everything above it is at machine precision.
    [(0.05, 1e-8), (0.1, 1e-11), (0.3, 1e-12), (0.5, 1e-12), (1.0, 1e-12), (2.0, 1e-12)],
)
def test_kww_shape_at_zero(beta, rel):
    # WHEN G(0) is the integral of exp(-u**beta), which is gamma(1 + 1/beta)

    # THEN
    value = _kww_shape(np.array([0.0]), beta)

    # EXPECT
    assert value[0] == pytest.approx(gamma(1.0 + 1.0 / beta), rel=rel)


def test_kww_shape_matches_lorentzian_at_beta_one():
    # WHEN beta = 1 the transform of exp(-u) is known in closed form
    w = sinh_grid(1e6)

    # THEN
    value = _kww_shape(w, 1.0)

    # EXPECT
    np.testing.assert_allclose(value, 1.0 / (1.0 + w**2), rtol=1e-9)


def test_kww_shape_matches_gaussian_at_beta_two():
    # WHEN beta = 2 the transform of exp(-u**2) is again a Gaussian
    w = np.linspace(-12.0, 12.0, 501)

    # THEN
    value = _kww_shape(w, 2.0)

    # EXPECT
    expected = 0.5 * np.sqrt(np.pi) * np.exp(-(w**2) / 4)
    np.testing.assert_allclose(value, expected, atol=1e-14)


@pytest.mark.parametrize('beta', [0.3, 0.6, 0.9])
def test_kww_shape_matches_the_large_w_asymptote(beta):
    # WHEN the leading term of the large-w expansion is
    # gamma(beta + 1) sin(pi beta / 2) / w**(beta + 1).  The next term is smaller by w**-beta, so
    # w has to be taken far out before the leading term alone is worth a few digits.
    w = np.array([1e9, 1e11])

    # THEN
    value = _kww_shape(w, beta)

    # EXPECT
    expected = gamma(beta + 1) * np.sin(np.pi * beta / 2) / w ** (beta + 1)
    np.testing.assert_allclose(value, expected, rtol=3e-3)


@pytest.mark.parametrize('beta', [0.3, 0.5, 0.8, 1.0, 1.6, 2.0])
def test_kww_shape_is_even_and_non_negative(beta):
    # WHEN
    w = sinh_grid(1e6, n_points=2001)

    # THEN
    value = _kww_shape(w, beta)

    # EXPECT
    np.testing.assert_allclose(value, _kww_shape(-w, beta), rtol=0, atol=0)
    assert np.all(value >= 0.0)


@pytest.mark.parametrize('beta', [0.3, 0.5, 0.8, 1.0, 1.6, 2.0])
def test_kww_shape_integrates_to_pi(beta):
    # WHEN the transform of a function that is 1 at t = 0 integrates to pi over w

    # THEN
    w = sinh_grid(1e8)
    integral = simpson(_kww_shape(w, beta), x=w)

    # EXPECT
    assert integral == pytest.approx(np.pi, rel=5e-3)


def test_kww_shape_is_blocked_without_changing_the_result():
    # WHEN a long energy axis is evaluated in blocks to bound the memory of the quadrature
    w = np.linspace(-40.0, 40.0, 997)
    reference = _kww_shape(w, 0.6)

    # THEN
    blocked = np.concatenate([_kww_shape(chunk, 0.6) for chunk in np.array_split(w, 7)])

    # EXPECT the block boundaries do not perturb any value
    np.testing.assert_array_equal(blocked, reference)


class TestStretchedExponential:
    @pytest.fixture
    def stretched_exponential(self):
        return StretchedExponential(
            name='StretchedName',
            display_name='TestStretched',
            area=2.0,
            center=0.5,
            relaxation_time=3.0,
            beta=0.7,
            x_unit='meV',
        )

    #############
    # Creation
    #############

    def test_init_no_inputs(self):
        # WHEN THEN
        stretched = StretchedExponential()

        # EXPECT
        assert stretched.display_name == 'StretchedExponential'
        assert stretched.area.value == pytest.approx(1.0)
        assert stretched.center.value == pytest.approx(0.0)
        assert stretched.relaxation_time.value == pytest.approx(1.0)
        assert stretched.beta.value == pytest.approx(1.0)
        assert stretched.x_unit == 'meV'
        assert stretched.y_unit == 'dimensionless'
        assert stretched.relaxation_time.unit == 'ps'
        assert stretched.beta.unit == 'dimensionless'
        assert stretched.center.fixed is True

    def test_initialization(self, stretched_exponential: StretchedExponential):
        # WHEN THEN EXPECT
        assert stretched_exponential.display_name == 'TestStretched'
        assert stretched_exponential.area.value == pytest.approx(2.0)
        assert stretched_exponential.center.value == pytest.approx(0.5)
        assert stretched_exponential.relaxation_time.value == pytest.approx(3.0)
        assert stretched_exponential.beta.value == pytest.approx(0.7)
        assert stretched_exponential.center.fixed is False

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            ({'area': 'invalid'}, 'area must be a number'),
            ({'center': 'invalid'}, 'center must be None or a number'),
            ({'relaxation_time': 'invalid'}, 'relaxation_time must be a number'),
            ({'beta': 'invalid'}, 'beta must be a number'),
            ({'x_unit': 123}, 'unit must be None, a string'),
            ({'y_unit': 123}, 'unit must be None, a string'),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=expected_message):
            StretchedExponential(**kwargs)

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            ({'relaxation_time': 0.0}, 'relaxation_time must be greater than zero'),
            ({'relaxation_time': -1.0}, 'relaxation_time must be greater than zero'),
            ({'relaxation_time': np.inf}, 'relaxation_time must be a finite number'),
            ({'beta': 0.0}, 'beta must be between'),
            ({'beta': 2.5}, 'beta must be between'),
            ({'beta': np.nan}, 'beta must be a finite number'),
        ],
    )
    def test_input_value_validation_raises(self, kwargs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match=expected_message):
            StretchedExponential(**kwargs)

    def test_negative_area_warns(self):
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match='may not be physically meaningful'):
            StretchedExponential(area=-2.0)

    def test_get_all_parameters(self, stretched_exponential: StretchedExponential):
        # WHEN THEN
        params = stretched_exponential.get_all_parameters()

        # EXPECT
        assert all(isinstance(param, Parameter) for param in params)
        assert {param.name for param in params} == {
            'StretchedName area',
            'StretchedName center',
            'StretchedName relaxation_time',
            'StretchedName beta',
        }

    def test_copy(self, stretched_exponential: StretchedExponential):
        # WHEN THEN
        stretched_copy = copy(stretched_exponential)

        # EXPECT
        assert stretched_copy is not stretched_exponential
        assert stretched_copy.display_name == stretched_exponential.display_name
        assert stretched_copy.area.value == stretched_exponential.area.value
        assert stretched_copy.center.value == stretched_exponential.center.value
        assert stretched_copy.relaxation_time.value == stretched_exponential.relaxation_time.value
        assert stretched_copy.beta.value == stretched_exponential.beta.value
        assert stretched_copy.x_unit == stretched_exponential.x_unit

    def test_repr(self, stretched_exponential: StretchedExponential):
        # WHEN THEN
        repr_str = repr(stretched_exponential)

        # EXPECT
        assert 'StretchedExponential' in repr_str
        assert 'name = StretchedName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'relaxation_time =' in repr_str
        assert 'beta =' in repr_str

    #############
    # Parameters
    #############

    @pytest.mark.parametrize(
        'prop, valid_value',
        [('area', 3.0), ('center', 0.6), ('relaxation_time', 4.0), ('beta', 0.9)],
    )
    def test_property_setters(
        self, stretched_exponential: StretchedExponential, prop, valid_value
    ):
        # WHEN: set a valid value
        setattr(stretched_exponential, prop, valid_value)
        # THEN EXPECT
        assert getattr(stretched_exponential, prop).value == valid_value

        # WHEN: set an invalid value — THEN EXPECT
        with pytest.raises(TypeError, match=' must be a number'):
            setattr(stretched_exponential, prop, 'invalid')

    def test_relaxation_time_must_be_positive(self, stretched_exponential: StretchedExponential):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='relaxation_time must be greater than zero'):
            stretched_exponential.relaxation_time = -1.0
        assert stretched_exponential.relaxation_time.value == pytest.approx(3.0)

    @pytest.mark.parametrize('value', [0.01, 2.5])
    def test_beta_outside_the_supported_range_raises(
        self, stretched_exponential: StretchedExponential, value
    ):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='beta must be between'):
            stretched_exponential.beta = value
        assert stretched_exponential.beta.value == pytest.approx(0.7)

    def test_area_setter_out_of_bounds_raises(self, stretched_exponential: StretchedExponential):
        # WHEN the fixture's area was created non-negative, so it carries min=0

        # THEN EXPECT a negative assignment raises instead of being silently clamped to 0
        with pytest.raises(ValueError, match='violates the parameter bounds'):
            stretched_exponential.area = -1.0
        assert stretched_exponential.area.value == pytest.approx(2.0)

    def test_center_is_fixed_if_set_to_None(self, stretched_exponential: StretchedExponential):
        # WHEN
        assert stretched_exponential.center.fixed is False

        # THEN
        stretched_exponential.center = None

        # EXPECT
        assert stretched_exponential.center.value == pytest.approx(0.0)
        assert stretched_exponential.center.fixed is True

    def test_width_is_the_energy_scale(self, stretched_exponential: StretchedExponential):
        # WHEN width is not stored but derived as Gamma = hbar / tau, in the component's x_unit
        expected = HBAR_MEV_PS / 3.0

        # THEN
        width = stretched_exponential.width

        # EXPECT
        assert width.value == pytest.approx(expected)
        assert str(width.unit) == 'meV'

    def test_width_tracks_the_relaxation_time(self, stretched_exponential: StretchedExponential):
        # WHEN
        before = stretched_exponential.width.value

        # THEN
        stretched_exponential.relaxation_time = 6.0

        # EXPECT doubling the relaxation time halves the width
        assert stretched_exponential.width.value == pytest.approx(before / 2.0)

    def test_width_is_read_only(self, stretched_exponential: StretchedExponential):
        # WHEN THEN EXPECT the relaxation time is the fittable parameter, not the width
        with pytest.raises(AttributeError):
            stretched_exponential.width = 0.5

    def test_width_follows_the_x_unit(self):
        # WHEN the component measures energy in microeV
        stretched = StretchedExponential(relaxation_time=5.0, beta=0.6, x_unit='ueV')

        # THEN
        width = stretched.width

        # EXPECT Gamma is expressed in that unit too
        assert width.value == pytest.approx(1e3 * HBAR_MEV_PS / 5.0)
        assert str(width.unit) == str(sc.Unit('ueV'))

    def test_width_with_a_non_energy_x_unit_raises(self):
        # WHEN THEN EXPECT hbar / relaxation_time cannot be expressed in metres
        with pytest.raises(UnitError, match='needs an energy x_unit'):
            _ = StretchedExponential(x_unit='m').width

    def test_width_is_the_lorentzian_hwhm_at_beta_one(self):
        # WHEN beta = 1 the transform is a Lorentzian, whose HWHM should be exactly width
        stretched = StretchedExponential(area=1.0, relaxation_time=4.0, beta=1.0)

        # THEN
        lorentzian = Lorentzian(area=1.0, width=stretched.width.value)

        # EXPECT
        x = np.linspace(-2.0, 2.0, 101)
        np.testing.assert_allclose(stretched.evaluate(x), lorentzian.evaluate(x), rtol=1e-12)

    #############
    # Evaluation
    #############

    def test_evaluate_reduces_to_a_lorentzian_at_beta_one(self):
        # WHEN beta = 1, the transform is a Lorentzian of HWHM hbar / tau
        stretched = StretchedExponential(area=2.5, relaxation_time=3.0, beta=1.0)
        lorentzian = Lorentzian(area=2.5, width=HBAR_MEV_PS / 3.0)
        x = np.linspace(-3.0, 3.0, 401)

        # THEN
        result = stretched.evaluate(x)

        # EXPECT
        np.testing.assert_allclose(result, lorentzian.evaluate(x), rtol=1e-10)

    def test_evaluate_reduces_to_a_gaussian_at_beta_two(self):
        # WHEN beta = 2, the transform is a Gaussian of standard deviation sqrt(2) hbar / tau
        stretched = StretchedExponential(area=2.5, relaxation_time=3.0, beta=2.0)
        gaussian = Gaussian(area=2.5, width=np.sqrt(2) * HBAR_MEV_PS / 3.0)
        x = np.linspace(-3.0, 3.0, 401)

        # THEN
        result = stretched.evaluate(x)

        # EXPECT
        np.testing.assert_allclose(result, gaussian.evaluate(x), atol=1e-13)

    def test_evaluate_peak_height(self, stretched_exponential: StretchedExponential):
        # WHEN the peak sits at the center and equals area * gamma(1 + 1/beta) / (pi * Gamma)
        energy_scale = HBAR_MEV_PS / 3.0

        # THEN
        result = stretched_exponential.evaluate(np.array([0.5]))

        # EXPECT
        expected = 2.0 * gamma(1.0 + 1.0 / 0.7) / (np.pi * energy_scale)
        assert result[0] == pytest.approx(expected, rel=1e-10)

    def test_evaluate_is_symmetric_about_the_center(
        self, stretched_exponential: StretchedExponential
    ):
        # WHEN
        offset = np.array([0.05, 0.3, 1.7])

        # THEN
        left = stretched_exponential.evaluate(0.5 - offset)
        right = stretched_exponential.evaluate(0.5 + offset)

        # EXPECT
        np.testing.assert_allclose(left, right, rtol=1e-12)

    @pytest.mark.parametrize('beta', [0.5, 0.7, 1.0, 1.5])
    def test_area_matches_parameter(self, beta):
        # WHEN the transform integrates to the area parameter over the whole axis
        stretched = StretchedExponential(area=2.0, relaxation_time=5.0, beta=beta)
        # The tails are heavy, so integrate on a grid that is dense near the peak and reaches far
        x = sinh_grid(1e8) * (HBAR_MEV_PS / 5.0)

        # THEN
        numerical_area = simpson(stretched.evaluate(x), x=x)

        # EXPECT
        assert numerical_area == pytest.approx(2.0, rel=5e-3)

    def test_relaxation_time_sets_the_width(self):
        # WHEN a longer relaxation time means a narrower line
        narrow = StretchedExponential(relaxation_time=20.0, beta=0.8)
        wide = StretchedExponential(relaxation_time=2.0, beta=0.8)

        # THEN
        narrow_peak = narrow.evaluate(np.array([0.0]))[0]
        wide_peak = wide.evaluate(np.array([0.0]))[0]

        # EXPECT the peak scales as tau, since the area is conserved
        assert narrow_peak == pytest.approx(10.0 * wide_peak, rel=1e-10)

    def test_evaluate_scipp_output(self, stretched_exponential: StretchedExponential):
        # WHEN
        x = np.linspace(-5, 5, 50)

        # THEN
        result = stretched_exponential.evaluate(x, output='scipp')

        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('dimensionless')
        np.testing.assert_allclose(result.values, stretched_exponential.evaluate(x))

    def test_evaluate_with_scipp_x_in_another_energy_unit(
        self, stretched_exponential: StretchedExponential
    ):
        # WHEN x carries microeV rather than the component's meV
        x = np.linspace(-2.0, 2.0, 51)

        # THEN
        result = stretched_exponential.evaluate(
            sc.array(dims=['energy'], values=x * 1e3, unit='microeV')
        )

        # EXPECT the same profile, since the output carries y_unit either way
        np.testing.assert_allclose(result, stretched_exponential.evaluate(x), rtol=1e-12)

    def test_evaluate_with_a_non_energy_x_unit_raises(self):
        # WHEN hbar / relaxation_time cannot be expressed in the x unit
        stretched = StretchedExponential(x_unit='m')

        # THEN EXPECT
        with pytest.raises(UnitError, match='needs an energy x_unit'):
            stretched.evaluate(np.array([0.0, 1.0]))

    @pytest.mark.parametrize('beta', [0.6, 0.8, 1.0, 1.5, 2.0])
    def test_evaluate_matches_a_naive_fft_transform(self, beta):
        # WHEN the same physics is computed the obvious way, by FFT-ing the sampled relaxation

        # THEN
        largest_gap = fft_comparison(beta, n_time=2**20)

        # EXPECT the two agree wherever the FFT itself is trustworthy
        assert largest_gap < 1e-3

    def test_the_naive_fft_converges_onto_evaluate(self):
        # WHEN the FFT reference is limited by its own time step, not by the implementation.  Its
        # error comes from the cusp of exp(-(t / tau)**beta) at t = 0, so the trapezoid sum
        # converges as dt**(1 + beta) -- refining dt by four should shrink it by 4**1.6 ~ 9.
        gaps = [fft_comparison(0.6, n_time=n) for n in (2**16, 2**18, 2**20)]

        # THEN
        ratios = [coarse / fine for coarse, fine in pairwise(gaps)]

        # EXPECT the gap closes at the predicted rate, so the FFT is converging onto our values
        assert all(ratio > 4.0 for ratio in ratios), (gaps, ratios)

    ##################
    # Unit conversion
    ##################

    def test_convert_x_unit(self, stretched_exponential: StretchedExponential):
        # WHEN THEN
        stretched_exponential.convert_x_unit('microeV')

        # EXPECT the time and the exponent are untouched: they carry no x unit
        assert stretched_exponential.x_unit == 'microeV'
        assert stretched_exponential.area.value == pytest.approx(2.0 * 1e3)
        assert stretched_exponential.center.value == pytest.approx(0.5 * 1e3)
        assert stretched_exponential.relaxation_time.value == pytest.approx(3.0)
        assert stretched_exponential.relaxation_time.unit == 'ps'
        assert stretched_exponential.beta.value == pytest.approx(0.7)

    def test_convert_x_unit_keeps_the_profile(self, stretched_exponential: StretchedExponential):
        # WHEN
        x = np.linspace(-2.0, 2.0, 51)
        before = stretched_exponential.evaluate(x)

        # THEN
        stretched_exponential.convert_x_unit('microeV')

        # EXPECT the same curve, read off the rescaled axis
        np.testing.assert_allclose(stretched_exponential.evaluate(x * 1e3), before, rtol=1e-12)

    def test_convert_x_unit_invalid_type_raises(self, stretched_exponential: StretchedExponential):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'x_unit must be a string or sc\.Unit'):
            stretched_exponential.convert_x_unit(123)

    def test_convert_x_unit_rollback_on_failure(self, stretched_exponential: StretchedExponential):
        # WHEN THEN
        with pytest.raises(UnitError):
            stretched_exponential.convert_x_unit('m')

        # EXPECT: state rolled back
        assert stretched_exponential.x_unit == 'meV'
        assert stretched_exponential.area.value == pytest.approx(2.0)
        assert stretched_exponential.center.value == pytest.approx(0.5)

    def test_convert_y_unit(self):
        # WHEN: x_unit='meV', y_unit='1/meV' → area_unit='dimensionless'
        stretched = StretchedExponential(area=1.0, x_unit='meV', y_unit='1/meV')

        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        stretched.convert_y_unit('1/eV')

        # EXPECT: y_unit updated and area value rescaled (1e3 factor)
        assert stretched.y_unit == '1/eV'
        assert stretched.area.value == pytest.approx(1e3)

    def test_convert_y_unit_invalid_type_raises(self, stretched_exponential: StretchedExponential):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            stretched_exponential.convert_y_unit(123)

    def test_convert_y_unit_rollback_on_failure(self):
        # WHEN
        stretched = StretchedExponential(area=1.0, x_unit='meV')

        # THEN
        with pytest.raises(UnitError):
            stretched.convert_y_unit('K')

        # EXPECT: state rolled back
        assert stretched.y_unit == 'dimensionless'
        assert stretched.area.value == pytest.approx(1.0)
