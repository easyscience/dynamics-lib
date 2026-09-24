# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

r"""
The stretched exponential (Kohlrausch-Williams-Watts) relaxation, Fourier transformed to energy.

The model is defined in time, as $I(t) = A e^{-(|t| / \tau)^\beta}$. To evaluate the Fourier
transform, we need to evaluate the following integral:

$$ G_\beta(w) = \int_0^\infty e^{-u^\beta} \cos(w u) \, du $$

The problem is that $\cos(w u)$ oscillates forever at constant amplitude, and $e^{-u^\beta}$ decays
extremely slowly once $\beta$ is small: at $\beta = 0.3$ it is still significant at $u \sim 10^5$.
Evaluating $G_\beta$ therefore means summing an enormous number of nearly cancelling oscillations.
Furthermore, large $w$ makes the oscillation fast, so the steps must be tiny, while small $\beta$
stretches the tail over more decades, so the range must be huge.

An FFT of the sampled relaxation cannot span that tail at small $\beta$ and, because $e^{-(t /
\tau)^\beta}$ has a cusp at $t = 0$, converges only as $(\Delta t)^{1 + \beta}$.
``scipy.stats.levy_stable`` is mathematically the same function, but it is 70-100x slower and not
accurate for $\alpha$ close to 1.

Instead, the integral is evaluated in the complex plane described in :func:`_kww_shape`.

The four helpers below are module-level functions rather than methods. :func:`_quadrature_nodes`
and :func:`_reduced_hwhm` must be, because ``@lru_cache`` on a method would key on ``self``: every
instance would rebuild its own grid and the cache would keep every component ever created alive.
:func:`_kww_shape` is a pure function of ``(w, beta)`` that touches no component state: area,
center and units are applied afterwards in ``_evaluate_values``, so it does not belong in the
class, and :func:`_hwhm_dependency_expression` only assembles a string from module constants.

:func:`_reduced_hwhm` is not on any evaluation path.  It solves the half width exactly, and the
closed form that ``width`` actually resolves through is fitted to it; it stays here as the
reference that defines `_HWHM_POLY_COEFFS` and as what the tests check that fit against.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from scipp import UnitError
from scipy.optimize import brentq

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import convert_value_unit
from easydynamics.utils.utils import hbar

if TYPE_CHECKING:
    import scipp as sc

MINIMUM_RELAXATION_TIME = 1e-10  # ps. Avoids a division by zero in hbar / tau
MINIMUM_BETA = 0.05  # Below this the quadrature grid can no longer resolve the time tail
MAXIMUM_BETA = 2.0  # Above this the transform is no longer positive, so unphysical

# Tuning of the exp-sinh quadrature used by _kww_shape. The step and the half-range were chosen
# together so the transform reproduces its analytic special cases (beta = 1 and beta = 2) to
# ~1e-11 relative over w in [0, 1e6]; see test_stretched_exponential.py.
_QUAD_STEP = 0.03
_QUAD_HALF_RANGE = 4.3
# exp(-u**beta) has fallen below exp(-_QUAD_DECADES) at the far end of the grid.
_QUAD_DECADES = 50.0
# The quadrature holds one (energies x nodes) temporary, so long energy axes are evaluated in
# blocks to keep that intermediate at a few tens of MB instead of scaling with the axis.
_MAX_BLOCK_ELEMENTS = 1 << 21
# The reduced half width spans 1.67 at beta = 2 down to ~7e-27 at beta = 0.05, so it is
# bracketed in log10(w).  The lower end stays clear of the subnormal range, where the
# 1 / (w sin theta) rescaling inside _kww_shape would overflow.
_HWHM_LOG_BRACKET = (-300.0, 3.0)

# The width is exposed as a dependent Parameter, and easyscience resolves a dependency from a
# string expression, so the root solved by _reduced_hwhm has to be written in closed form.  It
# very nearly is one: beta * ln(HWHM) - ln(beta) is a small, smooth function of beta alone, so
# HWHM = exp((ln(beta) + p(beta)) / beta) with p the polynomial below, in ascending powers.  The
# coefficients are a degree 8 least-squares fit over 400 log-spaced beta in [MINIMUM_BETA,
# MAXIMUM_BETA], and reproduce _reduced_hwhm to 1.4e-5 relative for beta >= 0.1.  Below that the
# fit degrades to ~1e-2, but so does the root it is fitted to, and both are tens of orders of
# magnitude under any energy grid step by then.  Regenerate, or check these against the root they
# are fitted to, with tools/fit_kww_hwhm_coefficients.py.
_HWHM_POLY_COEFFS = (
    -5.5129123891501144e-05,
    -0.24257355020449564,
    0.2875849387004671,
    -0.04773226324334423,
    -0.04385316873071495,
    0.11064865473723488,
    -0.09366816058690182,
    0.03431170767881221,
    -0.004659825239152873,
)


@lru_cache(maxsize=8)
def _quadrature_nodes(half_steps: int) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build the exp-sinh quadrature nodes and weights for ``2 * half_steps + 1`` points.

    The exp-sinh (double-exponential) rule maps the trapezoidal rule on the whole real line onto
    the half line via $s = e^{(\pi / 2) \sinh t}$, which makes the node density follow the decades
    of $s$ rather than its absolute size.  The grids are small and reused across evaluations, so
    they are cached rather than rebuilt on every call.

    Parameters
    ----------
    half_steps : int
        Number of steps of size ``_QUAD_STEP`` on each side of $t = 0$.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The node positions $s$ on the half line and their quadrature weights.
    """
    t = np.arange(-half_steps, half_steps + 1) * _QUAD_STEP
    nodes = np.exp(0.5 * np.pi * np.sinh(t))
    weights = _QUAD_STEP * nodes * (0.5 * np.pi) * np.cosh(t)
    return nodes, weights


def _kww_shape(w: np.ndarray, beta: float) -> np.ndarray:
    r"""
    Evaluate the reduced Fourier cosine transform of a stretched exponential.

    $$ G_\beta(w) = \int_0^\infty e^{-u^\beta} \cos(w u) \, du $$

    The physical spectrum is $\frac{A}{\pi \Gamma} G_\beta(x / \Gamma)$ with $\Gamma = \hbar /
    \tau$, so $w$ is the reduced energy $x / \Gamma$ and everything else is a scale factor.

    The integrand oscillates without decaying, so it is integrated along the rotated ray $u = s
    e^{i \theta}$ instead of the real axis.  Writing $\cos(w u) = \mathrm{Re}\, e^{i w u}$ makes
    the integrand analytic in a wedge around the positive real axis: $e^{-u^\beta}$ keeps decaying
    as long as $\arg(u) < \pi / (2 \beta)$, and the arc at infinity vanishes, so by Cauchy's
    theorem the ray may be swung up to that angle without changing the value.

    On the tilted ray $e^{i w u}$ becomes $e^{i w s \cos\theta} e^{-w s \sin\theta}$: the
    oscillation now decays with increasing $w$.  The faster the oscillation, the faster it is
    damped, so the number of oscillations before the integrand dies is bounded *independently of*
    $w$.

    The angle $\theta = \pi / (4 \beta)$ is half of the $\pi / (2 \beta)$ limit, a deliberate
    safety margin; it is capped at $0.45 \pi$ because for $\beta < 0.5$ the formula would exceed
    $\pi / 2$ and the ray would cross the imaginary axis, where $e^{-u^\beta}$ grows.

    Writing $e^{-u^\beta + i w u}$ out in real form on that ray, with the extra $\theta$ in the
    phase coming from $du = e^{i \theta} ds$, turns the integral into

    $$ G_\beta(w) = \int_0^\infty e^{-P(s)} \cos(Q(s) + \theta) \, ds $$

    with $P = \cos(\beta \theta) s^\beta + w \sin(\theta) s$ (the decay) and $Q = w \cos(\theta) s
    - \sin(\beta \theta) s^\beta$ (what is left of the oscillation), which an exp-sinh rule on a
    grid rescaled to the decay length of $P$ integrates to near machine precision for every $w$.

    The rescaling is the other half of the robustness.  $P$ has two terms, and whichever dies first
    sets the decay length: $\cos(\beta \theta)^{-1 / \beta}$ for the $s^\beta$ term, $1 / (w
    \sin\theta)$ for the $w s$ term.  Taking the smaller of the two and stretching the grid onto it
    means the integrand always falls off around $s = 1$, so a single node layout serves every $w$
    and every $\beta$.

    Accuracy is therefore governed by $w$ and $\beta$ alone. For $\beta \le 1$ the result tracks
    the exact power-law tail out to at least $w = 10^8$.  For $\beta > 1$ the profile decays fast
    enough to reach the floor of double precision, and values below roughly $10^{-11}$ of the peak
    are noise.

    Parameters
    ----------
    w : np.ndarray
        Reduced energies.  Only $|w|$ matters: the transform is even in $w$.
    beta : float
        Stretching exponent, in ``[MINIMUM_BETA, MAXIMUM_BETA]``.

    Returns
    -------
    np.ndarray
        The transform, clipped at zero.  It is non-negative for every $\beta \le 2$; the clip only
        removes rounding noise where the true value has underflowed.
    """
    abs_w = np.abs(w)

    theta = min(0.45 * np.pi, np.pi / (4.0 * beta))
    cos_rotated = np.cos(beta * theta)
    sin_rotated = np.sin(beta * theta)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Reach far enough out in s that exp(-cos(beta theta) s**beta) has died: small beta stretches
    # the tail over many more decades, so the grid has to follow it.  This inverts the node map
    # s = exp(pi / 2 sinh(t)) at the s where cos(beta theta) s**beta equals _QUAD_DECADES, i.e.
    # where the decaying factor has fallen to exp(-50).
    half_range = np.arcsinh((2.0 / np.pi) * np.log(_QUAD_DECADES / cos_rotated) / beta)
    half_steps = int(np.ceil(max(half_range, _QUAD_HALF_RANGE) / _QUAD_STEP))
    nodes, weights = _quadrature_nodes(half_steps)
    nodes_beta = nodes**beta

    # Rescale the grid onto whichever of the two exponents in P decays first, so the integrand
    # always falls off around s = 1 and the same node layout serves every w.  beta_scale is where
    # the s**beta term dies, w_scale where the w s term does; w_scale is infinite at w = 0, where
    # there is no oscillation to damp and beta_scale is the only length in the problem.
    beta_scale = cos_rotated ** (-1.0 / beta)
    w_scale = np.divide(1.0, abs_w * sin_theta, out=np.full(abs_w.shape, np.inf), where=abs_w > 0)
    scale = np.minimum(beta_scale, w_scale)

    out = np.empty(abs_w.shape)
    block = max(1, _MAX_BLOCK_ELEMENTS // nodes.size)
    for start in range(0, abs_w.size, block):
        stop = start + block
        block_w = abs_w[start:stop, np.newaxis]
        block_scale = scale[start:stop, np.newaxis]

        s = block_scale * nodes
        s_beta = block_scale**beta * nodes_beta
        decaying = cos_rotated * s_beta + (block_w * sin_theta) * s
        oscillating = (block_w * cos_theta) * s - sin_rotated * s_beta

        integral = (np.exp(-decaying) * np.cos(oscillating + theta) * weights).sum(axis=1)
        out[start:stop] = integral * scale[start:stop]

    return np.maximum(out, 0.0)


@lru_cache(maxsize=128)
def _reduced_hwhm(beta: float) -> float:
    r"""
    Solve for the half width at half maximum of $G_\beta$ in reduced energy $w = x / \Gamma$.

    $\Gamma = \hbar / \tau$ is the HWHM only at $\beta = 1$.  Below that the profile sharpens
    dramatically: the peak $G_\beta(0)$ grows while the area stays fixed, so the half width
    collapses far faster than $\Gamma$ suggests -- 0.22 $\Gamma$ at $\beta = 0.5$, 2.7e-4 $\Gamma$
    at $\beta = 0.2$, 8e-11 $\Gamma$ at $\beta = 0.1$.  There is no closed form, so the crossing of
    $G_\beta(w) = \tfrac{1}{2} G_\beta(0)$ is bracketed numerically.

    The search runs in $\log_{10} w$ because the root ranges over more than 25 decades across the
    supported $\beta$; a linear bracket could not resolve the small-$\beta$ end.  Each solve costs
    a few tens of single-point quadratures, under a millisecond, and the result is cached because
    it depends on $\beta$ alone.

    The root is well conditioned down to $\beta \approx 0.1$ and increasingly poorly below it: by
    $\beta = 0.05$ the profile takes some 60 e-folds of $w$ to fall by half, so $G_\beta$ is nearly
    flat in $\log w$ near the crossing and the result is only good to about a percent.  That is
    immaterial for the one thing the half width is used for -- comparing against the convolution's
    energy grid -- because at those $\beta$ it is already tens of orders of magnitude below any
    grid step, and the comparison lands the same way either way.

    Parameters
    ----------
    beta : float
        Stretching exponent, in ``[MINIMUM_BETA, MAXIMUM_BETA]``.

    Returns
    -------
    float
        The HWHM in units of $\Gamma$.  Multiply by $\Gamma$ to get an energy.
    """
    half_peak = 0.5 * _kww_shape(np.array([0.0]), beta)[0]

    def offset(log_w: float) -> float:
        return _kww_shape(np.array([10.0**log_w]), beta)[0] - half_peak

    return float(10.0 ** brentq(offset, *_HWHM_LOG_BRACKET, xtol=1e-12))


def _hwhm_dependency_expression() -> str:
    r"""
    Write the half width of :func:`_reduced_hwhm` as an easyscience dependency expression.

    The expression evaluates ``exp((log(beta) + p(beta)) / beta)`` from `_HWHM_POLY_COEFFS`, in
    Horner form, and multiplies it by $\hbar / \tau$ to turn the reduced half width into an energy.

    ``exp`` and ``log`` are not defined on DescriptorNumbers, so the reduced half width is built
    from ``b.value`` and is a plain float; the surrounding ``* hbar / tau`` is what makes the whole
    expression return a DescriptorNumber, as easyscience requires.  Referring to ``b.value`` rather
    than ``b`` strips beta's uncertainty, which is why the width carries only the relaxation time's
    contribution. When exp and log of DescriptorNumbers are supported, this can be rewritten to
    propagate beta's uncertainty too.

    Returns
    -------
    str
        The dependency expression, over the mapped names ``b``, ``hbar`` and ``tau``.
    """
    horner = repr(_HWHM_POLY_COEFFS[-1])
    for coefficient in _HWHM_POLY_COEFFS[-2::-1]:
        horner = f'({horner}) * b.value + {coefficient!r}'
    return f'exp((log(b.value) + ({horner})) / b.value) * hbar / tau'


class StretchedExponential(CreateParametersMixin, ModelComponent):
    r"""
    Model of a stretched exponential (Kohlrausch-Williams-Watts) relaxation, Fourier transformed
    from time to energy.

    The model is defined by its intermediate scattering function

    $$ I(t) = A \exp\left[-\left(\frac{|t|}{\tau}\right)^\beta\right] $$

    where $\tau$ is the relaxation time and $\beta$ the stretching exponent.  Here we calculate the
    Fourier transform:

    $$ S(x) = \frac{1}{2\pi\hbar} \int I(t)\, e^{-i (x - x_0) t / \hbar} \, \mathrm{d}t =
    \frac{A}{\pi \Gamma} \, G_\beta\!\left(\frac{x - x_0}{\Gamma}\right), \qquad \Gamma =
    \frac{\hbar}{\tau} $$

    with $G_\beta(w) = \int_0^\infty e^{-u^\beta} \cos(w u)\, \mathrm{d}u$.  $A$ is the area (the
    profile integrates to $A$ over $x$), $x_0$ is the center, and $\Gamma$ is the energy scale set
    by the relaxation time.  area has unit = x_unit * y_unit; center has unit = x_unit;
    relaxation_time has unit ps; beta is dimensionless.

    $\beta = 1$ recovers a Lorentzian of HWHM $\Gamma$ and $\beta = 2$ a Gaussian of standard
    deviation $\sqrt{2}\,\Gamma$; in between there is no closed form and the transform is evaluated
    numerically.  $\beta \le 1$ is the physically usual range.  $\beta$ is capped at 2 because
    $\exp(-|t|^\beta)$ stops being positive definite beyond it, so the transform would go negative.

    Note that the x-axis has to be an energy, since the relaxation time is turned into an energy
    scale through $\hbar$.

    Examples
    --------
    **Creating a stretched exponential**

    By default the center is fixed at 0 like a Lorentzian::
    ```python
    import numpy as np
    import easydynamics as edyn

    kww = edyn.StretchedExponential(area=1.0, relaxation_time=5.0, beta=0.7)
    x = np.linspace(-2, 2, 100)
    values = kww.evaluate(x)
    ```

    **Modifying parameters after construction**

    ```python
    import easydynamics as edyn

    kww = edyn.StretchedExponential(area=2.0, relaxation_time=10.0, beta=0.5, name='Polymer')
    kww.relaxation_time = 20.0
    kww.beta = 0.6
    ```
    """

    def __init__(
        self,
        area: Numeric = 1.0,
        center: Numeric | None = None,
        relaxation_time: Numeric = 1.0,
        beta: Numeric = 1.0,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'StretchedExponential',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        r"""
        Initialize the StretchedExponential component.

        Parameters
        ----------
        area : Numeric, default=1.0
            Integrated area under the transformed profile.  Unit is ``x_unit * y_unit``.
        center : Numeric | None, default=None
            Peak position in x_unit.  If None, defaults to 0 and the center parameter is fixed.
        relaxation_time : Numeric, default=1.0
            Relaxation time tau in ps.  Must be strictly positive.  It enters the spectrum as the
            energy scale $\Gamma = \hbar / \tau$.
        beta : Numeric, default=1.0
            Stretching exponent.  Must lie in ``[0.05, 2.0]``; $\beta = 1$ is a Lorentzian and
            $\beta = 2$ a Gaussian.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  Must be an energy, since $\hbar / \tau$ is converted into it.
            center is stored in this unit. area_unit = x_unit * y_unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='StretchedExponential'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier.  Auto-generated if None.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        self._area = self._create_area_parameter(
            area=area, name=name, x_unit=self.x_unit, y_unit=self.y_unit
        )
        self._center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, x_unit=self.x_unit
        )

        self._validate_relaxation_time(relaxation_time)
        self._relaxation_time = Parameter(
            name=name + ' relaxation_time',
            value=float(relaxation_time),
            unit='ps',
            min=MINIMUM_RELAXATION_TIME,
        )

        self._validate_beta(beta)
        self._beta = Parameter(
            name=name + ' beta',
            value=float(beta),
            unit='dimensionless',
            min=MINIMUM_BETA,
            max=MAXIMUM_BETA,
        )

        # hbar has to be in the dependency map, because carrying J*s through the expression is
        # what lets the unit algebra turn the reduced half width into an energy.  It is a private
        # copy rather than the shared ``utils.hbar`` deliberately: make_dependent_on attaches the
        # width as an observer of every mapped variable, so mapping the module-level constant
        # would append one observer per component to a list that is never emptied, keeping every
        # StretchedExponential ever built alive and lengthening each of its notifications.
        self._hbar = DescriptorNumber(name=name + ' hbar', value=hbar.value, unit=str(hbar.unit))
        # Built here rather than on first access so that get_all_parameters reports the same set
        # whether or not the width has been read yet.  A non-energy x_unit has no width to build
        # and must still construct: the UnitError is deferred to the first read, as for evaluate.
        self._width: Parameter | None = None
        with contextlib.suppress(UnitError):
            self._width = self._build_width()

    ################################
    # Validation
    ################################

    @staticmethod
    def _validate_relaxation_time(value: Numeric) -> None:
        """
        Check that a relaxation time is a finite, strictly positive number.

        Parameters
        ----------
        value : Numeric
            The candidate relaxation time.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not finite, or is smaller than ``MINIMUM_RELAXATION_TIME``.
        """
        if not isinstance(value, Numeric):
            raise TypeError('relaxation_time must be a number.')
        if not np.isfinite(value):
            raise ValueError('relaxation_time must be a finite number.')
        if float(value) < MINIMUM_RELAXATION_TIME:
            raise ValueError('relaxation_time must be greater than zero.')

    @staticmethod
    def _validate_beta(value: Numeric) -> None:
        """
        Check that a stretching exponent is a finite number inside the supported range.

        Parameters
        ----------
        value : Numeric
            The candidate stretching exponent.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not finite, or lies outside ``[MINIMUM_BETA, MAXIMUM_BETA]``.
        """
        if not isinstance(value, Numeric):
            raise TypeError('beta must be a number.')
        if not np.isfinite(value):
            raise ValueError('beta must be a finite number.')
        if not MINIMUM_BETA <= float(value) <= MAXIMUM_BETA:
            raise ValueError(f'beta must be between {MINIMUM_BETA} and {MAXIMUM_BETA}.')

    ################################
    # Properties
    ################################

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns
        -------
        Parameter
            The area Parameter with unit ``x_unit * y_unit``.
        """
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New area value (in current area unit = x_unit * y_unit).

        Notes
        -----
        A ``TypeError`` propagates from the shared value setter if *value* is not a numeric type,
        and a ``ValueError`` propagates from it if *value* violates the area parameter's bounds
        (e.g. a negative value when the area was created non-negative, giving it ``min=0``).
        """
        self._set_bounded_parameter_value(self._area, value, 'area')

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter.

        Returns
        -------
        Parameter
            The center (x_0) Parameter with unit ``x_unit``.
        """
        return self._center

    @center.setter
    def center(self, value: Numeric | None) -> None:
        """
        Parameters
        ----------
        value : Numeric | None
            New center value in x_unit.  If None, the center is set to 0 and the parameter is
            fixed.

        Raises
        ------
        TypeError
            If *value* is not None and not a numeric type.
        """
        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    @property
    def relaxation_time(self) -> Parameter:
        """
        Get the relaxation time parameter.

        Returns
        -------
        Parameter
            The relaxation time (tau) Parameter with unit ``ps``.
        """
        return self._relaxation_time

    @relaxation_time.setter
    def relaxation_time(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New relaxation time in the parameter's current unit.  Must be strictly positive.

        Notes
        -----
        A ``TypeError`` propagates from the validator if *value* is not a numeric type, and a
        ``ValueError`` propagates from it if *value* is not finite or not positive, or from the
        shared value setter if *value* violates the parameter's bounds.
        """
        self._validate_relaxation_time(value)
        self._set_bounded_parameter_value(self._relaxation_time, value, 'relaxation_time')

    @property
    def beta(self) -> Parameter:
        """
        Get the stretching exponent parameter.

        Returns
        -------
        Parameter
            The stretching exponent (beta) Parameter, dimensionless.
        """
        return self._beta

    @beta.setter
    def beta(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New stretching exponent.  Must lie in ``[MINIMUM_BETA, MAXIMUM_BETA]``.

        Notes
        -----
        A ``TypeError`` propagates from the validator if *value* is not a numeric type, and a
        ``ValueError`` propagates from it if *value* is not finite or lies outside the supported
        range, or from the shared value setter if *value* violates the parameter's bounds.
        """
        self._validate_beta(value)
        self._set_bounded_parameter_value(self._beta, value, 'beta')

    def _build_width(self) -> Parameter:
        r"""
        Build the dependent width Parameter in the component's current x_unit.

        Returns
        -------
        Parameter
            A Parameter made dependent on :attr:`beta` and :attr:`relaxation_time`.

        Raises
        ------
        UnitError
            If x_unit is not an energy, so $\hbar / \tau$ cannot be expressed in it.
        """
        width = Parameter(name=self.name + ' width', value=1.0, unit=str(self.x_unit))
        try:
            # easyscience propagates inf bounds through arithmetic, producing inf/inf=nan as a
            # transient intermediate; suppress the spurious numpy warnings as elsewhere.
            with np.errstate(invalid='ignore', divide='ignore'):
                width.make_dependent_on(
                    dependency_expression=_hwhm_dependency_expression(),
                    dependency_map={
                        'b': self._beta,
                        'hbar': self._hbar,
                        'tau': self._relaxation_time,
                    },
                    desired_unit=str(self.x_unit),
                )
        except UnitError as e:
            raise UnitError(
                f'{self.__class__.__name__} needs an energy x_unit so that hbar / '
                f'relaxation_time can be expressed in it, but got {self.x_unit}.'
            ) from e
        return width

    @property
    def width(self) -> Parameter:
        r"""
        Get the half width at half maximum of the peak.

        This is a *dependent* Parameter, derived from :attr:`relaxation_time` and :attr:`beta`. It
        is accurate to 1.4e-5 relative for $\beta \ge 0.1$. For better accuracy, use
        :func:`_reduced_hwhm` directly and multiply by $\hbar / \tau$ in the desired unit. Note
        that the uncertainty of the width may be underestimated, because the dependency expression
        uses ``b.value`` rather than ``b`` to compute the width, so the uncertainty of beta is not
        propagated into the width.

        Returns
        -------
        Parameter
            The HWHM expressed in the component's own x_unit.

        Notes
        -----
        A ``UnitError`` propagates from :meth:`_build_width` if x_unit is not an energy, so $\hbar
        / \tau$ cannot be expressed in it.
        """
        if self._width is None:
            self._width = self._build_width()
        return self._width

    ################################
    # Evaluation
    ################################

    def _energy_scale(self, eval_unit: str | None) -> float:
        r"""
        Get the energy scale $\Gamma = \hbar / \tau$ expressed in the evaluation unit.

        Parameters
        ----------
        eval_unit : str | None
            The unit x values are expressed in, or None for the component's own x_unit.

        Returns
        -------
        float
            Gamma in *eval_unit*.

        Raises
        ------
        UnitError
            If the target unit is not an energy, so $\hbar / \tau$ cannot be expressed in it.
        """
        target_unit = eval_unit if eval_unit is not None else self.x_unit
        tau_unit = self._relaxation_time.unit
        try:
            hbar_value = convert_value_unit(hbar.value, hbar.unit, f'({target_unit})*({tau_unit})')
        except UnitError as e:
            raise UnitError(
                f'{self.__class__.__name__} needs an energy x_unit so that hbar / '
                f'relaxation_time can be expressed in it, but got {target_unit}.'
            ) from e
        return hbar_value / self._relaxation_time.value

    def _evaluate_values(self, x_vals: np.ndarray, eval_unit: str | None) -> np.ndarray:
        r"""
        Evaluate the transformed stretched exponential at x_vals.

        $$ S(x) = \frac{A}{\pi \Gamma} \, G_\beta\!\left(\frac{x - x_0}{\Gamma}\right), \qquad
        \Gamma = \frac{\hbar}{\tau} $$

        where *A* is ``area``, *x*₀ is ``center``, *tau* is ``relaxation_time``, *beta* is the
        stretching exponent, and $G_\beta$ is the reduced transform computed by :func:`_kww_shape`.
        Parameters in the model's own units are temporarily converted to eval_unit for the
        computation.

        Parameters
        ----------
        x_vals : np.ndarray
            Raw x values expressed in eval_unit.
        eval_unit : str | None
            The unit of x_vals.

        Returns
        -------
        np.ndarray
            Evaluated values at x_vals.
        """
        center = self._resolve_param_value(self._center, eval_unit)
        area = self._resolve_param_value(self._area, self._eval_area_unit(eval_unit))
        energy_scale = self._energy_scale(eval_unit)

        shape = _kww_shape((x_vals - center) / energy_scale, self._beta.value)
        return area / (np.pi * energy_scale) * shape

    ################################
    # Unit conversion
    ################################

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        r"""
        Convert the center and the area to new_x_unit.

        The relaxation time carries a time unit and beta is dimensionless, so neither is affected;
        the energy scale $\hbar / \tau$ is re-derived in the new unit on every evaluation.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.
        """
        self._convert_x_unit_area_based(
            new_x_unit=new_x_unit,
            x_params=[self._center],
            area_param=self._area,
        )
        # The dependency resolves into the unit it was built with, so rebuild it in the new one.
        self._width = self._build_width()

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis (output) unit by rescaling the area parameter.

        The area is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(new_y_unit=new_y_unit, area_param=self._area)

    def __repr__(self) -> str:
        """
        Return a string representation of the StretchedExponential.

        Returns
        -------
        str
            A string representation of the StretchedExponential.
        """
        return (
            f'{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, '
            f'x_unit = {self.x_unit}, y_unit = {self.y_unit},\n '
            f'    area = {self.area},\n '
            f'    center = {self.center},\n '
            f'    relaxation_time = {self.relaxation_time},\n '
            f'    beta = {self.beta})'
        )
