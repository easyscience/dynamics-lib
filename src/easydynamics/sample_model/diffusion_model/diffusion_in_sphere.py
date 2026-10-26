# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import functools
import warnings

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from scipy.optimize import brentq
from scipy.special import spherical_jn as scipy_spherical_jn

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.diffusion_model.delta_lorentzian_diffusion_model_base import (
    DeltaLorentzianDiffusionModelBase,
)
from easydynamics.sample_model.diffusion_model.spherical_bessel_expression import (
    spherical_bessel_expression,
)
from easydynamics.utils.spherical_bessel import spherical_jn
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import hbar

# The roots x_nl of j_l'(x) = 0 below MAXIMUM_ROOT: the same 98 terms as Mantid's
# InelasticDiffSphere (initXnlCoeff),
# https://github.com/mantidproject/mantid/blob/main/Framework/CurveFitting/src/Functions/InelasticDiffSphere.cpp
# TODO: investigate shipping the roots as a table instead of computing them, and decide.  # ruff: ignore[line-contains-todo, missing-todo-author, missing-todo-link]
MAXIMUM_ROOT = 26.84
LIMIT_SWITCH = 1e-5  # Within this distance of a root, QR = x_nl, the 0/0 limit is used
MINIMUM_RADIUS = 1e-10  # In angstrom; a zero radius is unphysical
MINIMUM_DIFFUSION_COEFFICIENT = 1e-12  # In angstrom^2/ps, to keep the Lorentzian widths finite
MISSING_WEIGHT_TOLERANCE = 1e-3  # Truncating the sum above this loss of weight raises a warning
# Default number of terms, far fewer than Mantid's 98: each term adds two dependent Parameters per
# Q value, and their updates are slow in EasyScience Core 2.5.1. The first 10 terms miss less than
# 1e-3 of the weight up to QR of about 2 (4.3e-3 at QR = 3).
# TODO: return to all 98 terms once Core 3.0 makes dependent updates cheap.  # ruff: ignore[line-contains-todo, missing-todo-author, missing-todo-link]
DEFAULT_NUMBER_OF_LORENTZIANS = 10


@functools.cache
def sphere_roots() -> tuple[tuple[float, int, int], ...]:
    r"""
    Find the roots $x_{nl}$ of $j_l'(x) = 0$ between 0 and MAXIMUM_ROOT, sorted by x.

    n counts the roots of each l from 0, except for l = 0, whose root x = 0 is the elastic term:
    there n counts from 1, as in Bicout (2000). Computed once, on first use.

    Returns
    -------
    tuple[tuple[float, int, int], ...]
        (x, l, n) for each root, in increasing x.
    """
    grid = np.linspace(1e-6, MAXIMUM_ROOT, 20001)
    roots = []
    angular_order = 0
    while True:
        derivative = scipy_spherical_jn(angular_order, grid, derivative=True)
        changes = np.nonzero(np.sign(derivative[:-1]) * np.sign(derivative[1:]) < 0)[0]
        if changes.size == 0:
            break
        first_n = 1 if angular_order == 0 else 0
        for n, i in enumerate(changes, start=first_n):
            x = brentq(
                lambda z, order=angular_order: scipy_spherical_jn(order, z, derivative=True),
                grid[i],
                grid[i + 1],
                xtol=1e-15,
                rtol=1e-15,
            )
            roots.append((float(x), angular_order, n))
        angular_order += 1
    return tuple(sorted(roots))


@functools.cache
def _root_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Per-root constants of the weights, computed once.

    The weight of root x of order l is $(2l+1) \frac{6x^2}{x^2 - l(l+1)} J^2$ with $J(a) = \frac{a
    j_{l+1}(a) - l j_l(a)}{a^2 - x^2} = \frac{-a j_l'(a)}{a^2 - x^2}$, a = QR. At a = x both
    numerator and denominator vanish. From the spherical Bessel equation at $j_l'(x) = 0$: $j'' =
    -(1 - l(l+1)/x^2) j$ and $j''' = -(2/x) j'' - 2l(l+1) j / x^3$, so with $g = -a j_l'$ and $h =
    a^2 - x^2$: $J(x) = g'/h' = (x^2 - l(l+1)) j_l(x) / (2x^2)$ and $J'(x) = (2x g'' - 2 g') / (2
    (2x)^2)$, with $g' = -x j''$ and $g'' = -2j'' - x j'''$.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        x, l, the coefficient (2l+1) 6x^2 / (x^2 - l(l+1)), J(x) and J'(x), one value per root.
    """
    x = np.array([root[0] for root in sphere_roots()])
    angular_order = np.array([root[1] for root in sphere_roots()])
    l_term = angular_order * (angular_order + 1)
    coefficient = (2 * angular_order + 1) * 6 * x**2 / (x**2 - l_term)
    j = scipy_spherical_jn(angular_order, x)
    j2 = -(1 - l_term / x**2) * j
    j3 = -(2 / x) * j2 - 2 * l_term * j / x**3
    g1 = -x * j2
    g2 = -2 * j2 - x * j3
    J0 = g1 / (2 * x)
    J1 = (2 * x * g2 - 2 * g1) / (2 * (2 * x) ** 2)
    return x, angular_order, coefficient, J0, J1


class DiffusionInSphere(DeltaLorentzianDiffusionModelBase):
    r"""
    Model of diffusion inside an impermeable sphere.

    A particle diffuses with diffusion coefficient $D$ inside a sphere of radius $R$ that it cannot
    leave. The sphere is isotropic, so the scattering depends only on the magnitude of $Q$:

    $$ S(Q, E) = A_0(Q)\,\delta(E) + \sum_{(l, n)} A_{nl}(Q)\, L(E, \Gamma_{nl}) $$

    with

    $$ A_0(Q) = \left(\frac{3 j_1(QR)}{QR}\right)^2,\quad A_{nl}(Q) = (2l+1) \frac{6
    x_{nl}^2}{x_{nl}^2 - l(l+1)} \left(\frac{QR\, j_{l+1}(QR) - l\, j_l(QR)}{(QR)^2 -
    x_{nl}^2}\right)^2,\quad \Gamma_{nl} = \frac{\hbar x_{nl}^2 D}{R^2} $$

    where $j_l$ are the spherical Bessel functions, $x_{nl}$ the roots of $j_l'(x) = 0$ (no flux
    through the surface) and $L(E, \Gamma)$ a Lorentzian with half width at half maximum $\Gamma$.
    $A_0$ is the EISF and $A_{nl}$ the QISF of the root $x_{nl}$. The widths do not depend on $Q$;
    only the QISF do. At $QR = x_{nl}$ the QISF is a 0/0, and its exact limit is used there.

    The sum runs over the lowest ``number_of_lorentzians`` of the 98 roots below 26.84, the same
    roots as reference 3. A warning is raised if the truncation leaves out more than 1e-3 of the
    weight: with the default 10 terms this happens from QR of about 1.8, with all 98 from QR of
    about 9. Each term adds two dependent Parameters per Q value, which are slow to update in
    EasyScience Core 2.5.1, so the default is 10 terms.

    The Lorentzian of order $j$, from 1 to ``number_of_lorentzians``, belongs to the $j$-th root in
    increasing $x$, and ``roots`` gives its $(x, l, n)$. The radius is free by default, since the
    size of the sphere is usually what is fitted.

    References
    ----------
    1. F. Volino and A. J. Dianoux, Mol. Phys. 41, 271 (1980), doi:10.1080/00268978000102761.
    2. D. J. Bicout, Phys. Rev. E 62, 261 (2000), doi:10.1103/PhysRevE.62.261.
    3. Mantid, ``InelasticDiffSphere``, which uses the same 98 roots,
       https://github.com/mantidproject/mantid/blob/6ec66a6f524a7ec9d4258351fe3001c66c63dbe5/Framework/CurveFitting/src/Functions/InelasticDiffSphere.cpp#L53-L69.

    Examples
    --------
    **Creating a DiffusionInSphere model**

    The model creates one delta function and 10 Lorentzians per Q-value by default:
    ```python
    import numpy as np
    import easydynamics as edyn

    model = edyn.DiffusionInSphere(
        radius=1.0,  # angstrom
        diffusion_coefficient=0.1,  # angstrom^2/ps
        Q=np.linspace(0.3, 1.5, 7),  # QR up to 1.5, below the truncation warning
    )
    ```
    """

    _order_symbol = 'j'  # components are named '<lorentzian_name>_j<j>'

    def __init__(
        self,
        scale: Numeric = 1.0,
        radius: Numeric = 1.0,
        diffusion_coefficient: Numeric = 1.0,
        number_of_lorentzians: int = DEFAULT_NUMBER_OF_LORENTZIANS,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'DiffusionInSphere',
        display_name: str | None = 'DiffusionInSphere',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        delta_name: str = 'Delta function',
        delta_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new DiffusionInSphere model.

        A radius or diffusion coefficient that is not a number raises TypeError, one below its
        minimum raises ValueError, as do invalid delta names (from the base class). A
        number_of_lorentzians that is not an integer raises TypeError, one below 1 or above 98
        ValueError.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        radius : Numeric, default=1.0
            Radius R of the sphere in angstrom. Must be positive. Free by default.
        diffusion_coefficient : Numeric, default=1.0
            Diffusion coefficient D in angstrom^2/ps. Must be positive.
        number_of_lorentzians : int, default=DEFAULT_NUMBER_OF_LORENTZIANS
            Number of terms in the sum, the lowest roots first; at most 98, the roots below 26.84.
            Fewer terms make fits faster; the model warns if they omit too much weight.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='DiffusionInSphere'
            Name of the diffusion model.
        display_name : str | None, default='DiffusionInSphere'
            Display name of the diffusion model.
        lorentzian_name : str | None, default=None
            Base name of the Lorentzian components. The Lorentzian of order j is named
            '<lorentzian_name>_j<j>'. If None, it will be set to the name of the diffusion model.
        lorentzian_display_name : str | None, default=None
            Base display name of the Lorentzian components. If None, it will be set to
            lorentzian_name.
        delta_name : str, default='Delta function'
            Name of the delta function component.
        delta_display_name : str | None, default=None
            Display name of the delta function component. If None, it will be set to delta_name.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated.

        Raises
        ------
        ValueError
            If number_of_lorentzians is larger than the number of sphere roots available (98).
        """
        self._verify_radius(radius)
        self._verify_diffusion_coefficient(diffusion_coefficient)

        super().__init__(
            number_of_lorentzians,
            scale=scale,
            Q=Q,
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            lorentzian_name=lorentzian_name,
            lorentzian_display_name=lorentzian_display_name,
            delta_name=delta_name,
            delta_display_name=delta_display_name,
            unique_name=unique_name,
        )
        if self.number_of_lorentzians > len(sphere_roots()):
            raise ValueError(f'number_of_lorentzians must be at most {len(sphere_roots())}.')

        self._radius = Parameter(
            name='radius',
            value=float(radius),
            fixed=False,
            min=MINIMUM_RADIUS,
            unit='angstrom',
        )
        self._diffusion_coefficient = Parameter(
            name='diffusion_coefficient',
            value=float(diffusion_coefficient),
            fixed=False,
            min=MINIMUM_DIFFUSION_COEFFICIENT,
            unit='angstrom**2/ps',
        )

        self._hbar = hbar
        # Dimensionless one, which turns the plain numbers of the weight expressions into
        # (dimensionless) descriptors, as dependency expressions must evaluate to one.
        self._one = DescriptorNumber(name='one', value=1.0)

        # Dimensionless radius/angstrom for the dependency expressions. Deriving it through the
        # dependency graph keeps the weights correct if the radius is converted to another unit.
        with np.errstate(invalid='ignore'):
            self._radius_over_angstrom = Parameter.from_dependency(
                name='radius_over_angstrom',
                dependency_expression='radius / angstrom',
                dependency_map={'radius': self._radius, 'angstrom': angstrom},
            )
            self._radius_over_angstrom.set_desired_unit('dimensionless')

        self.create_component_collections()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def radius(self) -> Parameter:
        """
        Get the radius of the sphere.

        Returns
        -------
        Parameter
            Radius R.
        """
        return self._radius

    @radius.setter
    def radius(self, radius: Numeric) -> None:
        """
        Set the radius of the sphere.

        Parameters
        ----------
        radius : Numeric
            Radius R in the unit of the radius parameter (angstrom by default).
        """
        self._verify_radius(radius)
        self._radius.value = float(radius)

    @property
    def diffusion_coefficient(self) -> Parameter:
        """
        Get the diffusion coefficient.

        Returns
        -------
        Parameter
            Diffusion coefficient D.
        """
        return self._diffusion_coefficient

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, diffusion_coefficient: Numeric) -> None:
        """
        Set the diffusion coefficient.

        Parameters
        ----------
        diffusion_coefficient : Numeric
            Diffusion coefficient D in the unit of the parameter (angstrom^2/ps by default).
        """
        self._verify_diffusion_coefficient(diffusion_coefficient)
        self._diffusion_coefficient.value = float(diffusion_coefficient)

    @property
    def roots(self) -> tuple[tuple[float, int, int], ...]:
        r"""
        Get the roots $x_{nl}$ of the Lorentzians, in the order of the components.

        Returns
        -------
        tuple[tuple[float, int, int], ...]
            (x, l, n) of the Lorentzian of order j at index j - 1.
        """
        return sphere_roots()[: self.number_of_lorentzians]

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        r"""
        Calculate the half-width at half-maximum (HWHM) $\Gamma_{nl}$ of the Lorentzian of order
        $j$.

        The width is $\hbar x_{nl}^2 D / R^2$ and does not depend on Q.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order j of the Lorentzian, between 1 and number_of_lorentzians.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)
        radius = sc.values(self.radius.full_value)
        rate = self._hbar.full_value * sc.values(self.diffusion_coefficient.full_value) / radius**2
        x = sphere_roots()[int(order) - 1][0]
        return rate.to(unit=self.x_unit).value * x**2 * np.ones_like(Q)

    def calculate_EISF(self, Q: Q_type | None = None) -> np.ndarray:
        r"""
        Calculate the Elastic Incoherent Structure Factor (EISF), $(3 j_1(QR) / QR)^2$.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """
        QR = self._ensure_Q(Q) * self._radius_in_angstrom()
        with np.errstate(divide='ignore', invalid='ignore'):
            EISF = (3 * spherical_jn(1, QR) / QR) ** 2
        return np.where(QR == 0, 1.0, EISF)

    def calculate_QISF_order(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        r"""
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF) of one order, $A_{nl}(Q)$,
        the weight of the Lorentzian of order $j$ (root $x_{nl}$).

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order j of the Lorentzian, between 1 and number_of_lorentzians.

        Returns
        -------
        np.ndarray
            QISF values of that order (dimensionless).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)
        return self._weights(Q * self._radius_in_angstrom())[:, int(order) - 1]

    def calculate_QISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the total Quasi-Elastic Incoherent Structure Factor (QISF), the sum of the QISF
        of all number_of_lorentzians orders.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            QISF values (dimensionless).
        """
        Q = self._ensure_Q(Q)
        weights = self._weights(Q * self._radius_in_angstrom())
        return np.sum(weights[:, : self.number_of_lorentzians], axis=1)

    def calculate_missing_weight(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the spectral weight lost by truncating the sum at number_of_lorentzians terms.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            1 - EISF - QISF (dimensionless).
        """
        Q = self._ensure_Q(Q)
        return 1.0 - self.calculate_EISF(Q) - self.calculate_QISF(Q)

    def create_component_collections(self) -> list[ComponentCollection]:
        r"""
        Create ComponentCollections for the model at the current Q values.

        Each collection holds a delta function with area $K A_0(Q)$ (scale times EISF) and
        number_of_lorentzians Lorentzians with widths $\Gamma_{nl}$ and areas $K A_{nl}(Q)$ (scale
        times the QISF of that order), built by the base class. The sum is truncated, so a warning
        is raised if it omits more than MISSING_WEIGHT_TOLERANCE of the weight. The created
        collections are installed on the model, so the returned list is the live one.

        Returns
        -------
        list[ComponentCollection]
            One ComponentCollection per Q value.
        """
        collections = super().create_component_collections()
        self._warn_if_truncated()
        return collections

    def get_global_variables(self) -> list[Parameter]:
        """
        Get all global variables from the diffusion model.

        Returns
        -------
        list[Parameter]
            The scale, radius and diffusion coefficient.
        """
        return [self.scale, self.radius, self.diffusion_coefficient]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _radius_in_angstrom(self) -> float:
        """
        Get the radius in angstrom.

        Returns
        -------
        float
            The radius value in angstrom.
        """
        return sc.values(self.radius.full_value).to(unit='angstrom').value

    @staticmethod
    def _weights(QR: np.ndarray) -> np.ndarray:
        r"""
        Calculate the QISF $A_{nl}$ of all 98 roots at once.

        Parameters
        ----------
        QR : np.ndarray
            Q times R, dimensionless, with dims (Q,).

        Returns
        -------
        np.ndarray
            QISF with shape (len(QR), 98), the Lorentzian order j - 1 along the second axis.
        """
        x, angular_order, coefficient, J0, J1 = _root_arrays()
        a = np.asarray(QR, dtype=float)[:, None]
        orders = np.stack([angular_order, angular_order + 1])[:, None, :]  # j_l, j_(l+1) at once
        j_l, j_l1 = spherical_jn(orders, np.broadcast_to(a, (2, a.shape[0], x.size)))
        near = np.abs(a - x) < LIMIT_SWITCH
        with np.errstate(divide='ignore', invalid='ignore'):
            direct = (a * j_l1 - angular_order * j_l) / (a**2 - x**2)
        J = np.where(near, J0 + J1 * (a - x), direct)
        return coefficient * J**2

    def _create_weight(self, Q: float, order: int) -> Parameter:
        r"""
        Create the EISF (order 0) or the QISF of order $j$ as a dimensionless dependent Parameter
        that follows the radius.

        The dependency expressions cannot call scipy, so the spherical Bessel functions are written
        out in elementary functions (``spherical_bessel_expression``). Near the root, where the
        weight is a 0/0, the expression uses the limit and its first-order Taylor term, computed
        here once. Keeping the weight separate from the scale makes updates of the scale cheap.

        Parameters
        ----------
        Q : float
            Scattering vector in 1/angstrom.
        order : int
            The order j; 0 for the delta function (the EISF).

        Raises
        ------
        TypeError
            If Q is not a float.

        Returns
        -------
        Parameter
            The EISF or QISF, which becomes the weight of the component.
        """
        if not isinstance(Q, float):
            raise TypeError('Q must be a float.')

        # x is Q R and y = x**2, the names the Bessel expressions use
        lines = [f'x = {Q!r} * radius_ratio.value', 'y = x**2']
        if order == 0:
            lines += [f'j1 = {spherical_bessel_expression(1)}', 'one * (3 * j1 / x)**2']
        else:
            roots, angular_orders, coefficients, limits, slopes = _root_arrays()
            root = float(roots[order - 1])
            angular_order = int(angular_orders[order - 1])
            limit, slope = float(limits[order - 1]), float(slopes[order - 1])
            direct = f'(x * jl1 - {angular_order} * jl) / (y - {root**2!r})'
            near = f'({limit!r} + {slope!r} * (x - {root!r}))'
            lines += [
                f'jl = {spherical_bessel_expression(angular_order)}',
                f'jl1 = {spherical_bessel_expression(angular_order + 1)}',
                f'J = {near} if abs(x - {root!r}) < {LIMIT_SWITCH!r} else {direct}',
                f'one * {float(coefficients[order - 1])!r} * J**2',
            ]
        with np.errstate(invalid='ignore'):
            return Parameter.from_dependency(
                name=f'weight_j{order}_Q{Q:.2f}',
                dependency_expression='\n'.join(lines),
                dependency_map={'one': self._one, 'radius_ratio': self._radius_over_angstrom},
            )

    def _write_width_dependency_expression(self, order: int) -> str:
        """
        Write the dependency expression for the width of the Lorentzian of a given order.

        The radius enters through its dimensionless value in angstrom, which avoids Core's bound
        arithmetic dividing two infinite upper bounds (a RuntimeWarning on every update).

        Parameters
        ----------
        order : int
            The order j of the Lorentzian.

        Returns
        -------
        str
            Dependency expression for the width.
        """
        x_squared = float(sphere_roots()[order - 1][0] ** 2)
        return f'{x_squared!r} * hbar * D / angstrom**2 / radius_ratio.value**2'

    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map for the width.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """
        return {
            'hbar': self._hbar,
            'D': self.diffusion_coefficient,
            'angstrom': angstrom,
            'radius_ratio': self._radius_over_angstrom,
        }

    def _warn_if_truncated(self) -> None:
        """
        Warn if truncating the sum at number_of_lorentzians terms omits too much weight.

        The warning is raised if the missing weight exceeds MISSING_WEIGHT_TOLERANCE at any of the
        model's Q values. Without Q values there is nothing to check.
        """
        if self.Q is None:
            return
        if np.max(self.calculate_missing_weight(), initial=0.0) > MISSING_WEIGHT_TOLERANCE:
            warnings.warn(
                f'Truncating the sum at number_of_lorentzians={self.number_of_lorentzians} omits '
                f'more than {MISSING_WEIGHT_TOLERANCE} of the spectral weight at some Q values. '
                f'At most {len(sphere_roots())} terms are available (the roots below '
                f'{MAXIMUM_ROOT}); with all of them this happens from QR of about 9.',
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _verify_radius(radius: Numeric) -> None:
        """
        Verify a radius value.

        Parameters
        ----------
        radius : Numeric
            The radius to verify.

        Raises
        ------
        TypeError
            If radius is not a number.
        ValueError
            If radius is below MINIMUM_RADIUS.
        """
        if not isinstance(radius, Numeric):
            raise TypeError('radius must be a number.')
        if float(radius) < MINIMUM_RADIUS:
            raise ValueError(f'radius must be at least {MINIMUM_RADIUS}.')

    @staticmethod
    def _verify_diffusion_coefficient(diffusion_coefficient: Numeric) -> None:
        """
        Verify a diffusion coefficient value.

        Parameters
        ----------
        diffusion_coefficient : Numeric
            The diffusion coefficient to verify.

        Raises
        ------
        TypeError
            If diffusion_coefficient is not a number.
        ValueError
            If diffusion_coefficient is below MINIMUM_DIFFUSION_COEFFICIENT.
        """
        if not isinstance(diffusion_coefficient, Numeric):
            raise TypeError('diffusion_coefficient must be a number.')
        if float(diffusion_coefficient) < MINIMUM_DIFFUSION_COEFFICIENT:
            raise ValueError(
                f'diffusion_coefficient must be at least {MINIMUM_DIFFUSION_COEFFICIENT}.'
            )

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the DiffusionInSphere model.

        Returns
        -------
        str
            String representation of the model.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r},\n'
            f'x_unit={self.x_unit}, y_unit={self.y_unit},\n'
            f'    radius={self.radius},\n'
            f'    diffusion_coefficient={self.diffusion_coefficient},\n'
            f'    number_of_lorentzians={self.number_of_lorentzians},\n'
            f'    scale={self.scale})'
        )
