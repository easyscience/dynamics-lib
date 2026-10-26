# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

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

MINIMUM_RADIUS = 1e-10  # In angstrom; a zero radius is unphysical
MINIMUM_ROTATIONAL_DIFFUSION_COEFFICIENT = 1e-10  # In 1/ps, to keep the Lorentzian widths finite
# Each order adds two dependent Parameters per Q value, which are slow to update in EasyScience
# Core 2.5.1. The Bessel expressions are tested up to order 26.
MAXIMUM_L_MAX = 20
DEFAULT_L_MAX = 6  # Misses less than 1e-3 of the weight up to QR of about 4.3
MISSING_WEIGHT_TOLERANCE = 1e-3  # Truncating the sum above this loss of weight raises a warning


class IsotropicRotationalDiffusion(DeltaLorentzianDiffusionModelBase):
    r"""
    Model of isotropic rotational diffusion of a scatterer on the surface of a sphere.

    The scatterer stays at a fixed distance $R$ from a centre, for example a hydrogen atom in a
    molecule that tumbles freely, and moves over the surface of that sphere by rotational diffusion
    with rotational diffusion coefficient $D_R$. The motion is isotropic, so the scattering depends
    only on the magnitude of $Q$:

    $$ S(Q, E) = A_0(Q)\,\delta(E) + \sum_{l=1}^{l_{max}} A_l(Q)\, L(E, \Gamma_l) $$

    with

    $$ A_0(Q) = j_0^2(QR),\quad A_l(Q) = (2l+1)\, j_l^2(QR),\quad \Gamma_l = l(l+1)\hbar D_R $$

    where $j_l$ are the spherical Bessel functions and $L(E, \Gamma)$ a Lorentzian with half width
    at half maximum $\Gamma$. $A_0$ is the EISF and $A_l$ the QISF of order $l$. The widths do not
    depend on $Q$; only the QISF do.

    The EISF and the QISF of all orders sum to one, so cutting the infinite sum at $l_{max}$ =
    ``l_max`` leaves out some of the weight. A warning is raised if more than 1e-3 of the weight is
    left out at any of the model's Q values, and ``calculate_missing_weight`` returns the weight
    left out. With the default ``l_max`` of 6 this happens from QR of about 4.3, and with the
    largest, 20, from QR of about 17.

    The radius is fixed by default, since it is normally known from the molecular geometry.

    References
    ----------
    1. V. F. Sears, Can. J. Phys. 44, 1299 (1966), doi:10.1139/p66-109.
    2. V. F. Sears, Can. J. Phys. 45, 237 (1967), doi:10.1139/p67-025.
    3. M. Bée, Quasielastic Neutron Scattering, Adam Hilger, Bristol (1988).

    Examples
    --------
    **Creating an IsotropicRotationalDiffusion model**

    The model creates one delta function and ``l_max`` Lorentzians per Q-value:
    ```python
    import numpy as np
    import easydynamics as edyn

    model = edyn.IsotropicRotationalDiffusion(
        radius=2.0,  # angstrom
        rotational_diffusion_coefficient=0.1,  # 1/ps
        Q=np.linspace(0.3, 2, 7),
    )
    ```
    """

    _order_symbol = 'l'  # components are named '<lorentzian_name>_l<l>'

    def __init__(
        self,
        scale: Numeric = 1.0,
        radius: Numeric = 1.0,
        rotational_diffusion_coefficient: Numeric = 1.0,
        l_max: int = DEFAULT_L_MAX,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'IsotropicRotationalDiffusion',
        display_name: str | None = 'IsotropicRotationalDiffusion',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        delta_name: str = 'Delta function',
        delta_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new IsotropicRotationalDiffusion model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        radius : Numeric, default=1.0
            Radius R of the sphere in angstrom. Must be positive. Fixed by default.
        rotational_diffusion_coefficient : Numeric, default=1.0
            Rotational diffusion coefficient D_R in 1/ps. Must be positive.
        l_max : int, default=DEFAULT_L_MAX
            Order at which the sum is cut: the model has l_max Lorentzians per Q-value, of orders 1
            to l_max. Between 1 and MAXIMUM_L_MAX. Set at construction.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='IsotropicRotationalDiffusion'
            Name of the diffusion model.
        display_name : str | None, default='IsotropicRotationalDiffusion'
            Display name of the diffusion model.
        lorentzian_name : str | None, default=None
            Base name of the Lorentzian components. The Lorentzian of order l is named
            '<lorentzian_name>_l<l>'. If None, it will be set to the name of the diffusion model.
        lorentzian_display_name : str | None, default=None
            Base display name of the Lorentzian components. If None, it will be set to
            lorentzian_name.
        delta_name : str, default='Delta function'
            Name of the delta function component.
        delta_display_name : str | None, default=None
            Display name of the delta function component. If None, it will be set to delta_name.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated.
        """
        self._verify_radius(radius)
        self._verify_rotational_diffusion_coefficient(rotational_diffusion_coefficient)
        self._verify_l_max(l_max)

        super().__init__(
            int(l_max),
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

        self._radius = Parameter(
            name='radius',
            value=float(radius),
            fixed=True,
            min=MINIMUM_RADIUS,
            unit='angstrom',
        )
        self._rotational_diffusion_coefficient = Parameter(
            name='rotational_diffusion_coefficient',
            value=float(rotational_diffusion_coefficient),
            fixed=False,
            min=MINIMUM_ROTATIONAL_DIFFUSION_COEFFICIENT,
            unit='1/ps',
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
    def rotational_diffusion_coefficient(self) -> Parameter:
        """
        Get the rotational diffusion coefficient.

        Returns
        -------
        Parameter
            Rotational diffusion coefficient D_R.
        """
        return self._rotational_diffusion_coefficient

    @rotational_diffusion_coefficient.setter
    def rotational_diffusion_coefficient(self, rotational_diffusion_coefficient: Numeric) -> None:
        """
        Set the rotational diffusion coefficient.

        Parameters
        ----------
        rotational_diffusion_coefficient : Numeric
            Rotational diffusion coefficient D_R in the unit of the parameter (1/ps by default).
        """
        self._verify_rotational_diffusion_coefficient(rotational_diffusion_coefficient)
        self._rotational_diffusion_coefficient.value = float(rotational_diffusion_coefficient)

    @property
    def l_max(self) -> int:
        """
        Get the order at which the sum is cut.

        Returns
        -------
        int
            l_max, the number of Lorentzians per Q-value. Set at construction.
        """
        return self.number_of_lorentzians

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        r"""
        Calculate the half-width at half-maximum (HWHM) $\Gamma_l$ of the Lorentzian of order $l$.

        The width is $l(l+1)\hbar D_R$ and does not depend on Q.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order l of the Lorentzian, between 1 and l_max.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)

        width = self._hbar * self.rotational_diffusion_coefficient
        width.convert_unit(self.x_unit)
        return order * (order + 1) * width.value * np.ones_like(Q)

    def calculate_EISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF), $j_0^2(QR)$.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """
        Q = self._ensure_Q(Q)
        return spherical_jn(0, Q * self._radius_in_angstrom()) ** 2

    def calculate_QISF_order(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        r"""
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF) of one order, $A_l(Q)$.

        $A_l(Q) = (2l+1)\, j_l^2(QR)$ is the weight of the Lorentzian of order $l$.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order l of the Lorentzian, between 1 and l_max.

        Returns
        -------
        np.ndarray
            QISF values of that order (dimensionless).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)
        return (2 * order + 1) * spherical_jn(order, Q * self._radius_in_angstrom()) ** 2

    def calculate_QISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the total Quasi-Elastic Incoherent Structure Factor (QISF), the sum of the QISF
        of all l_max orders.

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
        orders = np.arange(1, self.l_max + 1)[:, None]
        QR = Q * self._radius_in_angstrom()
        return np.sum((2 * orders + 1) * spherical_jn(orders, QR) ** 2, axis=0)

    def calculate_missing_weight(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the weight left out by cutting the sum at l_max.

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

        Each collection holds a delta function with area $K A_0(Q)$ (scale times EISF) and l_max
        Lorentzians with widths $\Gamma_l$ and areas $K A_l(Q)$ (scale times the QISF of that
        order), built by the base class. The sum is cut at l_max, so a warning is raised if that
        leaves out more than MISSING_WEIGHT_TOLERANCE of the weight. The created collections are
        installed on the model, so the returned list is the live one.

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
            The scale, radius and rotational diffusion coefficient.
        """
        return [self.scale, self.radius, self.rotational_diffusion_coefficient]

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

    def _create_weight(self, Q: float, order: int) -> Parameter:
        r"""
        Create the EISF (order 0) or the QISF of order $l$ as a dimensionless dependent Parameter
        that follows the radius.

        The dependency expressions cannot call scipy, so $j_l$ is written out in elementary
        functions (``spherical_bessel_expression``). Keeping the weight separate from the scale
        makes updates of the scale cheap.

        Parameters
        ----------
        Q : float
            Scattering vector in 1/angstrom.
        order : int
            The order l; 0 for the delta function (the EISF).

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

        # x is Q R and y = x**2, the names the Bessel expression uses
        lines = [
            f'x = {Q!r} * radius_ratio.value',
            'y = x**2',
            f'jl = {spherical_bessel_expression(order)}',
            f'one * {2 * order + 1} * jl**2',
        ]
        with np.errstate(invalid='ignore'):
            return Parameter.from_dependency(
                name=f'weight_l{order}_Q{Q:.2f}',
                dependency_expression='\n'.join(lines),
                dependency_map={'one': self._one, 'radius_ratio': self._radius_over_angstrom},
            )

    def _write_width_dependency_expression(self, order: int) -> str:
        """
        Write the dependency expression for the width of the Lorentzian of a given order.

        Parameters
        ----------
        order : int
            The order l of the Lorentzian.

        Returns
        -------
        str
            Dependency expression for the width.
        """
        return f'{order * (order + 1)} * hbar * D_R'

    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map for the width.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """
        return {'hbar': self._hbar, 'D_R': self.rotational_diffusion_coefficient}

    def _warn_if_truncated(self) -> None:
        """
        Warn if cutting the sum at l_max leaves out too much weight.

        The warning is raised if the missing weight exceeds MISSING_WEIGHT_TOLERANCE at any of the
        model's Q values. Without Q values there is nothing to check.
        """
        if self.Q is None:
            return
        if np.max(self.calculate_missing_weight(), initial=0.0) > MISSING_WEIGHT_TOLERANCE:
            warnings.warn(
                f'Truncating the sum at l_max={self.l_max} omits more than '
                f'{MISSING_WEIGHT_TOLERANCE} of the spectral weight at some Q values. '
                f'Increase l_max (at most {MAXIMUM_L_MAX}).',
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
    def _verify_rotational_diffusion_coefficient(
        rotational_diffusion_coefficient: Numeric,
    ) -> None:
        """
        Verify a rotational diffusion coefficient value.

        Parameters
        ----------
        rotational_diffusion_coefficient : Numeric
            The rotational diffusion coefficient to verify.

        Raises
        ------
        TypeError
            If rotational_diffusion_coefficient is not a number.
        ValueError
            If rotational_diffusion_coefficient is below MINIMUM_ROTATIONAL_DIFFUSION_COEFFICIENT.
        """
        if not isinstance(rotational_diffusion_coefficient, Numeric):
            raise TypeError('rotational_diffusion_coefficient must be a number.')
        if float(rotational_diffusion_coefficient) < MINIMUM_ROTATIONAL_DIFFUSION_COEFFICIENT:
            raise ValueError(
                'rotational_diffusion_coefficient must be at least '
                f'{MINIMUM_ROTATIONAL_DIFFUSION_COEFFICIENT}.'
            )

    @staticmethod
    def _verify_l_max(l_max: int) -> None:
        """
        Verify an l_max value.

        Parameters
        ----------
        l_max : int
            The order at which to cut the sum.

        Raises
        ------
        TypeError
            If l_max is not an integer.
        ValueError
            If l_max is not between 1 and MAXIMUM_L_MAX.
        """
        if isinstance(l_max, bool) or not isinstance(l_max, (int, np.integer)):
            raise TypeError('l_max must be an integer.')
        if not 1 <= l_max <= MAXIMUM_L_MAX:
            raise ValueError(f'l_max must be between 1 and {MAXIMUM_L_MAX}.')

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the IsotropicRotationalDiffusion model.

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
            f'    rotational_diffusion_coefficient={self.rotational_diffusion_coefficient},\n'
            f'    l_max={self.l_max},\n'
            f'    scale={self.scale})'
        )
