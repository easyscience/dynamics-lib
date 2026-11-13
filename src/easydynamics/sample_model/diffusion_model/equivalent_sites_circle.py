# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.diffusion_model.delta_lorentzian_diffusion_model_base import (
    DeltaLorentzianDiffusionModelBase,
)
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import hbar

MINIMUM_RADIUS = 1e-10  # A zero radius is unphysical
MINIMUM_WIDTH = 1e-10  # To avoid division by zero
MAXIMUM_NUMBER_OF_SITES = 50


class EquivalentSitesCircle(DeltaLorentzianDiffusionModelBase):
    r"""
    Model of jumps between N equivalent sites on a circle.

    $$ S(Q, E) = A_0(Q)\,\delta(E) + \sum_{l=1}^{N-1} A_l(Q)\, L(E, \Gamma_l) $$

    with

    $$ A_l(Q) = \frac{1}{N} \sum_{k=1}^{N} j_0(Q r_k) \cos\left(\frac{2\pi l k}{N}\right), \quad
    r_k = 2R \sin\left(\frac{\pi k}{N}\right), \quad \Gamma_l = \frac{2\hbar}{\tau}
    \sin^2\left(\frac{\pi l}{N}\right) $$

    where $R$ is the radius of the circle, $\tau$ the residence time on a site, $j_0(x) = \sin(x) /
    x$, and $L(E, \Gamma)$ a Lorentzian with half width at half maximum $\Gamma$. $A_0$ is the EISF
    and $A_l$ ($l \geq 1$) the QISF of order $l$; they sum to one. For $N = 2$ this is the two-site
    jump model, with jump distance $2R$ and $\Gamma_1 = 2\hbar/\tau$. The width is also written as
    $4 \sin^2(\pi l / N) / \tau$, with a $\tau$ that is twice the residence time used here
    (reference 3).

    The radius is fixed by default, since it is normally known from the molecular geometry.

    References
    ----------
    1. J. D. Barnes, J. Chem. Phys. 58, 5193 (1973), doi:10.1063/1.1679130.
    2. M. Bée, Quasielastic Neutron Scattering, Adam Hilger, Bristol (1988).
    3. Mantid, ``InelasticDiffRotDiscreteCircle``, whose ``Decay`` parameter is this $2\tau$,
       https://github.com/mantidproject/mantid/blob/6ec66a6f524a7ec9d4258351fe3001c66c63dbe5/Framework/CurveFitting/src/Functions/InelasticDiffRotDiscreteCircle.cpp#L79-L108.

    Examples
    --------
    **Creating an EquivalentSitesCircle model**

    The model creates one delta function and N - 1 Lorentzians per Q-value:
    ```python
    import numpy as np
    import easydynamics as edyn

    Q = np.linspace(0.5, 2, 7)
    model = edyn.EquivalentSitesCircle(
        scale=1.0,
        radius=1.0,
        residence_time=2.0,
        number_of_sites=3,
        Q=Q,
    )
    component_collections = model.create_component_collections()
    ```
    """

    _order_symbol = 'l'  # components are named '<lorentzian_name>_l<l>'

    def __init__(
        self,
        scale: Numeric = 1.0,
        radius: Numeric = 1.0,
        residence_time: Numeric = 1.0,
        number_of_sites: int = 3,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'EquivalentSitesCircle',
        display_name: str | None = 'EquivalentSitesCircle',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        delta_name: str = 'Delta function',
        delta_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new EquivalentSitesCircle model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        radius : Numeric, default=1.0
            Radius of the circle in angstrom. Must be positive. Fixed by default.
        residence_time : Numeric, default=1.0
            Residence time tau on a site in ps.
        number_of_sites : int, default=3
            Number of equivalent sites N on the circle, between 2 and MAXIMUM_NUMBER_OF_SITES. Set
            at construction; not a fit parameter.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='EquivalentSitesCircle'
            Name of the diffusion model.
        display_name : str | None, default='EquivalentSitesCircle'
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

        Raises
        ------
        TypeError
            If radius or residence_time is not a number, if number_of_sites is not an integer or if
            delta_name or delta_display_name is not a string (or None).
        ValueError
            If radius or residence_time is smaller than its minimum, or if number_of_sites is not
            between 2 and MAXIMUM_NUMBER_OF_SITES.
        """
        self._verify_radius(radius)
        self._verify_residence_time(residence_time)

        if isinstance(number_of_sites, bool) or not isinstance(number_of_sites, (int, np.integer)):
            raise TypeError('number_of_sites must be an integer.')
        if not 2 <= number_of_sites <= MAXIMUM_NUMBER_OF_SITES:
            raise ValueError(f'number_of_sites must be between 2 and {MAXIMUM_NUMBER_OF_SITES}.')
        self._number_of_sites = int(number_of_sites)

        # N sites give N - 1 Lorentzians, of orders l = 1 ... N - 1
        super().__init__(
            self._number_of_sites - 1,
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
        self._residence_time = Parameter(
            name='residence_time',
            value=float(residence_time),
            fixed=False,
            min=MINIMUM_WIDTH,
            unit='ps',
        )

        self._hbar = hbar
        # Dimensionless one, which turns the plain numbers of the weight expressions into
        # (dimensionless) descriptors, as dependency expressions must evaluate to one.
        self._one = DescriptorNumber(name='one', value=1.0)

        # Dimensionless radius/angstrom for the weight dependency expressions. Deriving it through
        # the dependency graph keeps the weights correct if the radius is converted to another
        # unit (e.g. nm).
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
        Get the radius of the circle.

        Returns
        -------
        Parameter
            Radius R.
        """
        return self._radius

    @radius.setter
    def radius(self, radius: Numeric) -> None:
        """
        Set the radius of the circle.

        Parameters
        ----------
        radius : Numeric
            Radius R in the unit of the radius parameter (angstrom by default).
        """
        self._verify_radius(radius)
        self._radius.value = float(radius)

    @property
    def residence_time(self) -> Parameter:
        """
        Get the residence time on a site.

        Returns
        -------
        Parameter
            Residence time tau.
        """
        return self._residence_time

    @residence_time.setter
    def residence_time(self, residence_time: Numeric) -> None:
        """
        Set the residence time on a site.

        Parameters
        ----------
        residence_time : Numeric
            Residence time tau in ps.
        """
        self._verify_residence_time(residence_time)
        self._residence_time.value = float(residence_time)

    @property
    def number_of_sites(self) -> int:
        """
        Get the number of equivalent sites on the circle.

        Returns
        -------
        int
            The number of sites N. Set at construction.
        """
        return self._number_of_sites

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        r"""
        Calculate the half-width at half-maximum (HWHM) of the Lorentzian of a given order.

        The width is $(2\hbar/\tau) \sin^2(\pi l / N)$ and does not depend on Q.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order l of the Lorentzian, between 1 and N - 1.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)

        rate = 2 * self._hbar / self.residence_time
        rate.convert_unit(self.x_unit)
        angle = sc.scalar(np.pi * order / self.number_of_sites, unit='rad')
        return rate.value * sc.sin(angle).value ** 2 * np.ones_like(Q)

    def calculate_EISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF), $A_0(Q)$.

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
        return self._structure_factors(Q)['order', 0].values

    def calculate_QISF_order(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF) of one order, $A_l(Q)$, the
        weight of the Lorentzian of order $l$.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order l of the Lorentzian, between 1 and N - 1.

        Returns
        -------
        np.ndarray
            QISF values of that order (dimensionless).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)
        return self._structure_factors(Q)['order', int(order)].values

    def calculate_QISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the total Quasi-Elastic Incoherent Structure Factor (QISF), the sum of the QISF
        of all N - 1 orders.

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
        return sc.sum(self._structure_factors(Q)['order', 1:], 'order').values

    def get_global_variables(self) -> list[Parameter]:
        """
        Get all global variables from the diffusion model.

        Returns
        -------
        list[Parameter]
            The scale, radius and residence time.
        """
        return [self.scale, self.radius, self.residence_time]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _structure_factors(self, Q: np.ndarray) -> sc.Variable:
        r"""
        Calculate the EISF and the QISF, $A_l(Q)$ for all orders $l = 0 \ldots N - 1$, with Scipp.

        $A_l(Q) = \frac{1}{N} \sum_{k=1}^{N} j_0(Q r_k) \cos(2\pi l k / N)$ with $r_k = 2R\sin(\pi
        k / N)$. The variance of the radius is not propagated.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector in 1/angstrom.

        Returns
        -------
        sc.Variable
            Dimensionless $A_l(Q)$ with dims ``('Q', 'order')``; order 0 is the EISF.
        """
        N = self.number_of_sites
        site = sc.arange('k', 1.0, N + 1.0)
        order = sc.arange('order', 0.0, float(N))
        half_turn = sc.scalar(np.pi, unit='rad')

        radius = sc.values(self.radius.full_value).to(unit='angstrom')
        jump_distance = 2 * radius * sc.sin(half_turn * site / N)
        Q_variable = sc.array(dims=['Q'], values=Q, unit='1/angstrom')

        weights = sc.sum(
            sc.sinc(Q_variable * jump_distance) * sc.cos(2 * half_turn * order * site / N), 'k'
        )
        return (weights / N).transpose(['Q', 'order'])

    def _create_weight(self, Q: float, order: int) -> Parameter:
        r"""
        Create the EISF ($l = 0$) or the QISF of order $l$, $A_l(Q)$, as a dimensionless dependent
        Parameter that follows the radius.

        The dependency expressions cannot call Scipp, so the sum over sites is written out with
        NumPy's sinc, sinc(x) = sin(pi x) / (pi x). Keeping the weight separate from the scale
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
            The weight.
        """
        if not isinstance(Q, float):
            raise TypeError('Q must be a float.')

        N = self.number_of_sites
        terms = []
        for site in range(1, N + 1):
            # float(): NumPy 2 writes np.float64 as 'np.float64(...)', which asteval cannot parse
            coefficient = float(np.cos(2 * np.pi * order * site / N) / N)
            # j0(Q r_k) = sinc(Q r_k / pi) with r_k = 2 R sin(pi k / N)
            argument = float(Q * 2 * np.sin(np.pi * site / N) / np.pi)
            terms.append(f'{coefficient!r} * sinc({argument!r} * radius_ratio.value)')
        expression = f'one * ({" + ".join(terms)})'
        with np.errstate(invalid='ignore'):
            return Parameter.from_dependency(
                name=f'weight_l{order}_Q{Q:.2f}',
                dependency_expression=expression,
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
        factor = float(2 * np.sin(np.pi * order / self.number_of_sites) ** 2)
        return f'{factor!r} * hbar / tau'

    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map for the width.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """
        return {'hbar': self._hbar, 'tau': self.residence_time}

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
    def _verify_residence_time(residence_time: Numeric) -> None:
        """
        Verify a residence time value.

        Parameters
        ----------
        residence_time : Numeric
            The residence time to verify.

        Raises
        ------
        TypeError
            If residence_time is not a number.
        ValueError
            If residence_time is below MINIMUM_WIDTH.
        """
        if not isinstance(residence_time, Numeric):
            raise TypeError('residence_time must be a number.')
        if float(residence_time) < MINIMUM_WIDTH:
            raise ValueError(f'residence_time must be at least {MINIMUM_WIDTH}.')

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the EquivalentSitesCircle model.

        Returns
        -------
        str
            String representation of the model.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r},\n'
            f'x_unit={self.x_unit}, y_unit={self.y_unit},\n'
            f'    number_of_sites={self.number_of_sites},\n'
            f'    radius={self.radius},\n'
            f'    residence_time={self.residence_time},\n'
            f'    scale={self.scale})'
        )
