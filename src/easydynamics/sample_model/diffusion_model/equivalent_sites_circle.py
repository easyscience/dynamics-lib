# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import DeltaLorentzian
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.utils.fit_target import FitTarget
from easydynamics.utils.utils import CANONICAL_Q_UNIT
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import hbar

MINIMUM_RADIUS = 1e-10  # A zero radius is unphysical
MINIMUM_WIDTH = 1e-10  # To avoid division by zero
MAXIMUM_NUMBER_OF_SITES = 50


class EquivalentSitesCircle(DiffusionModelBase):
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
    jump model, with jump distance $2R$ and $\Gamma_1 = 2\hbar/\tau$.

    At each Q-value the model makes one ``DeltaLorentzian`` component. Its area follows the scale,
    its weights follow the radius and its widths follow the residence time, as dependent
    Parameters. The radius is fixed by default, since it is normally known from the molecular
    geometry.

    References
    ----------
    1. J. D. Barnes, J. Chem. Phys. 58, 5193 (1973), doi:10.1063/1.1679130.
    2. M. Bée, Quasielastic Neutron Scattering, Adam Hilger, Bristol (1988).

    Examples
    --------
    **Creating an EquivalentSitesCircle model**

    With N sites the model has N - 1 Lorentzians per Q-value:
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
    component_collections = model.get_component_collections()
    ```
    """

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
            Name of the DeltaLorentzian component. If None, it will be set to the name of the
            diffusion model.
        lorentzian_display_name : str | None, default=None
            Display name of the DeltaLorentzian component. If None, it will be set to
            lorentzian_name.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If number_of_sites is not an integer.
        ValueError
            If number_of_sites is not between 2 and MAXIMUM_NUMBER_OF_SITES.

        Notes
        -----
        A ``TypeError`` or ``ValueError`` also propagates from the checks of radius and
        residence_time if either is not a number or is below its minimum.
        """
        self._verify_radius(radius)
        self._verify_residence_time(residence_time)

        if isinstance(number_of_sites, bool) or not isinstance(number_of_sites, (int, np.integer)):
            raise TypeError('number_of_sites must be an integer.')
        if not 2 <= number_of_sites <= MAXIMUM_NUMBER_OF_SITES:
            raise ValueError(f'number_of_sites must be between 2 and {MAXIMUM_NUMBER_OF_SITES}.')

        super().__init__(
            scale=scale,
            Q=Q,
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            lorentzian_name=lorentzian_name,
            lorentzian_display_name=lorentzian_display_name,
            unique_name=unique_name,
        )

        self._number_of_sites = int(number_of_sites)
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

        # Dimensionless radius/angstrom for the weight expressions. Deriving it through the
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
        return rate.value * np.sin(np.pi * order / self.number_of_sites) ** 2 * np.ones_like(Q)

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

    def create_component_collections(self) -> list[ComponentCollection]:
        """
        Create the ComponentCollections of the model at the current Q values.

        Each collection holds one DeltaLorentzian with N - 1 Lorentzians. Its area follows the
        scale, its weights follow the radius and its widths follow the residence time. The created
        collections are installed on the model, so the returned list is the live one.

        Returns
        -------
        list[ComponentCollection]
            One ComponentCollection per Q value.
        """
        if self.Q is None:
            self._component_collections = []
            return self._component_collections

        collections = []
        for Q_value in self.Q.values:
            Q_value = float(Q_value)
            # easyscience propagates inf bounds through arithmetic, producing inf/inf=nan as a
            # transient intermediate. Python's min/max ignore nan so the final bounds are correct;
            # suppress the spurious numpy RuntimeWarning.
            with np.errstate(invalid='ignore'):
                component = DeltaLorentzian(
                    weights=[
                        self._create_weight(Q_value, order)
                        for order in range(self.number_of_sites)
                    ],
                    widths=[self._create_width(order) for order in range(1, self.number_of_sites)],
                    x_unit=self.x_unit,
                    y_unit=self.y_unit,
                    name=self.lorentzian_name,
                    display_name=self.lorentzian_display_name,
                )
                component.area.make_dependent_on(
                    dependency_expression='scale', dependency_map={'scale': self.scale}
                )

            collection = ComponentCollection(
                name=f'{self.name}_Q{Q_value:.2f}',
                display_name=f'{self.display_name}_Q{Q_value:.2f}',
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )
            collection.append_component(component)
            collections.append(collection)

        self._component_collections = collections
        return self._component_collections

    def get_fit_targets(self) -> list[FitTarget]:
        """
        Get the fittable predictions of the model as FitTargets.

        The model predicts ``'elastic_area'`` (``scale * EISF``) and, for every order from 1 to N -
        1, ``'area_<order>'`` (``scale`` times the QISF of that order) and ``'width_<order>'`` (its
        HWHM). The base class implementation is replaced because there is no single Lorentzian. The
        widths default to the dataset key of the component's width Parameter; the areas have no
        Parameter of their own, so supply their key in the FitBinding.

        Returns
        -------
        list[FitTarget]
            The fittable predictions of this model.
        """
        targets = [
            FitTarget(
                name='elastic_area',
                dataset_key=None,
                function=lambda Q, model=self, **_: model.calculate_EISF(Q) * model.scale.value,
                label=f'{self.display_name} elastic_area',
                x_unit=CANONICAL_Q_UNIT,
                y_unit=str(self.scale.unit),
            )
        ]
        for order in range(1, self.number_of_sites):
            targets.append(
                FitTarget(
                    name=f'area_{order}',
                    dataset_key=None,
                    function=lambda Q, model=self, order=order, **_: (
                        model.calculate_QISF_order(Q, order) * model.scale.value
                    ),
                    label=f'{self.display_name} area_{order}',
                    x_unit=CANONICAL_Q_UNIT,
                    y_unit=str(self.scale.unit),
                )
            )
            targets.append(
                FitTarget(
                    name=f'width_{order}',
                    dataset_key=f'{self.lorentzian_name} width_{order}',
                    function=lambda Q, model=self, order=order, **_: model.calculate_width(
                        Q, order
                    ),
                    label=f'{self.display_name} width_{order}',
                    x_unit=CANONICAL_Q_UNIT,
                    y_unit=str(self.x_unit),
                )
            )
        return targets

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _jump_geometry(self) -> tuple[sc.Variable, sc.Variable]:
        r"""
        Describe the N sites once, for both the direct and the dependent calculation of $A_l(Q)$.

        Site $k$ lies $k$ places away from site $N$ (the starting site, $k = N$ meaning no jump).
        Its chord is $d_k = 2R\sin(\pi k / N)$ and it enters the order-$l$ weight with the phase
        factor $\cos(2\pi l k / N) / N$, so $A_l(Q) = \sum_k j_0(Q d_k)\,\text{phase}_{lk}$.

        Returns
        -------
        tuple[sc.Variable, sc.Variable]
            ``chord``: $d_k / R$ with dims ``('k',)`` for $k = 1 \ldots N$, dimensionless.
            ``phase``: dims ``('order', 'k')`` for $l = 0 \ldots N - 1$, dimensionless.
        """
        N = self.number_of_sites
        site = sc.arange('k', 1.0, N + 1.0)
        order = sc.arange('order', 0.0, float(N))
        turn = sc.scalar(2 * np.pi, unit='rad')

        chord = 2 * sc.sin(turn * site / (2 * N))
        phase = sc.cos(turn * order * site / N) / N
        return chord, phase

    def _structure_factors(self, Q: np.ndarray) -> sc.Variable:
        r"""
        Calculate $A_l(Q)$ for all orders $l = 0 \ldots N - 1$ in one go.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector in 1/angstrom.

        Returns
        -------
        sc.Variable
            Dimensionless $A_l(Q)$ with dims ``('Q', 'order')``; order 0 is the EISF.
        """
        chord, phase = self._jump_geometry()
        # Converting here is what makes the result independent of the unit chosen for the radius.
        radius = sc.scalar(self.radius.value, unit=str(self.radius.unit)).to(unit='angstrom')
        Q_variable = sc.array(dims=['Q'], values=Q, unit=CANONICAL_Q_UNIT)

        overlap = sc.sinc(Q_variable * radius * chord)  # j0(Q d_k), dims (Q, k)
        return sc.sum(overlap * phase, 'k').transpose(['Q', 'order'])

    def _create_weight(self, Q: float, order: int) -> Parameter:
        r"""
        Create $A_l(Q)$ at one Q as a dimensionless dependent Parameter that follows the radius.

        A dependency expression cannot call Scipp, so the same sum as in ``_structure_factors`` is
        written out term by term from the same ``_jump_geometry``. NumPy's ``sinc`` is normalised,
        $\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)$, hence the division by pi.

        Parameters
        ----------
        Q : float
            Scattering vector in 1/angstrom.
        order : int
            The order l; 0 for the delta function (the EISF).

        Returns
        -------
        Parameter
            The weight.
        """
        chord, phase = self._jump_geometry()
        # float(): NumPy 2 writes np.float64 as 'np.float64(...)', which asteval cannot parse
        terms = [
            f'{float(coefficient)!r} * sinc({float(Q * distance / np.pi)!r} * radius_ratio.value)'
            for coefficient, distance in zip(
                phase['order', order].values, chord.values, strict=True
            )
        ]
        return Parameter.from_dependency(
            name=f'{self.lorentzian_name} weight_{order}',
            dependency_expression=f'one * ({" + ".join(terms)})',
            dependency_map={'one': self._one, 'radius_ratio': self._radius_over_angstrom},
        )

    def _create_width(self, order: int) -> Parameter:
        r"""
        Create the width $\Gamma_l = (2\hbar/\tau) \sin^2(\pi l / N)$ as a dependent Parameter that
        follows the residence time.

        Parameters
        ----------
        order : int
            The order l of the Lorentzian, between 1 and N - 1.

        Returns
        -------
        Parameter
            The width, in the x-unit of the model.
        """
        factor = float(2 * np.sin(np.pi * order / self.number_of_sites) ** 2)
        return Parameter.from_dependency(
            name=f'{self.lorentzian_name} width_{order}',
            dependency_expression=f'{factor!r} * hbar / tau',
            dependency_map={'hbar': self._hbar, 'tau': self.residence_time},
            desired_unit=self.x_unit,
        )

    def _verify_order(self, order: int) -> None:
        """
        Verify that order is a valid Lorentzian order.

        Parameters
        ----------
        order : int
            The order to verify.

        Raises
        ------
        TypeError
            If order is not an integer.
        ValueError
            If order is not between 1 and N - 1.
        """
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise TypeError('order must be an integer.')
        if not 1 <= order < self.number_of_sites:
            raise ValueError(f'order must be between 1 and {self.number_of_sites - 1}.')

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
