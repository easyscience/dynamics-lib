# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import math
import warnings

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.diffusion_model.delta_lorentzian_diffusion_model_base import (
    DeltaLorentzianDiffusionModelBase,
)
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import hbar

MINIMUM_DIFFUSION_COEFFICIENT = 1e-20  # In m^2/s, to keep the Lorentzian widths finite
MINIMUM_MEAN_U_SQUARED = 1e-10  # In angstrom^2; the widths scale with 1/<u^2>
MAXIMUM_NUMBER_OF_LORENTZIANS = 100  # As in QENSmodels
MISSING_WEIGHT_TOLERANCE = 1e-3  # Truncating the sum above this loss of weight raises a warning


class GaussianModel3D(DeltaLorentzianDiffusionModelBase):
    r"""
    Model of diffusion inside an isotropic three-dimensional harmonic well.

    A particle diffuses with diffusion coefficient $D$ in a harmonic potential that is the same in
    all three directions, so its position about the centre of the well has a Gaussian distribution.
    $\langle u^2 \rangle$ is the mean square distance from the centre, summed over the three
    directions; along any one direction it is $\langle u^2 \rangle / 3$. Because the well is
    isotropic, the scattering depends only on the magnitude of $Q$:

    $$ S(Q, E) = A_0(Q)\,\delta(E) + \sum_{i=1}^{n} A_i(Q)\, L(E, \Gamma_i) $$

    with

    $$ A_i(Q) = e^{-y} \frac{y^i}{i!}, \quad y = \frac{Q^2 \langle u^2 \rangle}{3}, \quad \Gamma_i
    = \frac{3 i \hbar D}{\langle u^2 \rangle} $$

    where $L(E, \Gamma)$ is a Lorentzian with half width at half maximum $\Gamma$. $A_0 = e^{-y}$
    is the EISF, the same Debye-Waller factor as in ``DeltaLorentz``, and $A_i$ is the QISF of
    order $i$. The widths $\Gamma_i$ grow linearly with $i$ and do not depend on $Q$; only the QISF
    do. The paper writes the model with the variance along one direction, $\langle u_x^2 \rangle =
    \langle u^2 \rangle / 3$.

    The QISF form a Poisson distribution in $i$ with mean $y$, and together with the EISF they sum
    to one. The infinite sum is therefore truncated at $n$ = ``number_of_lorentzians``, which must
    be well above $y$ at the largest $Q$. A warning is raised if the truncation leaves out more
    than 1e-3 of the total weight at any of the model's Q values, and ``calculate_missing_weight``
    returns the weight left out.

    References
    ----------
    F. Volino, J.-C. Perrin and S. Lyonnard, J. Phys. Chem. B 110, 11217 (2006),
    doi:10.1021/jp061103s.

    Examples
    --------
    **Creating a GaussianModel3D model**

    The model creates one delta function and ``number_of_lorentzians`` Lorentzians per Q-value:
    ```python
    import numpy as np
    import easydynamics as edyn

    Q = np.linspace(0.5, 2, 7)
    model = edyn.GaussianModel3D(
        scale=1.0,
        diffusion_coefficient=2e-9,  # m^2/s
        mean_u_squared=1.5,  # angstrom^2
        number_of_lorentzians=20,
        Q=Q,
    )
    component_collections = model.create_component_collections()
    ```
    """

    _order_symbol = 'i'  # components are named '<lorentzian_name>_i<i>'

    def __init__(
        self,
        scale: Numeric = 1.0,
        diffusion_coefficient: Numeric = 1.0,
        mean_u_squared: Numeric = 1.0,
        number_of_lorentzians: int = 20,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'GaussianModel3D',
        display_name: str | None = 'GaussianModel3D',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        delta_name: str = 'Delta function',
        delta_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new GaussianModel3D model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        diffusion_coefficient : Numeric, default=1.0
            Diffusion coefficient D in m^2/s.
        mean_u_squared : Numeric, default=1.0
            Mean square displacement <u^2> in angstrom^2, three times Volino's <u_x^2>.
        number_of_lorentzians : int, default=20
            Number of Lorentzians n per Q-value, i.e. the order at which the sum is truncated,
            between 1 and MAXIMUM_NUMBER_OF_LORENTZIANS. Set at construction.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='GaussianModel3D'
            Name of the diffusion model.
        display_name : str | None, default='GaussianModel3D'
            Display name of the diffusion model.
        lorentzian_name : str | None, default=None
            Base name of the Lorentzian components. The Lorentzian of order i is named
            '<lorentzian_name>_i<i>'. If None, it will be set to the name of the diffusion model.
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
            If diffusion_coefficient or mean_u_squared is not a number, if number_of_lorentzians is
            not an integer or if delta_name or delta_display_name is not a string (or None).
        ValueError
            If diffusion_coefficient or mean_u_squared is smaller than its minimum, or if
            number_of_lorentzians is not between 1 and MAXIMUM_NUMBER_OF_LORENTZIANS.
        """
        self._verify_diffusion_coefficient(diffusion_coefficient)
        self._verify_mean_u_squared(mean_u_squared)

        if isinstance(number_of_lorentzians, bool) or not isinstance(
            number_of_lorentzians, (int, np.integer)
        ):
            raise TypeError('number_of_lorentzians must be an integer.')
        if not 1 <= number_of_lorentzians <= MAXIMUM_NUMBER_OF_LORENTZIANS:
            raise ValueError(
                f'number_of_lorentzians must be between 1 and {MAXIMUM_NUMBER_OF_LORENTZIANS}.'
            )

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

        self._diffusion_coefficient = Parameter(
            name='diffusion_coefficient',
            value=float(diffusion_coefficient),
            fixed=False,
            min=MINIMUM_DIFFUSION_COEFFICIENT,
            unit='m**2/s',
        )
        self._mean_u_squared = Parameter(
            name='mean_u_squared',
            value=float(mean_u_squared),
            fixed=False,
            min=MINIMUM_MEAN_U_SQUARED,
            unit='angstrom**2',
        )

        self._hbar = hbar
        # Dimensionless one, which turns the plain numbers of the weight expressions into
        # (dimensionless) descriptors, as dependency expressions must evaluate to one.
        self._one = DescriptorNumber(name='one', value=1.0)

        # Dimensionless <u^2>/angstrom^2 for the weight dependency expressions. Deriving it
        # through the dependency graph keeps the weights correct if <u^2> is converted to another
        # unit (e.g. nm^2).
        with np.errstate(invalid='ignore'):
            self._mean_u_squared_over_angstrom_squared = Parameter.from_dependency(
                name='mean_u_squared_over_angstrom_squared',
                dependency_expression='mean_u_squared / angstrom**2',
                dependency_map={'mean_u_squared': self._mean_u_squared, 'angstrom': angstrom},
            )
            self._mean_u_squared_over_angstrom_squared.set_desired_unit('dimensionless')

        self.create_component_collections()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
            Diffusion coefficient D in m^2/s.
        """
        self._verify_diffusion_coefficient(diffusion_coefficient)
        self._diffusion_coefficient.value = float(diffusion_coefficient)

    @property
    def mean_u_squared(self) -> Parameter:
        """
        Get the mean square displacement.

        Returns
        -------
        Parameter
            Mean square displacement <u^2>.
        """
        return self._mean_u_squared

    @mean_u_squared.setter
    def mean_u_squared(self, mean_u_squared: Numeric) -> None:
        """
        Set the mean square displacement.

        Parameters
        ----------
        mean_u_squared : Numeric
            Mean square displacement <u^2> in the unit of the parameter (angstrom^2 by default).
        """
        self._verify_mean_u_squared(mean_u_squared)
        self._mean_u_squared.value = float(mean_u_squared)

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        r"""
        Calculate the half-width at half-maximum (HWHM) of the Lorentzian of a given order.

        The width is $3 i \hbar D / \langle u^2 \rangle$ and does not depend on Q.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order i of the Lorentzian, between 1 and number_of_lorentzians.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)

        width = (
            3
            * sc.values(self._hbar.full_value)
            * sc.values(self.diffusion_coefficient.full_value)
            / sc.values(self.mean_u_squared.full_value)
        ).to(unit=self.x_unit)
        return order * width.value * np.ones_like(Q)

    def calculate_EISF(self, Q: Q_type | None = None) -> np.ndarray:
        r"""
        Calculate the Elastic Incoherent Structure Factor (EISF), $\exp(-Q^2\langle u^2\rangle/3)$.

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
        return sc.exp(-self._y(Q)).values

    def calculate_QISF_order(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF) of one order, $A_i(Q) =
        e^{-y} y^i / i!$, the weight of the Lorentzian of order $i$.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order i of the Lorentzian, between 1 and number_of_lorentzians.

        Returns
        -------
        np.ndarray
            QISF values of that order (dimensionless).
        """
        self._verify_order(order)
        Q = self._ensure_Q(Q)
        return self._quasielastic_weights(Q)['order', int(order) - 1].values

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
        return sc.sum(self._quasielastic_weights(Q), 'order').values

    def calculate_missing_weight(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the spectral weight lost by truncating the sum at number_of_lorentzians.

        The weights of all orders sum to one, so this is 1 - EISF - QISF. If it is not small at the
        Q values of interest, increase number_of_lorentzians.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            The omitted weight (dimensionless).
        """
        Q = self._ensure_Q(Q)
        return 1.0 - self.calculate_EISF(Q) - self.calculate_QISF(Q)

    def create_component_collections(self) -> list[ComponentCollection]:
        r"""
        Create ComponentCollections for the model at the current Q values.

        Each collection holds a delta function with area $\mathrm{scale}\, A_0(Q)$ and
        number_of_lorentzians Lorentzians with widths $\Gamma_i$ and areas $\mathrm{scale}\,
        A_i(Q)$ (built by the base class). The sum over i is truncated, so a warning is raised if
        it omits more than MISSING_WEIGHT_TOLERANCE of the weight. The created collections are
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
            The scale, diffusion coefficient and mean square displacement.
        """
        return [self.scale, self.diffusion_coefficient, self.mean_u_squared]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _y(self, Q: np.ndarray) -> sc.Variable:
        r"""
        Calculate $y = Q^2 \langle u^2 \rangle / 3$ with Scipp.

        The variance of <u^2> is not propagated.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector in 1/angstrom.

        Returns
        -------
        sc.Variable
            Dimensionless y with dims ``('Q',)``.
        """
        Q_variable = sc.array(dims=['Q'], values=Q, unit='1/angstrom')
        mean_u_squared = sc.values(self.mean_u_squared.full_value)
        return (Q_variable**2 * mean_u_squared / 3).to(unit='dimensionless')

    def _quasielastic_weights(self, Q: np.ndarray) -> sc.Variable:
        r"""
        Calculate the QISF $A_i = e^{-y} y^i / i!$ of the orders $i = 1 \ldots n$ with Scipp.

        Computed in log space, $\exp(-y + i \ln y - \ln i!)$, so that neither $y^i$ nor $i!$
        overflows. Scipp has no factorial, so $\ln i!$ is the cumulative sum of $\ln k$. At $Q = 0$
        the logarithm is $-\infty$ and the weights are exactly 0.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector in 1/angstrom.

        Returns
        -------
        sc.Variable
            Dimensionless QISF with dims ``('Q', 'order')``.
        """
        order = sc.arange('order', 1.0, self.number_of_lorentzians + 1.0)
        log_factorial = sc.cumsum(sc.log(order), 'order')
        y = self._y(Q)
        weights = sc.exp(-y + order * sc.log(y) - log_factorial)
        return weights.transpose(['Q', 'order'])

    def _create_weight(self, Q: float, order: int) -> Parameter:
        r"""
        Create the EISF ($i = 0$) or the QISF of order $i$, $A_i(Q)$, as a dimensionless dependent
        Parameter that follows $\langle u^2 \rangle$.

        The dependency expressions cannot call Scipp, so the weight is written out in the same
        log-space form as ``_quasielastic_weights``. Keeping the weight separate from the scale
        makes updates of the scale cheap.

        Parameters
        ----------
        Q : float
            Scattering vector in 1/angstrom.
        order : int
            The order i; 0 for the delta function (the EISF).

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

        # float(): NumPy 2 writes np.float64 as 'np.float64(...)', which asteval cannot parse
        y = f'{float(Q**2 / 3)!r} * u2.value'
        if order == 0:
            expression = f'one * exp(-{y})'
        else:
            log_factorial = math.lgamma(order + 1)
            expression = f'y = {y}\none * exp(-y + {order} * log(y) - {log_factorial!r})'
        with np.errstate(invalid='ignore'):
            return Parameter.from_dependency(
                name=f'weight_i{order}_Q{Q:.2f}',
                dependency_expression=expression,
                dependency_map={
                    'one': self._one,
                    'u2': self._mean_u_squared_over_angstrom_squared,
                },
            )

    def _write_width_dependency_expression(self, order: int) -> str:
        """
        Write the dependency expression for the width of the Lorentzian of a given order.

        Parameters
        ----------
        order : int
            The order i of the Lorentzian.

        Returns
        -------
        str
            Dependency expression for the width.
        """
        # Dividing by the dimensionless value of <u^2>/angstrom^2 instead of by the <u^2>
        # Parameter avoids inf/inf in Core's bound arithmetic (D and <u^2> are both unbounded
        # above), which warns on every update during a fit.
        return f'{3 * order} * hbar * D / angstrom**2 / u2.value'

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
            'u2': self._mean_u_squared_over_angstrom_squared,
        }

    def _warn_if_truncated(self) -> None:
        """
        Warn if truncating the sum at number_of_lorentzians omits too much weight.

        The warning is raised if the missing weight exceeds MISSING_WEIGHT_TOLERANCE at any of the
        model's Q values. Without Q values there is nothing to check.
        """
        if self.Q is None:
            return
        if np.max(self.calculate_missing_weight(), initial=0.0) > MISSING_WEIGHT_TOLERANCE:
            warnings.warn(
                f'Truncating the sum at number_of_lorentzians={self.number_of_lorentzians} omits '
                f'more than {MISSING_WEIGHT_TOLERANCE} of the spectral weight at some Q values. '
                'Increase number_of_lorentzians.',
                UserWarning,
                stacklevel=3,
            )

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

    @staticmethod
    def _verify_mean_u_squared(mean_u_squared: Numeric) -> None:
        """
        Verify a mean square displacement value.

        Parameters
        ----------
        mean_u_squared : Numeric
            The mean square displacement to verify.

        Raises
        ------
        TypeError
            If mean_u_squared is not a number.
        ValueError
            If mean_u_squared is below MINIMUM_MEAN_U_SQUARED.
        """
        if not isinstance(mean_u_squared, Numeric):
            raise TypeError('mean_u_squared must be a number.')
        if float(mean_u_squared) < MINIMUM_MEAN_U_SQUARED:
            raise ValueError(f'mean_u_squared must be at least {MINIMUM_MEAN_U_SQUARED}.')

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the GaussianModel3D model.

        Returns
        -------
        str
            String representation of the model.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r},\n'
            f'x_unit={self.x_unit}, y_unit={self.y_unit},\n'
            f'    number_of_lorentzians={self.number_of_lorentzians},\n'
            f'    diffusion_coefficient={self.diffusion_coefficient},\n'
            f'    mean_u_squared={self.mean_u_squared},\n'
            f'    scale={self.scale})'
        )
