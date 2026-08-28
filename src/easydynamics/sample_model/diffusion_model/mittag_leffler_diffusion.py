# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import DiffusionDampedMittagLeffler
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.components.diffusion_damped_mittag_leffler import MAXIMUM_ALPHA
from easydynamics.sample_model.components.diffusion_damped_mittag_leffler import MINIMUM_ALPHA
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.utils.fit_target import FitTarget
from easydynamics.utils.utils import CANONICAL_Q_UNIT
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import convert_parameter_unit
from easydynamics.utils.utils import hbar
from easydynamics.utils.utils import verify_Q_index

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class MittagLefflerDiffusion(DiffusionModelBase):
    r"""
    Multiscale relaxation model of Hassani et al.: Mittag-Leffler internal dynamics damped by
    global translational diffusion.

    A. N. Hassani, L. Haris, M. Appel, T. Seydel, A. M. Stadler and G. R. Kneller, *Multiscale
    relaxation dynamics and diffusion of myelin basic protein in solution studied by quasielastic
    neutron scattering*, J. Chem. Phys. **156**, 025102 (2022),
    [doi:10.1063/5.0077100](https://doi.org/10.1063/5.0077100).

    The intermediate scattering function is written as an internal relaxation, described by a
    Mittag-Leffler function, times a global diffusion factor (their Eqs. 38-39),

    $$ F^{(+)}(t) = K e^{-\epsilon |t|} \left( EISF + (1 - EISF) E_\alpha(-(|t|/\tau_R)^\alpha)
    \right), \qquad \epsilon = \hbar D q^2 $$

    whose Fourier transform is their Eq. (41),

    $$ S^{(+)}(x) = K \left[ EISF \frac{1}{\pi} \frac{\epsilon}{x^2 + \epsilon^2}
    + (1 - EISF) \tilde{\phi}^{(\epsilon)}_{ML}(|x|) \right]. $$

    This model builds that sum at every Q as a
    [`ComponentCollection`][easydynamics.sample_model.ComponentCollection] of two components:

    - a [`Lorentzian`][easydynamics.sample_model.Lorentzian] carrying the elastic term, with area
      $K \cdot EISF$ and half width at half maximum $\epsilon = \hbar D q^2$. Note that the elastic
      line is a Lorentzian and not a delta function, because global diffusion broadens it.
    - a
      [`DiffusionDampedMittagLeffler`][easydynamics.sample_model.DiffusionDampedMittagLeffler]
      carrying the quasi-elastic term, with scale $K \cdot (1 - EISF)$, damping $\epsilon = \hbar D
      q^2$, width $\hbar/\tau_R$ and form parameter $\alpha$.

    The diffusion coefficient $D$ is global: it is the single parameter that ties the Q values
    together, exactly as in
    [`BrownianTranslationalDiffusion`][easydynamics.sample_model.BrownianTranslationalDiffusion].
    In the paper $\tau_R$, $\alpha$ and the EISF are instead fitted independently at every Q (their
    Fig. 5); pass ``allow_Q_variation={'A_0': True, 'relaxation_rate': True, 'alpha': True}`` to
    reproduce that. Q is assumed to be in 1/angstrom and $D$ in m^2/s.

    Examples
    --------
    **Creating a MittagLefflerDiffusion model with the paper's per-Q parameters**

    ```python
    import numpy as np
    import easydynamics as edyn

    Q = np.linspace(0.8, 1.8, 6)
    model = edyn.MittagLefflerDiffusion(
        scale=1.0,
        diffusion_coefficient=3.3e-11,
        A_0=0.05,
        relaxation_rate=0.02,
        alpha=0.85,
        allow_Q_variation={'A_0': True, 'relaxation_rate': True, 'alpha': True},
        Q=Q,
    )
    component_collections = model.get_component_collections()
    ```

    See also the tutorials.
    """

    def __init__(
        self,
        scale: Numeric = 1.0,
        diffusion_coefficient: Numeric = 1.0,
        A_0: Numeric = 0.0,
        relaxation_rate: Numeric = 1.0,
        alpha: Numeric = 1.0,
        allow_Q_variation: dict | None = None,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'MittagLefflerDiffusion',
        display_name: str | None = None,
        lorentzian_name: str = 'Elastic Lorentzian',
        lorentzian_display_name: str | None = None,
        mittag_leffler_name: str = 'Mittag-Leffler',
        mittag_leffler_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new MittagLefflerDiffusion model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor K for the model. Must be non-negative. Its unit is ``x_unit * y_unit``.
        diffusion_coefficient : Numeric, default=1.0
            Global translational diffusion coefficient D in m^2/s. Sets the diffusion damping
            ``epsilon = hbar * D * Q**2`` shared by both components. Must be non-negative.
        A_0 : Numeric, default=0.0
            Elastic incoherent structure factor (EISF), the elastic fraction of the intensity. Must
            lie in [0, 1]. The paper finds it close to zero over most of its Q range.
        relaxation_rate : Numeric, default=1.0
            Mittag-Leffler relaxation rate ``hbar / tau_R`` in x_unit. Must be strictly positive.
        alpha : Numeric, default=1.0
            Form parameter of the Mittag-Leffler relaxation function. Must lie in (0, 1]; alpha=1
            reduces the quasi-elastic term to a Lorentzian.
        allow_Q_variation : dict | None, default=None
            Dict describing which of ``'A_0'``, ``'relaxation_rate'`` and ``'alpha'`` are free at
            every Q instead of shared, with boolean values. If None, none of them vary with Q. The
            paper's analysis corresponds to all three being True.
        Q : Q_type | None, default=None
            Q values for the model in 1/angstrom. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='MittagLefflerDiffusion'
            Name of the diffusion model.
        display_name : str | None, default=None
            Display name of the diffusion model.
        lorentzian_name : str, default='Elastic Lorentzian'
            Name of the elastic Lorentzian component.
        lorentzian_display_name : str | None, default=None
            Display name of the elastic Lorentzian component. If None, it falls back to
            *lorentzian_name*.
        mittag_leffler_name : str, default='Mittag-Leffler'
            Name of the Mittag-Leffler component.
        mittag_leffler_display_name : str | None, default=None
            Display name of the Mittag-Leffler component. If None, it falls back to
            *mittag_leffler_name*.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If mittag_leffler_name is not a string, or if mittag_leffler_display_name is not a
            string or None.
        """
        super().__init__(
            scale=scale,
            x_unit=x_unit,
            y_unit=y_unit,
            Q=Q,
            lorentzian_name=lorentzian_name,
            lorentzian_display_name=lorentzian_display_name,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        self._hbar = hbar
        self._angstrom = angstrom

        self._diffusion_coefficient = self._create_diffusion_coefficient_parameter(
            diffusion_coefficient
        )
        self._A_0, self._A_1 = self._create_A0_A1_parameters(A_0)
        self._relaxation_rate = self._create_relaxation_rate_parameter(relaxation_rate)
        self._alpha = self._create_alpha_parameter(alpha)

        # --------------------------------------------------------------
        # names
        # --------------------------------------------------------------
        if not isinstance(mittag_leffler_name, str):
            raise TypeError('mittag_leffler_name must be a string.')

        if mittag_leffler_display_name is None:
            mittag_leffler_display_name = mittag_leffler_name

        if not isinstance(mittag_leffler_display_name, str):
            raise TypeError('mittag_leffler_display_name must be a string or None.')

        self._mittag_leffler_name = mittag_leffler_name
        self._mittag_leffler_display_name = mittag_leffler_display_name

        # --------------------------------------------------------------
        # Q variation
        # --------------------------------------------------------------
        self._allow_Q_variation = self._create_Q_variation_dict(allow_Q_variation)

        # create_component_collections creates the per-Q parameter lists itself, so the
        # components it builds are backed by the very parameters stored in those lists.
        self.create_component_collections()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def diffusion_coefficient(self) -> Parameter:
        """
        Get the global diffusion coefficient parameter D.

        Returns
        -------
        Parameter
            Diffusion coefficient D in m^2/s.
        """
        return self._diffusion_coefficient

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, diffusion_coefficient: Numeric) -> None:
        """
        Set the global diffusion coefficient parameter D.

        Parameters
        ----------
        diffusion_coefficient : Numeric
            The new value for D in m^2/s.

        Raises
        ------
        TypeError
            If diffusion_coefficient is not a number.
        ValueError
            If diffusion_coefficient is negative.
        """
        if not isinstance(diffusion_coefficient, Numeric):
            raise TypeError('diffusion_coefficient must be a number.')
        if float(diffusion_coefficient) < 0:
            raise ValueError('diffusion_coefficient must be non-negative.')
        self._diffusion_coefficient.value = float(diffusion_coefficient)

    @property
    def A_0(self) -> Parameter:
        """
        Get the elastic fraction parameter A_0 (the EISF).

        Returns
        -------
        Parameter
            The dimensionless A_0 parameter, bounded to [0, 1].
        """
        return self._A_0

    @A_0.setter
    def A_0(self, A_0: Numeric) -> None:
        """
        Set the elastic fraction parameter A_0.

        Parameters
        ----------
        A_0 : Numeric
            The new value for A_0. Must lie in [0, 1].

        Raises
        ------
        TypeError
            If A_0 is not a number.
        ValueError
            If A_0 is not between 0 and 1.
        """
        if not isinstance(A_0, Numeric):
            raise TypeError('A_0 must be a number.')
        if float(A_0) < 0 or float(A_0) > 1:
            raise ValueError('A_0 must be between 0 and 1.')
        self._A_0.value = float(A_0)

    @property
    def A_1(self) -> Parameter:
        """
        Get the quasi-elastic fraction parameter A_1 = 1 - A_0.

        Returns
        -------
        Parameter
            The dependent A_1 parameter.
        """
        return self._A_1

    @A_1.setter
    def A_1(self, _A_1: Numeric) -> None:
        """
        Reject assignment to the dependent A_1 parameter.

        Parameters
        ----------
        _A_1 : Numeric
            Ignored.

        Raises
        ------
        AttributeError
            Always; A_1 is derived from A_0 and must be changed through A_0.
        """
        raise AttributeError('A_1 is derived from A_0 and cannot be set directly. Set A_0.')

    @property
    def relaxation_rate(self) -> Parameter:
        """
        Get the Mittag-Leffler relaxation rate parameter hbar / tau_R.

        Returns
        -------
        Parameter
            The relaxation rate parameter with unit ``x_unit``.
        """
        return self._relaxation_rate

    @relaxation_rate.setter
    def relaxation_rate(self, relaxation_rate: Numeric) -> None:
        """
        Set the Mittag-Leffler relaxation rate parameter.

        Parameters
        ----------
        relaxation_rate : Numeric
            The new relaxation rate in x_unit. Must be strictly positive.

        Raises
        ------
        TypeError
            If relaxation_rate is not a number.
        ValueError
            If relaxation_rate is smaller than the minimum width.
        """
        if not isinstance(relaxation_rate, Numeric):
            raise TypeError('relaxation_rate must be a number.')
        if float(relaxation_rate) < MINIMUM_WIDTH:
            raise ValueError(f'relaxation_rate must be at least {MINIMUM_WIDTH}.')
        self._relaxation_rate.value = float(relaxation_rate)

    @property
    def alpha(self) -> Parameter:
        """
        Get the Mittag-Leffler form parameter alpha.

        Returns
        -------
        Parameter
            The dimensionless alpha parameter, bounded to (0, 1].
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: Numeric) -> None:
        """
        Set the Mittag-Leffler form parameter alpha.

        Parameters
        ----------
        alpha : Numeric
            The new form parameter. Must lie in (0, 1].

        Raises
        ------
        TypeError
            If alpha is not a number.
        ValueError
            If alpha does not lie in (0, 1].
        """
        if not isinstance(alpha, Numeric):
            raise TypeError('alpha must be a number.')
        if not MINIMUM_ALPHA <= float(alpha) <= MAXIMUM_ALPHA:
            raise ValueError('alpha must be greater than zero and at most one.')
        self._alpha.value = float(alpha)

    @property
    def mittag_leffler_name(self) -> str:
        """
        Get the name of the Mittag-Leffler component.

        Returns
        -------
        str
            Name of the Mittag-Leffler component.
        """
        return self._mittag_leffler_name

    @mittag_leffler_name.setter
    def mittag_leffler_name(self, mittag_leffler_name: str) -> None:
        """
        Set the name of the Mittag-Leffler component.

        Parameters
        ----------
        mittag_leffler_name : str
            The new name for the Mittag-Leffler component.

        Raises
        ------
        TypeError
            If mittag_leffler_name is not a string.
        """
        if not isinstance(mittag_leffler_name, str):
            raise TypeError('mittag_leffler_name must be a string.')
        self._mittag_leffler_name = mittag_leffler_name

    @property
    def mittag_leffler_display_name(self) -> str | None:
        """
        Get the display name of the Mittag-Leffler component.

        Returns
        -------
        str | None
            Display name of the Mittag-Leffler component, or None if not set.
        """
        return self._mittag_leffler_display_name

    @mittag_leffler_display_name.setter
    def mittag_leffler_display_name(self, mittag_leffler_display_name: str | None) -> None:
        """
        Set the display name of the Mittag-Leffler component.

        Parameters
        ----------
        mittag_leffler_display_name : str | None
            The new display name for the Mittag-Leffler component.

        Raises
        ------
        TypeError
            If mittag_leffler_display_name is not a string or None.
        """
        if not isinstance(mittag_leffler_display_name, (str, type(None))):
            raise TypeError('mittag_leffler_display_name must be a string or None.')
        self._mittag_leffler_display_name = mittag_leffler_display_name

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the diffusion damping epsilon = hbar * D * Q**2.

        This is both the half width at half maximum of the elastic Lorentzian and the ``damping``
        of the Mittag-Leffler component; the two share it by construction.

        Parameters
        ----------
        Q : Q_type, default=None
            Scattering vector in 1/angstrom. If None, the Q stored in the model is used.

        Returns
        -------
        np.ndarray
            Damping values in the unit of the model (e.g. meV).
        """
        Q = self._ensure_Q(Q)

        unit_conversion_factor = self._hbar * self.diffusion_coefficient / (self._angstrom**2)
        unit_conversion_factor.convert_unit(self.x_unit)
        return Q**2 * unit_conversion_factor.value

    def calculate_relaxation_rate(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the Mittag-Leffler relaxation rate hbar / tau_R at each Q.

        If the relaxation rate is allowed to vary with Q, the requested Q values are matched
        against the Q stored in the model and the corresponding per-Q rates are returned. Otherwise
        the shared rate is returned for every Q.

        Parameters
        ----------
        Q : Q_type, default=None
            Scattering vector in 1/angstrom. If None, the Q stored in the model is used.

        Returns
        -------
        np.ndarray
            Relaxation rates in the unit of the model (e.g. meV).

        Raises
        ------
        ValueError
            If Q-variation is enabled but Q has not been set on the model yet.
        """
        Q = self._ensure_Q(Q)

        if self._allow_Q_variation['relaxation_rate'] is True:
            if not self._relaxation_rate_list:
                raise ValueError(
                    'Relaxation rate Q-variation list is empty. '
                    'Set Q before calling calculate_relaxation_rate.'
                )
            indices = self._match_Q_indices(Q)
            return np.array([self._relaxation_rate_list[i].value for i in indices])

        return self.relaxation_rate.value * np.ones_like(Q)

    def calculate_alpha(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the Mittag-Leffler form parameter alpha at each Q.

        If alpha is allowed to vary with Q, the requested Q values are matched against the Q stored
        in the model and the corresponding per-Q values are returned. Otherwise the shared alpha is
        returned for every Q.

        Parameters
        ----------
        Q : Q_type, default=None
            Scattering vector in 1/angstrom. If None, the Q stored in the model is used.

        Returns
        -------
        np.ndarray
            Form parameters (dimensionless).

        Raises
        ------
        ValueError
            If Q-variation is enabled but Q has not been set on the model yet.
        """
        Q = self._ensure_Q(Q)

        if self._allow_Q_variation['alpha'] is True:
            if not self._alpha_list:
                raise ValueError(
                    'Alpha Q-variation list is empty. Set Q before calling calculate_alpha.'
                )
            indices = self._match_Q_indices(Q)
            return np.array([self._alpha_list[i].value for i in indices])

        return self.alpha.value * np.ones_like(Q)

    def calculate_EISF(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF), i.e. A_0.

        Parameters
        ----------
        Q : Q_type, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """
        Q = self._ensure_Q(Q)

        if self._allow_Q_variation['A_0'] is True:
            indices = self._match_Q_indices(Q)
            return np.array([self._A_0_list[i].value for i in indices])

        return self.A_0.value * np.ones_like(Q)

    def calculate_QISF(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF), i.e. A_1 = 1 - A_0.

        Parameters
        ----------
        Q : Q_type, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            QISF values (dimensionless).
        """
        Q = self._ensure_Q(Q)

        if self._allow_Q_variation['A_0'] is True:
            indices = self._match_Q_indices(Q)
            return np.array([self._A_1_list[i].value for i in indices])

        return self.A_1.value * np.ones_like(Q)

    def calculate_relaxation_rate_spectrum(self, rate: np.ndarray, Q: Q_type = None) -> np.ndarray:
        r"""
        Calculate the relaxation rate spectrum p(lambda) of Eq. (37) of Hassani et al.

        $$ p_{ML}(\lambda) = \frac{\sin(\pi\alpha)} {\pi\lambda((\lambda\tau_R)^{-\alpha} +
        (\lambda\tau_R)^{\alpha} + 2\cos(\pi\alpha))} $$

        This is the distribution of exponential relaxation rates whose superposition gives the
        Mittag-Leffler relaxation function, and is what the paper plots in its Fig. 6. It is a
        property of the internal dynamics alone, so the diffusion damping does not enter.

        Parameters
        ----------
        rate : np.ndarray
            Relaxation rates lambda at which to evaluate the spectrum, expressed as energies in
            x_unit. Must be strictly positive.
        Q : Q_type, default=None
            Scattering vector in 1/angstrom. If None, the Q stored in the model is used.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(Q), len(rate))`` holding p(lambda) for each Q, with unit
            ``1/x_unit``.

        Raises
        ------
        ValueError
            If any requested rate is not strictly positive.
        """
        rate = np.atleast_1d(np.asarray(rate, dtype=float))
        if np.any(rate <= 0):
            raise ValueError('rate must be strictly positive.')

        Q = self._ensure_Q(Q)
        alpha = self.calculate_alpha(Q)[:, np.newaxis]
        # lambda * tau_R, with tau_R expressed through the relaxation rate hbar/tau_R
        reduced_rate = rate[np.newaxis, :] / self.calculate_relaxation_rate(Q)[:, np.newaxis]

        numerator = np.sin(np.pi * alpha)
        denominator = (
            np.pi
            * rate[np.newaxis, :]
            * (reduced_rate**-alpha + reduced_rate**alpha + 2 * np.cos(np.pi * alpha))
        )
        return numerator / denominator

    def calculate_energy_barrier_distribution(
        self, barrier_height: np.ndarray, Q: Q_type = None
    ) -> np.ndarray:
        r"""
        Calculate the energy barrier distribution P(h) of Eq. (48) of Hassani et al.

        $$ P_{ML}(h) = \frac{2h\sin(\pi\alpha)} {\pi(e^{-\alpha h^2} + e^{\alpha h^2} +
        2\cos(\pi\alpha))} $$

        where $h = \Delta E / (k_B T)$ is the dimensionless barrier height of Zwanzig's rough
        harmonic potential. This is what the paper plots in the right panel of its Fig. 7. As
        $\alpha \to 1$ it collapses onto $\delta(h)$, a smooth potential; as $\alpha \to 0$ it
        broadens to include arbitrarily high barriers.

        Parameters
        ----------
        barrier_height : np.ndarray
            Dimensionless barrier heights h at which to evaluate the distribution.
        Q : Q_type, default=None
            Scattering vector in 1/angstrom. If None, the Q stored in the model is used.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(Q), len(barrier_height))`` holding P(h) for each Q.
        """
        barrier_height = np.atleast_1d(np.asarray(barrier_height, dtype=float))

        Q = self._ensure_Q(Q)
        alpha = self.calculate_alpha(Q)[:, np.newaxis]
        h = barrier_height[np.newaxis, :]

        # Factor exp(alpha * h**2) out of the denominator, so tall barriers underflow smoothly to
        # zero instead of overflowing exp() and leaving inf/inf behind.
        damping = np.exp(-alpha * h**2)
        numerator = 2 * h * np.sin(np.pi * alpha) * damping
        denominator = np.pi * (damping**2 + 1 + 2 * np.cos(np.pi * alpha) * damping)
        return numerator / denominator

    def create_component_collections(self) -> list[ComponentCollection]:
        r"""
        Create ComponentCollections for the MittagLefflerDiffusion model at the given Q values.

        Each collection holds the elastic Lorentzian (area $K \cdot EISF$, HWHM $\hbar D q^2$) and
        the Mittag-Leffler component (scale $K \cdot (1 - EISF)$, damping $\hbar D q^2$). The per-Q
        parameter lists are recreated here so the built components are backed by the very
        parameters stored in the lists, keeping ``calculate_*`` in sync with the components. The
        created collections are installed on the model, so the returned list is the live one.

        Returns
        -------
        list[ComponentCollection]
            List of ComponentCollections with a Lorentzian and a Mittag-Leffler component for each
            Q value.
        """
        if self.Q is None:
            self._A_0_list = []
            self._A_1_list = []
            self._relaxation_rate_list = []
            self._alpha_list = []
            self._component_collections = []
            return self._component_collections

        Q = self.Q.values

        if self._allow_Q_variation['A_0'] is True:
            self._A_0_list, self._A_1_list = self._create_A0_A1_parameter_lists()
        else:
            self._A_0_list = []
            self._A_1_list = []

        if self._allow_Q_variation['relaxation_rate'] is True:
            self._relaxation_rate_list = self._create_relaxation_rate_parameter_list()
        else:
            self._relaxation_rate_list = []

        if self._allow_Q_variation['alpha'] is True:
            self._alpha_list = self._create_alpha_parameter_list()
        else:
            self._alpha_list = []

        component_collection_list = [None] * len(Q)
        for i, Q_value in enumerate(Q):
            component_collection_list[i] = ComponentCollection(
                name=f'{self.name}_Q{Q_value:.2f}',
                display_name=f'{self.display_name}_Q{Q_value:.2f}',
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )

            # easyscience propagates inf bounds through arithmetic, producing inf/inf=nan
            # as a transient intermediate. Python's min/max ignore nan so the final bounds
            # are correct; suppress the spurious numpy RuntimeWarning.
            with np.errstate(invalid='ignore'):
                component_collection_list[i].append_component(
                    self._create_lorentzian_component(Q_value, i)
                )
                component_collection_list[i].append_component(
                    self._create_mittag_leffler_component(Q_value, i)
                )

        self._component_collections = component_collection_list
        return self._component_collections

    def get_fit_targets(self) -> list[FitTarget]:
        """
        Get the fittable predictions of the MittagLefflerDiffusion model as FitTargets.

        The model predicts three Q-dependent quantities: ``'width'`` (the shared damping ``hbar * D
        * Q**2``), ``'area'`` (the Mittag-Leffler weight ``scale * QISF(Q)``) and
        ``'elastic_area'`` (the elastic Lorentzian weight ``scale * EISF(Q)``). The base class
        implementation is replaced rather than extended, because here it is the *elastic* line that
        is a Lorentzian, so the base's ``'area'`` key would point at the wrong component.

        Returns
        -------
        list[FitTarget]
            The fittable predictions of this model.
        """
        return [
            FitTarget(
                name='area',
                dataset_key=f'{self.mittag_leffler_name} scale',
                function=lambda Q, model=self, **_: model.calculate_QISF(Q) * model.scale.value,
                label=f'{self.display_name} area',
                x_unit=CANONICAL_Q_UNIT,
                y_unit=str(self.scale.unit),
            ),
            FitTarget(
                name='width',
                dataset_key=f'{self.lorentzian_name} width',
                function=lambda Q, model=self, **_: model.calculate_width(Q),
                label=f'{self.display_name} width',
                x_unit=CANONICAL_Q_UNIT,
                y_unit=str(self.x_unit),
            ),
            FitTarget(
                name='elastic_area',
                dataset_key=f'{self.lorentzian_name} area',
                function=lambda Q, model=self, **_: model.calculate_EISF(Q) * model.scale.value,
                label=f'{self.display_name} elastic_area',
                x_unit=CANONICAL_Q_UNIT,
                y_unit=str(self.scale.unit),
            ),
        ]

    def get_global_variables(self) -> list[Parameter]:
        """
        Get all global variables from the diffusion model.

        Returns
        -------
        list[Parameter]
            A list of all global variables from the diffusion model.
        """
        variables = [self.scale, self.diffusion_coefficient]

        if self._allow_Q_variation['A_0'] is False:
            variables.append(self.A_0)
            variables.append(self.A_1)

        if self._allow_Q_variation['relaxation_rate'] is False:
            variables.append(self.relaxation_rate)

        if self._allow_Q_variation['alpha'] is False:
            variables.append(self.alpha)

        return variables

    def get_independent_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get the independent variables from the diffusion model.

        The per-Q relaxation rate and alpha parameters are the components' own parameters, so they
        are reached through the component collections; only the per-Q A_0/A_1 pairs, which the
        component areas merely depend on, are listed here.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the Q value for which to get the independent variables. If None,
            independent variables for all Q values will be included.

        Returns
        -------
        list[Parameter]
            List of independent variables in the model.
        """
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)

        variables = []
        if self._allow_Q_variation['A_0'] is True:
            if Q_index is None:
                variables.extend(self._A_0_list)
                variables.extend(self._A_1_list)
            else:
                variables.append(self._A_0_list[Q_index])
                variables.append(self._A_1_list[Q_index])

        return variables

    def get_all_variables(self, Q_index: int | None = None) -> list[DescriptorNumber]:
        """
        Get a list of all variables (Parameters and Descriptors) in the model.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the Q value for which to get the variables. If None, variables for all Q
            values will be included.

        Returns
        -------
        list[DescriptorNumber]
            List of all variables in the model.
        """
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)

        variables = self.get_global_variables()
        variables.extend(self.get_independent_variables(Q_index=Q_index))

        if Q_index is None:
            for component_collection in self._component_collections:
                variables.extend(component_collection.get_all_variables())
        else:
            variables.extend(self._component_collections[Q_index].get_all_variables())

        return variables

    # ------------------------------------------------------------------
    # Private methods for init
    # ------------------------------------------------------------------

    def _create_Q_variation_dict(self, allow_Q_variation: dict | None) -> dict:
        """
        Create the allow_Q_variation dict, ensuring it has the correct keys and default values.

        Parameters
        ----------
        allow_Q_variation : dict | None
            Dict describing whether to allow Q variation of A_0, relaxation_rate and alpha.

        Raises
        ------
        TypeError
            If allow_Q_variation is not a dict or None.
        ValueError
            If allow_Q_variation contains unknown keys.

        Returns
        -------
        dict
            A dict with keys 'A_0', 'relaxation_rate' and 'alpha'.
        """
        allow_Q_variation_default = {
            'A_0': False,
            'relaxation_rate': False,
            'alpha': False,
        }
        allowed_keys = set(allow_Q_variation_default)

        if allow_Q_variation is None:
            allow_Q_variation = {}
        if not isinstance(allow_Q_variation, dict):
            raise TypeError('allow_Q_variation must be a dict or None.')

        unknown_keys = set(allow_Q_variation) - allowed_keys
        if unknown_keys:
            raise ValueError(f'Unknown keys in allow_Q_variation: {unknown_keys}')

        return {**allow_Q_variation_default, **allow_Q_variation}

    @staticmethod
    def _create_diffusion_coefficient_parameter(diffusion_coefficient: Numeric) -> Parameter:
        """
        Create the global diffusion coefficient parameter.

        Parameters
        ----------
        diffusion_coefficient : Numeric
            The value for D in m^2/s.

        Raises
        ------
        TypeError
            If diffusion_coefficient is not a number.
        ValueError
            If diffusion_coefficient is negative.

        Returns
        -------
        Parameter
            The created diffusion coefficient parameter.
        """
        if not isinstance(diffusion_coefficient, Numeric):
            raise TypeError('diffusion_coefficient must be a number.')
        if float(diffusion_coefficient) < 0:
            raise ValueError('diffusion_coefficient must be non-negative.')
        return Parameter(
            name='diffusion_coefficient',
            value=float(diffusion_coefficient),
            fixed=False,
            unit='m**2/s',
            min=0.0,
        )

    @staticmethod
    def _create_A0_A1_parameters(A_0: Numeric) -> tuple[Parameter, Parameter]:
        """
        Create the shared A_0 and A_1 parameters.

        Parameters
        ----------
        A_0 : Numeric
            The value for the A_0 parameter.

        Raises
        ------
        TypeError
            If A_0 is not a number.
        ValueError
            If A_0 is not between 0 and 1.

        Returns
        -------
        tuple[Parameter, Parameter]
            A tuple containing the A_0 and A_1 parameters.
        """
        if not isinstance(A_0, Numeric):
            raise TypeError('A_0 must be a number.')
        if float(A_0) < 0 or float(A_0) > 1:
            raise ValueError('A_0 must be between 0 and 1.')

        A_0_parameter = Parameter(name='A_0', value=float(A_0), fixed=False, min=0.0, max=1.0)
        A_1_parameter = Parameter.from_dependency(
            name='A_1',
            dependency_expression='1 - A_0',
            dependency_map={'A_0': A_0_parameter},
        )
        return A_0_parameter, A_1_parameter

    def _create_relaxation_rate_parameter(self, relaxation_rate: Numeric) -> Parameter:
        """
        Create the shared relaxation rate parameter.

        Parameters
        ----------
        relaxation_rate : Numeric
            The value for the relaxation rate in x_unit.

        Raises
        ------
        TypeError
            If relaxation_rate is not a number.
        ValueError
            If relaxation_rate is less than the minimum width.

        Returns
        -------
        Parameter
            The created relaxation rate parameter.
        """
        if not isinstance(relaxation_rate, Numeric):
            raise TypeError('relaxation_rate must be a number.')
        if float(relaxation_rate) < MINIMUM_WIDTH:
            raise ValueError(f'relaxation_rate must be at least {MINIMUM_WIDTH}.')

        return Parameter(
            name='relaxation_rate',
            value=float(relaxation_rate),
            fixed=False,
            min=MINIMUM_WIDTH,
            unit=self.x_unit,
        )

    @staticmethod
    def _create_alpha_parameter(alpha: Numeric) -> Parameter:
        """
        Create the shared form parameter alpha.

        Parameters
        ----------
        alpha : Numeric
            The value for the form parameter.

        Raises
        ------
        TypeError
            If alpha is not a number.
        ValueError
            If alpha does not lie in (0, 1].

        Returns
        -------
        Parameter
            The created alpha parameter.
        """
        if not isinstance(alpha, Numeric):
            raise TypeError('alpha must be a number.')
        if not MINIMUM_ALPHA <= float(alpha) <= MAXIMUM_ALPHA:
            raise ValueError('alpha must be greater than zero and at most one.')

        return Parameter(
            name='alpha',
            value=float(alpha),
            fixed=False,
            min=MINIMUM_ALPHA,
            max=MAXIMUM_ALPHA,
            unit='dimensionless',
        )

    def _create_A0_A1_parameter_lists(self) -> tuple[list[Parameter], list[Parameter]]:
        """
        Create per-Q A_0 and A_1 parameters, seeded from the shared A_0.

        Returns
        -------
        tuple[list[Parameter], list[Parameter]]
            The per-Q A_0 parameters and the per-Q A_1 parameters derived from them.
        """
        A_0_list = []
        A_1_list = []
        for _ in range(len(self.Q)):
            # The per-Q amplitudes carry the model name so they do not collide with other
            # models' parameters. The name is the same at every Q on purpose: parameters are
            # tracked across Q by name (unique within a Q, shared across Q).
            a_0 = Parameter(
                name=f'{self.name} A_0',
                display_name='A_0',
                value=float(self.A_0.value),
                fixed=False,
                min=0.0,
                max=1.0,
            )
            a_1 = Parameter.from_dependency(
                name=f'{self.name} A_1',
                dependency_expression='1 - A_0',
                dependency_map={'A_0': a_0},
            )
            A_0_list.append(a_0)
            A_1_list.append(a_1)

        return A_0_list, A_1_list

    def _create_relaxation_rate_parameter_list(self) -> list[Parameter]:
        """
        Create per-Q relaxation rate parameters, seeded from the shared relaxation rate.

        Returns
        -------
        list[Parameter]
            The per-Q relaxation rate parameters, named after the Mittag-Leffler component's own
            width parameter so they slot straight into it.
        """
        return [
            Parameter(
                name=f'{self.mittag_leffler_name} width',
                value=float(self.relaxation_rate.value),
                fixed=False,
                min=MINIMUM_WIDTH,
                unit=self.x_unit,
            )
            for _ in range(len(self.Q))
        ]

    def _create_alpha_parameter_list(self) -> list[Parameter]:
        """
        Create per-Q alpha parameters, seeded from the shared alpha.

        Returns
        -------
        list[Parameter]
            The per-Q form parameters, named after the Mittag-Leffler component's own alpha
            parameter so they slot straight into it.
        """
        return [
            Parameter(
                name=f'{self.mittag_leffler_name} alpha',
                value=float(self.alpha.value),
                fixed=False,
                min=MINIMUM_ALPHA,
                max=MAXIMUM_ALPHA,
                unit='dimensionless',
            )
            for _ in range(len(self.Q))
        ]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _create_lorentzian_component(self, Q_value: float, Q_index: int) -> Lorentzian:
        """
        Build the elastic Lorentzian for one Q value.

        Its width is the diffusion damping ``hbar * D * Q**2`` and its area is ``scale * A_0``;
        both are always dependent parameters, since the elastic line is entirely determined by the
        global diffusion coefficient and the elastic fraction.

        Parameters
        ----------
        Q_value : float
            Scattering vector in 1/angstrom.
        Q_index : int
            Index of this Q value in the stored Q.

        Returns
        -------
        Lorentzian
            The configured elastic Lorentzian component.
        """
        component = Lorentzian(
            name=self.lorentzian_name,
            display_name=self.lorentzian_display_name,
            x_unit=self.x_unit,
            y_unit=self.y_unit,
        )
        component.width.make_dependent_on(
            dependency_expression=self._write_damping_dependency_expression(Q_value),
            dependency_map=self._write_damping_dependency_map_expression(),
            desired_unit=self.x_unit,
        )
        component.area.make_dependent_on(
            dependency_expression='scale * A_0',
            dependency_map=self._write_amplitude_dependency_map_expression(Q_index, elastic=True),
        )
        return component

    def _create_mittag_leffler_component(
        self, Q_value: float, Q_index: int
    ) -> DiffusionDampedMittagLeffler:
        """
        Build the Mittag-Leffler component for one Q value.

        Its damping is the diffusion damping ``hbar * D * Q**2`` and its scale is ``scale * A_1``,
        both always dependent. Its width and alpha are either the per-Q parameters from the
        corresponding lists, or made dependent on the shared parameters.

        Parameters
        ----------
        Q_value : float
            Scattering vector in 1/angstrom.
        Q_index : int
            Index of this Q value in the stored Q.

        Returns
        -------
        DiffusionDampedMittagLeffler
            The configured Mittag-Leffler component.
        """
        component = DiffusionDampedMittagLeffler(
            name=self.mittag_leffler_name,
            display_name=self.mittag_leffler_display_name,
            x_unit=self.x_unit,
            y_unit=self.y_unit,
        )

        if self._allow_Q_variation['relaxation_rate'] is True:
            component._width = self._relaxation_rate_list[Q_index]  # ruff: ignore[private-member-access]
        else:
            component.width.make_dependent_on(
                dependency_expression='relaxation_rate',
                dependency_map={'relaxation_rate': self.relaxation_rate},
                desired_unit=self.x_unit,
            )

        if self._allow_Q_variation['alpha'] is True:
            component._alpha = self._alpha_list[Q_index]  # ruff: ignore[private-member-access]
        else:
            component.alpha.make_dependent_on(
                dependency_expression='alpha',
                dependency_map={'alpha': self.alpha},
                desired_unit='dimensionless',
            )

        component.damping.make_dependent_on(
            dependency_expression=self._write_damping_dependency_expression(Q_value),
            dependency_map=self._write_damping_dependency_map_expression(),
            desired_unit=self.x_unit,
        )
        component.scale.make_dependent_on(
            dependency_expression='scale * A_1',
            dependency_map=self._write_amplitude_dependency_map_expression(Q_index, elastic=False),
        )
        return component

    def _on_Q_change(self) -> None:
        """
        Handle changes to the Q values.

        Rebuilds the component collections; the per-Q parameter lists are recreated inside
        ``create_component_collections``.
        """
        self.create_component_collections()

    def _convert_extra_x_unit_parameters(self, unit_str: str) -> None:
        """
        Convert the shared relaxation rate template to the new x-axis unit.

        The per-Q relaxation rate list (when Q-variation is enabled) holds the very Parameter
        objects used by the components, so those are converted in place with the collections.

        Parameters
        ----------
        unit_str : str
            The new x-axis unit as a string.
        """
        convert_parameter_unit(self._relaxation_rate, unit_str)

    def _write_damping_dependency_expression(self, Q: float) -> str:
        """
        Write the dependency expression for the diffusion damping ``hbar * D * Q**2``.

        Parameters
        ----------
        Q : float
            Scattering vector in 1/angstrom.

        Raises
        ------
        TypeError
            If Q is not a float.

        Returns
        -------
        str
            Dependency expression for the damping.
        """
        if not isinstance(Q, (float)):
            raise TypeError('Q must be a float.')

        # Q is given as a float, so we need to add the units
        return f'hbar * D * {Q}**2 * 1/(angstrom**2)'

    def _write_damping_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map for the diffusion damping.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the damping.
        """
        return {
            'D': self.diffusion_coefficient,
            'hbar': self._hbar,
            'angstrom': self._angstrom,
        }

    def _write_amplitude_dependency_map_expression(
        self, Q_index: int, elastic: bool
    ) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map for a component's amplitude.

        Parameters
        ----------
        Q_index : int
            Index of the Q value, used to pick the per-Q amplitude when Q-variation is enabled.
        elastic : bool
            True for the elastic Lorentzian's area (``scale * A_0``), False for the Mittag-Leffler
            component's scale (``scale * A_1``).

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the amplitude.
        """
        if self._allow_Q_variation['A_0'] is True:
            amplitude = self._A_0_list[Q_index] if elastic else self._A_1_list[Q_index]
        else:
            amplitude = self.A_0 if elastic else self.A_1

        return {'scale': self.scale, 'A_0' if elastic else 'A_1': amplitude}

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the MittagLefflerDiffusion model.

        Returns
        -------
        str
            String representation of the MittagLefflerDiffusion model.
        """
        return (
            f'MittagLefflerDiffusion(display_name={self.display_name}, '
            f'x_unit={self.x_unit}, y_unit={self.y_unit}, \n'
            f'    diffusion_coefficient={self.diffusion_coefficient}, \n'
            f'    A_0={self.A_0}, A_1={self.A_1}, \n'
            f'    relaxation_rate={self.relaxation_rate}, \n'
            f'    alpha={self.alpha}, \n'
            f'    scale={self.scale})'
        )
