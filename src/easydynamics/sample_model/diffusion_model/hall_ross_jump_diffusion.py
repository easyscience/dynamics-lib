# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import hbar

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class HallRossJumpDiffusion(DiffusionModelBase):
    r"""
    Model of Hall-Ross jump diffusion, with a Gaussian distribution of jump lengths.

    The model consists of a Lorentzian function for each Q-value, where the width is given by

    $$ \Gamma(Q) = \frac{\hbar}{\tau} \left( 1 - \exp\left(-\frac{Q^2 l^2}{2}\right) \right) $$

    where $\tau$ is the residence time and $l$ is the standard deviation of the jump-length
    distribution, the convention of reference 2. The mean-square jump length is $\langle r^2
    \rangle = 3 l^2$. At small $Q$ the width tends to $\hbar D Q^2$ with $D = l^2 / (2 \tau)$, and
    at large $Q$ to $\hbar / \tau$. $Q$ is assumed to have units of 1/angstrom. Creates
    ComponentCollections with Lorentzian components for given Q-values.

    References
    ----------
    1. P. L. Hall and D. K. Ross, *Incoherent neutron scattering functions for random jump
       diffusion in bounded and infinite media*, Mol. Phys. **42**, 673 (1981),
       [doi:10.1080/00268978100100521](https://doi.org/10.1080/00268978100100521).
    2. Mantid, ``HallRoss`` fit function, whose ``L`` parameter is this $l$,
       https://github.com/mantidproject/mantid/blob/17c1cd96c1d46b39bee21dc75a60017b658d689c/Framework/PythonInterface/plugins/functions/HallRoss.py#L30-L39.

    Examples
    --------
    **Creating a HallRossJumpDiffusion model**

    Pass the residence time (in ps) and jump length (in angstroms) along with Q values:
    ```python
    import numpy as np
    import easydynamics as edyn

    Q = np.linspace(0.5, 2, 7)
    diffusion_model = edyn.HallRossJumpDiffusion(
        scale=1.0,
        residence_time=1.0,
        jump_length=1.5,
        Q=Q,
    )
    component_collections = diffusion_model.create_component_collections()
    ```

    See also the tutorials.
    """

    def __init__(
        self,
        scale: Numeric = 1.0,
        residence_time: Numeric = 1.0,
        jump_length: Numeric = 1.0,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'HallRossJumpDiffusion',
        display_name: str | None = 'HallRossJumpDiffusion',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new HallRossJumpDiffusion model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        residence_time : Numeric, default=1.0
            Residence time parameter tau in ps.
        jump_length : Numeric, default=1.0
            Standard deviation l of the jump-length distribution in angstrom.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='HallRossJumpDiffusion'
            Name of the diffusion model.
        display_name : str | None, default='HallRossJumpDiffusion'
            Display name of the diffusion model.
        lorentzian_name : str | None, default=None
            Name of the Lorentzian component. If None, it will be set to the name of the diffusion
            model. By default, None.
        lorentzian_display_name : str | None, default=None
            Display name of the Lorentzian component. If None, it will be set to the
            lorentzian_name. By default, None
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated. By
            default, None.

        Raises
        ------
        TypeError
            If ``residence_time`` or ``jump_length`` is not a number.
        ValueError
            If ``jump_length`` is negative or ``residence_time`` is below ``MINIMUM_WIDTH``.
        """
        super().__init__(
            Q=Q,
            x_unit=x_unit,
            y_unit=y_unit,
            scale=scale,
            name=name,
            display_name=display_name,
            lorentzian_name=lorentzian_name,
            lorentzian_display_name=lorentzian_display_name,
            unique_name=unique_name,
        )

        if not isinstance(residence_time, Numeric):
            raise TypeError('residence_time must be a number.')

        if float(residence_time) < MINIMUM_WIDTH:
            raise ValueError(f'residence_time must be at least {MINIMUM_WIDTH}.')

        if not isinstance(jump_length, Numeric):
            raise TypeError('jump_length must be a number.')

        if float(jump_length) < 0:
            raise ValueError('jump_length must be non-negative.')

        residence_time = Parameter(
            name='residence_time',
            value=float(residence_time),
            fixed=False,
            unit='ps',
            min=MINIMUM_WIDTH,
        )

        jump_length = Parameter(
            name='jump_length',
            value=float(jump_length),
            fixed=False,
            unit='angstrom',
            min=0.0,
        )

        self._hbar = hbar
        self._angstrom = angstrom
        self._residence_time = residence_time
        self._jump_length = jump_length

        self._component_collections = self.create_component_collections()

    ################################
    # Properties
    ################################

    @property
    def residence_time(self) -> Parameter:
        """
        Get the residence time parameter tau.

        Returns
        -------
        Parameter
            Residence time tau.
        """
        return self._residence_time

    @residence_time.setter
    def residence_time(self, residence_time: Numeric) -> None:
        """
        Set the residence time parameter tau.

        Parameters
        ----------
        residence_time : Numeric
            Residence time tau in ps.

        Raises
        ------
        TypeError
            If residence_time is not a number.
        ValueError
            If residence_time is below ``MINIMUM_WIDTH``.
        """
        if not isinstance(residence_time, Numeric):
            raise TypeError('residence_time must be a number.')
        if float(residence_time) < MINIMUM_WIDTH:
            raise ValueError(f'residence_time must be at least {MINIMUM_WIDTH}.')

        self._residence_time.value = float(residence_time)

    @property
    def jump_length(self) -> Parameter:
        """
        Get the jump length parameter l.

        Returns
        -------
        Parameter
            Standard deviation l of the jump-length distribution in angstrom.
        """
        return self._jump_length

    @jump_length.setter
    def jump_length(self, jump_length: Numeric) -> None:
        """
        Set the jump length parameter l.

        Parameters
        ----------
        jump_length : Numeric
            Standard deviation l of the jump-length distribution in angstrom.

        Raises
        ------
        TypeError
            If jump_length is not a number.
        ValueError
            If jump_length is negative.
        """
        if not isinstance(jump_length, Numeric):
            raise TypeError('jump_length must be a number.')

        if float(jump_length) < 0:
            raise ValueError('jump_length must be non-negative.')
        self._jump_length.value = float(jump_length)

    ################################
    # Other methods
    ################################

    def calculate_width(self, Q: Q_type | None = None) -> np.ndarray:
        r"""
        Calculate the half-width at half-maximum (HWHM) for the diffusion model.

        $\Gamma(Q) = (\hbar / \tau) (1 - \exp(-Q^2 l^2 / 2))$, where $\tau$ is the residence time
        and $l$ is the standard deviation of the jump-length distribution.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom. Can be a single value or an array of values. If None,
            Q values stored in the model are used.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """
        Q = self._ensure_Q(Q)

        conversion_factor = self._jump_length / self._angstrom
        conversion_factor.convert_unit('dimensionless')

        prefactor = self._hbar / self._residence_time
        prefactor.convert_unit(self.x_unit)

        argument = Q * conversion_factor.value

        # -expm1(-x) is 1 - exp(-x) without the cancellation at small Q
        return prefactor.value * -np.expm1(-(argument**2) / 2)

    def calculate_EISF(self, Q: Q_type) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF), which is 0 for this model.

        Parameters
        ----------
        Q : Q_type
            Scattering vector in 1/angstrom. Can be a single value or an array of values.

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """
        Q = self._ensure_Q(Q)

        return np.zeros_like(Q)

    def calculate_QISF(self, Q: Q_type) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF), which is 1 for this model.

        Parameters
        ----------
        Q : Q_type
            Scattering vector in 1/angstrom. Can be a single value or an array of values.

        Returns
        -------
        np.ndarray
            QISF values (dimensionless).
        """
        Q = self._ensure_Q(Q)

        return np.ones_like(Q)

    def create_component_collections(
        self,
    ) -> list[ComponentCollection]:
        """
        Create ComponentCollection components for the diffusion model at given Q values.

        The created collections are installed on the model (they become the collections returned by
        ``get_component_collections``), so the returned list is the live one.

        Returns
        -------
        list[ComponentCollection]
            List of ComponentCollections with Hall-Ross Lorentzian components.
        """
        if self.Q is None:
            self._component_collections = []
            return self._component_collections

        Q = self.Q.values
        component_collection_list = [None] * len(Q)
        QISF = self.calculate_QISF(Q)

        # One Lorentzian per Q-value, with the Hall-Ross width and area equal to scale. No delta
        # function, as the EISF is 0.
        for i, Q_value in enumerate(Q):
            component_collection_list[i] = ComponentCollection(
                name=f'{self.name}_Q{Q_value:.2f}',
                display_name=f'{self.display_name}_Q{Q_value:.2f}',
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )

            lorentzian_component = Lorentzian(
                name=self.lorentzian_name,
                display_name=self.lorentzian_display_name,
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )

            dependency_expression = self._write_width_dependency_expression(Q[i])
            dependency_map = self._write_width_dependency_map_expression()

            # easyscience propagates inf bounds through arithmetic, producing inf/inf=nan
            # as a transient intermediate. Python's min/max ignore nan so the final bounds
            # are correct; suppress the spurious numpy RuntimeWarning.
            with np.errstate(invalid='ignore'):
                lorentzian_component.width.make_dependent_on(
                    dependency_expression=dependency_expression,
                    dependency_map=dependency_map,
                    desired_unit=self.x_unit,
                )

                area_dependency_map = self._write_area_dependency_map_expression()
                lorentzian_component.area.make_dependent_on(
                    dependency_expression=self._write_area_dependency_expression(QISF[i]),
                    dependency_map=area_dependency_map,
                )

            component_collection_list[i].append_component(lorentzian_component)

        self._component_collections = component_collection_list
        return self._component_collections

    ################################
    # Private methods
    ################################
    def _on_Q_change(self) -> None:
        """
        Update the component collections when Q changes. This is called automatically when the Q
        property is set. It regenerates the component collections based on the new Q values.
        """
        self._component_collections = self.create_component_collections()

    def _write_width_dependency_expression(self, Q: float) -> str:
        """
        Write the dependency expression for the width as a function of Q to make dependent
        Parameters.

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
            Dependency expression for the width.
        """
        if not isinstance(Q, (float)):
            raise TypeError('Q must be a float.')

        # Q is in 1/angstrom and l in angstrom; -expm1(-x) is 1 - exp(-x) without cancellation
        return f'(hbar / tau) * -expm1(-(({Q} * l.value) ** 2) / 2)'

    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """
        return {
            'tau': self.residence_time,
            'l': self.jump_length,
            'hbar': self._hbar,
        }

    ################################
    # dunder methods
    ################################

    def __repr__(self) -> str:
        """
        String representation of the HallRossJumpDiffusion model.

        Returns
        -------
        str
            String representation of the HallRossJumpDiffusion model.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r},\n'
            f'x_unit={self.x_unit}, y_unit={self.y_unit}, \n'
            f'    residence_time={self.residence_time},\n'
            f'    jump_length={self.jump_length}),\n'
            f'    scale={self.scale})'
        )
