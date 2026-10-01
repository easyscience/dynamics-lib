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


class ChudleyElliotJumpDiffusion(DiffusionModelBase):
    r"""
    Model of Chudley-Elliot jump diffusion.

    The model consists of a Lorentzian function for each Q-value, where the width is given by

    $$ \Gamma(Q) = \frac{\hbar}{\tau} \left( 1 - \frac{\sin(Q l)}{Q l} \right) $$

    where $\tau$ is the residence time and $l$ is the jump length. $Q$ is assumed to have units of
    1/angstrom. Creates ComponentCollections with Lorentzian components for given Q-values.

    Examples
    --------
    **Creating a ChudleyElliotJumpDiffusion model**

    Pass the residence time (in ps) and jump length (in angstroms) along with Q values:
    ```python
    import numpy as np
    import easydynamics as edyn

    Q = np.linspace(0.5, 2, 7)
    diffusion_model = edyn.ChudleyElliotJumpDiffusion(
        scale=1.0,
        residence_time=1.0,
        jump_length=1.5,
        Q=Q,
    )
    component_collections = diffusion_model.create_component_collections()
    ```
    """

    # TODO(markbujehein): Add tutorials # ruff: ignore[line-contains-todo,missing-todo-link]
    # 1a. add tutorial notebook for this model
    # 1b. add `See also the tutorials.` remark to bottom of docstring.

    def __init__(
        self,
        scale: Numeric = 1.0,
        residence_time: Numeric = 1.0,
        jump_length: Numeric = 1.0,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'ChudleyElliotJumpDiffusion',
        display_name: str | None = 'ChudleyElliotJumpDiffusion',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new ChudleyElliotJumpDiffusion model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        residence_time : Numeric, default=1.0
            Residence time parameter tau, by default 1.0
        jump_length : Numeric, default=1.0
            Jump length parameter l, by default 1.0
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='ChudleyElliotJumpDiffusion'
            Name of the diffusion model.
        display_name : str | None, default='ChudleyElliotJumpDiffusion'
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
            If ``residence_time`` or ``jump_length`` is negative.
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

        if float(residence_time) < 0:
            raise ValueError('residence_time must be non-negative.')

        if not isinstance(jump_length, Numeric):
            raise TypeError('jump_length must be a number.')

        if float(jump_length) < 0:
            raise ValueError('jump_length must be non-negative.')

        # Relaxation time tau
        residence_time = Parameter(
            name='residence_time',
            value=float(residence_time),
            fixed=False,
            unit='ps',
            min=0.0,
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
            If residence_time is negative.
        """
        if not isinstance(residence_time, Numeric):
            raise TypeError('residence_time must be a number.')
        if float(residence_time) < 0:
            raise ValueError('residence_time must be non-negative.')
        self._residence_time.value = float(residence_time)

    @property
    def jump_length(self) -> Parameter:
        """
        Get the jump length parameter l.

        Returns
        -------
        Parameter
            Jump length l in angstrom.
        """
        return self._jump_length

    @jump_length.setter
    def jump_length(self, jump_length: Numeric) -> None:
        """
        Set the jump length parameter l .

        Parameters
        ----------
        jump_length : Numeric
            Jump length l in angstrom.

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
        Calculate the half-width at half-maximum (HWHM) for the diffusion model. $\Gamma(Q) = \hbar
        / \tau * ( 1 - \sin(Q * l) / Q * l )$, where $tau$ is the residence time and $l$ is the
        jump length.

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

        return prefactor.value * (1 - np.sinc(argument / np.pi))

    def calculate_EISF(self, Q: Q_type) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF).

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
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF).

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
            List of ComponentCollections with Jump Diffusion Lorentzian components.
        """
        if self.Q is None:
            self._component_collections = []
            return self._component_collections

        Q = self.Q.values
        component_collection_list = [None] * len(Q)
        # In more complex models, this is used to scale the area of the
        # Lorentzians and the delta function.
        QISF = self.calculate_QISF(Q)

        # Create a Lorentzian component for each Q-value, with width
        # D*Q^2 and area equal to scale. No delta function, as the EISF
        # is 0.
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

            # Make the width dependent on Q
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

                # Make the area dependent on Q
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

        # Q is given as a float, so we need to add the units
        return f'(hbar / tau) * (1 - sin({Q} * l.value) / ({Q} * l.value) )'

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
            'angstrom': self._angstrom,
        }

    ################################
    # dunder methods
    ################################

    def __repr__(self) -> str:
        """
        String representation of the JumpTranslationalDiffusion model.

        Returns
        -------
        str
            String representation of the JumpTranslationalDiffusion model.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r},\n'
            f'x_unit={self.x_unit}, y_unit={self.y_unit}, \n'
            f'    residence_time={self.residence_time},\n'
            f'    jump_length={self.jump_length}),\n'
            f'    scale={self.scale})'
        )
