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


class JumpTranslationalDiffusion(DiffusionModelBase):
    r"""
    Model of Jump translational diffusion.

    The model consists of a Lorentzian function for each Q-value, where the width is given by

    $$ \Gamma(Q) = \frac{\hbar D Q^2}{1+D t Q^2}. $$

    where $D$ is the diffusion coefficient and $t$ is the relaxation time. Q is assumed to have
    units of 1/angstrom. Creates ComponentCollections with Lorentzian components for given
    Q-values.

    Examples
    --------
    **Creating a JumpTranslationalDiffusion model**

    Pass the diffusion coefficient (in m²/s) and relaxation time (in ps) along with Q values:
    ```python
    import numpy as np
    import easydynamics as edyn

    Q = np.linspace(0.5, 2, 7)
    diffusion_model = edyn.JumpTranslationalDiffusion(
        scale=1.0,
        diffusion_coefficient=2.4e-9,
        relaxation_time=1.0,
        Q=Q,
    )
    component_collections = diffusion_model.create_component_collections()
    ```

    See also the tutorials.
    """

    def __init__(
        self,
        scale: Numeric = 1.0,
        diffusion_coefficient: Numeric = 1.0,
        relaxation_time: Numeric = 1.0,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'JumpTranslationalDiffusion',
        display_name: str | None = 'JumpTranslationalDiffusion',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new JumpTranslationalDiffusion model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        diffusion_coefficient : Numeric, default=1.0
            Diffusion coefficient D in m^2/s.
        relaxation_time : Numeric, default=1.0
            Relaxation time t in ps.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='JumpTranslationalDiffusion'
            Name of the diffusion model.
        display_name : str | None, default='JumpTranslationalDiffusion'
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
            If scale, diffusion_coefficient, or relaxation_time  are not numbers.
        ValueError
            If scale, diffusion_coefficient, or relaxation_time are negative.
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

        if not isinstance(diffusion_coefficient, Numeric):
            raise TypeError('diffusion_coefficient must be a number.')

        if float(diffusion_coefficient) < 0:
            raise ValueError('diffusion_coefficient must be non-negative.')

        if not isinstance(relaxation_time, Numeric):
            raise TypeError('relaxation_time must be a number.')

        if float(relaxation_time) < 0:
            raise ValueError('relaxation_time must be non-negative.')

        diffusion_coefficient = Parameter(
            name='diffusion_coefficient',
            value=float(diffusion_coefficient),
            fixed=False,
            unit='m**2/s',
            min=0.0,
        )

        relaxation_time = Parameter(
            name='relaxation_time',
            value=float(relaxation_time),
            fixed=False,
            unit='ps',
            min=0.0,
        )

        self._hbar = hbar
        self._angstrom = angstrom
        self._diffusion_coefficient = diffusion_coefficient
        self._relaxation_time = relaxation_time

        self._component_collections = self.create_component_collections()

    ################################
    # Properties
    ################################

    @property
    def diffusion_coefficient(self) -> Parameter:
        """
        Get the diffusion coefficient parameter D.

        Returns
        -------
        Parameter
            Diffusion coefficient D.
        """
        return self._diffusion_coefficient

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, diffusion_coefficient: Numeric) -> None:
        """
        Set the diffusion coefficient parameter D.

        Parameters
        ----------
        diffusion_coefficient : Numeric
            Diffusion coefficient D in m^2/s.

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
    def relaxation_time(self) -> Parameter:
        """
        Get the relaxation time parameter t.

        Returns
        -------
        Parameter
            Relaxation time t in ps.
        """
        return self._relaxation_time

    @relaxation_time.setter
    def relaxation_time(self, relaxation_time: Numeric) -> None:
        """
        Set the relaxation time parameter t.

        Parameters
        ----------
        relaxation_time : Numeric
            Relaxation time t in ps.

        Raises
        ------
        TypeError
            If relaxation_time is not a number.
        ValueError
            If relaxation_time is negative.
        """
        if not isinstance(relaxation_time, Numeric):
            raise TypeError('relaxation_time must be a number.')

        if float(relaxation_time) < 0:
            raise ValueError('relaxation_time must be non-negative.')
        self._relaxation_time.value = float(relaxation_time)

    ################################
    # Other methods
    ################################

    def calculate_width(self, Q: Q_type | None = None) -> np.ndarray:
        r"""
        Calculate the half-width at half-maximum (HWHM) for the diffusion model. $\Gamma(Q) =
        Q^2/(1+D t Q^2)$, where $D$ is the diffusion coefficient and $t$ is the relaxation time.

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

        unit_conversion_factor_numerator = (
            self._hbar * self.diffusion_coefficient / (self._angstrom**2)
        )
        unit_conversion_factor_numerator.convert_unit(self.x_unit)

        numerator = unit_conversion_factor_numerator.value * Q**2

        unit_conversion_factor_denominator = (
            self.diffusion_coefficient / self._angstrom**2 * self.relaxation_time
        )
        unit_conversion_factor_denominator.convert_unit('dimensionless')

        denominator = 1 + unit_conversion_factor_denominator.value * Q**2

        return numerator / denominator

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

        The created collections are installed on the model (they become the collections returned
        by ``get_component_collections``), so the returned list is the live one.

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
        return f'hbar * D* {Q} **2/(angstrom**2)/(1 + (D * t* {Q} **2/(angstrom**2)))'

    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """
        return {
            'D': self.diffusion_coefficient,
            't': self.relaxation_time,
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
            f'    diffusion_coefficient={self.diffusion_coefficient},\n'
            f'    scale={self.scale})'
        )
