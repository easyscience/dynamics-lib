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


class BrownianTranslationalDiffusion(DiffusionModelBase):
    r"""
    Model of Brownian translational diffusion, consisting of a Lorentzian function for each
    Q-value, where the width is given by $D Q^2$, where $D$ is the diffusion coefficient. The area
    of the Lorentzians is given by the scale parameter multiplied by the QISF, which is 1 for this
    model. The EISF is 0 for this model, so there is no delta function component. Q is assumed to
    have units of 1/angstrom. Creates ComponentCollections with Lorentzian components for given
    Q-values.

    Examples
    --------
    **Creating a BrownianTranslationalDiffusion model**

    The model creates one Lorentzian per Q-value, with width $D Q^2$. Pass ``Q`` values at
    construction or later via ``create_component_collections``:
    ```python
    import numpy as np
    from easydynamics.sample_model.diffusion_model import BrownianTranslationalDiffusion

    Q = np.linspace(0.5, 2, 7)
    diffusion_model = BrownianTranslationalDiffusion(
        scale=1.0,
        diffusion_coefficient=2.4e-9,
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
        Q: Q_type | None = None,
        unit: str | sc.Unit = 'meV',
        name: str = 'BrownianTranslationalDiffusion',
        display_name: str | None = 'BrownianTranslationalDiffusion',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new BrownianTranslationalDiffusion model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        diffusion_coefficient : Numeric, default=1.0
            Diffusion coefficient D in m^2/s.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        unit : str | sc.Unit, default='meV'
            Unit of the diffusion model. Must be convertible to meV.
        name : str, default='BrownianTranslationalDiffusion'
            Name of the diffusion model.
        display_name : str | None, default='BrownianTranslationalDiffusion'
            Display name of the diffusion model.
        lorentzian_name : str | None, default=None
            Name of the Lorentzian component. If None, it will be set to the name of the diffusion
            model.
        lorentzian_display_name : str | None, default=None
            Display name of the Lorentzian component. If None, it will be set to the
            lorentzian_name.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated. By
            default, None.

        Raises
        ------
        TypeError
            If scale or diffusion_coefficient is not a number.

        ValueError
            If scale or diffusion_coefficient is negative.
        """
        super().__init__(
            Q=Q,
            unit=unit,
            scale=scale,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
            lorentzian_name=lorentzian_name,
            lorentzian_display_name=lorentzian_display_name,
        )

        if not isinstance(diffusion_coefficient, Numeric):
            raise TypeError('diffusion_coefficient must be a number.')

        if float(diffusion_coefficient) < 0:
            raise ValueError('diffusion_coefficient must be non-negative.')

        diffusion_coefficient = Parameter(
            name='diffusion_coefficient',
            value=float(diffusion_coefficient),
            fixed=False,
            unit='m**2/s',
            min=0.0,
        )

        self._hbar = hbar
        self._angstrom = angstrom
        self._diffusion_coefficient = diffusion_coefficient

        self._component_collections = self.create_component_collections()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def diffusion_coefficient(self) -> Parameter:
        """
        Get the diffusion coefficient parameter D.

        Returns
        -------
        Parameter
            Diffusion coefficient D in m^2/s.
        """
        return self._diffusion_coefficient

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, diffusion_coefficient: Numeric) -> None:
        """
        Set the diffusion coefficient parameter D.

        Parameters
        ----------
        diffusion_coefficient : Numeric
            The new value for the diffusion coefficient D in m^2/s.

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

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the half-width at half-maximum (HWHM) for the diffusion model.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """
        Q = self._ensure_Q(Q)

        unit_conversion_factor = self._hbar * self.diffusion_coefficient / (self._angstrom**2)
        unit_conversion_factor.convert_unit(self.unit)
        return Q**2 * unit_conversion_factor.value

    def calculate_EISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF) for the Brownian translational
        diffusion model.

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

        return np.zeros_like(Q)

    def calculate_QISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF).

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

        return np.ones_like(Q)

    def create_component_collections(
        self,
    ) -> list[ComponentCollection]:
        r"""
        Create ComponentCollection components for the Brownian translational diffusion model at
        given Q values.

        Returns
        -------
        list[ComponentCollection]
            List of ComponentCollections with Lorentzian components for each Q value. Each
            Lorentzian has a width given by $D*Q^2$ and an area given by the scale parameter
            multiplied by the QISF (which is 1 for this model).
        """
        Q = self.Q
        if Q is None:
            self._component_collections = []
            return self._component_collections

        component_collection_list = [None] * len(Q)
        # In more complex models, this is used to scale the area of the
        # Lorentzians and the delta function.
        QISF = self.calculate_QISF(Q)

        # Create a Lorentzian component for each Q-value, with
        # width D*Q^2 and area equal to scale.
        # No delta function, as the EISF is 0.
        for i, Q_value in enumerate(Q):
            component_collection_list[i] = ComponentCollection(
                name=f'{self.name}_Q{Q_value:.2f}',
                display_name=f'{self.display_name}_Q{Q_value:.2f}',
                unit=self.unit,
            )

            lorentzian_component = Lorentzian(
                name=self.lorentzian_name,
                display_name=self.lorentzian_display_name,
                unit=self.unit,
            )

            # Make the width dependent on Q
            dependency_expression = self._write_width_dependency_expression(Q[i])
            dependency_map = self._write_width_dependency_map_expression()

            lorentzian_component.width.make_dependent_on(
                dependency_expression=dependency_expression,
                dependency_map=dependency_map,
                desired_unit=self.unit,
            )

            # Make the area dependent on Q
            area_dependency_map = self._write_area_dependency_map_expression()
            lorentzian_component.area.make_dependent_on(
                dependency_expression=self._write_area_dependency_expression(QISF[i]),
                dependency_map=area_dependency_map,
            )

            component_collection_list[i].append_component(lorentzian_component)

        return component_collection_list

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

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
        return f'hbar * D* {Q} **2*1/(angstrom**2)'

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
            'hbar': self._hbar,
            'angstrom': self._angstrom,
        }

    def _write_area_dependency_expression(self, QISF: float) -> str:
        """
        Write the dependency expression for the area to make dependent Parameters.

        Parameters
        ----------
        QISF : float
            Quasielastic Incoherent Scattering Function.

        Raises
        ------
        TypeError
            If QISF is not a float.

        Returns
        -------
        str
            Dependency expression for the area.
        """
        if not isinstance(QISF, (float)):
            raise TypeError('QISF must be a float.')

        return f'{QISF} * scale'

    def _write_area_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the area.
        """
        return {
            'scale': self.scale,
        }

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the BrownianTranslationalDiffusion model.

        Returns
        -------
        str
            String representation of the BrownianTranslationalDiffusion model.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r},\n'
            f'    diffusion_coefficient={self.diffusion_coefficient},\n'
            f'    scale={self.scale})'
        )
