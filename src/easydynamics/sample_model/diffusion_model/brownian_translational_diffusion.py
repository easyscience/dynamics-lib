from numbers import Number
from typing import Dict, List, Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber, Parameter
from scipp.constants import hbar as scipp_hbar

from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.diffusion_model import DiffusionModel
from easydynamics.sample_model.sample_model import SampleModel

Numeric = Union[float, int]


class BrownianTranslationalDiffusion(DiffusionModel):
    """
    Model of Brownian translational diffusion, consisting of a Lorentzian
    function for each Q-value, where the width is given by :math:`DQ^2`.
    Q is assumed to have units of 1/angstrom.
    Creates SampleModels with Lorentzian components for given Q-values.

    Example usage:
    Q=np.linspace(0.5,2,7)
    energy=np.linspace(-2, 2, 501)
    scale=1.0
    diffusion_coefficient = 2.4e-9  # m^2/s
    diffusion_model=BrownianTranslationalDiffusion(name="DiffusionModel", scale=scale, diffusion_coefficient= diffusion_coefficient)
    sample_models=diffusion_model.create_sample_models(Q)
    See also the examples.
    """

    def __init__(
        self,
        name: Optional[str] = "BrownianTranslationalDiffusion",
        unit: Optional[Union[str, sc.Unit]] = "meV",
        scale: Optional[Union[Parameter, Numeric]] = 1.0,
        diffusion_coefficient: Optional[Union[Parameter, Numeric]] = 1.0,
        diffusion_unit: Optional[str] = "m**2/s",
    ):
        """
        Initialize a new BrownianTranslationalDiffusion model.

        Parameters
        ----------
        name : str
            Name of the diffusion model.
        unit : str or sc.Unit, optional
            Energy unit for the underlying Lorentzian components. Defaults to "meV".
        scale : float or Parameter, optional
            Scale factor for the diffusion model.
        diffusion_coefficient : float or Parameter, optional
            Diffusion coefficient D. If a number is provided, it is assumed to be in the unit given by diffusion_unit. Defaults to 1.0.
        diffusion_unit : str, optional
            Unit for the diffusion coefficient D. Default is "meV*Å**2". Options are 'meV*Å**2' or 'm**2/s'

        """
        if not isinstance(scale, (Parameter, Numeric)):
            raise TypeError("scale must be a Parameter or a number.")

        if not isinstance(diffusion_coefficient, (Parameter, Numeric)):
            raise TypeError("diffusion_coefficient must be a Parameter or a number.")

        if not isinstance(diffusion_unit, str):
            raise TypeError("diffusion_unit must be 'meV*Å**2' or 'm**2/s'.")

        if diffusion_unit == "meV*Å**2" or diffusion_unit == "meV*angstrom**2":
            # In this case, hbar is absorbed in the unit of D
            self._hbar = DescriptorNumber("hbar", 1.0)
        elif diffusion_unit == "m**2/s" or diffusion_unit == "m^2/s":
            self._hbar = DescriptorNumber.from_scipp("hbar", scipp_hbar)
        else:
            raise ValueError("diffusion_unit must be 'meV*Å**2' or 'm**2/s'.")

        if not isinstance(scale, Parameter):
            scale = Parameter(name="scale", value=float(scale), fixed=False)

        if not isinstance(diffusion_coefficient, Parameter):
            diffusion_coefficient = Parameter(
                name="diffusion_coefficient",
                value=float(diffusion_coefficient),
                fixed=False,
                unit=diffusion_unit,
            )
        super().__init__(
            name=name,
            unit=unit,
            scale=scale,
            diffusion_coefficient=diffusion_coefficient,
        )
        self._angstrom = DescriptorNumber("angstrom", 1e-10, unit="m")

    def calculate_width(self, Q: np.ndarray) -> np.ndarray:
        """
        Calculate the half-width at half-maximum (HWHM) for the diffusion model.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector in 1/angstrom

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """
        if not isinstance(Q, np.ndarray):
            raise TypeError("Q must be a numpy array.")

        width_list = []
        for Q_value in Q:
            # Q is given as a float, so we need to divide by angstrom**2 to get the right units
            width = (
                self._hbar
                * self.diffusion_coefficient
                * Q_value**2
                / (self._angstrom**2)
            )
            width.convert_unit(self.unit)
            width_list.append(width.value)
        width = np.array(width_list)

        return width

    def calculate_EISF(self, Q: np.ndarray) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF) for the Brownian translational diffusion model.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector in 1/angstrom

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """
        if not isinstance(Q, np.ndarray):
            raise TypeError("Q must be a numpy array.")
        EISF = np.zeros_like(Q)
        return EISF

    def calculate_QISF(self, Q: np.ndarray) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF).

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector in 1/angstrom

        Returns
        -------
        np.ndarray
            QISF values (dimensionless).
        """

        if not isinstance(Q, np.ndarray):
            raise TypeError("Q must be a numpy array.")
        QISF = np.ones_like(Q)
        return QISF

    def create_sample_models(
        self,
        Q: Union[Number, list, np.ndarray],
        component_name: str = "Lorentzian",
    ) -> List[SampleModel]:
        """
        Create SampleModel components for the Brownian translational diffusion model at given Q values.
        Args:
        ----------
        Q : Number, list, or np.ndarray
            Scattering vector values.
        component_name : str
            Name of the Lorentzian component.
        width_name : str
            Name of the width parameter.
        Returns
        -------
        List[SampleModel]
            List of SampleModels with Lorentzian components.
        """

        if isinstance(Q, (Number, list)):
            Q = np.array(Q)
        if not isinstance(Q, np.ndarray):
            raise TypeError("Q must be a number, list, or numpy array.")

        if Q.ndim != 1:
            raise ValueError("Q must be a 1-dimensional array.")

        if not isinstance(component_name, str):
            raise TypeError("component_name must be a string.")

        sample_model_list = [None] * len(Q)
        # In more complex models, this is used to scale the area of the Lorentzians and the delta function.
        QISF = self.calculate_QISF(Q)

        # Create a Lorentzian component for each Q-value, with width D*Q^2 and area equal to scale. No delta function, as the EISF is 0.
        for i in range(len(Q)):
            sample_model_list[i] = SampleModel(
                name=f"{self.name}_Q{Q[i]:.2f}", unit=self.unit
            )

            lorentzian_component = Lorentzian(
                name=component_name, area=self.scale * QISF[i], unit=self.unit
            )

            # Make the width dependent on Q
            dependency_expression = self._write_width_dependency_expression(Q[i])
            dependency_map = self._write_width_dependency_map_expression()

            lorentzian_component.width.make_dependent_on(
                dependency_expression=dependency_expression,
                dependency_map=dependency_map,
            )

            # Resolving the dependency can do weird things to the units, so we make sure it's correct.
            lorentzian_component.width.convert_unit(self.unit)
            sample_model_list[i].add_component(lorentzian_component)

        return sample_model_list

    def _write_width_dependency_expression(self, Q: float) -> str:
        """
        Write the dependency expression for the width as a function of Q to make dependent Parameters.
        """
        if not isinstance(Q, (float)):
            raise TypeError("Q must be a float.")

        # Q is given as a float, so we need to add the units
        return f"hbar * D* {Q} **2*1/(angstrom**2)"

    def _write_width_dependency_map_expression(self) -> Dict[str, str]:
        """
        Write the dependency map expression to make dependent Parameters.
        """
        return {
            "D": self.diffusion_coefficient,
            "hbar": self._hbar,
            "angstrom": self._angstrom,
        }

    def __repr__(self):
        """
        String representation of the BrownianTranslationalDiffusion model.
        """
        return f"BrownianTranslationalDiffusion(name={self.name}, diffusion_coefficient={self.diffusion_coefficient}, scale={self.scale})"
