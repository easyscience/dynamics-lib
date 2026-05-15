# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.diffusion_model_base import (
    DiffusionModelBase,
)
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class DeltaLorentz(DiffusionModelBase):
    r"""
    Model of Delta function and Lorentzian with intensities given by the Debye-Waller factor. $$ I
    = K \exp \left( \frac{-\langle u^2 \rangle Q^2}{3} \right)[A_0 \delta(E) + (A_1) L(E, \Gamma)]
    $$,

    where $K$ is the scale factor, $\langle u^2 \rangle$ is the mean square displacement, $Q$ is
    the scattering vector, $A_0$ and $A_1$ are the amplitudes of the delta function and Lorentzian,
    respectively, and $L(E, \Gamma)$ is the Lorentzian function with width $\Gamma$. $A_0+A_1=1$
    and $A_0$ is the EISF, while $A_1$ is the QISF. $A_0$ and $A_1$ can be Q-dependent or not.


    Examples
    --------
    >>> Q = np.linspace(0.5, 2, 7)
    >>> energy = np.linspace(-2, 2, 501)
    >>> scale = 1.0
    >>> mean_u_squared = 0.02
    >>> A_0 = 0.7
    >>> lorentzian_width = 1.0
    >>> model = DeltaLorentz(
    ...     display_name='DiffusionModel',
    ...     scale=scale,
    ...     mean_u_squared=mean_u_squared,
    ...     A_0=A_0,
    ...     lorentzian_width=lorentzian_width,
    ...     allow_Q_dependence=True,
    ... )
    >>> component_collections = model.create_component_collections(Q)

    See also the tutorials.
    """

    def __init__(
        self,
        scale: Numeric = 1.0,
        mean_u_squared: Numeric = 0.0,
        A_0: Numeric = 1.0,
        lorentzian_width: Numeric = 1.0,
        allow_Q_dependence: bool = False,
        unit: str | sc.Unit = "meV",
        display_name: str | None = "DeltaLorentz",
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new DeltaLorentz model.

        Parameters
        ----------
        unit : str | sc.Unit, default="meV"
            Unit of the diffusion model. Must be convertible to meV.
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        mean_u_squared : Numeric, default=0.0
            Mean square displacement in angstrom^2.
        A_0 : Numeric, default=1.0
            Amplitude of the delta function.
        lorentzian_width : Numeric, default=1.0
            Width of the Lorentzian function.
        allow_Q_dependence : bool, default=False
            Whether to allow Q-dependence of A_0 and A_1
        display_name : str | None, default="DeltaLorentz"
            Display name of the diffusion model.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated. By
            default, None.

        Raises
        ------
        TypeError
            If scale, mean_u_squared, A_0, or lorentzian_width is not a number.
        """
        if not isinstance(scale, Numeric):
            raise TypeError("scale must be a number.")

        if not isinstance(mean_u_squared, Numeric):
            raise TypeError("mean_u_squared must be a number.")

        if not isinstance(A_0, Numeric):
            raise TypeError("A_0 must be a number.")

        if not isinstance(lorentzian_width, Numeric):
            raise TypeError("lorentzian_width must be a number.")

        if not isinstance(allow_Q_dependence, bool):
            raise TypeError("allow_Q_dependence must be True or False.")

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
            scale=scale,
        )

        A_0 = Parameter(
            name="A_0",
            value=float(A_0),
            fixed=False,
            min=0.0,
            max=1.0,
        )
        self._A_0 = A_0

        A_1 = Parameter.from_dependency(
            name="A_1",
            dependency_expression="1 - A_0",
            dependency_map={"A_0": A_0},
        )
        self._A_1 = A_1

        self._allow_Q_dependence = allow_Q_dependence

        self._A_0_list = []
        self._A_1_list = []

        mean_u_squared = Parameter(
            name="mean_u_squared",
            value=float(mean_u_squared),
            fixed=False,
            min=0.0,
            unit="angstrom**2",
        )
        self._mean_u_squared = mean_u_squared

        lorentzian_width = Parameter(
            name="lorentzian_width",
            value=float(lorentzian_width),
            fixed=False,
            min=MINIMUM_WIDTH,
            unit=unit,
        )
        self._lorentzian_width = lorentzian_width

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean_u_squared(self) -> Parameter:
        """
        Get the mean square displacement parameter.

        Returns
        -------
        Parameter
            Mean square displacement in angstrom^2.
        """
        return self._mean_u_squared

    @mean_u_squared.setter
    def mean_u_squared(self, mean_u_squared: Numeric) -> None:
        """
        Set the mean square displacement parameter.

        Parameters
        ----------
        mean_u_squared : Numeric
            The new value for the mean square displacement in angstrom^2.

        Raises
        ------
        TypeError
            If mean_u_squared is not a number.
        ValueError
            If mean_u_squared is negative.
        """
        if not isinstance(mean_u_squared, Numeric):
            raise TypeError("mean_u_squared must be a number.")

        if float(mean_u_squared) < 0:
            raise ValueError("mean_u_squared must be non-negative.")
        self._mean_u_squared.value = float(mean_u_squared)

    @property
    def A_0(self) -> Parameter:
        """
        Get the amplitude of the delta function.

        Returns
        -------
        Parameter
            Amplitude of the delta function.
        """
        return self._A_0

    @A_0.setter
    def A_0(self, A_0: Numeric) -> None:
        """
        Set the amplitude of the delta function.

        Parameters
        ----------
        A_0 : Numeric
            The new value for the amplitude of the delta function. Must be between 0 and 1.

        Raises
        ------
        TypeError
            If A_0 is not a number.
        ValueError
            If A_0 is not between 0 and 1.
        """
        if not isinstance(A_0, Numeric):
            raise TypeError("A_0 must be a number.")

        if not (0 <= float(A_0) <= 1):
            raise ValueError("A_0 must be between 0 and 1.")
        self._A_0.value = float(A_0)

    @property
    def A_1(self) -> Parameter:
        """
        Get the amplitude of the Lorentzian function.

        Returns
        -------
        Parameter
            Amplitude of the Lorentzian function.
        """
        return self._A_1

    @A_1.setter
    def A_1(self, _A_1: Numeric) -> None:
        """
        A_1 cannot be set directly, as it is a dependent parameter defined as 1 - A_0. To change
        A_1, set A_0 to the desired value and A_1 will update accordingly.


        Parameters
        ----------
        _A_1 : Numeric
            The new value for the amplitude of the Lorentzian function. Is ignored

        Raises
        ------
         AttributeError If an attempt is made to set A_1 directly.
        """
        raise AttributeError(
            "A_1 is a dependent parameter and cannot be set directly. Set A_0 to change A_1 accordingly."
        )

    @property
    def lorentzian_width(self) -> Parameter:
        """
        Get the width of the Lorentzian function.

        Returns
        -------
        Parameter
            Width of the Lorentzian function.
        """
        return self._lorentzian_width

    @lorentzian_width.setter
    def lorentzian_width(self, lorentzian_width: Numeric) -> None:
        """
        Set the width of the Lorentzian function.

        Parameters
        ----------
        lorentzian_width : Numeric
            The new value for the width of the Lorentzian function. Must be a non-negative number.

        Raises
        ------
        TypeError
            If lorentzian_width is not a number.
        ValueError
            If lorentzian_width is less than the minimum allowed width.
        """
        if not isinstance(lorentzian_width, Numeric):
            raise TypeError("lorentzian_width must be a number.")

        if float(lorentzian_width) < MINIMUM_WIDTH:
            raise ValueError(f"lorentzian_width must be at least {MINIMUM_WIDTH}.")
        self._lorentzian_width.value = float(lorentzian_width)

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type) -> np.ndarray:
        """
        Calculate the half-width at half-maximum (HWHM) for the diffusion model.

        Parameters
        ----------
        Q : Q_type
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """

        Q = _validate_and_convert_Q(Q)

        return self.lorentzian_width.value * np.ones_like(Q)

    def calculate_EISF(self, Q: Q_type) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF) for the Brownian translational
        diffusion model.

        Parameters
        ----------
        Q : Q_type
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """

        # Need to handle units better
        Q = _validate_and_convert_Q(Q)
        if self._allow_Q_dependence is True:
            A_0_values = [A_0.value for A_0 in self._A_0_list]
        else:
            A_0_values = [self.A_0.value] * len(Q)
        return np.exp(-self.mean_u_squared.value * Q**2 / 3) * np.array(A_0_values)

    def calculate_QISF(self, Q: Q_type) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF).

        Parameters
        ----------
        Q : Q_type
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            QISF values (dimensionless).
        """

        Q = _validate_and_convert_Q(Q)
        if self._allow_Q_dependence is True:
            A_1_values = [A_1.value for A_1 in self._A_1_list]
        else:
            A_1_values = [self.A_1.value] * len(Q)
        return np.exp(-self.mean_u_squared.value * Q**2 / 3) * np.array(A_1_values)

    def create_component_collections(
        self,
        Q: Q_type,
        component_display_name: str = "DeltaLorentz component",
    ) -> list[ComponentCollection]:
        r"""
        Create ComponentCollection components for the DeltaLorentz model at given Q values.

        Parameters
        ----------
        Q : Q_type
            Scattering vector values.
        component_display_name : str, default="DeltaLorentz component"
            Name of the Lorentzian component.

        Raises
        ------
        TypeError
            If component_display_name is not a string.

        Returns
        -------
        list[ComponentCollection]
            List of ComponentCollections with Lorentzian and delta functioncomponents for each Q
            value.
        """
        Q = _validate_and_convert_Q(Q)

        if not isinstance(component_display_name, str):
            raise TypeError("component_name must be a string.")

        if self._allow_Q_dependence is True:
            A_0_list, A_1_list = self._create_A0_A1_parameters(self.A_0, Q)
            self._A_0_list = A_0_list
            self._A_1_list = A_1_list

        component_collection_list = [None] * len(Q)
        # In more complex models, this is used to scale the area of the
        # Lorentzians and the delta function.

        # Create a Lorentzian component for each Q-value, with
        # width D*Q^2 and area equal to scale.
        # No delta function, as the EISF is 0.
        for i, Q_value in enumerate(Q):
            component_collection_list[i] = ComponentCollection(
                display_name=f"{self.display_name}_Q{Q_value:.2f}", unit=self.unit
            )

            lorentzian_component = Lorentzian(
                display_name=component_display_name,
                unit=self.unit,
            )

            # Make the width dependent on Q
            lorentzian_component.width.make_dependent_on(
                dependency_expression=self._write_width_dependency_expression(Q_value),
                dependency_map=self._write_width_dependency_map_expression(),
                desired_unit=self.unit,
            )

            # Make the area dependent on Q
            if self._allow_Q_dependence is True:
                dependency_map = self._write_lorz_area_dependency_map_expression(i)
            else:
                dependency_map = self._write_lorz_area_dependency_map_expression(None)
            lorentzian_component.area.make_dependent_on(
                dependency_expression=self._write_lorz_area_dependency_expression(
                    Q_value
                ),
                dependency_map=dependency_map,
            )

            component_collection_list[i].append_component(lorentzian_component)

            delta_component = DeltaFunction(
                display_name="Delta function", unit=self.unit
            )
            if self._allow_Q_dependence is True:
                dependency_map = self._write_delta_area_dependency_map_expression(i)
            else:
                dependency_map = self._write_delta_area_dependency_map_expression(None)
            delta_component.area.make_dependent_on(
                dependency_expression=self._write_delta_area_dependency_expression(
                    Q_value
                ),
                dependency_map=dependency_map,
            )

            component_collection_list[i].append_component(delta_component)

        return component_collection_list

    def get_all_variables(self) -> list[DescriptorNumber]:

        if self._allow_Q_dependence is False:
            return super().get_all_variables()

        variables = [self.scale, self.mean_u_squared, self.lorentzian_width]
        variables.extend(self._A_0_list)
        variables.extend(self._A_1_list)
        return variables

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _create_A0_A1_parameters(
        self, A_0: Parameter, Q: Q_type
    ) -> tuple[list[Parameter], list[Parameter]]:
        """
        Create lists of A_0 and A_1 parameters for each Q value.
        Parameters
        ----------
        A_0 : Parameter
            The A_0 parameter to use as the base for creating the A_0 parameters for each Q value.
        Returns
        -------
        tuple[list[Parameter], list[Parameter]]
            A tuple containing two lists: the first list contains the A_0 parameters for each Q
            value, and the second list contains the A_1 parameters for each Q value.
        """
        A_0_list = []
        A_1_list = []
        for i, Q_value in enumerate(Q):
            A_0_list.append(
                Parameter(
                    name=f"A_0_Q{Q_value:.2f}",
                    value=float(A_0.value),
                    fixed=False,
                    min=0.0,
                    max=1.0,
                )
            )
            A_1_list.append(
                Parameter.from_dependency(
                    name=f"A_1_Q{Q_value:.2f}",
                    dependency_expression="1 - A_0",
                    dependency_map={"A_0": A_0_list[i]},
                )
            )

        self._A_0_list = A_0_list
        self._A_1_list = A_1_list

        return A_0_list, A_1_list

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
            raise TypeError("Q must be a float.")

        return "lorentzian_width"

    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """
        return {
            "lorentzian_width": self.lorentzian_width,
        }

    def _write_lorz_area_dependency_expression(self, Q) -> str:
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
        if not isinstance(Q, (float)):
            raise TypeError("Q must be a float.")

        return f"scale * exp(-mean_u_squared.value * {Q}**2 / 3) * A_1"

    def _write_lorz_area_dependency_map_expression(
        self, Q_index
    ) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the area.
        """
        if Q_index is None:
            return {
                "scale": self.scale,
                "mean_u_squared": self.mean_u_squared,
                "A_1": self.A_1,
            }

        return {
            "scale": self.scale,
            "mean_u_squared": self.mean_u_squared,
            "A_1": self._A_1_list[Q_index],
        }

    def _write_delta_area_dependency_expression(self, Q) -> str:
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
        if not isinstance(Q, (float)):
            raise TypeError("Q must be a float.")

        return f"scale * exp(-mean_u_squared.value * {Q}**2 / 3) * A_0"

    def _write_delta_area_dependency_map_expression(
        self,
        Q_index,
    ) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the area.
        """
        if Q_index is None:
            return {
                "scale": self.scale,
                "mean_u_squared": self.mean_u_squared,
                "A_0": self.A_0,
            }
        return {
            "scale": self.scale,
            "mean_u_squared": self.mean_u_squared,
            "A_0": self._A_0_list[Q_index],
        }

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the DeltaLorentz model.

        Returns
        -------
        str
            String representation of the DeltaLorentz model.
        """
        return (
            f"DeltaLorentz(display_name={self.display_name},"
            f"unit={self.unit}, \n"
            f"    mean_u_squared={self.mean_u_squared}, \n"
            f"    A_0={self.A_0}, A_1={self.A_1}, \n"
            f"    lorentzian_width={self.lorentzian_width}, \n"
            f"    scale={self.scale})"
        )
