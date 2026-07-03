# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.utils.fit_target import FitTarget
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import angstrom
from easydynamics.utils.utils import verify_Q_index

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class DeltaLorentz(DiffusionModelBase):
    r"""
    Model of Delta function and Lorentzian with intensities given by the Debye-Waller factor. $$ I
    = K \exp \left( \frac{-\langle u^2 \rangle Q^2}{3} \right)[A_0 \delta(E) + (A_1) L(E, \Gamma)]
    $$,

    where $K$ is the scale factor, $\langle u^2 \rangle$ is the mean square displacement, $Q$ is
    the scattering vector, $A_0$ and $A_1$ are the relative amplitudes of the delta function and
    Lorentzian, respectively, with the constraint that $A_0+A_1=1$, and $L(E, \Gamma)$ is the
    Lorentzian function with width $\Gamma$. $A_0$, $A_1$ and the width of the Lorentzian can be
    the same at all $Q$ or be allowed to vary with $Q$.


    Examples
    --------
    **Creating a DeltaLorentz model with Q-dependent parameters**

    Set ``allow_Q_variation`` to allow individual parameters to vary with Q:
    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    Q = np.linspace(0.5, 2, 7)
    model = sm.DeltaLorentz(
        display_name='DiffusionModel',
        scale=1.0,
        mean_u_squared=0.02,
        A_0=0.7,
        lorentzian_width=1.0,
        allow_Q_variation={'A_0': True, 'lorentzian_width': True},
        Q=Q,
    )
    component_collections = model.create_component_collections()
    ```

    See also the tutorials.
    """

    def __init__(
        self,
        scale: Numeric = 1.0,
        mean_u_squared: Numeric = 0.0,
        A_0: Numeric = 1.0,
        lorentzian_width: Numeric = 1.0,
        allow_Q_variation: dict | None = None,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'DeltaLorentz',
        display_name: str | None = None,
        lorentzian_name: str = 'Lorentzian',
        lorentzian_display_name: str | None = None,
        delta_name: str = 'Delta function',
        delta_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new DeltaLorentz model.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        mean_u_squared : Numeric, default=0.0
            Mean square displacement in angstrom^2.
        A_0 : Numeric, default=1.0
            Amplitude of the delta function.
        lorentzian_width : Numeric, default=1.0
            Width of the Lorentzian function.
        allow_Q_variation : dict | None, default=None
            Dict describing whether to allow Q variation of A_0 and the Lorentzian width. The dict
            can have the keys "A_0" and/or "lorentzian_width", with boolean values indicating
            whether to allow Q-dependence for each parameter. If None, no Q-dependence will be
            allowed.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        name : str, default='DeltaLorentz'
            Name of the diffusion model.
        display_name : str | None, default=None
            Display name of the diffusion model.
        lorentzian_name : str, default='Lorentzian'
            Name of the Lorentzian component. If None, it will be set to the name of the diffusion
            model.
        lorentzian_display_name : str | None, default=None
            Display name of the Lorentzian component. If None, it will be set to the display name
            of the diffusion model.
        delta_name : str, default='Delta function'
            Name of the delta function component.
        delta_display_name : str | None, default=None
            Display name of the delta function component. If None, it will be set to the display
            name of the delta function component.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated. By
            default, None.

        Raises
        ------
        TypeError
            If delta_name is not a string or if delta_display_name is not a string or None.
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
        self._mean_u_squared = self._create_mean_u_squared_parameter(mean_u_squared)

        # Dimensionless <u^2>/angstrom^2 for use inside exp() in the area dependency
        # expressions and the EISF/QISF calculations. Deriving it through the dependency
        # graph keeps those quantities correct if mean_u_squared is converted to another
        # unit (e.g. nm**2), which a raw `.value` would silently get wrong.
        with np.errstate(invalid='ignore'):
            self._mean_u_squared_over_angstrom_squared = Parameter.from_dependency(
                name='mean_u_squared_over_angstrom_squared',
                dependency_expression='mean_u_squared / angstrom**2',
                dependency_map={'mean_u_squared': self._mean_u_squared, 'angstrom': angstrom},
            )
            self._mean_u_squared_over_angstrom_squared.set_desired_unit('dimensionless')

        self._A_0, self._A_1 = self._create_A0_A1_parameters(A_0)

        self._lorentzian_width = self._create_lorentzian_width_parameter(lorentzian_width)

        # --------------------------------------------------------------
        # names
        # --------------------------------------------------------------
        if not isinstance(delta_name, str):
            raise TypeError('delta_name must be a string.')

        if delta_display_name is None:
            delta_display_name = delta_name

        if not isinstance(delta_display_name, str):
            raise TypeError('delta_display_name must be a string or None.')

        self._delta_name = delta_name
        self._delta_display_name = delta_display_name

        # --------------------------------------------------------------
        # Q variation
        # --------------------------------------------------------------
        self._allow_Q_variation = self._create_Q_variation_dict(allow_Q_variation)

        self._A_0_list = []
        self._A_1_list = []
        self._lorentzian_width_list = []
        if self.Q is not None:
            if self._allow_Q_variation['A_0'] is True:
                self._A_0_list, self._A_1_list = self._create_A0_A1_parameter_lists(self.A_0)

            if self._allow_Q_variation['lorentzian_width'] is True:
                self._lorentzian_width_list = self._create_lorentzian_width_parameter_list(
                    self.lorentzian_width,
                )

        self._component_collections = self.create_component_collections()

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
            raise TypeError('mean_u_squared must be a number.')

        if float(mean_u_squared) < 0:
            raise ValueError('mean_u_squared must be non-negative.')
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
            raise TypeError('A_0 must be a number.')

        if not (0 <= float(A_0) <= 1):
            raise ValueError('A_0 must be between 0 and 1.')
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
        AttributeError
            If an attempt is made to set A_1 directly.
        """
        raise AttributeError(
            'A_1 is a dependent parameter and cannot be set directly. '
            'Set A_0 to change A_1 accordingly.'
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
            raise TypeError('lorentzian_width must be a number.')

        if float(lorentzian_width) < MINIMUM_WIDTH:
            raise ValueError(f'lorentzian_width must be at least {MINIMUM_WIDTH}.')
        self._lorentzian_width.value = float(lorentzian_width)

    @property
    def delta_name(self) -> str:
        """
        Get the name of the delta function component.

        Returns
        -------
        str
            Name of the delta function component.
        """
        return self._delta_name

    @delta_name.setter
    def delta_name(self, delta_name: str) -> None:
        """
        Set the name of the delta function component.

        Parameters
        ----------
        delta_name : str
            The new name for the delta function component.

        Raises
        ------
        TypeError
            If delta_name is not a string.
        """
        if not isinstance(delta_name, str):
            raise TypeError('delta_name must be a string.')
        self._delta_name = delta_name

    @property
    def delta_display_name(self) -> str:
        """
        Get the display name of the delta function component.

        Returns
        -------
        str
            Display name of the delta function component.
        """
        return self._delta_display_name

    @delta_display_name.setter
    def delta_display_name(self, delta_display_name: str | None) -> None:
        """
        Set the display name of the delta function component.

        Parameters
        ----------
        delta_display_name : str | None
            The new display name for the delta function component.

        Raises
        ------
        TypeError
            If delta_display_name is not a string or None.
        """
        if not isinstance(delta_display_name, (str, type(None))):
            raise TypeError('delta_display_name must be a string or None.')
        self._delta_display_name = delta_display_name

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def calculate_width(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the half-width at half-maximum (HWHM) for the diffusion model. If the width is
        allowed to vary with Q then the Q stored in the model is used and the input is ignored. If
        the width is not allowed to vary then the same width is returned for all Q values.

        Parameters
        ----------
        Q : Q_type, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).

        Raises
        ------
        ValueError
            If Q-variation is enabled but Q has not been set on the model yet.
        """
        if self._allow_Q_variation['lorentzian_width'] is True:
            if not self._lorentzian_width_list:
                raise ValueError(
                    'Lorentzian width Q-variation list is empty. '
                    'Set Q before calling calculate_width.'
                )
            widths = [lorentzian_width.value for lorentzian_width in self._lorentzian_width_list]
            return np.array(widths)

        Q = self._ensure_Q(Q)

        widths = self.lorentzian_width.value * np.ones_like(Q)

        return np.array(widths)

    def calculate_EISF(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF) for the diffusion model.

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
        mean_u_squared = self._mean_u_squared_over_angstrom_squared.value
        if self._allow_Q_variation['A_0'] is True:
            A_0_values = [A_0_.value for A_0_ in self._A_0_list]
            return np.exp(-mean_u_squared * Q**2 / 3) * np.array(A_0_values)

        A_0_values = [self.A_0.value] * len(Q)
        return np.exp(-mean_u_squared * Q**2 / 3) * np.array(A_0_values)

    def calculate_QISF(self, Q: Q_type = None) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF).

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
        mean_u_squared = self._mean_u_squared_over_angstrom_squared.value
        if self._allow_Q_variation['A_0'] is True:
            A_1_values = [A_1_.value for A_1_ in self._A_1_list]
            return np.exp(-mean_u_squared * Q**2 / 3) * np.array(A_1_values)

        A_1_values = [self.A_1.value] * len(Q)
        return np.exp(-mean_u_squared * Q**2 / 3) * np.array(A_1_values)

    def create_component_collections(
        self,
    ) -> list[ComponentCollection]:
        r"""
        Create ComponentCollections  for the DeltaLorentz model at given Q values.

        Returns
        -------
        list[ComponentCollection]
            List of ComponentCollections with Lorentzian and delta function components for each Q
            value.
        """
        if self.Q is None:
            return []

        Q = self.Q.values

        if self._allow_Q_variation['A_0'] is True:
            A_0_list, A_1_list = self._create_A0_A1_parameter_lists(self.A_0)
            self._A_0_list = A_0_list
            self._A_1_list = A_1_list

        if self._allow_Q_variation['lorentzian_width'] is True:
            lorentzian_width_list = self._create_lorentzian_width_parameter_list(
                self.lorentzian_width
            )
            self._lorentzian_width_list = lorentzian_width_list

        component_collection_list = [None] * len(Q)
        for i, Q_value in enumerate(Q):
            component_collection_list[i] = ComponentCollection(
                display_name=f'{self.display_name}_Q{Q_value:.2f}',
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )

            # ------------------------------#
            # Create Lorentzian
            # ------------------------------#
            lorz_component = Lorentzian(
                name=self.lorentzian_name,
                display_name=self.lorentzian_display_name,
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )
            if self._allow_Q_variation['lorentzian_width'] is True:
                lorz_component._width = self._lorentzian_width_list[i]  # noqa: SLF001

            # If the width is allowed to vary with Q it is independent.
            # If the width is not allowed to vary with Q it must be made
            # dependent on the width parameter of the model.
            if self._allow_Q_variation['lorentzian_width'] is False:
                dependency_map = self._write_width_dependency_map_expression()

                lorz_component.width.make_dependent_on(
                    dependency_expression=self._write_lorz_width_dependency_expression(Q_value),
                    dependency_map=dependency_map,
                    desired_unit=self.x_unit,
                )
            # The area is always a dependent parameter in this model, as
            # it depends on the scale, mean_u_squared and A_1 parameters
            # of the model. If A_1 is allowed to vary with Q, the area
            # will also depend on the specific A_1 parameter for that Q
            # value. If A_1 is not allowed to vary with Q, the area will
            # depend on the single A_1 parameter of the model.
            if self._allow_Q_variation['A_0'] is True:
                dependency_map = self._write_lorz_area_dependency_map_expression(i)
            else:
                dependency_map = self._write_lorz_area_dependency_map_expression(None)

            lorz_component.area.make_dependent_on(
                dependency_expression=self._write_lorz_area_dependency_expression(Q_value),
                dependency_map=dependency_map,
            )

            component_collection_list[i].append_component(lorz_component)

            # ------------------------------#
            # Create delta function
            # ------------------------------#

            delta_component = DeltaFunction(
                name=self.delta_name,
                display_name=self.delta_display_name,
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )

            if self._allow_Q_variation['A_0'] is True:
                dependency_map = self._write_delta_area_dependency_map_expression(i)
            else:
                dependency_map = self._write_delta_area_dependency_map_expression(None)

            delta_component.area.make_dependent_on(
                dependency_expression=self._write_delta_area_dependency_expression(Q_value),
                dependency_map=dependency_map,
            )

            component_collection_list[i].append_component(delta_component)

        return component_collection_list

    def get_fit_targets(self) -> list[FitTarget]:
        """
        Get the fittable predictions of the DeltaLorentz model as FitTargets.

        Extends the base ``'area'`` and ``'width'`` predictions with ``'delta_area'`` (``scale *
        EISF(Q)``, the delta function's weight), whose default dataset key is derived from the
        delta component's name.

        Returns
        -------
        list[FitTarget]
            The fittable predictions of this model.
        """
        targets = super().get_fit_targets()
        targets.append(
            FitTarget(
                name='delta_area',
                dataset_key=f'{self.delta_name} area',
                function=lambda Q, model=self, **_: model.calculate_EISF(Q) * model.scale.value,
                label=f'{self.display_name} delta_area',
                x_unit='1/angstrom',
                y_unit=str(self.scale.unit),
            )
        )
        return targets

    def get_global_variables(self) -> list[Parameter]:
        """
        Get all global variables from the diffusion model.

        Returns
        -------
        list[Parameter]
            A list of all global variables from the diffusion model.
        """
        variables = [self.scale, self.mean_u_squared]

        if self._allow_Q_variation['lorentzian_width'] is False:
            variables.append(self.lorentzian_width)

        if self._allow_Q_variation['A_0'] is False:
            variables.append(self.A_0)
            variables.append(self.A_1)

        return variables

    def get_independent_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get the independent variables from the diffusion model. If Q_index is provided, only the
        independent variables for the specified Q value will be returned. If Q_index is None,
        independent variables for all Q values will be returned.

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

        # Also add all dependent variables from the component collections
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
        Create a dict for the allow_Q_variation attribute, ensuring that it has the correct keys
        and default values.

        Parameters
        ----------
        allow_Q_variation : dict | None
            Dict describing whether to allow Q variation of A_0 and the Lorentzian width.

        Raises
        ------
        TypeError
            If allow_Q_variation is not a dict or None.
        ValueError
            If allow_Q_variation contains unknown keys.

        Returns
        -------
        dict
            A dict with keys 'A_0' and 'lorentzian_width', indicating whether to allow Q variation
            for each parameter.
        """

        allow_Q_variation_default = {
            'A_0': False,
            'lorentzian_width': False,
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

    def _create_mean_u_squared_parameter(self, mean_u_squared: Numeric) -> Parameter:
        """
        Create the mean square displacement parameter.

        Parameters
        ----------
        mean_u_squared : Numeric
            The value for the mean square displacement in angstrom^2.

        Raises
        ------
        TypeError
            If mean_u_squared is not a number.
        ValueError
            If mean_u_squared is negative.

        Returns
        -------
        Parameter
            The created mean square displacement parameter.
        """

        if not isinstance(mean_u_squared, Numeric):
            raise TypeError('mean_u_squared must be a number.')

        if float(mean_u_squared) < 0:
            raise ValueError('mean_u_squared must be non-negative.')

        return Parameter(
            name='mean_u_squared',
            value=float(mean_u_squared),
            fixed=False,
            min=0.0,
            unit='angstrom**2',
        )

    def _create_A0_A1_parameters(self, A_0: Numeric) -> tuple[Parameter, Parameter]:
        """
        Create the A_0 and A_1 parameters.

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

        A_0 = Parameter(
            name='A_0',
            value=float(A_0),
            fixed=False,
            min=0.0,
            max=1.0,
        )

        A_1 = Parameter.from_dependency(
            name='A_1',
            dependency_expression='1 - A_0',
            dependency_map={'A_0': A_0},
        )
        return A_0, A_1

    def _create_lorentzian_width_parameter(self, lorentzian_width: Numeric) -> Parameter:
        """
        Create the Lorentzian width parameter.

        Parameters
        ----------
        lorentzian_width : Numeric
            The value for the Lorentzian width parameter.

        Raises
        ------
        TypeError
            If lorentzian_width is not a number.
        ValueError
            If lorentzian_width is less than the minimum width.

        Returns
        -------
        Parameter
            The created Lorentzian width parameter.
        """

        if not isinstance(lorentzian_width, Numeric):
            raise TypeError('lorentzian_width must be a number.')

        if float(lorentzian_width) < MINIMUM_WIDTH:
            raise ValueError(f'lorentzian_width must be at least {MINIMUM_WIDTH}.')

        return Parameter(
            name='lorentzian_width',
            value=float(lorentzian_width),
            fixed=False,
            min=MINIMUM_WIDTH,
            unit=self.x_unit,
        )

    def _create_A0_A1_parameter_lists(
        self,
        A_0: Parameter,
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
        for _ in range(len(self.Q)):
            a0 = Parameter(
                name='A_0',
                value=float(A_0.value),
                fixed=False,
                min=0.0,
                max=1.0,
            )

            a1 = Parameter.from_dependency(
                name='A_1',
                dependency_expression='1 - A_0',
                dependency_map={'A_0': a0},
            )

            A_0_list.append(a0)
            A_1_list.append(a1)

        return A_0_list, A_1_list

    def _create_lorentzian_width_parameter_list(
        self,
        lorentzian_width: Parameter,
    ) -> list[Parameter]:
        """
        Create a list of Lorentzian width parameters for each Q value.

        Parameters
        ----------
        lorentzian_width : Parameter
            The Lorentzian width parameter to use as the base for creating the Lorentzian width
            parameters for each Q value.

        Returns
        -------
        list[Parameter]
            A list containing the Lorentzian width parameters for each Q value.
        """
        return [
            Parameter(
                name=f'{self.lorentzian_name} width',
                value=float(lorentzian_width.value),
                fixed=False,
                min=MINIMUM_WIDTH,
                unit=self.x_unit,
            )
            for _ in range(len(self.Q))
        ]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _on_Q_change(self) -> None:
        """
        Handle changes to the Q values. Updates the A_0, A_1 and lorentzian_width parameters if
        they are allowed to vary with Q.
        """
        if self.Q is None:
            self._A_0_list = []
            self._A_1_list = []
            self._lorentzian_width_list = []
        else:
            if self._allow_Q_variation['A_0'] is True:
                self._A_0_list, self._A_1_list = self._create_A0_A1_parameter_lists(self.A_0)
            else:
                self._A_0_list = []
                self._A_1_list = []

            if self._allow_Q_variation['lorentzian_width'] is True:
                self._lorentzian_width_list = self._create_lorentzian_width_parameter_list(
                    self.lorentzian_width
                )
            else:
                self._lorentzian_width_list = []
        self._component_collections = self.create_component_collections()

    def _convert_extra_x_unit_parameters(self, unit_str: str) -> None:
        """
        Convert the Lorentzian width template to the new x-axis unit.

        The per-Q width list (when Q-variation is enabled) holds the very Parameter objects used by
        the components, so those are converted in place with the collections.

        Parameters
        ----------
        unit_str : str
            The new x-axis unit as a string.
        """
        self._lorentzian_width.convert_unit(unit_str)

    def _write_lorz_width_dependency_expression(self, Q: float) -> str:
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

        return 'lorentzian_width'

    def _write_width_dependency_map_expression(
        self,
    ) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """
        return {
            'lorentzian_width': self.lorentzian_width,
        }

    def _write_lorz_area_dependency_expression(self, Q: float) -> str:
        """
        Write the dependency expression for the area to make dependent Parameters.

        Parameters
        ----------
        Q : float
            Scattering vector in 1/angstrom .

        Raises
        ------
        TypeError
            If Q is not a float.

        Returns
        -------
        str
            Dependency expression for the area.
        """
        if not isinstance(Q, (float)):
            raise TypeError('Q must be a float.')

        # mean_u_squared_ratio is <u^2>/angstrom^2, kept dimensionless through the dependency
        # graph so the expression stays correct if mean_u_squared is converted to another unit.
        return f'scale * exp(-mean_u_squared_ratio.value * {Q}**2 / 3) * A_1'

    def _write_lorz_area_dependency_map_expression(
        self, Q_index: int | None
    ) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Parameters
        ----------
        Q_index : int | None
            Index of the Q value for which to write the dependency map. If None, write the
            dependency map for the case where A_1 is not Q-dependent.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the area.
        """
        if Q_index is None:
            return {
                'scale': self.scale,
                'mean_u_squared_ratio': self._mean_u_squared_over_angstrom_squared,
                'A_1': self.A_1,
            }

        return {
            'scale': self.scale,
            'mean_u_squared_ratio': self._mean_u_squared_over_angstrom_squared,
            'A_1': self._A_1_list[Q_index],
        }

    def _write_delta_area_dependency_expression(self, Q: float) -> str:
        """
        Write the dependency expression for the area to make dependent Parameters.

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
            Dependency expression for the area.
        """
        if not isinstance(Q, (float)):
            raise TypeError('Q must be a float.')

        # mean_u_squared_ratio is <u^2>/angstrom^2, kept dimensionless through the dependency
        # graph so the expression stays correct if mean_u_squared is converted to another unit.
        return f'scale * exp(-mean_u_squared_ratio.value * {Q}**2 / 3) * A_0'

    def _write_delta_area_dependency_map_expression(
        self,
        Q_index: int | None,
    ) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map expression to make dependent Parameters.

        Parameters
        ----------
        Q_index : int | None
            Index of the Q value for which to write the dependency map. If None, write the
            dependency map for the case where A_0 is not Q-dependent.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the area.
        """
        if Q_index is None:
            return {
                'scale': self.scale,
                'mean_u_squared_ratio': self._mean_u_squared_over_angstrom_squared,
                'A_0': self.A_0,
            }
        return {
            'scale': self.scale,
            'mean_u_squared_ratio': self._mean_u_squared_over_angstrom_squared,
            'A_0': self._A_0_list[Q_index],
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
            f'DeltaLorentz(display_name={self.display_name}, '
            f'x_unit={self.x_unit}, y_unit={self.y_unit}, \n'
            f'    mean_u_squared={self.mean_u_squared}, \n'
            f'    A_0={self.A_0}, A_1={self.A_1}, \n'
            f'    lorentzian_width={self.lorentzian_width}, \n'
            f'    scale={self.scale})'
        )
