# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import verify_Q_index


class DiffusionModelBase(EasyDynamicsModelBase):
    """Base class for constructing diffusion models."""

    def __init__(
        self,
        scale: Numeric = 1.0,
        Q: Q_type | None = None,
        unit: str | sc.Unit = 'meV',
        name: str = 'DiffusionModel',
        display_name: str | None = 'DiffusionModel',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new DiffusionModel.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        unit : str | sc.Unit, default='meV'
            Unit of the diffusion model. Must be convertible to meV.
        name : str, default='DiffusionModel'
            Name of the diffusion model.
        display_name : str | None, default='DiffusionModel'
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
            If scale is not a number.
        UnitError
            If unit is not a string or scipp Unit, or if it cannot be converted to meV.
        ValueError
            If scale is negative.
        """

        self._Q = _validate_and_convert_Q(Q)

        try:
            test = DescriptorNumber(name='test', value=1, unit=unit)
            test.convert_unit('meV')
        except Exception as e:
            raise UnitError(
                f'Invalid unit: {unit}. Unit must be a string or scipp Unit and convertible to meV.'  # noqa: E501
            ) from e

        if not isinstance(scale, Numeric):
            raise TypeError('scale must be a number.')

        if float(scale) < 0:
            raise ValueError('scale must be non-negative.')

        scale = Parameter(name='scale', value=float(scale), fixed=False, min=0.0, unit=unit)
        self._scale = scale

        super().__init__(unit=unit, name=name, display_name=display_name, unique_name=unique_name)

        if lorentzian_name is None:
            lorentzian_name = name

        if not isinstance(lorentzian_name, str):
            raise TypeError('lorentzian_name must be a string.')

        if lorentzian_display_name is None:
            lorentzian_display_name = lorentzian_name

        if not isinstance(lorentzian_display_name, str):
            raise TypeError('lorentzian_display_name must be a string or None.')

        self._lorentzian_name = lorentzian_name
        self._lorentzian_display_name = lorentzian_display_name

        if self.Q is None:
            self._component_collections = []
        else:
            self._component_collections = [ComponentCollection()] * len(self.Q)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scale(self) -> Parameter:
        """
        Get the scale parameter of the diffusion model.

        Returns
        -------
        Parameter
            Scale parameter of the diffusion model.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: Numeric) -> None:
        """
        Set the scale parameter of the diffusion model.

        Parameters
        ----------
        scale : Numeric
            The new value for the scale parameter. Must be a non-negative number.

        Raises
        ------
        TypeError
            If scale is not a number.
        ValueError
            If scale is negative.
        """
        if not isinstance(scale, Numeric):
            raise TypeError('scale must be a number.')

        if float(scale) < 0:
            raise ValueError('scale must be non-negative.')
        self._scale.value = float(scale)

    @property
    def Q(self) -> np.ndarray | None:
        """
        Get the Q values of the SampleModel.

        Returns
        -------
        np.ndarray | None
            The Q values of the SampleModel, or None if not set.
        """
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """
        Set the Q values of the SampleModel.

        If Q is already set, it throws an error if the new Q values are not similar to the old
        ones. To change Q values, first run clear_Q().

        Parameters
        ----------
        value : Q_type | None
            The new Q values to set. If None, Q values are not changed.

        Raises
        ------
        ValueError
            If the new Q values are not similar to the old ones when Q is already set.
        """
        if value is None:
            return
        old_Q = self._Q
        new_Q = _validate_and_convert_Q(value)

        if old_Q is None:
            self._Q = new_Q
            self._on_Q_change()
            return

        if len(old_Q) != len(new_Q) or not np.allclose(old_Q, new_Q):
            raise ValueError(
                'New Q values are not similar to the old ones. '
                'To change Q values, first run clear_Q().'
            )

    @property
    def lorentzian_name(self) -> str:
        """
        Get the name of the Lorentzian component.

        Returns
        -------
        str
            Name of the Lorentzian component.
        """
        return self._lorentzian_name

    @lorentzian_name.setter
    def lorentzian_name(self, lorentzian_name: str) -> None:
        """
        Set the name of the Lorentzian component.

        Parameters
        ----------
        lorentzian_name : str
            The new name for the Lorentzian component.

        Raises
        ------
        TypeError
            If lorentzian_name is not a string.
        """
        if not isinstance(lorentzian_name, str):
            raise TypeError('lorentzian_name must be a string.')
        self._lorentzian_name = lorentzian_name

    @property
    def lorentzian_display_name(self) -> str | None:
        """
        Get the display name of the Lorentzian component.

        Returns
        -------
        str | None
            Display name of the Lorentzian component, or None if not set.
        """
        return self._lorentzian_display_name

    @lorentzian_display_name.setter
    def lorentzian_display_name(self, lorentzian_display_name: str | None) -> None:
        """
        Set the display name of the Lorentzian component.

        Parameters
        ----------
        lorentzian_display_name : str | None
            The new display name for the Lorentzian component.

        Raises
        ------
        TypeError
            If lorentzian_display_name is not a string or None.
        """
        if not isinstance(lorentzian_display_name, (str, type(None))):
            raise TypeError('lorentzian_display_name must be a string or None.')
        self._lorentzian_display_name = lorentzian_display_name

    def clear_Q(self, confirm: bool = False) -> None:
        """
        Clear the Q values of the SampleModel, removing all component collections and their
        associated Parameters.

        Parameters
        ----------
        confirm : bool, default=False
            Confirmation to clear Q values.

        Raises
        ------
        ValueError
            If confirm is not True.
        """
        if not confirm:
            raise ValueError(
                'Clearing Q values requires confirmation. Set confirm=True to proceed.'
            )
        self._Q = None
        self._on_Q_change()

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def get_global_variables(self) -> list[Parameter]:
        """
        Get all global variables from the diffusion model.

        Returns
        -------
        list[Parameter]
            A list of all global variables from the diffusion model.
        """
        return super().get_all_variables()

    def get_independent_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get the independent variables from the diffusion model. If Q_index is provided, only the
        independent variables for the specified Q value will be returned. If Q_index is None,
        independent variables for all Q values will be returned. These are variables that are not
        global but also not part of the component collections.

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

        return []

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all variables from the diffusion model.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the ComponentCollection to get variables from. If None, all variables from
            all ComponentCollections are returned, in addition to the global variables.

        Returns
        -------
        list[Parameter]
            A list of all Parameters from the diffusion model.
        """
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)

        variables = self.get_global_variables()
        variables.extend(self.get_independent_variables(Q_index))

        if Q_index is None:
            for component_collection in self._component_collections:
                variables.extend(component_collection.get_all_variables())
        else:
            variables.extend(self._component_collections[Q_index].get_all_variables())
        return variables

    def get_all_parameters(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all Parameters from the diffusion model.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the ComponentCollection to get parameters from. If None, all parameters
            from all ComponentCollections are returned.

        Returns
        -------
        list[Parameter]
            A list of all Parameters from the diffusion model.
        """
        return [param for param in self.get_all_variables(Q_index) if isinstance(param, Parameter)]

    def get_fittable_parameters(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all fittable Parameters from the diffusion model.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the ComponentCollection to get fittable parameters from. If None, all
            fittable parameters from all ComponentCollections are returned.

        Returns
        -------
        list[Parameter]
            A list of all fittable Parameters from the diffusion model.
        """
        return [
            param
            for param in self.get_all_parameters(Q_index)
            if param.independent and not param.fixed
        ]

    def get_free_parameters(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all free Parameters from the diffusion model.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the ComponentCollection to get free parameters from. If None, all free
            parameters from all ComponentCollections are returned.

        Returns
        -------
        list[Parameter]
            A list of all free Parameters from the diffusion model.
        """
        return [param for param in self.get_fittable_parameters(Q_index) if not param.fixed]

    def get_fit_parameters(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all fit Parameters from the diffusion model. This is an alias for get_free_parameters.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the ComponentCollection to get fit parameters from. If None, all fit
            parameters from all ComponentCollections are returned.
        Returns
        -------
        list[Parameter]
            A list of all fit Parameters from the diffusion model.
        """
        return self.get_free_parameters(Q_index)

    def create_component_collections(self) -> list[ComponentCollection]:
        """
        Create the ComponentCollections for the diffusion model based on the current Q values.

        Returns
        -------
        list[ComponentCollection]
            A list of ComponentCollections corresponding to the current Q values.
        """
        if self.Q is None:
            self._component_collections = []
            return self._component_collections

        self._component_collections = [ComponentCollection()] * len(self.Q)

        return self._component_collections

    def get_component_collections(
        self, Q_index: int | None = None
    ) -> ComponentCollection | list[ComponentCollection]:
        """
        Get the ComponentCollection at the given Q index.

        Parameters
        ----------
        Q_index : int | None, default=None
            The index of the desired ComponentCollection. If None, all ComponentCollections are
            returned.

        Returns
        -------
        ComponentCollection | list[ComponentCollection]
            The ComponentCollection at the specified Q index. If Q_index is None, a list of all
            ComponentCollections is returned.
        """
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)
        if Q_index is None:
            return self._component_collections

        return self._component_collections[Q_index]

    # ------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------

    def _on_Q_change(self) -> None:
        """Handle changes to the Q values."""
        self.create_component_collections()

    def _ensure_Q(self, Q: Q_type) -> np.ndarray:
        """
        Convert Q to a numpy array, ensuring it is not None. Uses the stored Q if no input is
        given.

        Parameters
        ----------
        Q : Q_type
            The Q to be checked

        Returns
        -------
        np.ndarray
            The validated and converted Q values.

        Raises
        ------
        ValueError
            If the provided Q and self.Q are both None
        """
        if Q is None:
            Q = self.Q
        if Q is None:
            raise ValueError('Q must be provided either as an argument or set in the model.')

        return _validate_and_convert_Q(Q)

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the Diffusion model.

        Returns
        -------
        str
            String representation of the DiffusionModel.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r}, '
            f'unit={self.unit},\n'
            f'    scale={self.scale})'
        )
