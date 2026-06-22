# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import suppress

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.sample_model.model_base import ModelBase
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings
from easydynamics.utils import detailed_balance_factor
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q


class SampleModel(ModelBase):
    """
    SampleModel represents a model of a sample with components and diffusion models, parameterized
    by Q and optionally temperature. Generates ComponentCollections for each Q value, combining
    components from the base model and diffusion models.

    Applies detailed balancing based on temperature if provided.
    """

    def __init__(
        self,
        display_name: str = 'MySampleModel',
        unique_name: str | None = None,
        unit: str | sc.Unit = 'meV',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
        diffusion_models: DiffusionModelBase | list[DiffusionModelBase] | None = None,
        temperature: float | None = None,
        temperature_unit: str | sc.Unit = 'K',
        detailed_balance_settings: DetailedBalanceSettings | None = None,
    ) -> None:
        """
        Initialize the SampleModel.

        Parameters
        ----------
        display_name : str, default='MySampleModel'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        unit : str | sc.Unit, default='meV'
            Unit of the model. If None,.
        components : ModelComponent | ComponentCollection | None, default=None
            Template components of the model. If None, no components are added. These components
            are copied into ComponentCollections for each Q value.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        diffusion_models : DiffusionModelBase | list[DiffusionModelBase] | None, default=None
            Diffusion models to include in the SampleModel. If None, no diffusion models are added.
        temperature : float | None, default=None
            Temperature for detailed balancing. If None, no detailed balancing is applied. By
            default, None.
        temperature_unit : str | sc.Unit, default='K'
            Unit of the temperature.
        detailed_balance_settings : DetailedBalanceSettings | None, default=None
            Settings for detailed balancing.

        Raises
        ------
        TypeError
            If diffusion_models is not a DiffusionModelBase, a list of DiffusionModelBase, or None,
            or if temperature is not a number or None, or if detailed_balance_settings is not a
            DetailedBalanceSettings instance.
        ValueError
            If temperature is negative.
        """

        if diffusion_models is None:
            self._diffusion_models = []
        elif isinstance(diffusion_models, DiffusionModelBase):
            self._diffusion_models = [diffusion_models]
        else:
            if not isinstance(diffusion_models, list) or not all(
                isinstance(dm, DiffusionModelBase) for dm in diffusion_models
            ):
                raise TypeError(
                    'diffusion_models must be a DiffusionModelBase, '
                    'a list of DiffusionModelBase or None'
                )
            self._diffusion_models = diffusion_models

        Q = _validate_and_convert_Q(Q)
        for dm in self.diffusion_models:
            dm.Q = Q  # Ensure diffusion models have the same Q as the SampleModel

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
            components=components,
            Q=Q,
        )

        if temperature is None:
            self._temperature = None
        else:
            if not isinstance(temperature, Numeric):
                raise TypeError('temperature must be a number or None')

            if temperature < 0:
                raise ValueError('temperature must be non-negative')
            self._temperature = Parameter(
                name='Temperature',
                value=temperature,
                unit=temperature_unit,
                display_name='Temperature',
                fixed=True,
            )
        self._temperature_unit = temperature_unit

        if detailed_balance_settings is None:
            self._detailed_balance_settings = DetailedBalanceSettings()
        elif isinstance(detailed_balance_settings, DetailedBalanceSettings):
            self._detailed_balance_settings = detailed_balance_settings
        else:
            raise TypeError('detailed_balance_settings must be a DetailedBalanceSettings or None')

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def append_diffusion_model(self, diffusion_model: DiffusionModelBase) -> None:
        """
        Append a DiffusionModel to the SampleModel.

        Parameters
        ----------
        diffusion_model : DiffusionModelBase
            The DiffusionModel to append.

        Raises
        ------
        TypeError
            If the diffusion_model is not a DiffusionModelBase.
        """

        if not isinstance(diffusion_model, DiffusionModelBase):
            raise TypeError(
                f'diffusion_model must be a DiffusionModelBase, got {type(diffusion_model).__name__}'  # noqa: E501
            )
        diffusion_model.Q = self.Q
        self._diffusion_models.append(diffusion_model)
        self._component_collections_is_dirty = True

    def remove_diffusion_model(self, name: str) -> None:
        """
        Remove a DiffusionModel from the SampleModel by name.

        Parameters
        ----------
        name : str
            The name of the DiffusionModel to remove.

        Raises
        ------
        ValueError
            If no DiffusionModel with the given name is found.
        """
        for i, dm in enumerate(self.diffusion_models):
            if dm.name == name:
                del self.diffusion_models[i]
                self._component_collections_is_dirty = True
                return
        raise ValueError(
            f'No DiffusionModel with name {name} found. \n'
            f'The available names are: {[dm.name for dm in self.diffusion_models]}'
        )

    def clear_diffusion_models(self) -> None:
        """Clear all DiffusionModels from the SampleModel."""
        self.diffusion_models = []
        self._component_collections_is_dirty = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def diffusion_models(self) -> list[DiffusionModelBase]:
        """
        Get the diffusion models of the SampleModel.

        Returns
        -------
        list[DiffusionModelBase]
            The diffusion models of the SampleModel.
        """
        return self._diffusion_models

    @diffusion_models.setter
    def diffusion_models(
        self, value: DiffusionModelBase | list[DiffusionModelBase] | None
    ) -> None:
        """
        Set the diffusion models of the SampleModel.

        Parameters
        ----------
        value : DiffusionModelBase | list[DiffusionModelBase] | None
            The diffusion model(s) to set. Can be a single DiffusionModelBase, a list of
            DiffusionModelBase, or None to clear all diffusion models.

        Raises
        ------
        TypeError
            If value is not a DiffusionModelBase, a list of DiffusionModelBase, or None.
        """

        if value is None:
            self._diffusion_models = []
            self._on_diffusion_models_change()
            return
        if isinstance(value, DiffusionModelBase):
            value.Q = self.Q
            self._diffusion_models = [value]
            self._on_diffusion_models_change()
            return
        if not isinstance(value, list) or not all(
            isinstance(dm, DiffusionModelBase) for dm in value
        ):
            raise TypeError(
                'diffusion_models must be a DiffusionModelBase, a list of DiffusionModelBase, '
                'or None'
            )
        for dm in value:
            dm.Q = self.Q
        self._diffusion_models = value
        self._on_diffusion_models_change()

    @property
    def temperature(self) -> Parameter | None:
        """
        Get the temperature of the SampleModel.

        Returns
        -------
        Parameter | None
            The temperature Parameter of the SampleModel, or None if not set.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: Numeric | None) -> None:
        """
        Set the temperature of the SampleModel.

        Parameters
        ----------
        value : Numeric | None
            The temperature value to set. Can be a number or None to unset the temperature.

        Raises
        ------
        TypeError
            If value is not a number or None.
        ValueError
            If value is negative.
        """
        if value is None:
            self._temperature = None
            return

        if not isinstance(value, Numeric):
            raise TypeError('temperature must be a number or None')

        if value < 0:
            raise ValueError('temperature must be non-negative')

        if self._temperature is None:
            self._temperature = Parameter(
                name='Temperature',
                value=value,
                unit=self._temperature_unit,
                display_name='Temperature',
                fixed=True,
            )
        else:
            self._temperature.value = value

    @property
    def temperature_unit(self) -> str | sc.Unit:
        """
        Get the temperature unit of the SampleModel.

        Returns
        -------
        str | sc.Unit
            The unit of the temperature Parameter.
        """
        return self._temperature_unit

    @temperature_unit.setter
    def temperature_unit(self, _value: str | sc.Unit) -> None:
        """
        The temperature unit of the SampleModel is read-only.

        Parameters
        ----------
        _value : str | sc.Unit
            The unit to set for the temperature Parameter.

        Raises
        ------
        AttributeError
            Always, as temperature_unit is read-only.
        """

        raise AttributeError(
            f'Temperature_unit is read-only. Use convert_temperature_unit to change the unit between allowed types '  # noqa: E501
            f'or create a new {self.__class__.__name__} with the desired unit.'
        )

    def convert_temperature_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the temperature Parameter.

        Parameters
        ----------
        unit : str | sc.Unit
            The unit to convert the temperature Parameter to.

        Raises
        ------
        ValueError
            If temperature is not set or conversion fails.
        Exception
            If the provided unit is invalid or cannot be converted.
        """

        if self.temperature is None:
            raise ValueError('Temperature is not set, cannot convert unit.')

        old_unit = self.temperature.unit

        try:
            self.temperature.convert_unit(unit)
            self._temperature_unit = unit
        except Exception:
            # Attempt to rollback on failure
            with suppress(Exception):
                self.temperature.convert_unit(old_unit)
            raise

    @property
    def normalize_detailed_balance(self) -> bool:
        """
        Get whether to divide the detailed balance factor by temperature.

        Returns
        -------
        bool
            True if the detailed balance factor is divided by temperature, False otherwise.
        """
        return self.detailed_balance_settings.normalize_detailed_balance

    @normalize_detailed_balance.setter
    def normalize_detailed_balance(self, value: bool) -> None:
        """
        Set whether to divide the detailed balance factor by temperature.

        Parameters
        ----------
        value : bool
            True to divide the detailed balance factor by temperature, False otherwise.

        Raises
        ------
        TypeError
            If value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError('normalize_detailed_balance must be True or False')
        self.detailed_balance_settings.normalize_detailed_balance = value

    @property
    def use_detailed_balance(self) -> bool:
        """
        Get whether to apply detailed balance to the model.

        Returns
        -------
        bool
            True if detailed balance is applied, False otherwise.
        """
        return self.detailed_balance_settings.use_detailed_balance

    @use_detailed_balance.setter
    def use_detailed_balance(self, value: bool) -> None:
        """
        Set whether to apply detailed balance to the model.

        Parameters
        ----------
        value : bool
            True to apply detailed balance, False otherwise.

        Raises
        ------
        TypeError
            If value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError('use_detailed_balance must be True or False')
        self.detailed_balance_settings.use_detailed_balance = value

    @property
    def detailed_balance_settings(self) -> DetailedBalanceSettings:
        """
        Get the DetailedBalanceSettings of the SampleModel.

        Returns
        -------
        DetailedBalanceSettings
            The DetailedBalanceSettings of the SampleModel.
        """
        return self._detailed_balance_settings

    @detailed_balance_settings.setter
    def detailed_balance_settings(self, value: DetailedBalanceSettings) -> None:
        """
        Set the DetailedBalanceSettings of the SampleModel.

        Parameters
        ----------
        value : DetailedBalanceSettings
            The DetailedBalanceSettings to set.

        Raises
        ------
        TypeError
            If value is not a DetailedBalanceSettings.
        """
        if not isinstance(value, DetailedBalanceSettings):
            raise TypeError('detailed_balance_settings must be a DetailedBalanceSettings')
        self._detailed_balance_settings = value

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> list[np.ndarray]:
        """
        Evaluate the sample model at all Q for the given x values.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values to evaluate the model at. Can be a number, list, numpy array, scipp
            Variable, or scipp DataArray.

        Returns
        -------
        list[np.ndarray]
            List of evaluated model values for each Q.
        """

        y = super().evaluate(x)

        if self.temperature is not None and self.detailed_balance_settings.use_detailed_balance:
            DBF = detailed_balance_factor(
                energy=x,
                temperature=self.temperature,
                divide_by_temperature=self.detailed_balance_settings.normalize_detailed_balance,
                energy_unit=self.unit,
            )
            y = [yi * DBF for yi in y]

        return y

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all Parameters and Descriptors from all ComponentCollections in the SampleModel.

        Also includes temperature if set and all variables from diffusion models. Ignores the
        Parameters and Descriptors in self._components as these are just templates.

        Parameters
        ----------
        Q_index : int | None, default=None
            If specified, only get variables from the ComponentCollection at the given Q index. If
            None, get variables from all ComponentCollections.

        Returns
        -------
        list[Parameter]
            List of all Parameters and Descriptors, including temperature if set and all variables
            from diffusion models.
        """

        all_vars = super().get_all_variables(Q_index=Q_index)
        if self.temperature is not None:
            all_vars.append(self.temperature)

        for diffusion_model in self.diffusion_models:
            all_vars.extend(diffusion_model.get_global_variables())
            all_vars.extend(diffusion_model.get_independent_variables(Q_index=Q_index))

        return all_vars

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _generate_component_collections(self) -> None:
        """
        Generate ComponentCollections from the DiffusionModels for each Q and add the components
        from self._components.
        """
        super()._generate_component_collections()

        if self.Q is None:
            return
        # Generate components from diffusion models
        # and add to component collections
        for diffusion_model in self.diffusion_models:
            diffusion_collections = diffusion_model.get_component_collections()
            for target, source in zip(
                self._component_collections,
                diffusion_collections,
                strict=True,
            ):
                for component in source:
                    target.append_component(component)

    def _on_diffusion_models_change(self) -> None:
        """Handle changes to the diffusion models."""
        for diffusion_model in self.diffusion_models:
            diffusion_model.Q = self.Q
        self._component_collections_is_dirty = True

    def _on_Q_change(self) -> None:
        """Handle changes to the Q values."""
        for diffusion_model in self.diffusion_models:
            diffusion_model.clear_Q(confirm=True)
            diffusion_model.Q = self.Q
        self._component_collections_is_dirty = True

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a string representation of the SampleModel.

        Returns
        -------
        str
            A string representation of the SampleModel.
        """

        return (
            f'{self.__class__.__name__}(unique_name={self.unique_name}, unit={self.unit}), '
            f'Q = {self.Q}, \n '
            f'components = {self.components}, diffusion_models = {self.diffusion_models}, '
            f'temperature = {self.temperature}, '
            f'detailed_balance_settings = {self.detailed_balance_settings}'
        )
