# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.diffusion_model.diffusion_model_base import (
    DiffusionModelBase,
)
from easydynamics.sample_model.model_base import ModelBase
from easydynamics.utils import _detailed_balance_factor
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type

from .component_collection import ComponentCollection
from .components.model_component import ModelComponent


class SampleModel(ModelBase):
    """SampleModel represents a model of a sample with components and
    diffusion models, parameterized by Q and optionally temperature.
    Generates ComponentCollections for each Q value, combining
    components from the base model and diffusion models.

    Applies detailed balancing based on temperature if provided.

    Args:
        display_name (str): Display name of the model.
        unique_name (str | None): Unique name of the model. If None, a
            unique name will be generated.
        unit (str | sc.Unit | None): Unit of the model. If None,
            defaults to "meV".
        components (ModelComponent | ComponentCollection | None):
            Template components of the model. If None, no components are
            added. These components are copied into ComponentCollections
            for each Q value.
        Q (Number, list, np.ndarray, sc.array | None):
            Q values for the model. If None, Q is not set.
        diffusion_models (DiffusionModelBase | list[DiffusionModelBase]
            | None): Diffusion models to include in the SampleModel.
            If None, no diffusion models are added.
        temperature (float | None): Temperature for detailed balancing.
            If None, no detailed balancing is applied.
        temperature_unit (str | sc.Unit): Unit of the temperature.
            Defaults to "K".
        divide_by_temperature (bool): Whether to divide the detailed
            balance factor by temperature. Defaults to True.

    Attributes:
        unit (str | sc.Unit): Unit of the model.
        components (list[ModelComponent]): List of ModelComponents in
            the model.
        Q (np.ndarray | Numeric | list | ArrayLike | sc.Variable
            | None): Q values of the model.
        diffusion_models (list[DiffusionModelBase]): List of diffusion
            models in the SampleModel.
        temperature (Parameter | None): Temperature Parameter for
            detailed balancing, or None if not set.
        divide_by_temperature (bool): Whether to divide the detailed
            balance factor by temperature.
    """

    def __init__(
        self,
        display_name: str = "MySampleModel",
        unique_name: str | None = None,
        unit: str | sc.Unit = "meV",
        components: ComponentCollection | ModelComponent | None = None,
        Q: Q_type | None = None,
        diffusion_models: DiffusionModelBase | list[DiffusionModelBase] | None = None,
        temperature: float | None = None,
        temperature_unit: str | sc.Unit = "K",
        divide_by_temperature: bool = True,
    ):
        """Initialize the SampleModel.

        Args:
            display_name (str): Display name of the model.
            unique_name (str | None): Unique name of the model. If None,
                a unique name will be generated.
            unit (str | sc.Unit | None): Unit of the model. If None,
                defaults to "meV".
            components (ModelComponent | ComponentCollection | None):
                Template components of the model. If None, no components
                are added. These components are copied into
                ComponentCollections for each Q value.
            Q (Number, list, np.ndarray, sc.array | None):
                Q values for the model. If None, Q is not set.
            diffusion_models (DiffusionModelBase |
                list[DiffusionModelBase] | None): Diffusion models to
                include in the SampleModel. If None, no diffusion models
                are added.
            temperature (float | None): Temperature for detailed
                balancing. If None, no detailed balancing is applied.
            temperature_unit (str | sc.Unit): Unit of the temperature.
                Defaults to "K".
            divide_by_temperature (bool): Whether to divide the detailed
                balance factor by temperature. Defaults to True.

        Raises:
            TypeError: If diffusion_models is not a DiffusionModelBase,
                a list of DiffusionModelBase, or None.
            TypeError: If temperature is not a number or None.
            ValueError: If temperature is negative.
            TypeError: If divide_by_temperature is not a bool.
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
                    "diffusion_models must be a DiffusionModelBase, "
                    "a list of DiffusionModelBase or None"
                )
            self._diffusion_models = diffusion_models

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
                raise TypeError("temperature must be a number or None")

            if temperature < 0:
                raise ValueError("temperature must be non-negative")
            self._temperature = Parameter(
                name="Temperature",
                value=temperature,
                unit=temperature_unit,
                display_name="Temperature",
                fixed=True,
            )
        self._temperature_unit = temperature_unit

        if not isinstance(divide_by_temperature, bool):
            raise TypeError("divide_by_temperature must be True or False")
        self._divide_by_temperature = divide_by_temperature

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def append_diffusion_model(self, diffusion_model: DiffusionModelBase) -> None:
        """Append a DiffusionModel to the SampleModel.

        Args:
            diffusion_model (DiffusionModelBase): The DiffusionModel
                to append.

        Raises:
            TypeError: If the diffusion_model is not a
                DiffusionModelBase
        """

        if not isinstance(diffusion_model, DiffusionModelBase):
            raise TypeError(
                f"diffusion_model must be a DiffusionModelBase, got {type(diffusion_model).__name__}"  # noqa: E501
            )

        self._diffusion_models.append(diffusion_model)
        self._generate_component_collections()

    def remove_diffusion_model(self, name: "str") -> None:
        """Remove a DiffusionModel from the SampleModel by unique name.

        Args:
            name (str): The unique name of the DiffusionModel to remove.

        Raises:
            ValueError: If no DiffusionModel with the given unique name
                is found.
        """
        for i, dm in enumerate(self._diffusion_models):
            if dm.unique_name == name:
                del self._diffusion_models[i]
                self._generate_component_collections()
                return
        raise ValueError(
            f"No DiffusionModel with unique name {name} found. \n"
            f"The available unique names are: {[dm.unique_name for dm in self._diffusion_models]}"
        )

    def clear_diffusion_models(self) -> None:
        """Clear all DiffusionModels from the SampleModel."""
        self._diffusion_models.clear()
        self._generate_component_collections()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def diffusion_models(self) -> list[DiffusionModelBase]:
        """Get the diffusion models of the SampleModel.

        Returns:
            list[DiffusionModelBase]: The diffusion models of the
                SampleModel.
        """
        return self._diffusion_models

    @diffusion_models.setter
    def diffusion_models(
        self, value: DiffusionModelBase | list[DiffusionModelBase] | None
    ) -> None:
        """Set the diffusion models of the SampleModel.

        Args:
            value (DiffusionModelBase | list[DiffusionModelBase] |
                None):
                The diffusion model(s) to set. Can be a single
                DiffusionModelBase, a list of DiffusionModelBase, or
                None to clear all diffusion models.
        """

        if value is None:
            self._diffusion_models = []
            return
        if isinstance(value, DiffusionModelBase):
            self._diffusion_models = [value]
            return
        if not isinstance(value, list) or not all(
            isinstance(dm, DiffusionModelBase) for dm in value
        ):
            raise TypeError(
                "diffusion_models must be a DiffusionModelBase, a list of DiffusionModelBase, "
                "or None"
            )
        self._diffusion_models = value
        self._on_diffusion_models_change()

    @property
    def temperature(self) -> Parameter | None:
        """Get the temperature of the SampleModel.

        Returns:
            Parameter | None: The temperature Parameter of the
                SampleModel, or None if not set.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: Numeric | None) -> None:
        """Set the temperature of the SampleModel.

        Args:
            value (Numeric | None): The temperature value to set. Can be
                a number or None to unset the temperature.
        """
        if value is None:
            self._temperature = None
            return

        if not isinstance(value, Numeric):
            raise TypeError("temperature must be a number or None")

        if value < 0:
            raise ValueError("temperature must be non-negative")

        if self._temperature is None:
            self._temperature = Parameter(
                name="Temperature",
                value=value,
                unit=self._temperature_unit,
                display_name="Temperature",
                fixed=True,
            )
        else:
            self._temperature.value = value

    @property
    def temperature_unit(self) -> str | sc.Unit:
        """Get the temperature unit of the SampleModel.

        Returns:
            str | sc.Unit: The unit of the temperature Parameter.
        """
        return self._temperature_unit

    @temperature_unit.setter
    def temperature_unit(self, value: str | sc.Unit) -> None:
        """The temperature unit of the SampleModel is read-only.

        Args:
            value (str | sc.Unit): The unit to set for the temperature
                Parameter.

        Raises:
            AttributeError: Always, as temperature_unit is read-only.
        """

        raise AttributeError(
            f"Temperature_unit is read-only. Use convert_temperature_unit to change the unit between allowed types "  # noqa: E501
            f"or create a new {self.__class__.__name__} with the desired unit."
        )  # noqa: E501

    def convert_temperature_unit(self, unit: str | sc.Unit) -> None:
        """Convert the unit of the temperature Parameter.

        Args:
            unit (str | sc.Unit): The unit to convert the temperature
                Parameter to.

        Raises:
            ValueError: If temperature is not set or conversion fails.
        """

        if self._temperature is None:
            raise ValueError("Temperature is not set, cannot convert unit.")

        old_unit = self._temperature.unit

        try:
            self._temperature.convert_unit(unit)
            self._temperature_unit = unit
        except Exception as e:
            # Attempt to rollback on failure
            try:
                self._temperature.convert_unit(old_unit)
            except Exception:  # noqa: S110
                pass  # Best effort rollback
            raise e

    @property
    def divide_by_temperature(self) -> bool:
        """Get whether to divide the detailed balance factor by
        temperature.

        Returns:
            bool: True if the detailed balance factor is divided by
                temperature, False otherwise.
        """
        return self._divide_by_temperature

    @divide_by_temperature.setter
    def divide_by_temperature(self, value: bool) -> None:
        """Set whether to divide the detailed balance factor by
        temperature.

        Args:
            value (bool): True to divide the detailed balance factor by
                temperature, False otherwise.
        """
        if not isinstance(value, bool):
            raise TypeError("divide_by_temperature must be True or False")
        self._divide_by_temperature = value

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> list[np.ndarray]:
        """Evaluate the sample model at all Q for the given x values.

        Args:
            x (Numeric | list | np.ndarray | sc.Variable |
                sc.DataArray):
                The x values to evaluate the model at. Can be a number,
                list, numpy array, scipp Variable, or scipp DataArray.

        Returns:
            list[np.ndarray]: List of evaluated model values for each Q.
        """

        y = super().evaluate(x)

        if self._temperature is not None:
            DBF = _detailed_balance_factor(
                energy=x,
                temperature=self._temperature,
                divide_by_temperature=self._divide_by_temperature,
                energy_unit=self._unit,
            )
            y = [yi * DBF for yi in y]

        return y

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """Get all Parameters and Descriptors from all
        ComponentCollections in the SampleModel.

        Also includes temperature if set and all variables from
        diffusion models. Ignores the Parameters and Descriptors in
        self._components as these are just templates.

        Args:
            Q_index (int | None): If specified, only get variables from
                the ComponentCollection at the given Q index. If None,
                get variables from all ComponentCollections.
        """

        all_vars = super().get_all_variables(Q_index=Q_index)
        if self._temperature is not None:
            all_vars.append(self._temperature)

        for diffusion_model in self._diffusion_models:
            all_vars.extend(diffusion_model.get_all_variables())

        return all_vars

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _generate_component_collections(self) -> None:
        """Generate ComponentCollections from the DiffusionModels for
        each Q and add the components from self._components.
        """
        # TODO regenerate automatically if Q, diffusion models
        # or components have changed
        super()._generate_component_collections()

        if self._Q is None:
            return
        # Generate components from diffusion models
        # and add to component collections
        for diffusion_model in self._diffusion_models:
            diffusion_collections = diffusion_model.create_component_collections(
                Q=self._Q
            )
            for target, source in zip(
                self._component_collections, diffusion_collections
            ):
                for component in source.components:
                    target.append_component(component)

    def _on_diffusion_models_change(self) -> None:
        """Handle changes to the diffusion models."""
        self._generate_component_collections()

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation of the SampleModel.

        Returns:
            str: A string representation of the SampleModel.
        """

        return (
            f"{self.__class__.__name__}(unique_name={self.unique_name}, unit={self._unit}), "
            f"Q = {self._Q}, "
            f"components = {self._components}, diffusion_models = {self._diffusion_models}, "
            f"temperature = {self._temperature}, "
            f"divide_by_temperature = {self._divide_by_temperature}"
        )
