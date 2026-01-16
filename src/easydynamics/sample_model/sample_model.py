from copy import copy

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from numpy.typing import ArrayLike

from easydynamics.sample_model.diffusion_model import DiffusionModelBase
from easydynamics.sample_model.model_base import ModelBase
from easydynamics.utils import _detailed_balance_factor

from .component_collection import ComponentCollection
from .components.model_component import ModelComponent

Numeric = float | int
Q_type = np.ndarray | Numeric | list | ArrayLike


class SampleModel(ModelBase):
    """SampleModel represents a model of a sample with components and diffusion models,
    parameterized by Q and optionally temperature.
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
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
            components=components,
            Q=Q,
        )

        if diffusion_models is None:
            self._diffusion_models = []
        elif isinstance(diffusion_models, DiffusionModelBase):
            self._diffusion_models = [diffusion_models]
        else:
            if not isinstance(diffusion_models, list) or not all(
                isinstance(dm, DiffusionModelBase) for dm in diffusion_models
            ):
                raise TypeError(
                    "diffusion_models must be a DiffusionModelBase, a list of DiffusionModelBase or None"
                )
            self._diffusion_models = diffusion_models
        if temperature is None:
            self._temperature = None
        else:
            if not isinstance(temperature, Numeric):
                raise TypeError("temperature must be a number or None")
            self._temperature = Parameter(
                name="Temperature",
                value=temperature,
                unit=temperature_unit,
                display_name="Temperature",
            )
        self._temperature_unit = temperature_unit

        self._divide_by_temperature = divide_by_temperature

    # --------------------------------------------------------------------
    # Component management
    # --------------------------------------------------------------------

    def append_diffusion_model(self, diffusion_model: DiffusionModelBase) -> None:
        """Append a DiffusionModel to the SampleModel.

        Args:
            diffusion_model (DiffusionModelBase): The DiffusionModel to append.
        """

        if not isinstance(diffusion_model, DiffusionModelBase):
            raise TypeError(
                f"diffusion_model must be a DiffusionModelBase, got {type(diffusion_model).__name__}"
            )

        self._diffusion_models.append(diffusion_model)

    def clear_diffusion_models(self) -> None:
        """Clear all DiffusionModels from the SampleModel."""
        self._diffusion_models.clear()

    # --------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------

    @property
    def diffusion_models(self) -> list[DiffusionModelBase]:
        """Get the diffusion models of the SampleModel."""
        return self._diffusion_models

    @diffusion_models.setter
    def diffusion_models(
        self, value: DiffusionModelBase | list[DiffusionModelBase]
    ) -> None:
        """Set the diffusion models of the SampleModel."""

        if isinstance(value, DiffusionModelBase):
            self._diffusion_models = [value]
            return
        if not isinstance(value, list) or not all(
            isinstance(dm, DiffusionModelBase) for dm in value
        ):
            raise TypeError(
                "diffusion_models must be a DiffusionModelBase or a list of DiffusionModelBase"
            )
        self._diffusion_models = value

    @property
    def temperature(self) -> Parameter | None:
        """Get the temperature of the SampleModel."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: Numeric | None) -> None:
        """Set the temperature of the SampleModel."""
        if value is None:
            self._temperature = None
            return
        if not isinstance(value, Numeric):
            raise TypeError("temperature must be a number or None")
        if self._temperature is None:
            self._temperature = Parameter(
                name="Temperature",
                value=value,
                unit=self._temperature_unit,
                display_name="Temperature",
            )
        else:
            self._temperature.value = value

    @property
    def divide_by_temperature(self) -> bool:
        """Get whether to divide the detailed balance factor by temperature."""
        return self._divide_by_temperature

    @divide_by_temperature.setter
    def divide_by_temperature(self, value: bool) -> None:
        """Set whether to divide the detailed balance factor by temperature."""
        if not isinstance(value, bool):
            raise TypeError("divide_by_temperature must be True or False")
        self._divide_by_temperature = value

    # --------------------------------------------------------------------
    # Other methods
    # --------------------------------------------------------------------

    def evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> list[np.ndarray]:
        """
        Evaluate the sample model at all Q for the given x values

        Parameters
        ----------
        x : Number, list, np.ndarray, sc.Variable, or sc.DataArray
            Energy axis.

        Returns
        -------
        np.ndarray
            Evaluated model values.
        """

        if not self._component_collections:
            raise ValueError("No components in the model to evaluate.")
        y = [collection.evaluate(x) for collection in self._component_collections]

        if self._temperature is not None:
            # TODO handle units properly
            DBF = _detailed_balance_factor(
                x,
                self._temperature.value,
                sc.Unit("meV"),
                sc.Unit("K"),
                divide_by_temperature=self._divide_by_temperature,
            )
            y = [yi * DBF for yi in y]

        return y

    def generate_component_collections(self) -> None:
        """Generate ComponentCollections from the DiffusionModels for each Q and add the components from self._components."""

        # TODO update temporary name
        # TODO only regenerate if Q or diffusion models have changed

        if self._Q is None:
            raise ValueError("Q must be set before generating component collections.")

        self._component_collections = [ComponentCollection() for _ in self._Q]

        # Generate components from diffusion models and add to component collections
        for diffusion_model in self._diffusion_models:
            diffusion_collections = diffusion_model.create_component_collections(
                Q=self._Q, component_display_name="Temporary name"
            )
            for target, source in zip(
                self._component_collections, diffusion_collections
            ):
                for component in source.components:
                    target.append_component(component)

        # Add copies of components from self._components to each component collection
        for collection in self._component_collections:
            for component in self._components.components:
                collection.append_component(copy(component))

    def get_all_variables(self):
        """Get all Parameters and Descriptors from all ComponentCollections in the SampleModel.
        Also includes temperature if set and all variables from diffusion models.
        Ignores the Parameters and Descriptors in self._components as these are just templates."""
        all_vars = super().get_all_variables()
        all_vars.extend(self._temperature or [])

        diffusion_vars = [
            var
            for diffusion_model in self.diffusion_models
            for var in diffusion_model.get_all_variables()
        ]
        all_vars.extend(diffusion_vars)
        return all_vars

    # --------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # dunder methods
    # --------------------------------------------------------------------
