import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.diffusion_model import DiffusionModelBase
from easydynamics.sample_model.model_base import ModelBase
from easydynamics.utils import _detailed_balance_factor
from easydynamics.utils.utils import Numeric, Q_type

from .component_collection import ComponentCollection
from .components.model_component import ModelComponent


class SampleModel(ModelBase):
    """SampleModel represents a model of a sample with components and diffusion models,
    parameterized by Q and optionally temperature.
    Generates ComponentCollections for each Q value, combining components from the base model and diffusion models.
    Applies detailed balancing based on temperature if provided.
    Parameters
    ----------
    display_name : str
        Display name of the model.
    unique_name : str | None
        Unique name of the model. If None, a unique name will be generated.
    unit : str | sc.Unit | None
        Unit of the model. If None, unitless.
    components : ModelComponent | ComponentCollection | None
        Template components of the model. If None, no components are added. These components are copied into ComponentCollections for each Q value.
    Q : Number, list, np.ndarray or sc.array or None.
        Q values for the model. If None, Q is not set.
    diffusion_models : DiffusionModelBase | list[DiffusionModelBase] | None
        Diffusion models to include in the SampleModel. If None, no diffusion models are added
    temperature : float | None
        Temperature for detailed balancing. If None, no detailed balancing is applied.
    temperature_unit : str | sc.Unit
        Unit of the temperature. Defaults to "K".
    divide_by_temperature : bool
        Whether to divide the detailed balance factor by temperature. Defaults to True.
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

        if not isinstance(divide_by_temperature, bool):
            raise TypeError("divide_by_temperature must be True or False")
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

    def remove_diffusion_model(self, name: "str") -> None:
        """Remove a DiffusionModel from the SampleModel by name.

        Args:
            name (str): The unique name of the DiffusionModel to remove.
        """
        for i, dm in enumerate(self._diffusion_models):
            if dm.unique_name == name:
                del self._diffusion_models[i]
                return
        raise ValueError(
            f"No DiffusionModel with name {name} found. The available names are: {[dm.unique_name for dm in self._diffusion_models]}"
        )

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
        self, value: DiffusionModelBase | list[DiffusionModelBase] | None
    ) -> None:
        """Set the diffusion models of the SampleModel."""
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
                "diffusion_models must be a DiffusionModelBase, a list of DiffusionModelBase, or None"
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

        if value < 0:
            raise ValueError("temperature must be non-negative")

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
    def temperature_unit(self) -> str | sc.Unit:
        """Get the temperature unit of the SampleModel."""
        return self._temperature_unit

    @temperature_unit.setter
    def temperature_unit(self, value: str | sc.Unit) -> None:
        raise AttributeError(
            f"Temperature_unit is read-only. Use convert_temperature_unit to change the unit between allowed types "
            f"or create a new {self.__class__.__name__} with the desired unit."
        )

    def convert_temperature_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the temperature Parameter.
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
            except Exception:
                pass  # Best effort rollback
            raise e

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
        list[np.ndarray]
            List of evaluated model values for each Q.
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

    def generate_component_collections(self) -> None:
        """Generate ComponentCollections from the DiffusionModels for each Q and add the components from self._components."""
        # TODO regenerate automatically if Q, diffusion models or components have changed
        super().generate_component_collections()

        # Generate components from diffusion models and add to component collections
        if self._diffusion_models is not None:
            for diffusion_model in self._diffusion_models:
                diffusion_collections = diffusion_model.create_component_collections(
                    Q=self._Q
                )
                for target, source in zip(
                    self._component_collections, diffusion_collections
                ):
                    for component in source.components:
                        target.append_component(component)

    def get_all_variables(self):
        """Get all Parameters and Descriptors from all ComponentCollections in the SampleModel.
        Also includes temperature if set and all variables from diffusion models.
        Ignores the Parameters and Descriptors in self._components as these are just templates."""
        all_vars = super().get_all_variables()
        if self._temperature is not None:
            all_vars.append(self._temperature)

        for diffusion_model in self.diffusion_models:
            all_vars.extend(diffusion_model.get_all_variables())

        return all_vars

    # --------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # dunder methods
    # --------------------------------------------------------------------

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(unique_name={self.unique_name}, unit={self.unit}), Q = {self.Q}, "
            f"components = {self.components}, diffusion_models = {self.diffusion_models}, "
            f"temperature = {self.temperature}, divide_by_temperature = {self.divide_by_temperature}"
        )
