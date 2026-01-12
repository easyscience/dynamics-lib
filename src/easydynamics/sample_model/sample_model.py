# The current SampleModel will be renamed to ComponentCollection or something similar. It will take a list (perhaps sc.array) of Q among other things.

# The SampleModel will allow the user to append DiffusionModels to a list of DiffusionModels and append ModelComponents to a ComponentCollection.

# There will also be a list of ComponentCollections, where each is a copy of the ComponentCollection that the user supplied. The user will also be allowed to work directly with this list. The list is the same length as Q; each ComponentCollection corresponds to a single Q.

# Behind the scenes, it will have a list of ComponentCollection, which contains all the user supplied ComponentCollections.

# The DiffusionModel will also be able to generate components. It may be best to keep them in a separate list of ComponentCollections, just to make sure they don't accidentally get overwritten or changed by the user. It should be possible to append a DiffusionModel without actually generating the components it contains., Fitting entire diffusion models is very difficult until you have a very good understanding of your data, and can take very long - in preliminary tests, fitting sequentially took about 3 seconds, and a DiffusionModel from an ok, but not great, starting point took 15 minutes.

# Perhaps it will have an explicit generate_diffusion_model_components or some such.

# It should have a calculate and plot method to plot the model of the scattering of the sample before convolution. I suppose that they could take two lists of ComponentCollections and make a single ComponentCollection out of them?

# It will have an optional Temperature, which when not None will include detailed balance calculations.

# It will eventually also have to support taking a list of temperatures and allow models to vary as function of temperature - not entirely sure how that will work. Perhaps we will actually have a list of list of ComponentCollection, where one is over temperature, and the other is over Q. Let's deal with that when we get there.

# SampleModel will inherit from SampleModelBase or something similar.

from copy import copy

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from numpy.typing import ArrayLike

from easydynamics.sample_model.diffusion_model import DiffusionModelBase
from easydynamics.sample_model.sample_model_base import SampleModelBase
from easydynamics.utils import _detailed_balance_factor

from .component_collection import ComponentCollection
from .components.model_component import ModelComponent

Numeric = float | int
Q_type = np.ndarray | Numeric | list | ArrayLike


# Base class for sample models - contains a list of ComponentCollection as function of Q and probably not much else


class SampleModel(SampleModelBase):
    def __init__(
        self,
        unit: str | sc.Unit = "meV",
        display_name: str = "MySampleModel",
        unique_name: str | None = None,
        components: ComponentCollection | None = None,
        diffusion_models: list[DiffusionModelBase] | None = None,
        Q: np.ndarray | None = None,
        temperature: float | None = None,
        temperature_unit: str | sc.Unit = "K",
        divide_by_temperature: bool = False,
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        if unit is not None and not isinstance(unit, (str, sc.Unit)):
            raise TypeError(
                f"unit must be None, a string, or a scipp Unit, got {type(unit).__name__}"
            )
        self._unit = unit

        if components is None:
            self._components = ComponentCollection()
        else:
            if not isinstance(components, ComponentCollection):
                raise TypeError(
                    f"components must be a ComponentCollection or None, got {type(components).__name__}"
                )
            self._components = components

        if diffusion_models is None:
            self._diffusion_models = []
        else:
            if not isinstance(diffusion_models, list) or not all(
                isinstance(dm, DiffusionModelBase) for dm in diffusion_models
            ):
                raise TypeError(
                    "diffusion_models must be a list of DiffusionModelBase or None"
                )
            self._diffusion_models = diffusion_models

        if Q is None:
            self._Q = None
        else:
            self._Q = self._validate_and_convert_Q(Q)

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

    def add_diffusion_model(self, diffusion_model: DiffusionModelBase) -> None:
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

    def add_component(self, component: ModelComponent) -> None:
        """Append a ModelComponent to the SampleModel.

        Args:
            component (ModelComponent): The ModelComponent to append.
        """
        self._components.add_component(component)

    def remove_component(self, unique_name: str) -> None:
        """Remove a ModelComponent from the SampleModel by its unique name.

        Args:
            unique_name (str): The unique name of the ModelComponent to remove.
        """
        self._components.remove_component(unique_name)

    def clear_components(self) -> None:
        """Clear all ModelComponents from the SampleModel."""
        self._components.clear_components()

    # --------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------

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

    @property
    def Q(self) -> np.ndarray | None:
        """Get the Q values of the SampleModel."""
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """Set the Q values of the SampleModel."""
        if value is None:
            self._Q = None
        else:
            self._Q = self._validate_and_convert_Q(value)

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

    # --------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------

    def _generate_component_collections(self) -> None:
        """Generate ComponentCollections from the DiffusionModels for each Q and add the components from self._components."""

        # TODO update temporary name

        self._component_collections = [ComponentCollection() for _ in self._Q]

        for diffusion_model in self._diffusion_models:
            diffusion_collections = diffusion_model.create_component_collections(
                Q=self._Q, component_display_name="Temporary name"
            )
            for target, source in zip(
                self._component_collections, diffusion_collections
            ):
                for component in source.components:
                    target.add_component(component)

        for collection in self._component_collections:
            # Add copies of components from self._components to each collection
            for component in self._components.components:
                collection.add_component(copy(component))

    # --------------------------------------------------------------------
    # dunder methods
    # --------------------------------------------------------------------
