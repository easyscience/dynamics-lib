from copy import copy

import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase as EasyScienceModelBase
from numpy.typing import ArrayLike

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import _validate_and_convert_Q, _validate_unit

Numeric = float | int

Q_type = np.ndarray | Numeric | list | ArrayLike | sc.Variable


class ModelBase(EasyScienceModelBase):
    """Base class for Sample Models.

    Contains common functionality for models with components and Q dependence.

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
    Q : Q_type | None
        Q values for the model. If None, Q is not set.
    """

    def __init__(
        self,
        display_name: str = "MyModelBase",
        unique_name: str | None = None,
        unit: str | sc.Unit | None = "meV",
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )
        self._unit = _validate_unit(unit)

        if components is not None and not isinstance(
            components, (ModelComponent, ComponentCollection)
        ):
            raise TypeError(
                f"components must be a ModelComponent, a ComponentCollection or None, got {type(components).__name__}"
            )

        self._components = ComponentCollection()
        if isinstance(components, (ModelComponent, ComponentCollection)):
            self.append_component(components)

            self._Q = _validate_and_convert_Q(Q)

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
            Evaluated model values.
        """

        if not self._component_collections:
            raise ValueError("No components in the model to evaluate.")
        y = [collection.evaluate(x) for collection in self._component_collections]

        return y

    def generate_component_collections(self) -> None:
        """Generate ComponentCollections for each Q value."""
        # TODO only regenerate if Q or diffusion models have changed

        if self._Q is None:
            raise ValueError("Q must be set before generating component collections.")

        self._component_collections = [ComponentCollection() for _ in self._Q]

        # Add copies of components from self._components to each component collection
        for collection in self._component_collections:
            for component in self._components.components:
                collection.append_component(copy(component))

    def get_all_variables(self):
        """Get all Parameters and Descriptors from all ComponentCollections in the ModelBase.
        Ignores the Parameters and Descriptors in self._components as these are just templates."""

        all_vars = [
            var
            for collection in self._component_collections
            for var in collection.get_all_variables()
        ]
        return all_vars

    # --------------------------------------------------------------------
    # Component management
    # --------------------------------------------------------------------
    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """Append a ModelComponent or ComponentCollection to the SampleModel.

        Args:
            component (ModelComponent | ComponentCollection): The ModelComponent or ComponentCollection to append.
        """
        self._components.append_component(component)

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
    def unit(self) -> str | sc.Unit:
        """
        Get the unit of the ComponentCollection.

        Returns
        -------
        str or sc.Unit or None
        """
        return self._unit

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )  # noqa: E501

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the ComponentCollection and all its components.
        """

        old_unit = self._unit

        try:
            for component in self.components:
                component.convert_unit(unit)
            self._unit = unit
        except Exception as e:
            # Attempt to rollback on failure
            try:
                for component in self.components:
                    component.convert_unit(old_unit)
            except Exception:
                pass  # Best effort rollback
            raise e

    @property
    def components(self) -> list[ModelComponent]:
        """Get the components of the SampleModel."""
        return self._components.components

    @components.setter
    def components(self, value: ModelComponent | ComponentCollection) -> None:
        """Set the components of the SampleModel."""
        if not isinstance(value, (ModelComponent, ComponentCollection)):
            raise TypeError(
                "components must be a ModelComponent or a ComponentCollection"
            )

        self.clear_components()
        self.append_component(value)

    @property
    def Q(self) -> np.ndarray | None:
        """Get the Q values of the SampleModel."""
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """Set the Q values of the SampleModel."""
        self._Q = _validate_and_convert_Q(value)

    # --------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # dunder methods
    # --------------------------------------------------------------------

    def __repr__(self):
        return f"{self.__class__.__name__}(unique_name={self.unique_name}, unit={self.unit}), Q = {self.Q}, components = {self.components}"
