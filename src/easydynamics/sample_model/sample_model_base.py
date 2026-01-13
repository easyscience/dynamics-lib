# from easyscience.variable import DescriptorBase, Parameter
# from .components.model_component import ModelComponent
from copy import copy

import numpy as np
import scipp as sc

# from easyscience.job.theoreticalmodel import TheoreticalModelBase
from easyscience.base_classes.model_base import ModelBase
from numpy.typing import ArrayLike

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent

Numeric = float | int

# Base class for sample models - contains a list of ComponentCollection as function of Q and probably not much else
Q_type = np.ndarray | Numeric | list | ArrayLike


class SampleModelBase(ModelBase):
    def __init__(
        self,
        display_name: str = "MySampleModelBase",
        unique_name: str | None = None,
        unit: str | sc.Unit = "meV",
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
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
        elif isinstance(components, ModelComponent):
            self._components = ComponentCollection()
            self._components.append_component(components)
        else:
            if not isinstance(components, ComponentCollection):
                raise TypeError(
                    f"components must be a ModelComponent, a ComponentCollection or None, got {type(components).__name__}"
                )
            self._components = components

        if Q is None:
            self._Q = None
        else:
            self._Q = self._validate_and_convert_Q(Q)

    # --------------------------------------------------------------------
    # Component management
    # ---
    def append_component(self, component: ModelComponent) -> None:
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

    def append_components_from_collection(
        self, component_collection: ComponentCollection
    ) -> None:
        """Append a ComponentCollection to the SampleModel.
        Args:
        component_collection (ComponentCollection): The ComponentCollection to append.
        """
        if not isinstance(component_collection, ComponentCollection):
            raise TypeError(
                f"component_collection must be a ComponentCollection, got {type(component_collection).__name__}"
            )
        for component in component_collection.components:
            self._components.add_component(component)

    def clear_components(self) -> None:
        """Clear all ModelComponents from the SampleModel."""
        self._components.clear_components()

    # --------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------

    @property
    def components(self) -> ComponentCollection:
        """Get the components of the SampleModel."""
        return self._components

    @components.setter
    def components(self, value: ModelComponent | ComponentCollection) -> None:
        """Set the components of the SampleModel."""
        if isinstance(value, ModelComponent):
            self._components = ComponentCollection()
            self._components.append_component(value)
            return
        if not isinstance(value, ComponentCollection):
            raise TypeError(
                "components must be a ModelComponent or a ComponentCollection"
            )
        self._components = value

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
    # Private methods
    # --------------------------------------------------------------------

    def _validate_and_convert_Q(self, Q: Q_type) -> np.ndarray:
        """
        Validate and convert Q to a numpy array.
        Parameters
        ----------
        Q : Number, list, or np.ndarray
            Scattering vector values in 1/angstrom.
        Returns
        -------
        np.ndarray
            Q as a numpy array.
        """
        if isinstance(Q, Numeric):
            Q = np.array([Q])
        if isinstance(Q, list):
            Q = np.array(Q)
        if not isinstance(Q, np.ndarray):
            raise TypeError("Q must be a number, list, or numpy array.")

        if Q.ndim > 1:
            raise ValueError("Q must be a 1-dimensional array.")

        return Q

    def generate_component_collections(self) -> None:
        """Generate ComponentCollections for each Q value."""
        # TODO only regenerate if Q or diffusion models have changed

        if self._Q is None:
            raise ValueError("Q must be set before generating component collections.")

        self._component_collections = [ComponentCollection() for _ in self._Q]

        # Add copies of components from self._components to each component collection
        for collection in self._component_collections:
            for component in self._components.components:
                collection.add_component(copy(component))
