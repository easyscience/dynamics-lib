import numpy as np
import scipp as sc
from numpy.typing import ArrayLike

from easydynamics.sample_model.sample_model_base import SampleModelBase

from .component_collection import ComponentCollection
from .components import DeltaFunction, Polynomial
from .components.model_component import ModelComponent

Numeric = float | int
Q_type = np.ndarray | Numeric | list | ArrayLike


class ResolutionModel(SampleModelBase):
    """ResolutionMmodel represents a model of the instrment resolution in an experiment at various Q."""

    def __init__(
        self,
        display_name: str = "MySampleModel",
        unique_name: str | None = None,
        unit: str | sc.Unit = "meV",
        components: ComponentCollection | ModelComponent | None = None,
        Q: Q_type | None = None,
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
            components=components,
            Q=Q,
        )

    def append_component(self, component: ModelComponent):
        """Append a component to the ResolutionModel. Does not allow DeltaFunction or Polynomial components, as these are not physical resolution components.
        Args:
            component (ModelComponent): Component to append.
        Raises:
            TypeError: If the component is a DeltaFunction or Polynomial.
        """
        if isinstance(component, (DeltaFunction, Polynomial)):
            raise TypeError(
                f"component in ResolutionModel cannot be a {component.__class__.__name__}"
            )
        super().append_component(component)

