import numpy as np
import scipp as sc
from numpy.typing import ArrayLike

from easydynamics.sample_model.sample_model_base import SampleModelBase

from .component_collection import ComponentCollection
from .components.model_component import ModelComponent

Numeric = float | int
Q_type = np.ndarray | Numeric | list | ArrayLike


class BackgroundModel(SampleModelBase):
    """BackgroundModel represents a model of the background in an experiment at various Q."""

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
