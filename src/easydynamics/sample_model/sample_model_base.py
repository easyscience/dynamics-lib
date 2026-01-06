import numpy as np
import scipp as sc

# from easyscience.job.theoreticalmodel import TheoreticalModelBase
from easyscience.base_classes.model_base import ModelBase

# from easyscience.variable import DescriptorBase, Parameter
# from .components.model_component import ModelComponent
from .component_collection import ComponentCollection

Numeric = float | int

# Base class for sample models - contains a list of ComponentCollection as function of Q and probably not much else


class SampleModelBase(ModelBase):
    def __init__(
        self,
        unit: str | sc.Unit = "meV",
        display_name: str = "MySampleModelBase",
        unique_name: str | None = None,
        components: ComponentCollection | None = None,
        Q: np.ndarray | None = None,
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
        self._components = (
            components if components is not None else ComponentCollection()
        )
