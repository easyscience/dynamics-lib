# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc

from easydynamics.sample_model.model_base import ModelBase
from easydynamics.utils.utils import Q_type

from .component_collection import ComponentCollection
from .components import DeltaFunction
from .components import Polynomial
from .components.model_component import ModelComponent


class ResolutionModel(ModelBase):
    """
    ResolutionModel represents a model of the instrment resolution in an experiment at various Q.
    """

    def __init__(
        self,
        display_name: str = 'MyResolutionModel',
        unique_name: str | None = None,
        unit: str | sc.Unit = 'meV',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        """
        Initialize a ResolutionModel.

        Parameters
        ----------
        display_name : str, default='MyResolutionModel'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        unit : str | sc.Unit, default='meV'
            Unit of the model.
        components : ModelComponent | ComponentCollection | None, default=None
            Template components of the model. If None, no components are added. These components
            are copied into ComponentCollections for each Q value.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        """

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
            components=components,
            Q=Q,
        )

    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """
        Append a component to the ResolutionModel.

        Does not allow DeltaFunction or Polynomial components, as these are not physical resolution
        components.

        Parameters
        ----------
        component : ModelComponent | ComponentCollection
            Component(s) to append.

        Raises
        ------
        TypeError
            If the component is a DeltaFunction or Polynomial.
        """
        if isinstance(component, ComponentCollection):
            components = component.components
        else:
            components = (component,)

        for comp in components:
            if isinstance(comp, (DeltaFunction, Polynomial)):
                raise TypeError(
                    f'Component in ResolutionModel cannot be a {comp.__class__.__name__}'
                )

        super().append_component(component)
