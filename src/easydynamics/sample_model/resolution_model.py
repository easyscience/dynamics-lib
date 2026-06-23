# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import scipp as sc

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Polynomial
from easydynamics.sample_model.components.exponential import Exponential
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.model_base import ModelBase
from easydynamics.sample_model.sample_model import SampleModel
from easydynamics.utils.utils import Q_type


class ResolutionModel(ModelBase):
    """
    ResolutionModel represents a model of the instrument resolution in an experiment at various Q.

    Examples
    --------
    **Creating a Gaussian resolution model**

    A single Gaussian is the most common resolution model. Note that ``DeltaFunction``,
    ``Polynomial``, and ``Exponential`` components are not allowed in a ResolutionModel:
    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    Q = np.linspace(0.5, 2, 7)
    resolution_model = sm.ResolutionModel(
        components=sm.Gaussian(width=0.05, area=1.0),
        Q=Q,
    )
    energy = np.linspace(-2, 2, 100)
    resolution = resolution_model.evaluate(energy)
    ```

    **Building a resolution model from a fitted SampleModel**

    After fitting vanadium data with a SampleModel, use ``from_sample_model`` to convert it
    directly into a ResolutionModel:
    ```python
    resolution_model = sm.ResolutionModel.from_sample_model(fitted_sample_model)
    ```
    """

    def __init__(
        self,
        display_name: str = 'MyResolutionModel',
        unique_name: str | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            x_unit=x_unit,
            y_unit=y_unit,
            components=components,
            Q=Q,
        )

    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """
        Append a component to the ResolutionModel.

        Does not allow DeltaFunction, Polynomial, or Exponential components.
        """
        components = component if isinstance(component, ComponentCollection) else (component,)

        for comp in components:
            if isinstance(comp, (DeltaFunction, Polynomial, Exponential)):
                raise TypeError(
                    f'Component in ResolutionModel cannot be a {comp.__class__.__name__}'
                )

        super().append_component(component)

    @classmethod
    def from_sample_model(
        cls,
        sample_model: SampleModel,
        normalize_area: bool = True,
        fix_parameters: bool = True,
    ) -> 'ResolutionModel':
        """Create a ResolutionModel from a SampleModel."""
        if not isinstance(sample_model, SampleModel):
            raise TypeError(
                f'sample_model must be an instance of SampleModel. Got {type(sample_model)}.'
            )

        if not isinstance(normalize_area, bool):
            raise TypeError('normalize_area must be True or False.')

        if not isinstance(fix_parameters, bool):
            raise TypeError('fix_parameters must be True or False.')

        resolution_model = cls(
            display_name=sample_model.display_name,
            x_unit=sample_model.x_unit,
            y_unit=sample_model.y_unit,
            components=sample_model.components,
            Q=sample_model.Q,
        )

        if sample_model.Q is not None:
            resolution_model._ensure_component_collections_current()
            for index in range(len(sample_model.Q)):
                resolution_model._component_collections[index] = copy(
                    sample_model.get_component_collection(Q_index=index)
                )
        if normalize_area:
            resolution_model.normalize_area()

        if fix_parameters:
            resolution_model.fix_all_parameters()

        return resolution_model

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'unit={self.unit}, '
            f'Q_len={None if self._Q is None else len(self._Q)}, '
            f'components={self.components})'
        )
