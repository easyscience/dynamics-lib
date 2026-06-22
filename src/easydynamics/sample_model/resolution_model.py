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
        """
        Create a ResolutionModel from a SampleModel.

        Parameters
        ----------
        sample_model : SampleModel
            SampleModel to create the ResolutionModel from.
        normalize_area : bool, default=True
            Whether to normalize the components in the ResolutionModel to have area 1.
        fix_parameters : bool, default=True
            Whether to fix the parameters in the ResolutionModel.

        Returns
        -------
        'ResolutionModel'
            ResolutionModel created from the SampleModel.

        Raises
        ------
        TypeError
            If sample_model is not a SampleModel, or if normalize_area or fix_parameters are not
            bool.
        """
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
            unit=sample_model.unit,
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
