# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc

from easydynamics.sample_model.model_base import ModelBase
from easydynamics.utils.utils import Q_type

from .component_collection import ComponentCollection
from .components.model_component import ModelComponent


class BackgroundModel(ModelBase):
    """BackgroundModel represents a model of the background in an
    experiment at various Q.
    """

    def __init__(
        self,
        display_name: str | None = 'MyBackgroundModel',
        unique_name: str | None = None,
        unit: str | sc.Unit = 'meV',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        """Initialize the BackgroundModel.

        Parameters
        ----------
        display_name : str | None, optional
            Display name of the model. By default, 'MyBackgroundModel'.
        unique_name : str | None, optional
            Unique name of the model. If None,
            a unique name will be generated. By default, None.
        unit : str | sc.Unit, optional
            Unit of the model. By default, 'meV'.
        components : ModelComponent | ComponentCollection | None, optional
            Template components of the model. If None, no components
            are added. These components are copied into
            ComponentCollections for each Q value. By default, None.
        Q : Q_type | None, optional
            Q values for the model. If None, Q is not
            set. By default, None.
        """
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
            components=components,
            Q=Q,
        )
