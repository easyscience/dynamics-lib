# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.model_base import ModelBase
from easydynamics.utils.utils import Q_type


class BackgroundModel(ModelBase):
    """
    BackgroundModel represents a model of the background in an experiment at various Q.
    """

    def __init__(
        self,
        display_name: str | None = 'MyBackgroundModel',
        unique_name: str | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        """
        Initialize the BackgroundModel.

        Parameters
        ----------
        display_name : str | None, default='MyBackgroundModel'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy, Q, etc.).
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity).
        components : ModelComponent | ComponentCollection | None, default=None
            Template components of the model. If None, no components are added. These components
            are copied into ComponentCollections for each Q value.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        """
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            x_unit=x_unit,
            y_unit=y_unit,
            components=components,
            Q=Q,
        )
