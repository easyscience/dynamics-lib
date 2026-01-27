# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc

from easydynamics.sample_model.model_base import ModelBase
from easydynamics.utils.utils import Q_type

from .component_collection import ComponentCollection
from .components.model_component import ModelComponent


class BackgroundModel(ModelBase):
    """BackgroundModel represents a model of the background in an
    experiment at various Q.

    Parameters
    ----------
    display_name : str
        Display name of the model.
    unique_name : str | None
        Unique name of the model. If None, a unique name will be
        generated.
    unit : str | sc.Unit | None
        Unit of the model. If None, unitless.
    components : ModelComponent | ComponentCollection | None
        Template components of the model. If None, no components are
        added.
        These components are copied into ComponentCollections for each
        Q value.
    Q : Q_type | None
        Q values for the model. If None, Q is not set.
    """

    def __init__(
        self,
        display_name: str = 'MyBackgroundModel',
        unique_name: str | None = None,
        unit: str | sc.Unit = 'meV',
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
