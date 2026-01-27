# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase
from easyscience.variable import DescriptorNumber
from numpy.typing import ArrayLike
from scipp import UnitError

from easydynamics.utils.utils import Numeric

Q_type = np.ndarray | Numeric | list | ArrayLike


class DiffusionModelBase(ModelBase):
    """Base class for constructing diffusion models."""

    def __init__(
        self,
        display_name='MyDiffusionModel',
        unique_name: str | None = None,
        unit: str | sc.Unit = 'meV',
    ):
        """Initialize a new DiffusionModel.

        Parameters
        ----------
        display_name : str
            Display name of the diffusion model.
        unit : str or sc.Unit, optional
            Unit of the diffusion model. Defaults to "meV".
        """

        try:
            test = DescriptorNumber(name='test', value=1, unit=unit)
            test.convert_unit('meV')
        except Exception as e:
            raise UnitError(
                f'Invalid unit: {unit}. Unit must be a string or scipp Unit and convertible to meV.'  # noqa: E501
            ) from e

        super().__init__(display_name=display_name, unique_name=unique_name)
        self._unit = unit

    @property
    def unit(self) -> str:
        """Get the unit of the DiffusionModel.

        Returns
        -------
        str or sc.Unit or None
        """
        return str(self._unit)

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f'Unit is read-only. Use convert_unit to change the unit between allowed types '
                f'or create a new {self.__class__.__name__} with the desired unit.'
            )
        )  # noqa: E501

    def __repr__(self):
        """String representation of the Diffusion model."""
        return f'{self.__class__.__name__}(display_name={self.display_name}, unit={self.unit})'
