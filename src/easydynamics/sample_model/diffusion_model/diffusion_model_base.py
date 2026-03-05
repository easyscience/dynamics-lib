# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc
from easyscience.base_classes.model_base import ModelBase
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.utils.utils import Numeric


class DiffusionModelBase(ModelBase):
    """Base class for constructing diffusion models.

    Args:
        display_name (str): Display name of the diffusion model.
        unique_name (str | None): Unique name of the diffusion model.
            If None, a unique name will be generated.
        scale (Numeric): Scale factor for the diffusion model. Must be a
            non-negative number. Defaults to 1.0.
        unit (str | sc.Unit): Unit of the diffusion model. Must be
            convertible to meV. Defaults to "meV".

    Attributes:
        unit (str | sc.Unit): Unit of the diffusion model.
        scale (Parameter): Scale parameter of the diffusion model.
    """

    def __init__(
        self,
        display_name='MyDiffusionModel',
        unique_name: str | None = None,
        scale: Numeric = 1.0,
        unit: str | sc.Unit = 'meV',
    ):
        """Initialize a new DiffusionModel.

        Args:
            display_name (str): Display name of the diffusion model.
            unique_name (str | None): Unique name of the diffusion
                model. If None, a unique name will be generated.
            scale (Numeric): Scale factor for the diffusion model. Must
                be a non-negative number. Defaults to 1.0.
            unit (str | sc.Unit): Unit of the diffusion model. Must be
                convertible to meV. Defaults to "meV".

        Raises:
            TypeError: If scale is not a number.
            UnitError: If unit is not a string or scipp Unit, or if it
                cannot be converted to meV.
        """
        if not isinstance(scale, Numeric):
            raise TypeError('scale must be a number.')

        scale = Parameter(name='scale', value=float(scale), fixed=False, min=0.0)

        try:
            test = DescriptorNumber(name='test', value=1, unit=unit)
            test.convert_unit('meV')
        except Exception as e:
            raise UnitError(
                f'Invalid unit: {unit}. Unit must be a string or scipp Unit and convertible to meV.'  # noqa: E501
            ) from e

        super().__init__(display_name=display_name, unique_name=unique_name)
        self._unit = unit
        self._scale = scale

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def unit(self) -> str:
        """Get the unit of the energy axis of the DiffusionModel.

        Returns:
            (str | sc.Unit | None): Unit of the DiffusionModel.
        """
        return str(self._unit)

    @unit.setter
    def unit(self, unit_str: str) -> None:
        """The unit of the energy axis is read-only. To change the unit,
        use convert_unit or create a new DiffusionModel with the desired
        unit.

        Args:
            unit_str (str): The new unit to set (ignored)

        Raises:
            AttributeError: Always, since the unit is read-only.
        """
        raise AttributeError(
            (
                f'Unit is read-only. Use convert_unit to change the unit between allowed types '
                f'or create a new {self.__class__.__name__} with the desired unit.'
            )
        )  # noqa: E501

    @property
    def scale(self) -> Parameter:
        """Get the scale parameter of the diffusion model.

        Returns:
            Parameter: scale parameter of the diffusion model
        """
        return self._scale

    @scale.setter
    def scale(self, scale: Numeric) -> None:
        """Set the scale parameter of the diffusion model.

        Args:
            scale (Numeric): The new value for the scale parameter. Must
                be a non-negative number.

        Raises:
            TypeError: If scale is not a number.
            ValueError: If scale is negative.
        """
        if not isinstance(scale, Numeric):
            raise TypeError('scale must be a number.')

        if float(scale) < 0:
            raise ValueError('scale must be non-negative.')
        self._scale.value = scale

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation of the Diffusion model.

        Returns:
            str: String representation of the DiffusionModel.
        """
        return f'{self.__class__.__name__}(display_name={self.display_name}, unit={self.unit})'
