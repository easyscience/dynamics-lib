# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.utils.utils import Numeric


class DiffusionModelBase(EasyDynamicsModelBase):
    """Base class for constructing diffusion models."""

    def __init__(
        self,
        display_name: str | None = 'MyDiffusionModel',
        unique_name: str | None = None,
        scale: Numeric = 1.0,
        unit: str | sc.Unit = 'meV',
    ) -> None:
        """
        Initialize a new DiffusionModel.

        Parameters
        ----------
        display_name : str | None, default='MyDiffusionModel'
            Display name of the diffusion model.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated. By
            default, None.
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        unit : str | sc.Unit, default='meV'
            Unit of the diffusion model. Must be convertible to meV.

        Raises
        ------
        TypeError
            If scale is not a number.
        UnitError
            If unit is not a string or scipp Unit, or if it cannot be converted to meV.
        """

        try:
            test = DescriptorNumber(name='test', value=1, unit=unit)
            test.convert_unit('meV')
        except Exception as e:
            raise UnitError(
                f'Invalid unit: {unit}. Unit must be a string or scipp Unit and convertible to meV.'  # noqa: E501
            ) from e

        if not isinstance(scale, Numeric):
            raise TypeError('scale must be a number.')

        scale = Parameter(name='scale', value=float(scale), fixed=False, min=0.0, unit=unit)

        super().__init__(display_name=display_name, unique_name=unique_name, unit=unit)
        self._scale = scale

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scale(self) -> Parameter:
        """
        Get the scale parameter of the diffusion model.

        Returns
        -------
        Parameter
            Scale parameter of the diffusion model.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: Numeric) -> None:
        """
        Set the scale parameter of the diffusion model.

        Parameters
        ----------
        scale : Numeric
            The new value for the scale parameter. Must be a non-negative number.

        Raises
        ------
        TypeError
            If scale is not a number.
        ValueError
            If scale is negative.
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
        """
        String representation of the Diffusion model.

        Returns
        -------
        str
            String representation of the DiffusionModel.
        """
        return f'{self.__class__.__name__}(display_name={self.display_name}, unit={self.unit})'
