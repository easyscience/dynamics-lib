# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q


class DiffusionModelBase(EasyDynamicsModelBase):
    """Base class for constructing diffusion models."""

    def __init__(
        self,
        scale: Numeric = 1.0,
        Q: Q_type | None = None,
        unit: str | sc.Unit = "meV",
        name: str = "DiffusionModel",
        display_name: str | None = "MyDiffusionModel",
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a new DiffusionModel.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        unit : str | sc.Unit, default='meV'
            Unit of the diffusion model. Must be convertible to meV.
        name : str, default='DiffusionModel'
            Name of the diffusion model.
        display_name : str | None, default='MyDiffusionModel'
            Display name of the diffusion model.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated. By
            default, None.

        Raises
        ------
        TypeError
            If scale is not a number.
        UnitError
            If unit is not a string or scipp Unit, or if it cannot be converted to meV.
        """

        self._Q = _validate_and_convert_Q(Q)

        try:
            test = DescriptorNumber(name="test", value=1, unit=unit)
            test.convert_unit("meV")
        except Exception as e:
            raise UnitError(
                f"Invalid unit: {unit}. Unit must be a string or scipp Unit and convertible to meV."  # noqa: E501
            ) from e

        if not isinstance(scale, Numeric):
            raise TypeError("scale must be a number.")

        scale = Parameter(
            name="scale", value=float(scale), fixed=False, min=0.0, unit=unit
        )

        super().__init__(
            unit=unit, name=name, display_name=display_name, unique_name=unique_name
        )
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
            raise TypeError("scale must be a number.")

        if float(scale) < 0:
            raise ValueError("scale must be non-negative.")
        self._scale.value = float(scale)

    @property
    def Q(self) -> np.ndarray | None:
        """
        Get the Q values of the SampleModel.

        Returns
        -------
        np.ndarray | None
            The Q values of the SampleModel, or None if not set.
        """
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """
        Set the Q values of the SampleModel.

        If Q is already set, it throws an error if the new Q values are not similar to the old
        ones. To change Q values, first run clear_Q().

        Parameters
        ----------
        value : Q_type | None
            The new Q values to set. If None, Q values are not changed.

        Raises
        ------
        ValueError
            If the new Q values are not similar to the old ones when Q is already set.
        """
        if value is None:
            return
        old_Q = self._Q
        new_Q = _validate_and_convert_Q(value)

        if old_Q is None:
            self._Q = new_Q
            self._on_Q_change()
            return

        if len(old_Q) != len(new_Q) or not np.allclose(old_Q, new_Q):
            raise ValueError(
                "New Q values are not similar to the old ones. "
                "To change Q values, first run clear_Q()."
            )

    def clear_Q(self, confirm: bool = False) -> None:
        """
        Clear the Q values of the SampleModel, removing all component collections and their
        associated Parameters.

        Parameters
        ----------
        confirm : bool, default=False
            Confirmation to clear Q values.

        Raises
        ------
        ValueError
            If confirm is not True.
        """
        if not confirm:
            raise ValueError(
                "Clearing Q values requires confirmation. Set confirm=True to proceed."
            )
        self._Q = None
        self._on_Q_change()

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
        return (
            f"{self.__class__.__name__}(name={self.name}, display_name={self.display_name}, "
            f"unit={self.unit}), \n"
            f"    scale={self.scale})"
        )
