# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.utils.utils import Numeric

from .model_component import ModelComponent

EPSILON = 1e-8  # small number to avoid floating point issues


class DeltaFunction(CreateParametersMixin, ModelComponent):
    """
    Delta function.

    Evaluates to zero everywhere, except in convolutions, where it acts as an identity. This is
    handled by the Convolution method. If the center is not provided, it will be centered at 0 and
    fixed, which is typically what you want in QENS.
    """

    def __init__(
        self,
        center: Numeric | Parameter | None = None,
        area: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = 'meV',
        display_name: str | None = 'DeltaFunction',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Delta function.

        Parameters
        ----------
        center : Numeric | Parameter | None, default=None
            Center of the delta function. If None. By default, None.
        area : Numeric | Parameter, default=1.0
            Total area under the curve. By default, 1.0.
        unit : str | sc.Unit, default='meV'
            Unit of the parameters. By default, 'meV'.
        display_name : str | None, default='DeltaFunction'
            Name of the component. By default, 'DeltaFunction'.
        unique_name : str | None, default=None
            Unique name of the component. If None, a unique_name is automatically generated. By
            default, None.
        """
        # Validate inputs and create Parameters if not given
        super().__init__(
            display_name=display_name,
            unit=unit,
            unique_name=unique_name,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=display_name, unit=self._unit)
        center = self._create_center_parameter(
            center=center, name=display_name, fix_if_none=True, unit=self._unit
        )

        self._area = area
        self._center = center

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns
        -------
        Parameter
            The area parameter.
        """

        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Set the value of the area parameter.

        Parameters
        ----------
        value : Numeric
            The new value for the area parameter.

        Raises
        ------
        TypeError :
            If the value is not a number.
        """

        if not isinstance(value, Numeric):
            raise TypeError('area must be a number')
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter.

        Returns
        -------
        Parameter
            The center parameter.
        """

        return self._center

    @center.setter
    def center(self, value: Numeric | None) -> None:
        """
        Set the center parameter value.

        Parameters
        ----------
        value : Numeric | None
            The new value for the center parameter. If None. By default, 0 and is fixed.

        Raises
        ------
        TypeError :
            If the value is not a number or None.
        """

        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        """
        Evaluate the Delta function at the given x values.

        The Delta function evaluates to zero everywhere, except at the center. Its numerical
        integral is equal to the area. It acts as an identity in convolutions.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the Delta function.

        Returns
        -------
        np.ndarray
            The evaluated Delta function at the given x values.
        """

        # x assumed sorted, 1D numpy array
        x = self._prepare_x_for_evaluate(x)
        model = np.zeros_like(x, dtype=float)
        center = self.center.value
        area = self.area.value

        if x.min() - EPSILON <= center <= x.max() + EPSILON:
            # nearest index
            i = np.argmin(np.abs(x - center))

            # left half-width
            if i == 0:  # noqa: SIM108
                left = x[1] - x[0] if x.size > 1 else 0.5
            else:
                left = x[i] - x[i - 1]

            # right half-width
            if i == x.size - 1:  # noqa: SIM108
                right = x[-1] - x[-2] if x.size > 1 else 0.5
            else:
                right = x[i + 1] - x[i]

            # effective bin width: half left + half right
            bin_width = 0.5 * (left + right)

            model[i] = area / bin_width

        return model

    def __repr__(self) -> str:
        """
        Return a string representation of the Delta function.

        Returns
        -------
        str
            A string representation of the Delta function.
        """

        return f'DeltaFunction(unique_name = {self.unique_name}, unit = {self._unit},\n \
        area = {self.area},\n center = {self.center}'
