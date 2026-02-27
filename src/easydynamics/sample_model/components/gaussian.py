# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.utils.utils import Numeric

from .model_component import ModelComponent


class Gaussian(CreateParametersMixin, ModelComponent):
    """
    Model of a Gaussian function.

    The intensity is given by $I(x) = \frac{A}{\\sigma \\sqrt{2\\pi}}
    e^{-\frac{1}{2} \\left(\frac{x - x_0}{\\sigma}\right)^2}$,
    where $A$ is the area, $x_0$ is the center, and $\\sigma$ is the
    width. If the center is not provided, it will be centered at 0 and
    fixed, which is typically what you want in QENS.

    Args:
        area (Int | float | Parameter): Area of the Gaussian.
        center (Int | float | None | Parameter): Center of the Gaussian.
            If None, defaults to 0 and is fixed
        width (Int | float | Parameter): Standard deviation.
        unit (str | sc.Unit): Unit of the parameters. Defaults to "meV".
        display_name (str | None): Name of the component.
        unique_name (str | None): Unique name of the component. if None,
            a unique_name is automatically generated.
    """

    def __init__(
        self,
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter | None = None,
        width: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = "meV",
        display_name: str | None = "Gaussian",
        unique_name: str | None = None,
    ):
        # Validate inputs and create Parameters if not given
        super().__init__(
            display_name=display_name,
            unit=unit,
            unique_name=unique_name,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(
            area=area, name=display_name, unit=self._unit
        )
        center = self._create_center_parameter(
            center=center, name=display_name, fix_if_none=True, unit=self._unit
        )
        width = self._create_width_parameter(
            width=width, name=display_name, unit=self._unit
        )

        self._area = area
        self._center = center
        self._width = width

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns:
            Parameter: The area parameter.
        """

        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Set the value of the area parameter.

        Args:
            value (Numeric): The new value for the area parameter.

        Raises:
            TypeError: If the value is not a number.
        """

        if not isinstance(value, Numeric):
            raise TypeError("area must be a number")
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter.

        Returns:
            Parameter: The center parameter.
        """

        return self._center

    @center.setter
    def center(self, value: Numeric) -> None:
        """
        Set the center parameter value.

        Args:
            value (Numeric | None): The new value for the center
            parameter. If None, defaults to 0 and is fixed.

        Raises:
            TypeError: If the value is not a number or None.
        """

        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number")
        self._center.value = value

    @property
    def width(self) -> Parameter:
        """
        Get the width parameter.

        Returns:
            Parameter: The width parameter.
        """
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """
        Set the center parameter value.

        Args:
            value (Numeric | None): The new value for the center
            parameter. If None, defaults to 0 and is fixed.

        Raises:
            TypeError: If the value is not a number or None.
        """
        if not isinstance(value, Numeric):
            raise TypeError("width must be a number")
        self._width.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
    ) -> np.ndarray:
        """Evaluate the Gaussian at the given x values.

        If x is a scipp Variable, the unit of the Gaussian will be
        converted to match x.
        The intensity is given by $I(x) = \frac{A}
        {\\sigma \\sqrt{2\\pi}}
        e^{-\frac{1}{2} \\left(\frac{x - x_0}{\\sigma}\right)^2}$,

        Args:
            x (Numeric or list or np.ndarray or sc.Variable or
                sc.DataArray):
                The x values at which to evaluate the Gaussian.

        Returns:
            np.ndarray: The intensity of the Gaussian at the given x
                values.
        """

        x = self._prepare_x_for_evaluate(x)

        normalization = 1 / (np.sqrt(2 * np.pi) * self.width.value)
        exponent = -0.5 * ((x - self.center.value) / self.width.value) ** 2

        return self.area.value * normalization * np.exp(exponent)

    def __repr__(self) -> str:
        """
        Return a string representation of the Gaussian.

        Returns:
            str: A string representation of the Gaussian.

        """

        return f"Gaussian(unique_name = {self.unique_name}, unit = {self._unit},\n \
            area = {self.area},\n center = {self.center},\n width = {self.width})"
