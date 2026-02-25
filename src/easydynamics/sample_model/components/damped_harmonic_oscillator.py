# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.utils.utils import Numeric

from .model_component import ModelComponent


class DampedHarmonicOscillator(CreateParametersMixin, ModelComponent):
    r"""
    Model of a Damped Harmonic Oscillator (DHO).

    The intensity is given by
    $I(x) = 2*A*x_0^2*\gamma/\pi / ( (x^2 - x_0^2)^2 + (2*\gamma*x)^2 )$


    Args:
        area (Int | float): Area under the curve.
        center (Int | float): Resonance frequency, approximately the
            peak position.
        width (Int | float): Damping constant, approximately the
            half width at half max (HWHM) of the peaks.
        unit (str | sc.Unit): Unit of the parameters.
            Defaults to "meV".
        display_name (str | None): Display name of the component.
        unique_name (str | None): Unique name of the component.
            If None, a unique_name is automatically generated.

    Attributes:
        area (Parameter): Area under the curve.
        center (Parameter): Resonance frequency, approximately the
            peak position.
        width (Parameter): Damping constant, approximately the
            half width at half max (HWHM) of the peaks.
        unit (str | sc.Unit): Unit of the parameters.
        display_name (str | None): Display name of the component.
        unique_name (str | None): Unique name of the component.
    """

    def __init__(
        self,
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter = 1.0,
        width: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = "meV",
        display_name: str | None = "DampedHarmonicOscillator",
        unique_name: str | None = None,
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(
            area=area, name=display_name, unit=self._unit
        )
        center = self._create_center_parameter(
            center=center,
            name=display_name,
            fix_if_none=False,
            unit=self._unit,
            enforce_minimum_center=True,
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
        Set the value of the center parameter.

        Args:
            value (Numeric): The new value for the center parameter.

        Raises:
            TypeError: If the value is not a number.
            ValueError: If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number")

        if value <= 0:
            raise ValueError("center must be positive")
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
        Set the value of the width parameter.

        Args:
            value (Numeric): The new value for the width parameter.

        Raises:
            TypeError: If the value is not a number.
            ValueError: If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError("width must be a number")
        self._width.value = value

    def evaluate(
        self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray
    ) -> np.ndarray:
        r"""
        Evaluate the Damped Harmonic Oscillator at the given x values.

        If x is a scipp Variable, the unit of the DHO will be converted
        to match x. The intensity is given by $I(x) =
        2*A*x_0^2*\gamma/\pi / ( (x^2 - x_0^2)^2 + (2*\gamma*x)^2 )$

        Args:
            x (Numeric or list or np.ndarray or sc.Variable or
            sc.DataArray):
                The x values at which to evaluate the DHO.

        Returns:
            np.ndarray: The intensity of the DHO at the given x values.
        """

        x = self._prepare_x_for_evaluate(x)

        normalization = 2 * self.center.value**2 * self.width.value / np.pi
        # No division by zero here, width>0 enforced in setter
        denominator = (x**2 - self.center.value**2) ** 2 + (
            2 * self.width.value * x
        ) ** 2

        return self.area.value * normalization / (denominator)

    def __repr__(self):
        """
        Return a string representation of the Damped Harmonic
        Oscillator.

        Returns:
            str: A string representation of the Damped Harmonic
            Oscillator.
        """
        return f"DampedHarmonicOscillator(display_name = {self.display_name}, unit = {self._unit},\n \
        area = {self.area},\n center = {self.center},\n width = {self.width})"
