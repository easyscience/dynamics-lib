from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from .model_component import ModelComponent

Numeric = Union[float, int]

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class Lorentzian(ModelComponent):
    """
    Lorentzian function: area*width / (pi * ( (x - center)^2 + width^2 ) )
    If the center is not provided, it will be centered at 0 and fixed, which is typically what you want in QENS.

    Args:
        name (str): Name of the component.
        area (Int, float or Parameter): Area of the Lorentzian.
        center (Int, float, None or Parameter): Peak center. If None, defaults to 0 and is fixed.
        width (Int, float or Parameter): Half Width at Half Maximum (HWHM)
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
    """

    def __init__(
        self,
        name: str = "Lorentzian",
        area: Numeric = 1.0,
        center: Union[Numeric, None] = None,
        width: Numeric = 1.0,
        unit: Union[str, sc.Unit] = "meV",
    ):
        # Validate inputs and create Parameters if not given
        # this method lives in ModelComponent since it's the same for all components
        self.validate_unit(unit)

        # Area
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")
        if isinstance(area, Numeric):
            area = Parameter(name=name + " area", value=float(area), unit=unit)

        if area.value < 0:
            warnings.warn(
                "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
                    name
                )
            )
        else:
            area.min = 0.0

        # Center
        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number, or a Parameter.")

        if center is None:
            center = Parameter(name=name + " center", value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Numeric):
            center = Parameter(name=name + " center", value=float(center), unit=unit)

        # Width
        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if isinstance(width, Numeric):
            if float(width) < MINIMUM_WIDTH:
                raise ValueError("The width of a Lorentzian must be greater than zero.")
            width = Parameter(
                name=name + " width", value=float(width), unit=unit, min=MINIMUM_WIDTH
            )
        else:
            if width.value <= 0:
                raise ValueError("The width of a Lorentzian must be greater than zero.")
            width.min = MINIMUM_WIDTH

        super().__init__(
            name=name,
            unit=unit,
            area=area,
            center=center,
            width=width,
        )

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Lorentzian at the given x values.
        If x is a scipp Variable, the unit of the Lorentzian will be converted to match x.
        The Lorentzian evaluates to area*width / (pi * ( (x - center)^2 + width^2 ) )"""

        x = self._prepare_x_for_evaluate(x)

        normalization = self.width.value / np.pi
        denominator = (x - self.center.value) ** 2 + self.width.value**2

        return self.area.value * normalization / denominator

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str or sc.Unit): The new unit to convert to.
        """

        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.width.convert_unit(unit)
        self._unit = unit

    def __repr__(self):
        return f"Lorentzian(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
