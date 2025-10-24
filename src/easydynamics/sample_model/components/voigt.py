from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.special import voigt_profile

from .model_component import ModelComponent

Numeric = Union[float, int]

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class Voigt(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.
    If the center is not provided, it will be centered at 0 and fixed, which is typically what you want in QENS.

    Args:
        name (str): Name of the component.
        center (Int or float or None): Center of the Voigt profile.
        gaussian_width (Int or float): Standard deviation of the Gaussian part.
        lorentzian_width (Int or float): Half width at half max (HWHM) of the Lorentzian part.
        area (Int or float): Total area under the curve.
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
    """

    def __init__(
        self,
        name: Optional[str] = "Voigt",
        area: Optional[Union[Numeric, Parameter]] = 1.0,
        center: Optional[Union[Numeric, Parameter, None]] = None,
        gaussian_width: Optional[Union[Numeric, Parameter]] = 1.0,
        lorentzian_width: Optional[Union[Numeric, Parameter]] = 1.0,
        unit: Optional[Union[str, sc.Unit]] = "meV",
    ):
        # Validate inputs
        # this method lives in ModelComponent since it's the same for all components
        self.validate_unit(unit)

        # Area
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")
        if isinstance(area, Numeric):
            area = Parameter(name=name + " area", value=float(area), unit=unit)

        if area.value < 0:
            warnings.warn(
                "The area of the Voigt with name {} is negative, which may not be physically meaningful.".format(
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

        # Gaussian width
        if not isinstance(gaussian_width, (Numeric, Parameter)):
            raise TypeError("gaussian_width must be a number or a Parameter.")

        if isinstance(gaussian_width, Numeric):
            if float(gaussian_width) < MINIMUM_WIDTH:
                raise ValueError(
                    "The gaussian_width of a Voigt must be greater than zero."
                )
            gaussian_width = Parameter(
                name=name + " gaussian_width",
                value=float(gaussian_width),
                unit=unit,
                min=MINIMUM_WIDTH,
            )
        else:
            if gaussian_width.value <= 0:
                raise ValueError(
                    "The gaussian_width of a Voigt must be greater than zero."
                )
            gaussian_width.min = MINIMUM_WIDTH

        # Lorentzian width
        if not isinstance(lorentzian_width, (Numeric, Parameter)):
            raise TypeError("lorentzian_width must be a number or a Parameter.")

        if isinstance(lorentzian_width, Numeric):
            if float(lorentzian_width) < MINIMUM_WIDTH:
                raise ValueError(
                    "The lorentzian_width of a Voigt must be greater than zero."
                )
            lorentzian_width = Parameter(
                name=name + " lorentzian_width",
                value=float(lorentzian_width),
                unit=unit,
                min=MINIMUM_WIDTH,
            )
        else:
            if lorentzian_width.value <= 0:
                raise ValueError(
                    "The lorentzian_width of a Voigt must be greater than zero."
                )
            lorentzian_width.min = MINIMUM_WIDTH

        super().__init__(
            name=name,
            unit=unit,
            area=area,
            center=center,
            gaussian_width=gaussian_width,
            lorentzian_width=lorentzian_width,
        )

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Voigt at the given x values.
        If x is a scipp Variable, the unit of the Voigt will be converted to match x.
        The Voigt evaluates to the convolution of a Gaussian with sigma gaussian_width and a Lorentzian with half width at half max lorentzian_width, centered at center, with area equal to area."""

        x = self._prepare_x_for_evaluate(x)

        return self.area.value * voigt_profile(
            x - self.center.value,
            self.gaussian_width.value,
            self.lorentzian_width.value,
        )

    def __repr__(self):
        return f"Voigt(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n gaussian_width = {self.gaussian_width},\n lorentzian_width = {self.lorentzian_width})"
