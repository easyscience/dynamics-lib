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
        name: str = "Voigt",
        area: Numeric = 1.0,
        center: Union[Numeric, None] = None,
        gaussian_width: Numeric = 1.0,
        lorentzian_width: Numeric = 1.0,
        unit: Union[str, sc.Unit] = "meV",
    ):
        # Validate inputs
        if not isinstance(area, Numeric):
            raise TypeError("area must be a number.")

        area = float(area)
        if area < 0:
            warnings.warn(
                "The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(
                    name
                )
            )

        if center is not None and not isinstance(center, Numeric):
            raise TypeError("center must be None or a number.")

        if isinstance(center, Numeric):
            center = float(center)

        if not isinstance(gaussian_width, Numeric):
            raise TypeError("gaussian_width must be a number.")

        gaussian_width = float(gaussian_width)
        if gaussian_width <= 0:
            raise ValueError("The gaussian_width of a Voigt must be greater than zero.")

        if not isinstance(lorentzian_width, Numeric):
            raise TypeError("lorentzian_width must be a number.")

        lorentzian_width = float(lorentzian_width)
        if lorentzian_width <= 0:
            raise ValueError(
                "The lorentzian_width of a Voigt must be greater than zero."
            )

        super().__init__(name=name, unit=unit)

        # Create Parameters from floats
        self._area = Parameter(name=name + " area", value=area, unit=unit)
        if area > 0:
            self._area.min = 0.0

        if center is None:
            self._center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        else:
            self._center = Parameter(name=name + " center", value=center, unit=unit)

        self._gaussian_width = Parameter(
            name=name + " gaussian_width",
            value=gaussian_width,
            unit=unit,
            min=MINIMUM_WIDTH,
        )

        self._lorentzian_width = Parameter(
            name=name + " lorentzian_width",
            value=lorentzian_width,
            unit=unit,
            min=MINIMUM_WIDTH,
        )

    @property
    def area(self) -> Parameter:
        """Return the area parameter."""
        return self._area

    @area.setter
    def area(self, value: Numeric):
        """Set the area parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("area must be a number.")
        value = float(value)
        if value < 0:
            warnings.warn(
                "The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """Return the center parameter."""
        return self._center

    @center.setter
    def center(self, value: Numeric):
        """Set the center parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number.")
        self._center.value = float(value)

    @property
    def gaussian_width(self) -> Parameter:
        """Return the gaussian_width parameter."""
        return self._gaussian_width

    @gaussian_width.setter
    def gaussian_width(self, value: Numeric):
        """Set the gaussian_width parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("gaussian_width must be a number.")
        value = float(value)
        if value <= 0:
            raise ValueError("The gaussian_width of a Voigt must be greater than zero.")
        self._gaussian_width.value = value

    @property
    def lorentzian_width(self) -> Parameter:
        """Return the lorentzian_width parameter."""
        return self._lorentzian_width

    @lorentzian_width.setter
    def lorentzian_width(self, value: Numeric):
        """Set the lorentzian_width parameter."""
        if not isinstance(value, Numeric):
            raise TypeError("lorentzian_width must be a number.")
        value = float(value)
        if value <= 0:
            raise ValueError(
                "The lorentzian_width of a Voigt must be greater than zero."
            )
        self._lorentzian_width.value = value

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Voigt at the given x values.
        If x is a scipp Variable, the unit of the Voigt will be converted to match x.
        The Voigt evaluates to the convolution of a Gaussian with sigma gaussian_width and a Lorentzian with half width at half max lorentzian_width, centered at center, with area equal to area."""

        x = self._prepare_x_for_evaluate(x)

        return self._area.value * voigt_profile(
            x - self._center.value,
            self._gaussian_width.value,
            self._lorentzian_width.value,
        )

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """
        self._area.convert_unit(unit)
        self._center.convert_unit(unit)
        self._gaussian_width.convert_unit(unit)
        self._lorentzian_width.convert_unit(unit)
        self._unit = unit

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self._area, self._center, self._gaussian_width, self._lorentzian_width]

    def copy(self, name: Optional[str] = None) -> Voigt:
        """
        Return a deep copy of this component with independent parameters.
        """
        if name is None:
            name = "copy of " + self.name

        model_copy = Voigt(
            name=name,
            area=self._area.value,
            center=self._center.value,
            gaussian_width=self._gaussian_width.value,
            lorentzian_width=self._lorentzian_width.value,
            unit=self._unit,
        )
        model_copy._area.fixed = self._area.fixed
        model_copy._center.fixed = self._center.fixed
        model_copy._gaussian_width.fixed = self._gaussian_width.fixed
        model_copy._lorentzian_width.fixed = self._lorentzian_width.fixed

        return model_copy

    def __repr__(self):
        return f"Voigt(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center},\n _gaussian_width = {self._gaussian_width},\n _lorentzian_width = {self._lorentzian_width})"
