from __future__ import annotations

from scipy.special import voigt_profile

from typing import Union

import numpy as np

from easyscience.variable import Parameter

from .model_component import ModelComponent

import scipp as sc

import warnings

Numeric = Union[float, int]


class Voigt(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (Int or float or None): Center of the Voigt profile.
        gaussian_width (Int or float): Standard deviation of the Gaussian part.
        lorentzian_width (Int or float): Half width at half max (HWHM) of the Lorentzian part.
        area (Int or float): Total area under the curve.
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

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")

        super().__init__(name=name)

        self._unit = unit  # Set the unit for the component

        # Create Parameters from floats
        self._area = Parameter(name=name + " area", value=area, unit=unit)

        if center is None:
            self._center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        else:
            self._center = Parameter(name=name + " center", value=center, unit=unit)

        self.gaussian_width = Parameter(
            name=name + " gaussian_width", value=gaussian_width, unit=unit, min=0.0
        )

        self.lorentzian_width = Parameter(
            name=name + " lorentzian_width",
            value=lorentzian_width,
            unit=unit,
            min=0.0,
        )

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.gaussian_width.value <= 0:
            raise ValueError("The gaussian_width of a Voigt must be greater than zero.")
        if self.lorentzian_width.value <= 0:
            raise ValueError(
                "The lorentzian_width of a Voigt must be greater than zero."
            )
        if self._area.value < 0:
            warnings.warn(
                "The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self._unit is not None and x.unit != self._unit:
                try:
                    self.convert_unit(x.unit.name)
                except Exception as e:
                    raise ValueError(
                        f"Input x has unit {x.unit}, but Voigt component has unit {self._unit}. Failed to convert Voigt to {x.unit}."
                    ) from e
                warnings.warn(
                    f"Input x has unit {x.unit}, but Voigt component has unit {self._unit}. Converting Voigt to {x.unit}."
                )
        else:
            x_in = x
        return self._area.value * voigt_profile(
            x_in - self._center.value,
            self.gaussian_width.value,
            self.lorentzian_width.value,
        )

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """
        self._area.convert_unit(unit)
        self._center.convert_unit(unit)
        self.gaussian_width.convert_unit(unit)
        self.lorentzian_width.convert_unit(unit)
        self._unit = unit

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self._area, self._center, self.gaussian_width, self.lorentzian_width]

    def copy(self) -> Voigt:
        model_copy = Voigt(
            name=self.name,
            area=self._area.value,
            center=self._center.value,
            gaussian_width=self.gaussian_width.value,
            lorentzian_width=self.lorentzian_width.value,
            unit=self._unit,
        )
        model_copy.area.fixed = self._area.fixed
        model_copy.center.fixed = self._center.fixed
        model_copy.gaussian_width.fixed = self.gaussian_width.fixed
        model_copy.lorentzian_width.fixed = self.lorentzian_width.fixed

        return model_copy

    def __repr__(self):
        return f"Voigt(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center},\n gaussian_width = {self.gaussian_width},\n lorentzian_width = {self.lorentzian_width})"
