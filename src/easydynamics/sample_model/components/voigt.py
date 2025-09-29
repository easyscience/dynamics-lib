from __future__ import annotations

from scipy.special import voigt_profile

from typing import Union

import numpy as np

from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent

import scipp as sc

import warnings

Numeric = Union[float, int]


class Voigt(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (Numeric or Parameter or None): Center of the Voigt profile.
        gaussian_width (Numeric or Parameter): Standard deviation of the Gaussian part.
        lorentzian_width (Numeric or Parameter): Half width at half max (HWHM) of the Lorentzian part.
        area (Numeric or Parameter): Total area under the curve.
    """

    def __init__(
        self,
        name: str = "Voigt",
        area: Union[Numeric, Parameter] = 1.0,
        center: Union[Numeric, Parameter, None] = None,
        gaussian_width: Union[Numeric, Parameter] = 1.0,
        lorentzian_width: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(gaussian_width, (Numeric, Parameter)):
            raise TypeError("gaussian_width must be a number or a Parameter.")

        if not isinstance(lorentzian_width, (Numeric, Parameter)):
            raise TypeError("lorentzian_width must be a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(gaussian_width, Numeric):
            if gaussian_width <= 0:
                raise ValueError(
                    "The gaussian_width of a Voigt must be greater than zero."
                )
            gaussian_width = float(gaussian_width)

        if isinstance(lorentzian_width, Numeric):
            if lorentzian_width <= 0:
                raise ValueError(
                    "The lorentzian_width of a Voigt must be greater than zero."
                )
            lorentzian_width = float(lorentzian_width)

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )
            area = float(area)

        super().__init__(name=name)

        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        elif isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(gaussian_width, Numeric):
            self.gaussian_width = Parameter(
                name=name + " gaussian_width", value=gaussian_width, unit=unit, min=0.0
            )
        else:
            self.gaussian_width = gaussian_width

        if isinstance(lorentzian_width, Numeric):
            self.lorentzian_width = Parameter(
                name=name + " lorentzian_width",
                value=lorentzian_width,
                unit=unit,
                min=0.0,
            )
        else:
            self.lorentzian_width = lorentzian_width

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.gaussian_width.value <= 0:
            raise ValueError("The gaussian_width of a Voigt must be greater than zero.")
        if self.lorentzian_width.value <= 0:
            raise ValueError(
                "The lorentzian_width of a Voigt must be greater than zero."
            )
        if self.area.value < 0:
            warnings.warn(
                "The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(
                    f"Input x has unit {x.unit}, but Voigt component has unit {self.unit}. Converting Voigt to {x.unit}."
                )
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return self.area.value * voigt_profile(
            x_in - self.center.value,
            self.gaussian_width.value,
            self.lorentzian_width.value,
        )

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """
        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.gaussian_width.convert_unit(unit)
        self.lorentzian_width.convert_unit(unit)
        self.unit = unit

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center, self.gaussian_width, self.lorentzian_width]

    def copy(self) -> Voigt:
        model_copy = Voigt(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            gaussian_width=self.gaussian_width.value,
            lorentzian_width=self.lorentzian_width.value,
            unit=self.unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.gaussian_width.fixed = self.gaussian_width.fixed
        model_copy.lorentzian_width.fixed = self.lorentzian_width.fixed

        return model_copy

    def __repr__(self):
        return f"Voigt(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center},\n gaussian_width = {self.gaussian_width},\n lorentzian_width = {self.lorentzian_width})"
