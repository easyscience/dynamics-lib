from typing import Union, List

import numpy as np

from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent

import scipp as sc

import warnings

Numeric = Union[float, int]


class Gaussian(ModelComponent):
    """
    Gaussian function. Creates new EasyScience Parameters if floats are provided, otherwise uses the provided Parameters.

    Args:
        area (Numeric or Parameter): Area of the Gaussian. Has the same unit as the x axis
        center (Numeric or Parameter or None): Center of the Gaussian. If None, defaults to 0 and is fixed
        width (Numeric or Parameter): Standard deviation.
    """

    def __init__(
        self,
        name: str = "Gaussian",
        area: Union[Numeric, Parameter] = 1.0,
        center: Union[Numeric, Parameter, None] = None,
        width: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs - throw errors before any Parameters are created
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(width, Numeric):
            if width <= 0:
                raise ValueError("The width of a Gaussian must be greater than zero.")
            width = float(width)

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(
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

        if isinstance(width, Numeric):
            self.width = Parameter(
                name=name + " width", value=width, unit=unit, min=0.0
            )
        else:
            self.width = width

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.width.value <= 0:
            raise ValueError("The width of a Gaussian must be greater than zero.")
        if self.area.value < 0:
            warnings.warn(
                "The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(
                    f"Input x has unit {x.unit}, but Gaussian component has unit {self.unit}. Converting Gaussian to {x.unit}."
                )
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return (
            self.area.value
            * 1
            / (np.sqrt(2 * np.pi) * self.width.value)
            * np.exp(-0.5 * ((x_in - self.center.value) / self.width.value) ** 2)
        )

    def get_parameters(self) -> List[Parameter]:
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center, self.width]

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """

        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.width.convert_unit(unit)
        self.unit = unit

    def copy(self) -> "Gaussian":
        """
        Return a deep copy of this component with independent parameters.
        """

        model_copy = Gaussian(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit,
        )

        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.width.fixed = self.width.fixed
        return model_copy

    def __repr__(self):
        return f"Gaussian(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
