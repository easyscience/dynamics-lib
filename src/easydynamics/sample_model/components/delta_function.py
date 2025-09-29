from typing import Union

from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent


import warnings

Numeric = Union[float, int]


class DeltaFunction(ModelComponent):
    """
    Delta function. Evaluates to zero everywhere, except in convolutions, where it acts as an identity. This is handled in the ResolutionHandler.

    Args:
        center (Numeric or Parameter or None): Center of the delta function. If None, defaults to 0 and is fixed.
        area (Numeric or Parameter): Total area under the curve.
    """

    def __init__(
        self,
        name: str = "DeltaFunction",
        center: Union[None, Numeric, Parameter] = None,
        area: Union[Numeric, Parameter] = 1.0,
        unit="meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )
            area = float(area)

        if isinstance(center, Numeric):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        elif isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x):
        if self.area.value < 0:
            warnings.warn(
                "The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )
        # TODO: Consider adding support for evaluation without resolution convolution
        return 0 * x

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center]

    def convert_unit(self, unit):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """
        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.unit = unit

    def copy(self) -> "DeltaFunction":
        """
        Return a deep copy of this component with independent parameters.
        """
        model_copy = DeltaFunction(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            unit=self.unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        return model_copy

    def __repr__(self):
        return f"DeltaFunction(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center}"
