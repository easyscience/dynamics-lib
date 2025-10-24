import warnings
from typing import Union

from easyscience.variable import Parameter

Numeric = Union[int, float]

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class ValidationMixin:
    """Provides `_validate_area` for components that define an 'area' parameter."""

    def _validate_area(self, area: Union[Numeric, Parameter], name: str) -> Parameter:
        """Validate and convert an area-like input to a Parameter object."""
        if not isinstance(area, (Parameter, Numeric)):
            raise TypeError("area must be a number or a Parameter.")

        if isinstance(area, Numeric):
            area = Parameter(name=name + " area", value=float(area), unit=self._unit)

        if area.value < 0:
            warnings.warn(
                f"The area of {name} is negative, which may not be physically meaningful."
            )
        else:
            area.min = 0.0

        return area

    def _validate_center(
        self,
        center: Union[Numeric, Parameter, None],
        name: str,
        fix_if_none: bool,
    ) -> Parameter:
        """Validate and convert a center-like input to a Parameter object."""
        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number, or a Parameter.")

        if center is None:
            center = Parameter(
                name=name + " center",
                value=0.0,
                unit=self._unit,
                fixed=fix_if_none,
            )
        elif isinstance(center, Numeric):
            center = Parameter(
                name=name + " center", value=float(center), unit=self._unit
            )

        return center

    def _validate_width(
        self,
        width: Union[Numeric, Parameter],
        name: str,
        param_name: str = "width",
    ) -> Parameter:
        """Validate and convert a width-like input to a Parameter object."""
        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError(f"{param_name} must be a number or a Parameter.")

        # Width
        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError(f"{param_name} must be a number or a Parameter.")

        if isinstance(width, Numeric):
            if float(width) < MINIMUM_WIDTH:
                raise ValueError(
                    f"The {param_name} of a {self.__class__.__name__} must be greater than zero."
                )
            width = Parameter(
                name=name + " " + param_name,
                value=float(width),
                unit=self._unit,
                min=MINIMUM_WIDTH,
            )
        else:
            if width.value <= 0:
                raise ValueError(
                    f"The {param_name} of a {self.__class__.__name__} must be greater than zero."
                )
            width.min = MINIMUM_WIDTH

        return width
