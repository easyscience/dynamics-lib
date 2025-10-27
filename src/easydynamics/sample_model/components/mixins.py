import warnings
from typing import Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

Numeric = Union[int, float]


class CreateParametersMixin:
    """Provides parameter creation and validation methods for model components.

    This mixin provides methods to create and validate common physics parameters
    (area, center, width) with appropriate bounds and type checking.
    """

    def _create_area_parameter(
        self,
        area: Union[Numeric, Parameter],
        name: str,
        unit: Union[str, sc.Unit] = "meV",
        minimum_area: float = 0.0,
    ) -> Parameter:
        """Validate and convert a number to a Parameter describing the area
        of a function. If the area is negative, a warning is raised.
        If the area is non-negative, its minimum is set to 0 to avoid it
        accidentally becoming negative during fitting.
        args:
            area (Numeric or Parameter): The area value or Parameter.
            name (str): The name of the model component.
            unit (str or sc.Unit): The unit of the area Parameter.
            minimum_area (float): The minimum allowed value for the area Parameter.
        returns:
            Parameter: The validated area Parameter.
        raises:
            TypeError: If area is not a number or a Parameter.
            Warning: If area is negative.
        """
        if not isinstance(area, (Parameter, Numeric)):
            raise TypeError("area must be a number or a Parameter.")

        if isinstance(area, Numeric):
            if not np.isfinite(area):
                raise ValueError("area must be a finite number or a Parameter")

            area = Parameter(name=name + " area", value=float(area), unit=unit)

        if area.value < 0:
            warnings.warn(
                f"The area of {name} is negative, which may not be physically meaningful."
            )
        else:
            area.min = minimum_area

        return area

    def _create_center_parameter(
        self,
        center: Union[Numeric, Parameter, None],
        name: str,
        fix_if_none: bool,
        unit: Union[str, sc.Unit] = "meV",
    ) -> Parameter:
        """Validate and convert a number to a Parameter describing the center of a function.
        args:
            center (Numeric, Parameter, or None): The center value or Parameter.
            name (str): The name of the model component.
            fix_if_none (bool): Whether to fix the center Parameter if center is None.
            unit (str or sc.Unit): The unit of the center Parameter.
        returns:
            Parameter: The validated center Parameter.
        raises:
            TypeError: If center is not None, a number, or a Parameter.
        """
        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number, or a Parameter.")

        if center is None:
            center = Parameter(
                name=name + " center",
                value=0.0,
                unit=unit,
                fixed=fix_if_none,
            )
        elif isinstance(center, Numeric):
            if not np.isfinite(center):
                raise ValueError("center must be None, a finite number or a Parameter")

            center = Parameter(name=name + " center", value=float(center), unit=unit)

        return center

    def _create_width_parameter(
        self,
        width: Union[Numeric, Parameter],
        name: str,
        param_name: str = "width",
        unit: Union[str, sc.Unit] = "meV",
        minimum_width: float = 0.0,
    ) -> Parameter:
        """Validate and convert a number to a Parameter describing the width of a function.
        args:
            width (Numeric or Parameter): The width value or Parameter.
            name (str): The name of the model component.
            param_name (str): The name of the width parameter.
            unit (str or sc.Unit): The unit of the width Parameter.
            minimum_width (float): The minimum allowed value for the width Parameter.
        returns:
            Parameter: The validated width Parameter.
        raises:
            TypeError: If width is not a number or a Parameter.
            ValueError: If width is non-positive.
        """
        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError(f"{param_name} must be a number or a Parameter.")

        if isinstance(width, Numeric):
            if not np.isfinite(width):
                raise ValueError(f"{param_name} must be a finite number or a Parameter")

            if float(width) < minimum_width:
                raise ValueError(
                    f"The {param_name} of a {self.__class__.__name__} must be greater than zero."
                )
            width = Parameter(
                name=name + " " + param_name,
                value=float(width),
                unit=unit,
                min=minimum_width,
            )
        else:
            if width.value <= 0:
                raise ValueError(
                    f"The {param_name} of a {self.__class__.__name__} must be greater than zero."
                )
            width.min = minimum_width

        return width
