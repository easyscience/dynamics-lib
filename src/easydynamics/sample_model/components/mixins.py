# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.utils.utils import Numeric

MINIMUM_WIDTH = 1e-10  # To avoid division by zero
MINIMUM_AREA = 0.0  # To avoid negative areas
DHO_MINIMUM_CENTER = 1e-10  # To avoid zero center in DHO


class CreateParametersMixin:
    """Provides parameter creation and validation methods for model
    components.

    This mixin provides methods to create and validate common physics
    parameters (area, center, width) with appropriate bounds and type
    checking.
    """

    def _create_area_parameter(
        self,
        area: Numeric | Parameter,
        name: str,
        unit: str | sc.Unit = 'meV',
        minimum_area: float = MINIMUM_AREA,
    ) -> Parameter:
        """Validate and convert a number to a Parameter describing the
        area of a function.

        If the area is negative, a warning is raised.
        If the area is non-negative, its minimum is set to 0 to avoid it
        accidentally becoming negative during fitting.

        Args:
            area (Numeric | Parameter): The area value or Parameter.
            name (str): The name of the model component.
            unit (str | sc.Unit, default='meV'): The unit of the area Parameter.
            minimum_area (float, default=MINIMUM_AREA): The minimum allowed area.

        Returns:
            Parameter: The validated area Parameter.

        Raises:
            TypeError: If area is not a number or a Parameter.
            ValueError: If area is not a finite number or if the area
                Parameter has a non-finite value.
        """
        if not isinstance(area, (Parameter, Numeric)):
            raise TypeError('area must be a number or a Parameter.')

        if isinstance(area, Numeric):
            if not np.isfinite(area):
                raise ValueError('area must be a finite number or a Parameter')

            area = Parameter(name=name + ' area', value=float(area), unit=unit)

        if area.value < 0:
            warnings.warn(
                f'The area of {name} is negative, which may not be physically meaningful.',
                UserWarning,
                stacklevel=3,
            )
        else:
            area.min = minimum_area

        return area

    def _create_center_parameter(
        self,
        center: Numeric | Parameter | None,
        name: str,
        fix_if_none: bool,
        unit: str | sc.Unit = 'meV',
        enforce_minimum_center: bool = False,
    ) -> Parameter:
        """Validate and convert a number to a Parameter describing the
        center of a function.

        Args:
            center (Numeric | Parameter | None): The center value or
                Parameter.
            name (str): The name of the model component.
            fix_if_none (bool): Whether to fix the center Parameter
                if center is None.
            unit (str | sc.Unit, default='meV'): The unit of the center
                Parameter.
            enforce_minimum_center (bool, default=False): Whether to
                enforce a minimum center value to avoid zero center in
                DHO.

        Returns:
            Parameter: The validated center Parameter.

        Raises:
            TypeError: If center is not None, a number, or a Parameter.
            ValueError: If center is a number but not finite, or if
                center is a Parameter but has a non-finite value.
        """
        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError('center must be None, a number, or a Parameter.')

        if center is None:
            center = Parameter(
                name=name + ' center',
                value=0.0,
                unit=unit,
                fixed=fix_if_none,
            )
        elif isinstance(center, Numeric):
            if not np.isfinite(center):
                raise ValueError('center must be None, a finite number or a Parameter')

            center = Parameter(name=name + ' center', value=float(center), unit=unit)
        if enforce_minimum_center:
            center.min = DHO_MINIMUM_CENTER
        return center

    def _create_width_parameter(
        self,
        width: Numeric | Parameter,
        name: str,
        param_name: str = 'width',
        unit: str | sc.Unit = 'meV',
        minimum_width: float = MINIMUM_WIDTH,
    ) -> Parameter:
        """Validate and convert a number to a Parameter describing the
        width of a function.

        Args:
            width (Numeric | Parameter): The width value or Parameter.
            name (str): The name of the model component.
            param_name (str, default='width'): The name of the width parameter.
            unit (str | sc.Unit, default='meV'): The unit of the width Parameter.
            minimum_width (float, default=MINIMUM_WIDTH): The minimum
                allowed width.

        Returns:
            Parameter: The validated width Parameter.

        Raises:
            TypeError: If width is not a number or a Parameter.
            ValueError: If width is non-positive.
        """
        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError(f'{param_name} must be a number or a Parameter.')

        if isinstance(width, Numeric):
            if not np.isfinite(width):
                raise ValueError(f'{param_name} must be a finite number or a Parameter')

            if float(width) < minimum_width:
                raise ValueError(
                    f'The {param_name} of a {self.__class__.__name__} must be greater than zero.'
                )
            width = Parameter(
                name=name + ' ' + param_name,
                value=float(width),
                unit=unit,
                min=minimum_width,
            )
        else:
            if width.value <= 0:
                raise ValueError(
                    f'The {param_name} of a {self.__class__.__name__} must be greater than zero.'
                )
            width.min = minimum_width

        return width
