from __future__ import annotations

from scipy.special import voigt_profile

from typing import Union, Optional

import numpy as np

from easyscience.variable import Parameter

from .model_component import ModelComponent

import scipp as sc
from scipp import UnitError

import warnings

Numeric = Union[float, int]

MINIMUM_WIDTH = 1e-10  # To avoid division by zero


class Voigt(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (Int or float or None): Center of the Voigt profile.
        _gaussian_width (Int or float): Standard deviation of the Gaussian part.
        _lorentzian_width (Int or float): Half width at half max (HWHM) of the Lorentzian part.
        area (Int or float): Total area under the curve.
    """

    def __init__(
        self,
        name: str = "Voigt",
        area: Numeric = 1.0,
        center: Union[Numeric, None] = None,
        _gaussian_width: Numeric = 1.0,
        _lorentzian_width: Numeric = 1.0,
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

        if not isinstance(_gaussian_width, Numeric):
            raise TypeError("_gaussian_width must be a number.")

        _gaussian_width = float(_gaussian_width)
        if _gaussian_width <= 0:
            raise ValueError(
                "The _gaussian_width of a Voigt must be greater than zero."
            )

        if not isinstance(_lorentzian_width, Numeric):
            raise TypeError("_lorentzian_width must be a number.")

        _lorentzian_width = float(_lorentzian_width)
        if _lorentzian_width <= 0:
            raise ValueError(
                "The _lorentzian_width of a Voigt must be greater than zero."
            )

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError("unit must be a string or a scipp unit.")

        super().__init__(name=name)

        self._unit = unit  # Set the unit for the component

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
            name=name + " _gaussian_width",
            value=_gaussian_width,
            unit=unit,
            min=MINIMUM_WIDTH,
        )

        self._lorentzian_width = Parameter(
            name=name + " _lorentzian_width",
            value=_lorentzian_width,
            unit=unit,
            min=MINIMUM_WIDTH,
        )

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        """Evaluate the Voigt at the given x values.
        If x is a scipp Variable, the unit of the Voigt will be converted to match x.
        The Voigt evaluates to the convolution of a Gaussian with sigma gaussian_width and a Lorentzian with half width at half max lorentzian_width, centered at center, with area equal to area."""

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self._unit is not None and x.unit != self._unit:
                try:
                    self.convert_unit(x.unit.name)
                except Exception as e:
                    raise UnitError(
                        f"Input x has unit {x.unit}, but Voigt component has unit {self._unit}. Failed to convert Voigt to {x.unit}."
                    ) from e
                warnings.warn(
                    f"Input x has unit {x.unit}, but Voigt component has unit {self._unit}. Converting Voigt to {x.unit}."
                )
        else:
            x_in = x

        if any(np.isnan(x_in)):
            raise ValueError("Input x contains NaN values.")

        if any(np.isinf(x_in)):
            raise ValueError("Input x contains infinite values.")

        return self._area.value * voigt_profile(
            x_in - self._center.value,
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
            _gaussian_width=self._gaussian_width.value,
            _lorentzian_width=self._lorentzian_width.value,
            unit=self._unit,
        )
        model_copy._area.fixed = self._area.fixed
        model_copy._center.fixed = self._center.fixed
        model_copy.__gaussian_width.fixed = self._gaussian_width.fixed
        model_copy.__lorentzian_width.fixed = self._lorentzian_width.fixed

        return model_copy

    def __repr__(self):
        return f"Voigt(name = {self.name}, unit = {self._unit},\n area = {self._area},\n center = {self._center},\n _gaussian_width = {self._gaussian_width},\n _lorentzian_width = {self._lorentzian_width})"
