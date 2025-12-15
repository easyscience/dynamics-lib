from __future__ import annotations

from typing import Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.special import voigt_profile

from easydynamics.sample_model.components.mixins import CreateParametersMixin

from .model_component import ModelComponent

Numeric = float | int


class Voigt(CreateParametersMixin, ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.
    If the center is not provided, it will be centered at 0 and fixed, which is typically what you want in QENS.

    Args:
        display_name (str): Name of the component.
        center (Int or float or None): Center of the Voigt profile.
        gaussian_width (Int or float): Standard deviation of the Gaussian part.
        lorentzian_width (Int or float): Half width at half max (HWHM) of the Lorentzian part.
        area (Int or float): Total area under the curve.
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
    """

    def __init__(
        self,
        display_name: str = "Voigt",
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter | None = None,
        gaussian_width: Numeric | Parameter = 1.0,
        lorentzian_width: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = "meV",
    ):
        super().__init__(
            display_name=display_name,
            unit=unit,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(
            area=area, name=display_name, unit=self._unit
        )
        center = self._create_center_parameter(
            center=center, name=display_name, fix_if_none=True, unit=self._unit
        )
        gaussian_width = self._create_width_parameter(
            width=gaussian_width,
            name=display_name,
            param_name="gaussian_width",
            unit=self._unit,
        )
        lorentzian_width = self._create_width_parameter(
            width=lorentzian_width,
            name=display_name,
            param_name="lorentzian_width",
            unit=self._unit,
        )

        self._area = area
        self._center = center
        self._gaussian_width = gaussian_width
        self._lorentzian_width = lorentzian_width

    @property
    def area(self) -> Parameter:
        """Get the area parameter."""
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """Set the area parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError("area must be a number")
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """Get the center parameter."""
        return self._center

    @center.setter
    def center(self, value: Numeric) -> None:
        """Set the center parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number")
        self._center.value = value

    @property
    def gaussian_width(self) -> Parameter:
        """Get the width parameter."""
        return self._gaussian_width

    @gaussian_width.setter
    def gaussian_width(self, value: Numeric) -> None:
        """Set the width parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError("gaussian_width must be a number")
        self._gaussian_width.value = value

    @property
    def lorentzian_width(self) -> Parameter:
        """Get the width parameter."""
        return self._lorentzian_width

    @lorentzian_width.setter
    def lorentzian_width(self, value: Numeric) -> None:
        """Set the width parameter value."""
        if not isinstance(value, Numeric):
            raise TypeError("lorentzian_width must be a number")
        self._lorentzian_width.value = value

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
        return f"Voigt(display_name = {self.display_name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n gaussian_width = {self.gaussian_width},\n lorentzian_width = {self.lorentzian_width})"
