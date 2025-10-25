from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.special import voigt_profile

from easydynamics.sample_model.components.mixins import CreateParametersMixin

from .model_component import ModelComponent

Numeric = Union[float, int]


class Voigt(CreateParametersMixin, ModelComponent):
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
        name: Optional[str] = "Voigt",
        area: Optional[Union[Numeric, Parameter]] = 1.0,
        center: Optional[Union[Numeric, Parameter, None]] = None,
        gaussian_width: Optional[Union[Numeric, Parameter]] = 1.0,
        lorentzian_width: Optional[Union[Numeric, Parameter]] = 1.0,
        unit: Optional[Union[str, sc.Unit]] = "meV",
    ):
        # Validate inputs and create Parameters if not given
        self.validate_unit(unit)
        self._unit = unit

        area = self._create_area_parameter(area, name)
        center = self._create_center_parameter(center, name, fix_if_none=True)
        gaussian_width = self._create_width_parameter(
            gaussian_width, name, param_name="gaussian_width"
        )
        lorentzian_width = self._create_width_parameter(
            lorentzian_width, name, param_name="lorentzian_width"
        )

        super().__init__(
            name=name,
            unit=unit,
            area=area,
            center=center,
            gaussian_width=gaussian_width,
            lorentzian_width=lorentzian_width,
        )

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
        return f"Voigt(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n gaussian_width = {self.gaussian_width},\n lorentzian_width = {self.lorentzian_width})"
