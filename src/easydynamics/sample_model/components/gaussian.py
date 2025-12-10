from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin

from .model_component import ModelComponent

Numeric = Union[float, int]


class Gaussian(CreateParametersMixin, ModelComponent):
    """
    Gaussian function: area/(width*sqrt(2pi)) * exp(-0.5*((x - center)/width)^2)
    If the center is not provided, it will be centered at 0 and fixed, which is typically what you want in QENS.

    Args:
        name (str): Name of the component.
        area (Int, float or Parameter): Area of the Gaussian.
        center (Int, float, None or Parameter): Center of the Gaussian. If None, defaults to 0 and is fixed
        width (Int, float or Parameter): Standard deviation.
        unit (str or sc.Unit): Unit of the parameters. Defaults to "meV".
    """

    def __init__(
        self,
        display_name: Optional[str] = "Gaussian",
        area: Optional[Union[Numeric, Parameter]] = 1.0,
        center: Optional[Union[Numeric, Parameter, None]] = None,
        width: Optional[Union[Numeric, Parameter]] = 1.0,
        unit: Optional[Union[str, sc.Unit]] = "meV",
    ):
        # Validate inputs and create Parameters if not given
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
        width = self._create_width_parameter(
            width=width, name=display_name, unit=self._unit
        )

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """Evaluate the Gaussian at the given x values.
        If x is a scipp Variable, the unit of the Gaussian will be converted to match x.
        The Gaussian evaluates to area/(width*sqrt(2pi)) * exp(-0.5*((x - center)/width)^2)"""

        x = self._prepare_x_for_evaluate(x)

        normalization = 1 / (np.sqrt(2 * np.pi) * self.width.value)
        exponent = -0.5 * ((x - self.center.value) / self.width.value) ** 2

        return self.area.value * normalization * np.exp(exponent)

    def __repr__(self):
        return f"Gaussian(name = {self.name}, unit = {self._unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"
