# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter
from scipy.special import voigt_profile

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.utils.utils import Numeric

from .model_component import ModelComponent


class Voigt(CreateParametersMixin, ModelComponent):
    r"""
    Voigt profile, a convolution of Gaussian and Lorentzian.

    If the center is not provided, it will be centered at 0 and fixed, which is typically what you
    want in QENS.

    Use scipy.special.voigt_profile to evaluate the Voigt profile.
    """

    def __init__(
        self,
        area: Numeric | Parameter = 1.0,
        center: Numeric | Parameter | None = None,
        gaussian_width: Numeric | Parameter = 1.0,
        lorentzian_width: Numeric | Parameter = 1.0,
        unit: str | sc.Unit = 'meV',
        display_name: str | None = 'Voigt',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a Voigt component.

        Parameters
        ----------
        area : Numeric | Parameter, default=1.0
            Total area under the curve. By default, 1.0.
        center : Numeric | Parameter | None, default=None
            Center of the Voigt profile. By default, None.
        gaussian_width : Numeric | Parameter, default=1.0
            Standard deviation of the Gaussian part. By default, 1.0.
        lorentzian_width : Numeric | Parameter, default=1.0
            Half width at half max (HWHM) of the Lorentzian part. By default, 1.0.
        unit : str | sc.Unit, default='meV'
            Unit of the parameters. By default, 'meV'.
        display_name : str | None, default='Voigt'
            Display name of the component. By default, 'Voigt'.
        unique_name : str | None, default=None
            Unique name of the component. If None, a unique_name is automatically generated. By
            default, None.
        """

        super().__init__(
            display_name=display_name,
            unit=unit,
            unique_name=unique_name,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=display_name, unit=self._unit)
        center = self._create_center_parameter(
            center=center, name=display_name, fix_if_none=True, unit=self._unit
        )
        gaussian_width = self._create_width_parameter(
            width=gaussian_width,
            name=display_name,
            param_name='gaussian_width',
            unit=self._unit,
        )
        lorentzian_width = self._create_width_parameter(
            width=lorentzian_width,
            name=display_name,
            param_name='lorentzian_width',
            unit=self._unit,
        )

        self._area = area
        self._center = center
        self._gaussian_width = gaussian_width
        self._lorentzian_width = lorentzian_width

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns
        -------
        Parameter
            The area parameter.
        """
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Set the value of the area parameter.

        Parameters
        ----------
        value : Numeric
            The new value for the area parameter.

        Raises
        ------
        TypeError :
            If the value is not a number.
        """
        if not isinstance(value, Numeric):
            raise TypeError('area must be a number')
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter.

        Returns
        -------
        Parameter
            The center parameter.
        """
        return self._center

    @center.setter
    def center(self, value: Numeric | None) -> None:
        """
        Set the value of the center parameter.

        Parameters
        ----------
        value : Numeric | None
            The new value for the center parameter. If None. By default, 0 and is fixed.

        Raises
        ------
        TypeError :
            If the value is not a number.
        """
        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    @property
    def gaussian_width(self) -> Parameter:
        """
        Get the Gaussian width parameter.

        Returns
        -------
        Parameter
            The Gaussian width parameter.
        """
        return self._gaussian_width

    @gaussian_width.setter
    def gaussian_width(self, value: Numeric) -> None:
        """
        Set the width parameter value.

        Parameters
        ----------
        value : Numeric
            The new value for the width parameter.

        Raises
        ------
        TypeError :
            If the value is not a number.
        ValueError :
            If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('gaussian_width must be a number')
        if float(value) <= 0:
            raise ValueError('gaussian_width must be positive')
        self._gaussian_width.value = value

    @property
    def lorentzian_width(self) -> Parameter:
        """
        Get the Lorentzian width parameter (HWHM).

        Returns
        -------
        Parameter
            The Lorentzian width parameter.
        """
        return self._lorentzian_width

    @lorentzian_width.setter
    def lorentzian_width(self, value: Numeric) -> None:
        """
        Set the value of the Lorentzian width parameter.

        Parameters
        ----------
        value : Numeric
            The new value for the Lorentzian width parameter.

        Raises
        ------
        TypeError :
            If the value is not a number.
        ValueError :
            If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('lorentzian_width must be a number')
        if float(value) <= 0:
            raise ValueError('lorentzian_width must be positive')
        self._lorentzian_width.value = value

    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        r"""
        Evaluate the Voigt at the given x values.

        If x is a scipp Variable, the unit of the Voigt will be converted to match x. The Voigt
        evaluates to the convolution of a Gaussian with sigma gaussian_width and a Lorentzian with
        half width at half max lorentzian_width, centered at center, with area equal to area.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the Voigt.

        Returns
        -------
        np.ndarray
            The intensity of the Voigt at the given x values.
        """

        x = self._prepare_x_for_evaluate(x)

        return self.area.value * voigt_profile(
            x - self.center.value,
            self.gaussian_width.value,
            self.lorentzian_width.value,
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the Voigt.

        Returns
        -------
        str
            A string representation of the Voigt.
        """

        return f'Voigt(unique_name = {self.unique_name}, unit = {self._unit},\n \
        area = {self.area},\n \
        center = {self.center},\n \
        gaussian_width = {self.gaussian_width},\n \
        lorentzian_width = {self.lorentzian_width})'
