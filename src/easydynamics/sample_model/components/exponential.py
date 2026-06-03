# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric


class Exponential(CreateParametersMixin, ModelComponent):
    r"""
    Model of an exponential function.

    The intensity is given by

    $$ I(x) = A e^{B (x-x_0)}, $$

    where $A$ is the amplitude, $x_0$ is the center, and $B$ describes the rate of decay or growth.
    """

    def __init__(
        self,
        amplitude: Numeric = 1.0,
        center: Numeric | None = None,
        rate: Numeric = 1.0,
        unit: str | sc.Unit = 'meV',
        name: str = 'Exponential',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Exponential component.

        Parameters
        ----------
        amplitude : Numeric, default=1.0
            Amplitude of the Exponential.
        center : Numeric | None, default=None
            Center of the Exponential. If None, the center is fixed at 0.
        rate : Numeric, default=1.0
            Decay or growth constant of the Exponential.
        unit : str | sc.Unit, default='meV'
            Unit of the parameters.
        name : str, default='Exponential'
            Name of the component for indexing.
        display_name : str | None, default=None
            Display name of the component.
        unique_name : str | None, default=None
            Unique name of the component. If None, a unique_name is automatically generated. By
            default, None.

        Raises
        ------
        TypeError
            If amplitude, center, or rate are not numbers or Parameters.
        ValueError
            If amplitude, center or rate are not finite numbers.
        """
        # Validate inputs and create Parameters if not given
        super().__init__(
            unit=unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        if not isinstance(amplitude, (Parameter, Numeric)):
            raise TypeError('amplitude must be a number or a Parameter.')

        if isinstance(amplitude, Numeric):
            if not np.isfinite(amplitude):
                raise ValueError('amplitude must be a finite number or a Parameter')

            amplitude = Parameter(name=name + ' amplitude', value=float(amplitude), unit=unit)

        center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, unit=self._unit
        )

        if not isinstance(rate, (Parameter, Numeric)):
            raise TypeError('rate must be a number or a Parameter.')

        if isinstance(rate, Numeric):
            if not np.isfinite(rate):
                raise ValueError('rate must be a finite number or a Parameter')

            rate = Parameter(name=name + ' rate', value=float(rate), unit='1/' + str(unit))

        self._amplitude = amplitude
        self._center = center
        self._rate = rate

    @property
    def amplitude(self) -> Parameter:
        """
        Get the amplitude parameter.

        Returns
        -------
        Parameter
            The amplitude parameter.
        """

        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: Numeric) -> None:
        """
        Set the value of the amplitude parameter.

        Parameters
        ----------
        value : Numeric
            The new value for the amplitude parameter.

        Raises
        ------
        TypeError
            If the value is not a number.
        """

        if not isinstance(value, Numeric):
            raise TypeError('amplitude must be a number')
        self._amplitude.value = value

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
        Set the center parameter value.

        Parameters
        ----------
        value : Numeric | None
            The new value for the center parameter.

        Raises
        ------
        TypeError
            If the value is not a number.
        """

        if value is None:
            value = 0.0
            self._center.fixed = True

        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    @property
    def rate(self) -> Parameter:
        """
        Get the rate parameter.

        Returns
        -------
        Parameter
            The rate parameter.
        """
        return self._rate

    @rate.setter
    def rate(self, value: Numeric) -> None:
        """
        Set the rate parameter value.

        Parameters
        ----------
        value : Numeric
            The new value for the rate parameter.

        Raises
        ------
        TypeError
            If the value is not a number.
        """
        if not isinstance(value, Numeric):
            raise TypeError('rate must be a number')

        self._rate.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
    ) -> np.ndarray:
        r"""
        Evaluate the Exponential at the given x values.

        If x is a scipp Variable, the unit of the Exponential will be converted to match x. The
        intensity is given by $$ I(x) = A \exp\left( r (x - x_0) \right) $$

        where $A$ is the amplitude, $x_0$ is the center, and $r$ is the rate.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the Exponential.

        Returns
        -------
        np.ndarray
            The intensity of the Exponential at the given x values.
        """

        x = self._prepare_x_for_evaluate(x)
        exponent = self.rate.value * (x - self.center.value)

        return self.amplitude.value * np.exp(exponent)

    def convert_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the unit of the Parameters in the component.

        Parameters
        ----------
        unit : str | sc.Unit
            The new unit to convert to.

        Raises
        ------
        TypeError
            If unit is not a string or sc.Unit.
        Exception
            If conversion fails for any parameter.
        """

        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError('unit must be a string or sc.Unit')

        old_unit = self._unit
        pars = [self.amplitude, self.center]
        try:
            for p in pars:
                p.convert_unit(unit)
            self.rate.convert_unit('1/' + str(unit))
            self._unit = unit
        except Exception as e:
            # Attempt to rollback on failure
            try:
                for p in pars:
                    p.convert_unit(old_unit)
                self.rate.convert_unit('1/' + str(old_unit))
            except Exception:  # noqa: S110
                pass  # Best effort rollback
            raise e

    def __repr__(self) -> str:
        """
        Return a string representation of the Exponential.

        Returns
        -------
        str
            A string representation of the Exponential.
        """

        return (
            f'Exponential(name = {self.name}, display_name = {self.display_name}, '
            f'unit = {self._unit},\n '
            f'    amplitude = {self.amplitude},\n '
            f'    center = {self.center},\n '
            f'    rate = {self.rate})'
        )
