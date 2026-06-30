# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipp as sc

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

if TYPE_CHECKING:
    from easyscience.variable import Parameter


class DampedHarmonicOscillator(CreateParametersMixin, ModelComponent):
    r"""
    Model of a Damped Harmonic Oscillator (DHO).

    $$ I(x) = \frac{2 A x_0^2 \gamma}{\pi \left( (x^2 - x_0^2)^2 + (2\gamma x)^2 \right)} $$

    where $A$ is the area (``area``), $x_0$ is the center (``center``), and $\gamma$ is the half width at half max (``width``). area has unit = x_unit *
    y_unit; center and width have unit = x_unit.

    Examples
    --------
    **Creating a Damped Harmonic Oscillator**

    The ``center`` parameter is the resonance frequency, which must be positive. Both phonon peaks
    (at ±center) are captured by the model:
    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    dho = sm.DampedHarmonicOscillator(area=1.0, center=10.0, width=1.0)
    x = np.linspace(-20, 20, 200)
    values = dho.evaluate(x)
    ```

    **Modifying parameters after construction**

    ```python
    import easydynamics.sample_model as sm

    dho = sm.DampedHarmonicOscillator(area=2.0, center=5.0, width=0.5, name='Phonon')
    dho.area = 3.0
    dho.center = 8.0
    dho.width = 0.3
    ```
    """

    def __init__(
        self,
        area: Numeric = 1.0,
        center: Numeric = 1.0,
        width: Numeric = 1.0,
        x_unit: str | sc.Unit = "meV",
        y_unit: str | sc.Unit = "dimensionless",
        name: str = "DampedHarmonicOscillator",
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Damped Harmonic Oscillator component.

        Parameters
        ----------
        area : Numeric, default=1.0
            Integrated area under the DHO profile.  Unit is ``x_unit * y_unit``.
        center : Numeric, default=1.0
            Resonance frequency (x_0) in x_unit; approximately the peak position.  Must be strictly positive; a minimum of
            ``DHO_MINIMUM_CENTER`` (1e-10) is enforced.
        width : Numeric, default=1.0
            Damping coefficient (gamma) in x_unit.  Must be strictly positive. Approximately equal
            to the HWHM of each peak.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  center and width are stored in this unit. area_unit = x_unit *
            y_unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='DampedHarmonicOscillator'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier.  Auto-generated if None.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            unique_name=unique_name,
            x_unit=x_unit,
            y_unit=y_unit,
        )

        # These methods live in CreateParametersMixin
        self._area = self._create_area_parameter(
            area=area, name=name, x_unit=self._x_unit, y_unit=self._y_unit
        )
        self._center = self._create_center_parameter(
            center=center,
            name=name,
            fix_if_none=False,
            x_unit=self._x_unit,
            enforce_minimum_center=True,
        )
        self._width = self._create_width_parameter(
            width=width, name=name, x_unit=self._x_unit
        )

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns
        -------
        Parameter
            The area Parameter with unit ``x_unit * y_unit``.
        """
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New area value (in current area unit = x_unit * y_unit).

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        """
        if not isinstance(value, Numeric):
            raise TypeError("area must be a number")
        self._area.value = value

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter (resonance frequency).

        Returns
        -------
        Parameter
            The resonance frequency (x_0) Parameter with unit ``x_unit``.
        """
        return self._center

    @center.setter
    def center(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New resonance frequency in x_unit.  Must be strictly positive.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError("center must be a number")
        if float(value) <= 0:
            raise ValueError("center must be positive")
        self._center.value = value

    @property
    def width(self) -> Parameter:
        """
        Get the width parameter (damping coefficient).

        Returns
        -------
        Parameter
            The damping coefficient (gamma) Parameter with unit ``x_unit``.
        """
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New damping coefficient in x_unit.  Must be strictly positive.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError("width must be a number")
        if float(value) <= 0:
            raise ValueError("width must be positive")
        self._width.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = "numpy",
    ) -> np.ndarray | sc.Variable:
        r"""
        Evaluate the DHO at x.

        $$ I(x) = \frac{2 A x_0^2 \gamma}{\pi \left( (x^2 - x_0^2)^2 + (2\gamma x)^2 \right)} $$

        where *A* is ``area``, *x*₀ is ``center`` (resonance frequency), and *gamma* is ``width``
        (damping coefficient). Here *I* is the scattered intensity. Parameters in the model's own
        units are temporarily converted to x's unit for the computation.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Input x values.
        output : str, default='numpy'
            'numpy' returns np.ndarray; 'scipp' returns sc.Variable with y_unit.

        Returns
        -------
        np.ndarray | sc.Variable
            Evaluated DHO values at x.
        """
        x_vals, detected_unit, dim = self._prepare_x_for_evaluate(x)
        eval_unit = detected_unit or self._x_unit
        eval_area_unit = str(sc.Unit(eval_unit) * sc.Unit(self._y_unit))

        center = self._resolve_param_value(self._center, eval_unit)
        width = self._resolve_param_value(self._width, eval_unit)
        area = self._resolve_param_value(self._area, eval_area_unit)

        normalization = 2 * center**2 * width / np.pi
        denominator = (x_vals**2 - center**2) ** 2 + (2 * width * x_vals) ** 2
        # denominator cannot reach zero: center > 0 enforced by DHO_MINIMUM_CENTER
        result = area * normalization / denominator

        if output == "scipp":
            return sc.array(dims=[dim], values=result, unit=self._y_unit)
        return result

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert x-axis parameters (center, width) and area to new_x_unit.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.
        """
        self._convert_x_unit_area_based(
            new_x_unit=new_x_unit,
            x_params=[self._center, self._width],
            area_param=self._area,
        )

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit by rescaling the area parameter.

        The area is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(new_y_unit=new_y_unit, area_param=self._area)

    def __repr__(self) -> str:
        """
        Return a string representation of the Damped Harmonic Oscillator.

        Returns
        -------
        str
            A string representation of the Damped Harmonic Oscillator.
        """
        return (
            f"{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, "
            f"x_unit = {self._x_unit}, y_unit = {self._y_unit},\n "
            f"    area = {self.area},\n "
            f"    center = {self.center},\n "
            f"    width = {self.width})"
        )
