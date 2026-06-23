# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

if TYPE_CHECKING:
    import scipp as sc
    from easyscience.variable import Parameter


class DampedHarmonicOscillator(CreateParametersMixin, ModelComponent):
    r"""
    Model of a Damped Harmonic Oscillator (DHO).

    The intensity is given by $$ I(x) = \frac{2 A x_0^2 \gamma}{\pi \left( (x^2 - x_0^2)^2 + (2
    \gamma x)^2 \right)}, $$ where $A$ is the area, $x_0$ is the center, and $\gamma$ is the width.

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
        unit: str | sc.Unit = 'meV',
        name: str = 'DampedHarmonicOscillator',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Damped Harmonic Oscillator.

        Parameters
        ----------
        area : Numeric, default=1.0
            Area under the curve.
        center : Numeric, default=1.0
            Resonance frequency, approximately the peak position.
        width : Numeric, default=1.0
            Damping constant, approximately the half width at half max (HWHM) of the peaks. By
            default, 1.0.
        unit : str | sc.Unit, default='meV'
            Unit of the parameters.
        name : str, default='DampedHarmonicOscillator'
            Name of the component for indexing.
        display_name : str | None, default=None
            Display name of the component.
        unique_name : str | None, default=None
            Unique name of the component. If None, a unique_name is automatically generated. By
            default, None.
        """

        super().__init__(
            name=name,
            display_name=display_name,
            unique_name=unique_name,
            unit=unit,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=name, unit=self._unit)
        center = self._create_center_parameter(
            center=center,
            name=name,
            fix_if_none=False,
            unit=self._unit,
            enforce_minimum_center=True,
        )

        width = self._create_width_parameter(width=width, name=name, unit=self._unit)

        self._area = area
        self._center = center
        self._width = width

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
        TypeError
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
    def center(self, value: Numeric) -> None:
        """
        Set the value of the center parameter.

        Parameters
        ----------
        value : Numeric
            The new value for the center parameter.

        Raises
        ------
        TypeError
            If the value is not a number.
        ValueError
            If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')

        if float(value) <= 0:
            raise ValueError('center must be positive')
        self._center.value = value

    @property
    def width(self) -> Parameter:
        """
        Get the width parameter.

        Returns
        -------
        Parameter
            The width parameter.
        """
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """
        Set the value of the width parameter.

        Parameters
        ----------
        value : Numeric
            The new value for the width parameter.

        Raises
        ------
        TypeError
            If the value is not a number.
        ValueError
            If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('width must be a number')

        if float(value) <= 0:
            raise ValueError('width must be positive')

        self._width.value = value

    def evaluate(self, x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray) -> np.ndarray:
        r"""
        Evaluate the Damped Harmonic Oscillator at the given x values.

        If x is a scipp Variable, the unit of the DHO will be converted to match x. The intensity
        is given by $$ I(x) = \frac{2 A x_0^2 \gamma}{\pi \left( (x^2 - x_0^2)^2 + (2 \gamma x)^2
        \right)}, $$ where $A$ is the area, $x_0$ is the center, and $\gamma$ is the width.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the DHO.

        Returns
        -------
        np.ndarray
            The intensity of the DHO at the given x values.
        """

        x = self._prepare_x_for_evaluate(x)

        normalization = 2 * self.center.value**2 * self.width.value / np.pi
        # No division by zero here, width>0 enforced in setter
        denominator = (x**2 - self.center.value**2) ** 2 + (2 * self.width.value * x) ** 2

        return self.area.value * normalization / (denominator)

    def __repr__(self) -> str:
        """
        Return a string representation of the Damped Harmonic Oscillator.

        Returns
        -------
        str
            A string representation of the Damped Harmonic Oscillator.
        """
        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r}, '
            f'unit={self._unit},\n'
            f'    area={self.area},\n'
            f'    center={self.center},\n'
            f'    width={self.width})'
        )
