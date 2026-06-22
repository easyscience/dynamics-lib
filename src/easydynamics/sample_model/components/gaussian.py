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


class Gaussian(CreateParametersMixin, ModelComponent):
    r"""
    Model of a Gaussian function.

    The intensity is given by

    $$ I(x) = \frac{A}{\sigma \sqrt{2\pi}} \exp\left( -\frac{1}{2} \left(\frac{x -
    x_0}{\sigma}\right)^2 \right) $$

    where $A$ is the area, $x_0$ is the center, and $\sigma$ is the width.

    If the center is not provided, it will be centered at 0 and fixed, which is typically what you
    want in QENS.

    Examples
    --------
    **Creating a Gaussian with a fixed center (typical QENS use)**

    By default the center is fixed at 0, which is the typical setup for a QENS elastic line:
    ```python
    import numpy as np
    from easydynamics.sample_model.components import Gaussian

    g = Gaussian(area=1.0, width=0.5)
    x = np.linspace(-2, 2, 100)
    values = g.evaluate(x)
    ```

    **Creating a Gaussian with a free center and modifying parameters**

    Pass a numeric value for ``center`` to leave it free during fitting, and use the property
    setters to update parameter values after construction:
    ```python
    from easydynamics.sample_model.components import Gaussian

    g = Gaussian(area=2.0, center=0.5, width=0.3, name='Peak')
    g.area = 3.0
    g.width = 0.2
    ```
    """

    def __init__(
        self,
        area: Numeric = 1.0,
        center: Numeric | None = None,
        width: Numeric = 1.0,
        unit: str | sc.Unit = 'meV',
        name: str = 'Gaussian',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Gaussian component.

        Parameters
        ----------
        area : Numeric, default=1.0
            Area of the Gaussian.
        center : Numeric | None, default=None
            Center of the Gaussian. If None, defaults to 0 and is fixed.
        width : Numeric, default=1.0
            Standard deviation.
        unit : str | sc.Unit, default='meV'
            Unit of the parameters.
        name : str, default='Gaussian'
            Name of the component for indexing.
        display_name : str | None, default=None
            Name of the component.
        unique_name : str | None, default=None
            Unique name of the component. if None, a unique_name is automatically generated. By
            default, None.
        """
        # Validate inputs and create Parameters if not given
        super().__init__(
            unit=unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        # These methods live in ValidationMixin
        area = self._create_area_parameter(area=area, name=name, unit=self._unit)
        center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, unit=self._unit
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
    def center(self, value: Numeric | None) -> None:
        """
        Set the center parameter value.

        Parameters
        ----------
        value : Numeric | None
            The new value for the center parameter. If None, defaults to 0 and is fixed.

        Raises
        ------
        TypeError
            If the value is not a number or None.
        """

        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    @property
    def width(self) -> Parameter:
        """
        Get the width parameter (standard deviation).

        Returns
        -------
        Parameter
            The width parameter.
        """
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """
        Set the width parameter value.

        Parameters
        ----------
        value : Numeric
            The new value for the width parameter.

        Raises
        ------
        TypeError
            If the value is not a number or None.
        ValueError
            If the value is not positive.
        """
        if not isinstance(value, Numeric):
            raise TypeError('width must be a number')

        if float(value) <= 0:
            raise ValueError('width must be positive')

        self._width.value = value

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
    ) -> np.ndarray:
        r"""
        Evaluate the Gaussian at the given x values.

        If x is a scipp Variable, the unit of the Gaussian will be converted to match x. The
        intensity is given by $$ I(x) = \frac{A}{\sigma \sqrt{2\pi}} \exp\left( -\frac{1}{2}
        \left(\frac{x - x_0}{\sigma}\right)^2 \right) $$

        where $A$ is the area, $x_0$ is the center, and $\sigma$ is the width.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            The x values at which to evaluate the Gaussian.

        Returns
        -------
        np.ndarray
            The intensity of the Gaussian at the given x values.
        """

        x = self._prepare_x_for_evaluate(x)

        normalization = 1 / (np.sqrt(2 * np.pi) * self.width.value)
        exponent = -0.5 * ((x - self.center.value) / self.width.value) ** 2

        return self.area.value * normalization * np.exp(exponent)

    def __repr__(self) -> str:
        """
        Return a string representation of the Gaussian.

        Returns
        -------
        str
            A string representation of the Gaussian.
        """

        return (
            f'{self.__class__.__name__}('
            f'name={self.name!r}, display_name={self.display_name!r}, '
            f'unit={self._unit},\n'
            f'    area={self.area},\n'
            f'    center={self.center},\n'
            f'    width={self.width})'
        )
