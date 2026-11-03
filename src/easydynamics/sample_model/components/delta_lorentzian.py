# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from easyscience.variable import DescriptorBase
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

# Absolute tolerance for deciding whether the center falls inside the x range, as in
# DeltaFunction. It only absorbs floating-point noise at the grid edges.
EPSILON = 1e-8

if TYPE_CHECKING:
    from collections.abc import Sequence

    import scipp as sc


class DeltaLorentzian(CreateParametersMixin, ModelComponent):
    r"""
    Model of one delta function and n Lorentzians sharing a center.

    $$ I(x) = A \left[ w_0\, \delta(x - x_0) + \sum_{j=1}^{n} w_j\, \frac{1}{\pi}
    \frac{\Gamma_j}{(x - x_0)^2 + \Gamma_j^2} \right] $$

    where $A$ is the total area, $x_0$ is the center, $w_0$ is the weight of the delta function,
    and $w_j$ and $\Gamma_j$ are the weight and half width at half max (HWHM) of the Lorentzian of
    order $j$. The weights are relative: the delta function has area $A w_0$ and the Lorentzian of
    order $j$ has area $A w_j$. They need not sum to one, but they do when $A$ is the total
    scattering of a diffusion model.

    area has unit = x_unit * y_unit; center and widths have unit = x_unit; weights are
    dimensionless. If the center is not provided, it will be centered at 0 and fixed, which is
    typically what you want in QENS. The number of Lorentzians is set at construction.

    This is the component to use at one Q-value for models made of a delta function and several
    Lorentzians. It evaluates as a ``DeltaFunction`` plus ``Lorentzian`` components would, and the
    delta function is handled as in ``DeltaFunction``. It should not be confused with the diffusion
    model ``DeltaLorentz``, which has one Lorentzian and handles a range of Q-values.

    Examples
    --------
    **Creating a delta function and two Lorentzians**

    The weights are ordered ``[w0, w1, ..., wn]`` with the delta function first; there is one width
    per Lorentzian:
    ```python
    import numpy as np
    import easydynamics as edyn

    component = edyn.DeltaLorentzian(
        area=2.0,
        weights=[0.5, 0.3, 0.2],
        widths=[0.1, 0.4],
    )
    x = np.linspace(-2, 2, 101)
    values = component.evaluate(x)
    ```

    **Letting other parameters drive the component**

    Weights and widths may be ``Parameter`` objects, for example dependent parameters. They are
    used as they are, so the component follows them:
    ```python
    from easyscience.variable import Parameter
    import easydynamics as edyn

    tau = Parameter(name='tau', value=0.5, unit='meV', min=1e-10)
    width = Parameter.from_dependency(
        name='width', dependency_expression='2 * tau', dependency_map={'tau': tau}
    )
    component = edyn.DeltaLorentzian(weights=[0.0, 1.0], widths=[width])
    tau.value = 1.0  # the width of the Lorentzian is now 2.0
    ```
    """

    def __init__(
        self,
        area: Numeric = 1.0,
        center: Numeric | None = None,
        weights: Sequence[Numeric | Parameter] = (0.5, 0.5),
        widths: Sequence[Numeric | Parameter] = (1.0,),
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'DeltaLorentzian',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the DeltaLorentzian component.

        Parameters
        ----------
        area : Numeric, default=1.0
            Total area of the component. Unit is ``x_unit * y_unit``.
        center : Numeric | None, default=None
            Position of the delta function and the Lorentzians in x_unit. If None, defaults to 0
            and the center parameter is fixed.
        weights : Sequence[Numeric | Parameter], default=(0.5, 0.5)
            Relative weights ``[w0, w1, ..., wn]``: the delta function first, then one per
            Lorentzian. Must have one element more than ``widths`` and be non-negative. Numbers
            become dimensionless Parameters; Parameter instances are used as they are.
        widths : Sequence[Numeric | Parameter], default=(1.0,)
            Half widths at half maximum ``[G1, ..., Gn]`` of the Lorentzians in x_unit. Must
            contain at least one element, each strictly positive. Numbers become Parameters;
            Parameter instances are used as they are.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis. center and widths are stored in this unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='DeltaLorentzian'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting. Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier. Auto-generated if None.

        Raises
        ------
        ValueError
            If widths is empty, if weights does not have one element more than widths, if a weight
            is negative or not finite, or if a width is not strictly positive.

        Notes
        -----
        A ``TypeError`` propagates from the validation if area or center is not a number, if
        weights or widths is not a list, tuple or ndarray, or if an element of either is neither a
        number nor a Parameter.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        self._validate_sequence(weights, 'weights')
        self._validate_sequence(widths, 'widths')
        if len(widths) == 0:
            raise ValueError('At least one width must be provided.')
        if len(weights) != len(widths) + 1:
            raise ValueError(
                'weights must have one more element than widths: one weight for the delta '
                'function and one for each Lorentzian.'
            )

        self._area = self._create_area_parameter(
            area=area, name=name, x_unit=self.x_unit, y_unit=self.y_unit
        )
        self._center = self._create_center_parameter(
            center=center, name=name, fix_if_none=True, x_unit=self.x_unit
        )
        self._weights = [self._create_weight_parameter(w, name, i) for i, w in enumerate(weights)]
        self._widths = [
            w
            if isinstance(w, Parameter)
            else self._create_width_parameter(
                width=w, name=name, param_name=f'width_{j}', x_unit=self.x_unit
            )
            for j, w in enumerate(widths, start=1)
        ]

    @property
    def number_of_lorentzians(self) -> int:
        """
        Get the number of Lorentzians.

        Returns
        -------
        int
            The number n of Lorentzians. Set at construction.
        """
        return len(self._widths)

    @property
    def area(self) -> Parameter:
        """
        Get the area parameter.

        Returns
        -------
        Parameter
            The total area Parameter with unit ``x_unit * y_unit``.
        """
        return self._area

    @area.setter
    def area(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New area value (in current area unit = x_unit * y_unit).

        Notes
        -----
        A ``TypeError`` propagates from the shared value setter if *value* is not a numeric type,
        and a ``ValueError`` propagates from it if *value* violates the area parameter's bounds.
        """
        self._set_bounded_parameter_value(self._area, value, 'area')

    @property
    def center(self) -> Parameter:
        """
        Get the center parameter.

        Returns
        -------
        Parameter
            The center Parameter with unit ``x_unit``.
        """
        return self._center

    @center.setter
    def center(self, value: Numeric | None) -> None:
        """
        Parameters
        ----------
        value : Numeric | None
            New center value in x_unit. If None, the center is set to 0 and the parameter is fixed.

        Raises
        ------
        TypeError
            If *value* is not None and not a numeric type.
        """
        if value is None:
            value = 0.0
            self._center.fixed = True
        if not isinstance(value, Numeric):
            raise TypeError('center must be a number')
        self._center.value = value

    @property
    def weights(self) -> list[Parameter]:
        """
        Get the weights of the delta function and the Lorentzians.

        Returns
        -------
        list[Parameter]
            A shallow copy of the weights ``[w0, w1, ..., wn]``, the delta function first. The
            Parameters are the model's own; modifying the list does not affect the component.
        """
        return list(self._weights)

    @property
    def widths(self) -> list[Parameter]:
        """
        Get the half widths at half maximum of the Lorentzians.

        Returns
        -------
        list[Parameter]
            A shallow copy of the widths ``[G1, ..., Gn]`` with unit ``x_unit``. The Parameters are
            the model's own; modifying the list does not affect the component.
        """
        return list(self._widths)

    def get_all_variables(self) -> list[DescriptorBase]:
        """
        Get all variables of the component.

        The weights and widths are held in lists, which EasyScience does not look into, so they are
        listed here.

        Returns
        -------
        list[DescriptorBase]
            The area, center, weights and widths.
        """
        return [self._area, self._center, *self._weights, *self._widths]

    def _evaluate_values(self, x_vals: np.ndarray, eval_unit: str | None) -> np.ndarray:
        """
        Evaluate the delta function and the Lorentzians at x_vals.

        Parameters in the model's own units are temporarily converted to eval_unit for the
        computation.

        Parameters
        ----------
        x_vals : np.ndarray
            Raw x values expressed in eval_unit.
        eval_unit : str | None
            The unit of x_vals.

        Returns
        -------
        np.ndarray
            The sum of the delta function and the Lorentzians at x_vals.

        Raises
        ------
        ValueError
            If x_vals contains a single point, which defines no bin width for the delta function.
        """
        center = self._resolve_param_value(self._center, eval_unit)
        area = self._resolve_param_value(self._area, self._eval_area_unit(eval_unit))
        weights = np.array([weight.value for weight in self._weights])
        widths = np.array([self._resolve_param_value(width, eval_unit) for width in self._widths])

        if x_vals.size == 1:
            raise ValueError(
                'A DeltaLorentzian cannot be evaluated at a single x value: the height of its '
                'delta function is area / bin_width, and a single point defines no bin width. '
                'Evaluate on a grid of at least two x values.'
            )

        # Lorentzians: broadcast the order against x, dims (order, x), then sum over the orders
        normalization = widths[:, np.newaxis] / np.pi
        denominator = (x_vals[np.newaxis, :] - center) ** 2 + widths[:, np.newaxis] ** 2
        model = (area * weights[1:, np.newaxis] * normalization / denominator).sum(axis=0)

        # Delta function, as in DeltaFunction: a single spike of height area / bin_width in the
        # bin nearest the center, if the center is within the x range
        if x_vals.min() - EPSILON <= center <= x_vals.max() + EPSILON:
            # Bin widths only make sense on a sorted grid; compute there and map the spike back
            # to the original position of the nearest x value.
            order = np.argsort(x_vals)
            x_sorted = x_vals[order]
            i = np.argmin(np.abs(x_sorted - center))
            left = x_sorted[i] - x_sorted[i - 1] if i > 0 else x_sorted[1] - x_sorted[0]
            if i == x_sorted.size - 1:
                right = x_sorted[-1] - x_sorted[-2]
            else:
                right = x_sorted[i + 1] - x_sorted[i]
            bin_width = 0.5 * (left + right)
            model[order[i]] += area * weights[0] / bin_width

        return model

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert x-axis parameters (center, widths) and area to new_x_unit.

        The weights are dimensionless and are not changed.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit. Must be dimensionally compatible with the current x_unit.
        """
        self._convert_x_unit_area_based(
            new_x_unit=new_x_unit,
            x_params=[self._center, *self._widths],
            area_param=self._area,
        )

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis (output) unit by rescaling the area parameter.

        The area is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(new_y_unit=new_y_unit, area_param=self._area)

    @staticmethod
    def _validate_sequence(values: object, label: str) -> None:
        """
        Validate that values is a sequence of numbers or Parameters.

        Parameters
        ----------
        values : object
            The object to validate.
        label : str
            Name of the argument used in error messages.

        Raises
        ------
        TypeError
            If values is not a list, tuple or ndarray, or an element is neither a number nor a
            Parameter.
        """
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError(
                f'{label} must be a sequence (list/tuple/ndarray) of numbers or Parameters.'
            )
        if not all(isinstance(v, (Numeric, Parameter)) for v in values):
            raise TypeError(f'{label} must contain numbers or Parameters.')

    @staticmethod
    def _create_weight_parameter(weight: Numeric | Parameter, name: str, order: int) -> Parameter:
        """
        Create the dimensionless Parameter of a weight, or use the given Parameter as it is.

        Parameters
        ----------
        weight : Numeric | Parameter
            The weight value, or an existing Parameter.
        name : str
            Name of the component, used to label the Parameter.
        order : int
            Order of the weight; 0 for the delta function.

        Returns
        -------
        Parameter
            The weight Parameter, bounded below by zero if it was created here.

        Raises
        ------
        ValueError
            If a numeric weight is negative or not finite.
        """
        if isinstance(weight, Parameter):
            return weight
        if not np.isfinite(weight):
            raise ValueError('weights must be finite numbers.')
        if weight < 0:
            raise ValueError('weights must be non-negative.')
        return Parameter(
            name=f'{name} weight_{order}',
            value=float(weight),
            unit='dimensionless',
            min=0.0,
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the DeltaLorentzian.

        Returns
        -------
        str
            A string representation of the DeltaLorentzian.
        """
        return (
            f'{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, '
            f'x_unit = {self.x_unit}, y_unit = {self.y_unit},\n'
            f'    area = {self.area},\n'
            f'    center = {self.center},\n'
            f'    weights = {self.weights},\n'
            f'    widths = {self.widths})'
        )
