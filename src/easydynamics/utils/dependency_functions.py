# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Functions for use in EasyScience dependency expressions.

Temporary: EasyScience cannot apply functions such as ``exp`` to a Parameter in a dependency
expression yet. Delete this module, and its use in ``ModelBase``, once it can.
"""

from collections.abc import Callable

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber as EasyScienceDescriptorNumber

from easydynamics.utils.utils import Numeric
from easydynamics.variable import DescriptorNumber


class DependencyFunction(DescriptorNumber):
    """
    A function that can be called on a Parameter in an EasyScience dependency expression.

    ``Parameter.make_dependent_on`` only accepts DescriptorNumbers in its dependency map, and
    resolves the names in the expression from that map. A callable DescriptorNumber mapped to e.g.
    ``'exp'`` is therefore what the expression calls, instead of the numpy function, which cannot
    handle a Parameter.
    """

    def __init__(
        self,
        name: str,
        function: Callable[[float], float],
        derivative: Callable[[float], float],
    ) -> None:
        """
        Initialize the DependencyFunction.

        Parameters
        ----------
        name : str
            The name of the function.
        function : Callable[[float], float]
            The function of a dimensionless number.
        derivative : Callable[[float], float]
            The derivative of the function, used to propagate the variance.
        """
        super().__init__(0.0, name=name)
        self._function = function
        self._derivative = derivative

    def __call__(
        self, argument: EasyScienceDescriptorNumber | Numeric
    ) -> DescriptorNumber | float:
        """
        Apply the function.

        Parameters
        ----------
        argument : EasyScienceDescriptorNumber | Numeric
            The argument of the function. A DescriptorNumber must have a unit that is convertible
            to dimensionless.

        Returns
        -------
        DescriptorNumber | float
            The dimensionless function value, with the variance of a DescriptorNumber argument
            propagated to first order. A plain number gives a plain number.
        """
        # Arithmetic in a dependency expression yields EasyScience objects, not EasyDynamics ones.
        if not isinstance(argument, EasyScienceDescriptorNumber):
            return self._function(argument)
        dimensionless = sc.to_unit(argument.full_value, 'dimensionless')
        return DescriptorNumber(
            name=self.name,
            value=self._function(dimensionless.value),
            variance=self._derivative(dimensionless.value) ** 2 * dimensionless.variance,
        )


def _sinc(x: float) -> float:
    """
    Calculate the unnormalized sinc function, sin(x)/x.

    Parameters
    ----------
    x : float
        The argument.

    Returns
    -------
    float
        sin(x)/x, which is 1 at x = 0.
    """
    return float(np.sinc(x / np.pi))


def _sinc_derivative(x: float) -> float:
    """
    Calculate the derivative of the unnormalized sinc function.

    Parameters
    ----------
    x : float
        The argument.

    Returns
    -------
    float
        (cos(x) - sin(x)/x)/x, which is 0 at x = 0.
    """
    if x == 0:
        return 0.0
    return float((np.cos(x) - _sinc(x)) / x)


def create_dependency_functions() -> dict[str, DependencyFunction]:
    """
    Create the functions to add to the dependency map of ``Parameter.make_dependent_on``.

    New objects are created on each call: ``make_dependent_on`` attaches the dependent Parameter as
    an observer of every mapped object, which must not accumulate on shared ones.

    Returns
    -------
    dict[str, DependencyFunction]
        ``exp``, ``sin``, ``cos`` and ``sinc`` (the unnormalized sin(x)/x), keyed by name.
    """
    functions = {
        'exp': (np.exp, np.exp),
        'sin': (np.sin, np.cos),
        'cos': (np.cos, lambda x: -np.sin(x)),
        'sinc': (_sinc, _sinc_derivative),
    }
    return {
        name: DependencyFunction(name, function, derivative)
        for name, (function, derivative) in functions.items()
    }
