# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numbers

import numpy as np
import scipp as sc
from easyscience.variable import Parameter as EasyScienceParameter

from easydynamics.base_classes.name_mixin import NameMixin


class Parameter(NameMixin, EasyScienceParameter):
    """
    A class representing a parameter in the EasyDynamics framework. It adds the 'name' attribute to
    the base Parameter class from EasyScience.
    """

    def __init__(
        self,
        value: numbers.Number,
        *,
        name: str = 'MyEasyDynamicsParameter',
        unit: str | sc.Unit = '',
        variance: numbers.Number | None = 0.0,
        min: numbers.Number | None = -np.inf,
        max: numbers.Number | None = np.inf,
        fixed: bool = False,
        unique_name: str | None = None,
        description: str | None = None,
        url: str | None = None,
        display_name: str | None = None,
        **kwargs: object,  # Additional keyword arguments (used for (de)serialization)
    ) -> None:
        """
        This class is an extension of a ``DescriptorNumber``.

        Where the descriptor was for static objects, a ``Parameter`` is for dynamic objects. A
        parameter has the ability to be used in fitting and has additional fields to facilitate
        this.

        All arguments after ``value`` are keyword-only.

        Parameters
        ----------
        value : numbers.Number
            Value of this object.
        name : str, default="MyEasyDynamicsParameter"
            Name of the Parameter.
        unit : str | sc.Unit, default=""
            This object can have a physical unit associated with it. By default, ''.
        variance : numbers.Number | None, default=0.0
            The variance of the value. By default, 0.0.
        min : numbers.Number | None, default=-np.inf
            The minimum value for fitting. By default, -np.inf.
        max : numbers.Number | None, default=np.inf
            The maximum value for fitting. By default, np.inf.
        fixed : bool, default=False
            If the parameter is free to vary during fitting. By default, False.
        unique_name : str | None, default=None
            Unique identifier for this object. By default, None.
        description : str | None, default=None
            A brief summary of what this object is. By default, None.
        url : str | None, default=None
            Lookup url for documentation/information. By default, None.
        display_name : str | None, default=None
            The name of the object as it should be displayed. Falls back to ``unique_name`` when
            not given. By default, None.
        **kwargs : object
            Additional keyword arguments used during serialization.

        Raises
        ------
        TypeError
            If any numeric or boolean input has the wrong type.
        ValueError
            If ``value`` falls outside the provided bounds or if the bounds are invalid.
        """
        super().__init__(
            value=value,
            name=name,
            unit=unit,
            variance=variance,
            min=min,
            max=max,
            fixed=fixed,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            **kwargs,
        )
