# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numbers

import scipp as sc
from easyscience.variable import DescriptorNumber as EasyScienceDescriptorNumber

from easydynamics.base_classes.name_mixin import NameMixin


class DescriptorNumber(NameMixin, EasyScienceDescriptorNumber):
    """
    A class representing a descriptor number in the EasyDynamics framework. It adds the 'name'
    attribute to the base DescriptorNumber class from EasyScience.
    """

    def __init__(
        self,
        value: numbers.Number,
        *,
        name: str = 'MyEasyDynamicsDescriptorNumber',
        unit: str | sc.Unit = '',
        variance: numbers.Number | None = 0.0,
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
        name : str, default='MyEasyDynamicsDescriptorNumber'
            Name of the Parameter.
        unit : str | sc.Unit, default=''
            This object can have a physical unit associated with it. By default, ''.
        variance : numbers.Number | None, default=0.0
            The variance of the value. By default, 0.0.
        unique_name : str | None, default=None
            Unique identifier for this object. By default, None.
        description : str | None, default=None
            A brief summary of what this object is. By default, None.
        url : str | None, default=None
            Lookup url for documentation/information. By default, None.
        display_name : str | None, default=None
            The name of the object as it should be displayed. Falls back to ``name`` when not
            given. By default, None.
        **kwargs : object
            Additional keyword arguments used during serialization.
        """
        if display_name is None:
            display_name = name

        super().__init__(
            value=value,
            name=name,
            unit=unit,
            variance=variance,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            **kwargs,
        )
