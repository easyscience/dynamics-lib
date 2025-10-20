import warnings
from copy import copy
from itertools import chain
from typing import List, Optional, Union

import numpy as np
import scipp as sc
from easyscience.base_classes import CollectionBase
from easyscience.global_object.undo_redo import NotarizedDict
from easyscience.job.theoreticalmodel import TheoreticalModelBase
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.utils import _detailed_balance_factor as detailed_balance_factor

from .components.model_component import ModelComponent

Numeric = Union[float, int]


class SampleModel(CollectionBase, TheoreticalModelBase):
    """
    A model of the scattering from a sample, combining multiple model components.
    Optionally applies detailed balancing.

    Attributes
    ----------
    temperature : Parameter
        Temperature parameter for detailed balance.
    use_detailed_balance : bool
        Whether to apply detailed balance.
    normalize_detailed_balance : bool
        Whether to normalize the detailed balance by temperature.
    name : str
        Name of the SampleModel.
    unit : str or sc.Unit
        Unit of the SampleModel.
    components : List[ModelComponent]
        List of model components in the SampleModel.

    """

    def __init__(
        self,
        name: str = "MySampleModel",
        unit: Optional[Union[str, sc.Unit]] = "meV",
        temperature: Optional[Union[Numeric, sc.Variable]] = None,
        temperature_unit: Optional[str] = "K",
    ):
        """
        Initialize a new SampleModel.

        Parameters
        ----------
        name : str
            Name of the sample model.
        temperature : Number, sc.Variable or None, optional
            Temperature for detailed balance.
        temperature_unit : str, default 'K'
            Unit of the temperature.
        """

        CollectionBase.__init__(self, name=name)
        TheoreticalModelBase.__init__(self, name=name)
        if not isinstance(self._kwargs, NotarizedDict):
            self._kwargs = NotarizedDict()

        # If temperature is given, create a Parameter and enable detailed balance.
        if temperature is None:
            self._temperature = None
            self._use_detailed_balance = False
        elif isinstance(temperature, sc.Variable):
            self._temperature = Parameter(
                name="temperature",
                value=temperature.value,
                unit=temperature.unit,
                fixed=True,
            )
        else:
            self._temperature = Parameter(
                name="temperature", value=temperature, unit=temperature_unit, fixed=True
            )
            self._use_detailed_balance = True

        self._normalize_detailed_balance = (
            True  # Whether to normalize by temperature when using detailed balance.
        )
        self._unit = unit

    ##############################################
    #       Methods for managing components     #
    ##############################################

    def add_component(
        self, component: ModelComponent, name: Optional[str] = None
    ) -> None:
        """
        Add a model component to the SampleModel. Component names must be unique.
        Parameters
        ----------
        component : ModelComponent
            The model component to add.
        name : str, optional
            Name to assign to the component. If None, uses the component's own name.
        """
        if not isinstance(component, ModelComponent):
            raise TypeError("component must be an instance of ModelComponent.")

        if name is None:
            name = component.name
        if name in self.list_component_names():
            raise ValueError(f"Component with name '{name}' already exists.")

        component.name = name

        self.insert(index=len(self), value=component)

    def remove_component(self, name: str):
        """
        Remove a model component by name.
        """
        # Find index where item.name == name
        indices = [i for i, item in enumerate(list(self)) if item.name == name]
        if not indices:
            raise KeyError(f"No component named '{name}' exists in the model.")
        del self[indices[0]]

    def list_component_names(self) -> List[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        List[str]
            Component names.
        """

        return [item.name for item in list(self)]

    def clear_components(self):
        """
        Remove all components from the model.
        """

        for _ in range(len(self)):
            del self[0]

    def normalize_area(self) -> None:
        # Useful for convolutions.
        """
        Normalize the areas of all components so they sum to 1.
        """
        if not self.components:
            raise ValueError("No components in the model to normalize.")

        area_params = []
        total_area = 0.0

        for component in list(self):
            if hasattr(component, "area"):
                area_params.append(component.area)
                total_area += component.area.value
            else:
                warnings.warn(
                    f"Component '{component.name}' does not have an 'area' attribute and will be skipped in normalization."
                )

        if total_area == 0:
            raise ValueError("Total area is zero; cannot normalize.")

        if not np.isfinite(total_area):
            raise ValueError("Total area is not finite; cannot normalize.")

        for param in area_params:
            param.value /= total_area

    def convert_unit(self, unit: Union[str, sc.Unit]):
        """
        Convert the unit of the SampleModel and all its components.
        """
        self._unit = unit
        # for component in self.components.values():
        for component in list(self):
            component.convert_unit(unit)

    @property
    def components(self) -> List[ModelComponent]:
        """
        Get the list of components in the SampleModel.

        Returns
        -------
        List[ModelComponent]
        """
        return list(self)

    @property
    def unit(self) -> Optional[Union[str, sc.Unit]]:
        """
        Get the unit of the SampleModel.

        Returns
        -------
        str or sc.Unit or None
        """
        return self._unit

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )  # noqa: E501

    ##########################################################
    #       Methods for temperature and detailed balance     #
    ##########################################################

    @property
    def temperature(self) -> Optional[Parameter]:
        """
        Get the temperature.

        Returns
        -------
        Parameter
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: Optional[Numeric]) -> None:
        """
        Set the temperature.

        Parameters
        ----------
        value : Number
            Temperature value. If None, removes temperature and disables detailed balance.
        """
        # If None, disable detailed balance and remove temperature parameter.
        if value is None:
            self._use_detailed_balance = False
            self._temperature = None
            return

        if not isinstance(value, Numeric):
            raise TypeError("Temperature must be a number or None.")
        value = float(value)

        if value < 0:
            raise ValueError("Temperature must be non-negative.")

        if isinstance(self._temperature, Parameter):
            self._temperature.value = value
        else:
            self._temperature = Parameter(
                name="temperature", value=value, unit="K", fixed=True
            )

    def convert_temperature_unit(self, new_unit: Union[str, sc.Unit]) -> None:
        """
        Convert the temperature parameter to a new unit.

        Parameters
        ----------
        new_unit : str or sc.Unit
            The new unit for the temperature.
        """
        if self._temperature is None:
            raise ValueError("Temperature is not set; cannot convert units.")

        try:
            self._temperature.convert_unit(new_unit)
        except UnitError as e:
            raise UnitError(f"Failed to convert temperature to unit '{new_unit}': {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during unit conversion: {e}")

    @property
    def use_detailed_balance(self) -> bool:
        """
        True if detailed balance is enabled, otherwise False.

        Returns
        -------
        bool
        """
        return self._use_detailed_balance

    @use_detailed_balance.setter
    def use_detailed_balance(self, value: bool) -> None:
        """
        If True, enables the use of detailed balance. Otherwise disables it.

        Parameters
        ----------
        value : bool
            True to enable, False to disable.
        """
        if self._temperature is None:
            raise ValueError("Temperature must be set to use detailed balance.")
        self._use_detailed_balance = value

    @property
    def normalize_detailed_balance(self) -> bool:
        """
        If True, detailed balance will be normalized by temperature. If False, it will not be normalized.

        """
        return self._normalize_detailed_balance

    @normalize_detailed_balance.setter
    def normalize_detailed_balance(self, value: bool) -> None:
        """
        If True, normalizes the detailed balance by temperature.

        Parameters
        ----------
        value : bool
            True to normalize, False otherwise.
        """
        self._normalize_detailed_balance = value

    ##########################################################
    #       Evaluate        #
    ##########################################################

    def evaluate(
        self, x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray]
    ) -> np.ndarray:
        """
        Evaluate the sum of all components, optionally applying detailed balance.

        Parameters
        ----------
        x : Number, list, np.ndarray, sc.Variable, or sc.DataArray
            Energy axis.

        Returns
        -------
        np.ndarray
            Evaluated model values.
        """

        if not self.components:
            raise ValueError("No components in the model to evaluate.")
        result = None
        for component in list(self):
            value = component.evaluate(x)
            result = value if result is None else result + value

        if (
            self.use_detailed_balance
            and self._temperature is not None
            and self._temperature.value >= 0
        ):
            result *= detailed_balance_factor(
                energy=x,
                temperature=self._temperature,
                divide_by_temperature=self._normalize_detailed_balance,
            )

        return result

    def evaluate_component(
        self,
        x: Union[Numeric, list, np.ndarray, sc.Variable, sc.DataArray],
        name: str,
    ) -> np.ndarray:
        """
        Evaluate a single component by name, optionally applying detailed balance.

        Parameters
        ----------
        x : Number, list, np.ndarray, sc.Variable, or sc.DataArray
            Energy axis.
        name : str
            Component name.

        Returns
        -------
        np.ndarray
            Evaluated values for the specified component.
        """
        if not self.components:
            raise ValueError("No components in the model to evaluate.")

        if not isinstance(name, str):
            raise TypeError(
                (f"Component name must be a string, got {type(name)} instead.")
            )

        matches = [comp for comp in list(self) if comp.name == name]
        if not matches:
            raise KeyError(f"No component named '{name}' exists.")

        component = matches[0]

        result = component.evaluate(x)
        if (
            self.use_detailed_balance
            and self._temperature is not None
            and self._temperature.value >= 0
        ):
            result *= detailed_balance_factor(
                energy=x,
                temperature=self._temperature,
                divide_by_temperature=self._normalize_detailed_balance,
            )

        return result

    ##############################################
    #       Methods for managing parameters     #
    ##############################################

    def get_parameters(self) -> List[Parameter]:
        """
        Return all parameters in the SampleModel.

        Returns
        -------
        List[Parameter]
        """
        # Create generator for temperature parameter
        temp_params = (self._temperature,) if self._temperature is not None else ()

        # Create generator for component parameters
        comp_params = (param for comp in list(self) for param in comp.get_parameters())

        # Chain them together and return as list
        return list(chain(temp_params, comp_params))

    def get_fit_parameters(self) -> List[Parameter]:
        """
        Get all fit parameters, removing fixed and dependent parameters.

        Returns:
            List[Parameter]: A list of fit parameters.
        """

        def is_fit_parameter(param: Parameter) -> bool:
            """Check if a parameter can be used for fitting."""
            return not getattr(param, "fixed", False) and getattr(
                param, "_independent", True
            )

        return [param for param in self.get_parameters() if is_fit_parameter(param)]

        # return fit_parameters

    def fix_all_parameters(self) -> None:
        """
        Fix all free parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = True

    def free_all_parameters(self) -> None:
        """
        Free all fixed parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = False

    ##############################################
    #       dunder methods                      #
    ##############################################

    def __copy__(self) -> "SampleModel":
        """
        Create a deep copy of the SampleModel with independent parameters.

        Returns
        -------
        SampleModel
            A new instance with copied components and parameters.
        """
        name = "copy of " + self.name

        new_model = SampleModel(
            name=name,
            temperature=self._temperature.value if self._temperature else None,
            unit=self.unit,
        )

        if self._temperature:
            new_model.use_detailed_balance = self.use_detailed_balance
            new_model.normalize_detailed_balance = self.normalize_detailed_balance

        for comp in list(self):
            new_model.add_component(component=copy(comp), name=comp.name)
            new_model[comp.name].name = comp.name  # Remove 'copy of ' prefix
            for par in new_model[comp.name].get_parameters():
                par.name = par.name.removeprefix("copy of ")

        return new_model

    def __repr__(self) -> str:
        """
        Return a string representation of the SampleModel.

        Returns
        -------
        str
        """
        comp_names = ", ".join(c.name for c in self) or "No components"

        temp_str = ""
        if (
            getattr(self, "_use_detailed_balance", False)
            and getattr(self, "_temperature", None) is not None
        ):
            temp = self._temperature
            temp_str = f" | Temperature: {temp.value} {temp.unit}"

        return f"<SampleModel name='{self.name}' | Components: {comp_names}{temp_str}>"
