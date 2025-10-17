import warnings
from collections.abc import MutableMapping
from copy import copy
from typing import Dict, List, Optional, Union

import numpy as np
import scipp as sc
from easyscience.base_classes import ObjBase
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.utils import _detailed_balance_factor as detailed_balance_factor

from .components.model_component import ModelComponent

Numeric = Union[float, int]


class SampleModel(ObjBase, MutableMapping):
    """
    A model of the scattering from a sample, combining multiple model components.
    Optionally applies detailed balancing.

    Attributes
    ----------
    components : dict
        Dictionary of model components keyed by name.
    temperature : Parameter
        Temperature parameter for detailed balance.
    use_detailed_balance : bool
        Whether to apply detailed balance.
    normalize_detailed_balance : bool
        Whether to normalize the detailed balance by temperature.
    name : str
        Name of the SampleModel.
    """

    def __init__(
        self,
        name: str = "MySampleModel",
        temperature: Optional[Union[Numeric, None]] = None,
        temperature_unit: Optional[str] = "K",
    ):
        """
        Initialize a new SampleModel.

        Parameters
        ----------
        name : str
            Name of the sample model.
        temperature : Number or None, optional
            Temperature for detailed balance.
        temperature_unit : str, default 'K'
            Unit of the temperature.
        """

        self.components: Dict[str, ModelComponent] = {}
        super().__init__(name=name)
        # If temperature is given, create a Parameter and enable detailed balance.
        if temperature is not None:
            self._temperature = Parameter(
                name="temperature", value=temperature, unit=temperature_unit, fixed=True
            )
            self._use_detailed_balance = True
        else:
            self._temperature = None
            self._use_detailed_balance = False

        self._normalize_detailed_balance = (
            True  # Whether to normalize by temperature when using detailed balance.
        )

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
        if name is None:
            name = component.name
        if name in self.components:
            raise ValueError(f"Component with name '{name}' already exists.")

        if not isinstance(component, ModelComponent):
            raise TypeError("component must be an instance of ModelComponent.")

        self.components[name] = component

    def remove_component(self, name: str):
        """
        Remove a model component by name.

        Parameters
        ----------
        name : str
            Name of the component to remove.
        """

        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists in the model.")
        del self.components[name]

    def list_components(self) -> List[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        List[str]
            Component names.
        """

        return list(self.components.keys())

    def clear_components(self):
        """
        Remove all components from the model.
        """

        self.components.clear()

    def normalize_area(self) -> None:
        # Useful for convolutions.
        """
        Normalize the areas of all components so they sum to 1.
        """
        if not self.components:
            raise ValueError("No components in the model to normalize.")

        area_params = []
        total_area = 0.0

        for component in self.components.values():
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

    ##########################################################
    #       Methods for temperature and detailed balance     #
    ##########################################################

    @property
    def temperature(self) -> Union[Parameter, None]:
        """
        Get the temperature.

        Returns
        -------
        Parameter
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: Union[Numeric, None]) -> None:
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
        for component in self.components.values():
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
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists.")

        result = self.components[name].evaluate(x)
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
        if isinstance(self._temperature, Parameter):
            params = [self._temperature]
        else:
            params = []
        for comp in self.components.values():
            params.extend(comp.get_parameters())
        return params

    def get_fit_parameters(self) -> List[Parameter]:
        """
        Get all fit parameters, removing fixed and dependent parameters.

        Returns:
            List[Parameter]: A list of fit parameters.
        """

        parameters = self.get_parameters()
        fit_parameters = []

        for parameter in parameters:
            is_not_fixed = not getattr(parameter, "fixed", False)
            is_independent = getattr(parameter, "_independent", True)

            if is_not_fixed and is_independent:
                fit_parameters.append(parameter)

        return fit_parameters

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
        )

        if self._temperature:
            new_model.use_detailed_balance = self.use_detailed_balance

        for comp in self.components.values():
            new_model.add_component(component=copy(comp), name=comp.name)
            new_model[comp.name].name = comp.name  # Remove 'copy of ' prefix
            for par in new_model[comp.name].get_parameters():
                par.name = par.name.removeprefix("copy of ")

        return new_model

    ##############################################
    #       dict-like behaviour                  #
    ##############################################

    def __getitem__(self, key: str) -> ModelComponent:
        """
        Access a component by name.

        Parameters
        ----------
        key : str
            Name of the component.

        Returns
        -------
        ModelComponent
        """
        return self.components[key]

    def __setitem__(self, key: str, value: ModelComponent) -> None:
        """
        Set or replace a component.

        Parameters
        ----------
        key : str
            Name of the component.
        value : ModelComponent
            The component to assign.
        """
        if not isinstance(value, ModelComponent):
            raise TypeError("Value must be an instance of ModelComponent.")
        self.components[key] = value

    def __delitem__(self, key: str) -> None:
        """
        Remove a component by name.
        Parameters
        ----------
        key : str
            Name of the component to remove.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")

        if key not in self.components:
            raise KeyError(f"No component named '{key}' exists in the model.")

        self.remove_component(key)

    def __contains__(self, name: str) -> bool:
        """
        Check if a component exists in the model.

        Parameters
        ----------
        name : str
            Name of the component.

        Returns
        -------
        bool
        """
        return name in self.components

    def __iter__(self) -> iter:
        """Iterate over component names."""
        return iter(self.components)

    def __len__(self) -> int:
        """Return the number of components in the model."""
        return len(self.components)

    def __repr__(self) -> str:
        """
        Return a string representation of the SampleModel.

        Returns
        -------
        str
        """
        comp_names = ", ".join(self.components.keys()) or "No components"
        temp_str = (
            f" | Temperature: {self._temperature.value} {self._temperature.unit}"
            if self._use_detailed_balance
            else ""
        )
        return f"<SampleModel name='{self.name}' | Components: {comp_names}{temp_str}>"
