import warnings
from typing import Dict, List, Union, Tuple

import numpy as np

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

from easydynamics.utils import detailed_balance_factor
from .components import ModelComponent



class SampleModel(ObjBase):
    """
    A model of the scattering from a sample, combining multiple model components.
    Optionally applies detailed balancing.

    Attributes
    ----------
    components : dict
        Dictionary of model components keyed by name.
    """
    def __init__(self, name: str = "MySampleModel", temperature: Union[float, None] = None):
        """
        Initialize a new SampleModel.

        Parameters
        ----------
        name : str
            Name of the sample model.
        temperature : float or None, optional
        """
                
        self.components: Dict[str, ModelComponent] = {}
        super().__init__(name=name)
        if temperature is not None:
            self._temperature = Parameter(name="temperature", value=temperature, unit='K', fixed=True)
            self._use_detailed_balance = True
        else:
            self._temperature=None
            self._use_detailed_balance = False

    def add_component(self, component: ModelComponent):
        """
        Add a model component to the SampleModel. Component names must be unique.
        """
        if component.name in self.components:
            raise ValueError(f"Component with name '{component.name}' already exists.")
        self.components[component.name] = component

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

    def __setitem__(self, key: str, value: ModelComponent):
        """
        Set or replace a component by name using dictionary-like syntax.

        Parameters
        ----------
        key : str
            Name of the component.
        value : ModelComponent
            The component to assign.
        """
        self.components[key] = value

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

    def __repr__(self):
        """
        Return a string representation of the SampleModel.

        Returns
        -------
        str
        """
        comp_names = ", ".join(self.components.keys()) or "No components"
        temp_str = (f" | Temperature: {self._temperature.value} {self._temperature.unit}"
                    if self._use_detailed_balance else "")
        return (f"<SampleModel name='{self.name}' | "
                f"Components: {comp_names}{temp_str}>")

    @property
    def temperature(self) -> Parameter:
        """
        Access the temperature parameter.

        Returns
        -------
        Parameter
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: Union[float, None], unit: str = 'K'):
        """
        Set the temperature and enables detailed balance if value is non-negative.

        Parameters
        ----------
        value : float
            Temperature value.
        unit : str, default 'K'
            Unit of the temperature.
        """
        if value is None:
            self._use_detailed_balance = False
            self._temperature = None
            return

        if value < 0:
            raise ValueError("Temperature must be non-negative.")

        if isinstance(self._temperature, Parameter):
            self._temperature.value = value
        else:
            self._temperature = Parameter(name="temperature", value=value, unit=unit, fixed=True)

        if not self.use_detailed_balance:
            self.use_detailed_balance = value >= 0

    @property
    def use_detailed_balance(self) -> bool:
        """
        Indicates whether detailed balance is enabled.

        Returns
        -------
        bool
        """
        return self._use_detailed_balance

    @use_detailed_balance.setter
    def use_detailed_balance(self, value: bool):
        """
        Enable or disable the use of detailed balance.

        Parameters
        ----------
        value : bool
            True to enable, False to disable.
        """
        self._use_detailed_balance = value

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the sum of all components, optionally applying detailed balance.

        Parameters
        ----------
        x : np.ndarray
            Energy axis (e.g., in meV).

        Returns
        -------
        np.ndarray
            Evaluated model values.
        """
        result = np.zeros_like(x, dtype=float)
        for component in self.components.values():
            result += component.evaluate(x)

        if self.use_detailed_balance and self._temperature.value >= 0:
            result *= detailed_balance_factor(x, self._temperature.value)

        return result

    def evaluate_component(self, name: str, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a single component by name, optionally applying detailed balance.

        Parameters
        ----------
        name : str
            Component name.
        x : np.ndarray
            Energy axis.

        Returns
        -------
        np.ndarray
            Evaluated values for the specified component.

        Raises
        ------
        KeyError
            If the component is not found.
        """
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists.")

        result = self.components[name].evaluate(x)
        if self._use_detailed_balance and self._temperature.value >= 0:
            result *= detailed_balance_factor(x, self._temperature.value)

        return result

    def normalize_area(self):
        """
        Normalize the areas of all components so they sum to 1.
        """
        area_params = []
        total_area = 0.0

        for component in self.components.values():
            for param in component.get_parameters():
                if 'area' in param.name.lower():
                    area_params.append(param)
                    total_area += param.value

        if total_area == 0:
            raise ValueError("Total area is zero; cannot normalize.")

        for param in area_params:
            param.value /= total_area

    def get_parameters(self) -> List[Parameter]:
        """
        Return all parameters from the model, including temperature.

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
    
    def get_fit_parameters(self):
        """
        Get all fit parameters, removing fixed and dependent parameters.

        Returns:
            List[Parameter]: A list of fit parameters.
        """

        parameters = self.get_parameters()
        fit_parameters = []
        
        for parameter in parameters:
            is_not_fixed = not getattr(parameter, 'fixed', False)
            is_independent = getattr(parameter, '_independent', True)
            
            if is_not_fixed and is_independent:
                fit_parameters.append(parameter)
        
        return fit_parameters

    
    def fix_all_parameters(self):
        """
        Fix all unfixed parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = True

    def free_all_parameters(self):
        """
        Free all fixed parameters in the model.
        """
        for param in self.get_parameters():
            param.fixed = False

    def fix_all_component_parameters(self,component_name: str):
        """
        Fix all unfixed parameters in the specified component.
        """
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found.")
        
        self.components[component_name].fix_all_parameters()

    def free_all_component_parameters(self, component_name: str):
        """
        Free all fixed parameters in the specified component.
        """
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found.")

        self.components[component_name].free_all_parameters()

    def fix_component_parameter(self,component_name: str, parameter_name: str):
        """
        Fix a specific parameter in the specified component.
        """
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found.")

        component = self.components[component_name]
        param = component.get_parameter(parameter_name)
        if param is None:
            raise ValueError(f"Parameter '{parameter_name}' not found in component '{component_name}'.")

        param.fixed = True

    def free_component_parameter(self, component_name: str, parameter_name: str):
        """
        Free a specific parameter in the specified component.
        """
        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' not found.")

        component = self.components[component_name]
        param = component.get_parameter(parameter_name)
        if param is None:
            raise ValueError(f"Parameter '{parameter_name}' not found in component '{component_name}'.")

        param.fixed = False

    def update_values_from(
        self,
        other: "SampleModel",
        *,
        only_free: bool = True,
    )-> Dict[str, Tuple[float, float]]:
        """
        Overwrite this model's Parameter.values from another SampleModel, matching by
        component name and Parameter.name. This is used to copy fit results when doing sequential fitting.  

        Parameters
        ----------
        other : SampleModel
            Source of values.
        only_free : bool, default True
            If True, skip Parameters in *self* that are fixed.

        Returns
        -------
        Dict[str, Tuple[float, float]]
            Mapping key -> (old_value, new_value), where key is
             "<component>.<Parameter.name>".

        """
        if not isinstance(other, SampleModel):
            raise TypeError("other must be a SampleModel")

        report: Dict[str, Tuple[float, float]] = {}

        # Check that components are the same
        self_names = set(self.components.keys())
        other_names = set(other.components.keys())

        if self_names != other_names:
            missing = self_names - other_names
            extra   = other_names - self_names
            raise ValueError(
                f"Component name mismatch.\n"
                f"  Missing in source: {missing or '{}'}\n"
                f"  Extra in source:   {extra or '{}'}"
            )


        # Go through components
        for cname in self_names:
            c_self  = self.components[cname]
            c_other = other.components[cname]

            # Check that parameters are the same
            self_params  = {p.name: p for p in c_self.get_parameters()}
            other_params = {p.name: p for p in c_other.get_parameters()}

            if set(self_params) != set(other_params):
                missing = set(self_params) - set(other_params)
                extra   = set(other_params) - set(self_params)
                raise ValueError(
                    f"Parameter name mismatch in component '{cname}'.\n"
                    f"  Missing in source: {missing or '{}'}\n"
                    f"  Extra in source:   {extra or '{}'}"
                )


            for pname in set(self_params):
                p_self  = self_params[pname]
                p_other = other_params[pname]

                if only_free and getattr(p_self, "fixed", False):
                    continue

                # Units: convert units to other's unit if they differ
                u_self  = getattr(p_self, "unit", None)
                u_other = getattr(p_other, "unit", None)
                if u_self != u_other:
                    p_self.convert_unit(u_other)

                # Update value, but save the old one.
                old = p_self.value
                p_self.value = p_other.value
                report[f"{cname}.{pname}"] = (old, p_self.value)

        return report


    
    def copy(self) -> "SampleModel":
        """
        Create a deep copy of the SampleModel with independent parameters.

        Returns
        -------
        SampleModel
            A new instance with copied components and parameters.
        """
        
        new_model = SampleModel(name=self.name, temperature=self._temperature.value if self._temperature else None)

        new_model.use_detailed_balance = self._use_detailed_balance

        for comp in self.components.values():
            new_model.add_component(comp.copy())

        return new_model
