import warnings
from typing import Dict, List, Union, Tuple

import numpy as np

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

from easydynamics.utils import detailed_balance_factor
from .components import ModelComponent



class SampleModel(ObjBase):
    """
    Represents a combined model composed of multiple model components.

    This class allows components to be added, removed, and evaluated together.
    Optionally applies detailed balancing. 

    Attributes
    ----------
    components : dict
        Dictionary of model components keyed by name.
    """
    def __init__(self, name="MySampleModel", temperature=None):
        """
        Initialize a new SampleModel.

        Parameters
        ----------
        name : str
            Name of the sample model.
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
        Add a model component to the SampleModel.

        Raises
        ------
        ValueError
            If a component with the same name already exists.
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

        Raises
        ------
        KeyError
            If the component does not exist.
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
        Access a component by name using dictionary-like indexing.

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
    def temperature(self, value: Union[float, None]):
        """
        Set the temperature.

        If a negative value is provided, detailed balance is disabled and a warning is issued.

        Parameters
        ----------
        value : float
            Temperature value in Kelvin.
        """
        if value is None:
            self._use_detailed_balance = False
            self._temperature = None
            return
            # warnings.warn("Temperature is negative. Disabling detailed balance.") #TODO: Uncomment this line if you want to enable the warning

        if value < 0:
            raise ValueError("Temperature must be non-negative.")

        if isinstance(self._temperature, Parameter):
            self._temperature.value = value
        else:
            self._temperature = Parameter(name="temperature", value=value, unit='K', fixed=True)

        if not self.use_detailed_balance:
            self.use_detailed_balance = value >= 0
            # print(f"Detailed balance set to {self.use_detailed_balance} for temperature {value} K") # #TODO: Uncomment this line if you want to enable the warning

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

        Modifies the 'area' parameters directly. Raises an error if total area is zero.
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
        Get all fit parameters, filtering out fixed parameters.

        Returns:
            List[Parameter]: A list of unfixed fit parameters.
        """
        return [param for param in self.get_parameters() if not getattr(param, 'fixed', False)]
    

    def update_values_from(
        self,
        other: "SampleModel",
        *,
        only_unfixed: bool = True,
        strict_components: bool = True,
        strict_params: bool = True,
        include_temperature: bool = False,
        require_same_units: bool = True,
        convert_units: bool = False,
    )-> Dict[str, Tuple[float, float]]:
        """
        Overwrite this model's Parameter.values from another SampleModel, matching by
        component name and Parameter.name.

        Parameters
        ----------
        other : SampleModel
            Source of values.
        only_unfixed : bool, default True
            If True, skip Parameters in *self* that are fixed.
        strict_components : bool, default True
            If True, require identical component name sets; else use intersection.
        strict_params : bool, default True
            If True, require identical parameter name sets per component; else use intersection.
        include_temperature : bool, default False
            If True, also copy temperature (if present on both).
        require_same_units : bool, default True
            If True, fail on unit mismatch; otherwise allow.
        convert_units : bool, default False
            If True and units differ, attempt self-param conversion to other's unit.

        Returns
        -------
        Dict[str, Tuple[float, float]]
            Mapping key -> (old_value, new_value), where key is
            "temperature" or "<component>.<Parameter.name>".

        Notes
        -----
        - Only .value is overwritten. Parameter objects are not replaced.
        - Parameter matching uses the full Parameter.name as you define it
          in components (e.g. 'Lorentziancenter', 'Gaussianwidth', 'Polynomial_c0').
        """
        if not isinstance(other, SampleModel):
            raise TypeError("other must be a SampleModel")

        report: Dict[str, Tuple[float, float]] = {}

        # --- Component set checks
        self_names = set(self.components.keys())
        other_names = set(other.components.keys())

        if strict_components and self_names != other_names:
            missing = self_names - other_names
            extra   = other_names - self_names
            raise ValueError(
                f"Component name mismatch.\n"
                f"  Missing in source: {missing or '{}'}\n"
                f"  Extra in source:   {extra or '{}'}"
            )

        common_components = self_names & other_names if not strict_components else self_names

        # --- Optional: temperature
        if include_temperature and isinstance(self._temperature, Parameter) and isinstance(other._temperature, Parameter):
            if (not only_unfixed) or (not getattr(self._temperature, "fixed", False)):
                if getattr(self._temperature, "unit", None) != getattr(other._temperature, "unit", None):
                    if require_same_units and not convert_units:
                        raise ValueError(
                            f"Unit mismatch for temperature: {self._temperature.unit} vs {other._temperature.unit}"
                        )
                    if convert_units and hasattr(self._temperature, "convert_unit"):
                        self._temperature.convert_unit(other._temperature.unit)
                old = self._temperature.value
                self._temperature.value = other._temperature.value
                report["temperature"] = (old, self._temperature.value)

        # --- Per-component parameters
        for cname in common_components:
            c_self  = self.components[cname]
            c_other = other.components[cname]

            self_params  = {p.name: p for p in c_self.get_parameters()}
            other_params = {p.name: p for p in c_other.get_parameters()}

            if strict_params and set(self_params) != set(other_params):
                missing = set(self_params) - set(other_params)
                extra   = set(other_params) - set(self_params)
                raise ValueError(
                    f"Parameter name mismatch in component '{cname}'.\n"
                    f"  Missing in source: {missing or '{}'}\n"
                    f"  Extra in source:   {extra or '{}'}"
                )

            common_param_names = set(self_params) & set(other_params) if not strict_params else set(self_params)

            for pname in common_param_names:
                p_self  = self_params[pname]
                p_other = other_params[pname]

                if only_unfixed and getattr(p_self, "fixed", False):
                    continue

                # Units (best effort)
                u_self  = getattr(p_self, "unit", None)
                u_other = getattr(p_other, "unit", None)
                if u_self != u_other:
                    if require_same_units and not convert_units:
                        raise ValueError(
                            f"Unit mismatch for {cname}.{pname}: {u_self} vs {u_other}"
                        )
                    if convert_units and hasattr(p_self, "convert_unit") and u_other is not None:
                        p_self.convert_unit(u_other)

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
        # new_model.temperature = self._temperature.value
        new_model.use_detailed_balance = self._use_detailed_balance


        for comp in self.components.values():
            new_model.add_component(comp.copy())

            

        return new_model