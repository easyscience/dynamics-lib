from typing import Dict
import numpy as np
from .components import ModelComponent


from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase




class SampleModel(ObjBase):
    """
    Represents a combined model composed of multiple model components.

    Methods:
        add_component(component): Add a new model component.
        evaluate(x): Sum and evaluate all components at given x.
    """
    def __init__(self,name):
        self.components: Dict[str, ModelComponent] = {}
        super().__init__(name=name)
        self._temperature=Parameter(name="temperature", value=-1, unit='K',fixed=True)  # Default temperature in Kelvin
        self.use_detailed_balance = True

    def add_component(self, component: ModelComponent):
        """
        Add a model component to the sample model.

        Args:
            component (ModelComponent): An instance of a model component.
        """

        if component.name in self.components:
            raise ValueError(f"Component with name '{component.name}' already exists.")
        self.components[component.name] = component

    def remove_component(self, name: str):
        """
        Remove a model component by name.

        Args:
            name (str): Name of the component to remove.

        Raises:
            KeyError: If no component with the given name exists.
        """
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists in the model.")
        del self.components[name]


    def __getitem__(self, key: str) -> ModelComponent:
        """Allow access via SampleModel['ComponentName']"""
        return self.components[key]

    def __setitem__(self, key: str, value: ModelComponent):
        """Allow assignment via SampleModel['ComponentName'] = component"""
        self.components[key] = value


    def temperature(self, temperature: float, unit='K'):
        """
        Set the temperature for the SampleModel.

        Args:
            temperature (float): Temperature value to set.
        """
        self._temperature.convert_unit(unit)
        self._temperature.value = temperature


    def get_temperature(self) -> Parameter:
        """
        Get the current temperature of the SampleModel.

        Returns:
            Parameter: The temperature
        """
        return self._temperature
    
    def use_detailed_balance(self, use: bool = True):
        """
        Set whether to use detailed balance in the model evaluation.

        Args:
            use (bool): If True, detailed balance is used.
        """
        self.use_detailed_balance = use



    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the full sample model by summing the contribution from all components.

        Args:
            x (np.ndarray): Input energy axis (e.g. in meV).

        Returns:
            np.ndarray: Total model output evaluated at `x`.
        """
        result = np.zeros_like(x, dtype=float)

        for component in self.components.values():
            # result += component.evaluate( x - self.offset.value)
            result += component.evaluate(x)

        if self.use_detailed_balance and self._temperature.value >= 0:
            result *= self.detailed_balance_factor(x, self._temperature.value)

        return result
    

    def normalize_area(self):
        """
        Normalize the SampleModel so the total area of all components sums to 1.
        This modifies the area parameters in-place.
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



    @staticmethod
    def detailed_balance_factor(omega_meV, temperature_K):
        """
        Compute ω * (n + 1), where n is the Bose-Einstein occupation number.
        
        This expression arises in detailed balance factors in neutron and light scattering.

        Parameters
        ----------
        omega_meV : float or np.ndarray
            Energy transfer in meV.
        temperature_K : float
            Temperature in Kelvin. Must be >= 0.
        
        Returns
        -------
        result : float or np.ndarray
            The value of ω * (n + 1), safely evaluated even for T=0.
        """
        if temperature_K < 0:
            raise ValueError("Temperature must be non-negative.")

        omega_meV = np.asarray(omega_meV, dtype=np.float64)

        if temperature_K == 0:
            return  np.maximum(omega_meV, 0.0)

        k_B_meV_per_K = 8.617333262e-2  # Boltzmann constant in meV/K

        beta = 1.0 / (k_B_meV_per_K * temperature_K)
        x = beta * omega_meV

        result = np.empty_like(omega_meV)

        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            exp_x = np.exp(x)
            denom = np.expm1(x)  # More stable than exp(x) - 1
            safe = denom != 0
            result[safe] = omega_meV[safe] * exp_x[safe] / denom[safe]
            result[~safe] = k_B_meV_per_K * temperature_K  # Limit as ω → 0

        return result
    
    def evaluate_component(self, name: str, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a specific component by name.

        Args:
            name (str): Name of the component to evaluate.
            x (np.ndarray): Input values (e.g. energy axis).

        Returns:
            np.ndarray: Evaluated component values.

        Raises:
            KeyError: If the named component does not exist.
        """
        if name not in self.components:
            raise KeyError(f"No component named '{name}' exists.")
        
        component = self.components[name]
        # return component.evaluate(x - self.offset.value)
        result = component.evaluate(x)
        if self.use_detailed_balance and self._temperature.value >= 0:
            result *= self.detailed_balance_factor(x, self._temperature.value)

        return result

    
    def get_parameters(self):
        """
        Get all parameters from the model components.

        Returns:
            List[Parameter]: List of parameters from all components.
        """
        params = []
        for comp in self.components.values():
            params.extend(comp.get_parameters())
        return params
    
    def name(self):
        """
        Get the name of the sample model.

        Returns:
            str: Name of the sample model.
        """
        return self.name
    
    def set_name(self, name: str):
        """
        Set the name of the sample model.

        Args:
            name (str): New name for the sample model.
        """
        self.name = name





    def __repr__(self):
        """
        String representation of the SampleModel object.

        Returns:
            str: Representation of the SampleModel object.
        """
        return f"SampleModel"