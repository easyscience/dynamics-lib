from abc import abstractmethod
from typing import Callable, Dict

import numpy as np
from scipy.special import voigt_profile

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

import warnings

from numbers import Number

#TODO: Allow specification of units for parameters in components

class ModelComponent(ObjBase):
    """
    Abstract base class for all model components.
    """

    def __init__(self, name='ModelComponent'):
        super().__init__(name=name)
        self.unit=None  

    def fix_all_parameters(self):
        """Fix all parameters in the model component."""
        for p in self.get_parameters():
            p.fixed = True

    def fit_all_parameters(self):
        """Fit all parameters in the model component."""
        for p in self.get_parameters():
            p.fixed = False

    def get_parameter(self, parameter_name):
        """
        Get a specific parameter by name (explicit or partial match).
        
        Args:
            parameter_name (str): Name of the parameter, or partial name to match.
        
        Returns:
            Parameter: The matched parameter.
        
        Raises:
            ValueError: If no matching or ambiguous parameter is found.
        """
        # First, attempt exact match
        for p in self.get_parameters():
            if p.name == parameter_name:
                return p

        # If exact match is not found, attempt partial match
        matches = [p for p in self.get_parameters() if parameter_name in p.name]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous parameter name '{parameter_name}' matches multiple parameters: {[p.name for p in matches]}")
        else:
            raise ValueError(f"Parameter '{parameter_name}' not found.")

    def set_parameter_value(self, parameter_name, value, unit=None):
        """
        Set the value of a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        if unit is not None:
            param.convert_unit(unit)
        param.value = value

    def set_parameter_bounds(self, parameter_name, min=None, max=None, unit=None):
        """
        Set the bounds of a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        if unit is not None:
            param.convert_unit(unit)
        if min is not None:
            param.min = min
        if max is not None:
            param.max = max

    def fix_parameter(self, parameter_name):
        """
        Fix a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        param.fixed = True

    def free_parameter(self, parameter_name):
        """
        Free a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        param.fixed = False

    def convert_unit(self, unit):
        """
        Convert the unit of the Parameters in the component.
        
        Args:
            unit (str): The new unit to convert to.
        """
        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.width.convert_unit(unit)
        self.unit = unit  

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the model component at positions x.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Evaluated function values.
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """
        Get all parameters from the model component.

        Returns
        -------
        List[Parameter]
            List of parameters in the component.
        """
        pass

    @abstractmethod
    def copy(self) -> "ModelComponent":
        """
        Return a deep copy of this component with independent parameters.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class GaussianComponent(ModelComponent):
    """
    Gaussian function. Creates new EasyScience Parameters if floats are provided, otherwise uses the provided Parameters.

    Args:
        area (float): area of the Gaussian.
        center (float): Center of the Gaussian. If None, defaults to 0 and is fixed
        width (float): Standard deviation.
    """

    def __init__(self, name='Gaussian', area=1.0, center=None, width=1.0, unit='meV'):
        
        # Validate inputs - throw errors before any Parameters are created
        if not isinstance(area, (Number, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")

        if center is not None and not isinstance(center, (Number, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")

        if not isinstance(width, (Number, Parameter)):
            raise TypeError("width must be a number or an EasyScience Parameter.")

        if isinstance(width,Number):
            if width <= 0:
                raise ValueError("The width of a Gaussian must be greater than zero.")
            width=float(width)

        if isinstance(area,Number):
            if area < 0:
                warnings.warn("The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)


        if isinstance(center,Number):
            center = float(center)
        
        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component

        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name= name+ ' center', value=0.0, unit=unit,fixed=True)
        elif isinstance(center,Number):
            self.center = Parameter(name=name+ ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(width,Number):
            self.width = Parameter(name=name+ ' width', value=width, unit=unit,min=0.0)
        else:
            self.width=width

        if isinstance(area,Number):
            self.area = Parameter(name=name+ ' area', value=area, unit=unit,min=0.0)
        else:
            self.area=area

    def evaluate(self, x):
        #TODO: Handle units properly
        if self.width.value <= 0:
            raise ValueError("The width of a Gaussian must be greater than zero.")
        if self.area.value < 0:
            warnings.warn("The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(self.name))
        return self.area.value * 1/(np.sqrt(2 * np.pi) * self.width.value) * np.exp(-0.5 * ((x - self.center.value) / self.width.value) ** 2)
    

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]
    
    def copy(self) -> "GaussianComponent":

        ModelCopy=GaussianComponent(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit
        )

        ModelCopy.area.fixed = self.area.fixed
        ModelCopy.center.fixed = self.center.fixed
        ModelCopy.width.fixed = self.width.fixed
        return ModelCopy

    def __repr__(self):
        return f"GaussianComponent(name={self.name}, area={self.area}, center={self.center}, width={self.width})"


class LorentzianComponent(ModelComponent):
    """
    Lorentzian function. Creates new EasyScience Parameters if floats are provided, otherwise uses the provided Parameters.

    Args:
        area (float): Area of the Lorentzian.
        center (float): Peak center.
        width (float): HWHM (Half Width at Half Maximum).
    """

    def __init__(self, name='Lorentzian', area=1.0, center=None, width=1.0, unit='meV'):

        
        # Validate inputs
        if not isinstance(area, (Number, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")

        if center is not None and not isinstance(center, (Number, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")

        if not isinstance(width, (Number, Parameter)):
            raise TypeError("width must be a number or an EasyScience Parameter.")

        if isinstance(width, Number):
            if width <= 0:
                raise ValueError("The width of a Lorentzian must be greater than zero.")
            width=float(width)

        if isinstance(area, Number):
            if area < 0:
                warnings.warn("The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)


        if isinstance(center, Number):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component

        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name=name + ' center', value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Number):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(width, Number):
            self.width = Parameter(name=name + ' width', value=width, unit=unit,min=0.0)
        else:
            self.width=width

        if isinstance(area, Number):
            self.area = Parameter(name=name + ' area', value=area, unit=unit,min=0.0)
        else:
            self.area=area

    def evaluate(self, x):
            #TODO: Handle units properly
        if self.width.value <= 0:
            raise ValueError("Width must be greater than 0 for Lorentzian.")
        if self.area.value < 0:
            warnings.warn("The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(self.name))
        return self.area.value * (self.width.value/np.pi / ((x - self.center.value)**2 + self.width.value**2))


    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]
    
    def copy(self) -> "LorentzianComponent":

        ModelCopy =LorentzianComponent(
            name=self.name, 
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit
        )   
        ModelCopy.area.fixed = self.area.fixed
        ModelCopy.center.fixed = self.center.fixed 
        ModelCopy.width.fixed = self.width.fixed
        return ModelCopy


    def __repr__(self):
        return f"LorentzianComponent(name={self.name}, area={self.area}, center={self.center}, width={self.width})"


class VoigtComponent(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (float): Center of the Voigt profile.
        width (float): Standard deviation of the Gaussian part.
        gamma (float): HWHM of the Lorentzian part.
        area (float): Total area under the curve.
    """

    def __init__(self, name='Voigt', area=1.0, center=None, Gwidth=1.0, Lwidth=1.0, unit='meV'):
        # Validate inputs
        if not isinstance(area, (Number, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")
        if center is not None and not isinstance(center, (Number, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")
        if not isinstance(Gwidth, (Number, Parameter)):
            raise TypeError("Gwidth must be a number or an EasyScience Parameter.")
        if not isinstance(Lwidth, (Number, Parameter)):
            raise TypeError("Lwidth must be a number or an EasyScience Parameter.")
        if isinstance(Gwidth, Number):
            if Gwidth <= 0:
                raise ValueError("Gwidth must be greater than 0 for Voigt profile.")
            Gwidth=float(Gwidth)
        if isinstance(Lwidth, Number):
            if Lwidth <= 0:
                raise ValueError("Lwidth must be greater than 0 for Voigt profile.")
            Lwidth=float(Lwidth)
        if isinstance(area, Number):
            if area < 0:
                warnings.warn("The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)
        
        super().__init__(name=name)


        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name=name + ' center', value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Number):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)

        if isinstance(Gwidth, Number):
            self.Gwidth = Parameter(name=name + ' Gwidth', value=Gwidth, unit=unit,min=0.0)
        else:
            self.Gwidth=Gwidth

        if isinstance(Lwidth, Number):
            self.Lwidth = Parameter(name=name + ' Lwidth', value=Lwidth, unit=unit,min=0.0)
        else:
            self.Lwidth=Lwidth

        if isinstance(area, Number):
            self.area = Parameter(name=name + ' area', value=area, unit=unit,min=0.0)
        else:
            self.area=area

    def evaluate(self, x):
        if self.Gwidth.value <= 0:
            raise ValueError("Gwidth must be greater than 0 for Voigt profile.")
        if self.Lwidth.value <= 0:
            raise ValueError("Lwidth must be greater than 0 for Voigt profile.")
        if self.area.value < 0:
            warnings.warn("The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(self.name))
        return self.area.value * voigt_profile(x - self.center.value, self.Gwidth.value, self.Lwidth.value)

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.Gwidth, self.Lwidth]
    
    def copy(self) -> "VoigtComponent":

        ModelCopy = VoigtComponent(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            Gwidth=self.Gwidth.value,
            Lwidth=self.Lwidth.value,
            unit=self.unit
        )
        ModelCopy.area.fixed = self.area.fixed
        ModelCopy.center.fixed = self.center.fixed
        ModelCopy.Gwidth.fixed = self.Gwidth.fixed
        ModelCopy.Lwidth.fixed = self.Lwidth.fixed

        return ModelCopy

    def __repr__(self):
        return f"VoigtComponent(name={self.name}, area={self.area}, center={self.center}, Gwidth={self.Gwidth}, Lwidth={self.Lwidth})"


class DeltaFunctionComponent(ModelComponent):
    """
    Delta function.

    Args:
        center (float): Mean of the Gaussian.
        area (float): Total area under the curve.
    """

    def __init__(self, name='DeltaFunction', center=None, area=1.0, unit='meV'):
        # Validate inputs
        if not isinstance(area, (Number, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")
        if center is not None and not isinstance(center, (Number, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")
        if isinstance(area, Number):
            if area < 0:
                warnings.warn("The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)
        if isinstance(center, Number):
            center = float(center)
        super().__init__(name=name)
        self.unit = unit
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name=name + ' center', value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Number):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(area, Number):
            self.area = Parameter(name=name + ' area', value=area, unit=unit,min=0.0)
        else:
            self.area=area


    def evaluate(self, x):

        if self.area.value < 0:
            warnings.warn("The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(self.name))
        #TODO: Handle units properly. Also handle area if we want users to be able to plot it without resolution convolution
        return 0*x
    
    
    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center]
    
    def convert_unit(self, unit):
        """
        Convert the unit of the Parameters in the component.
        
        Args:
            unit (str): The new unit to convert to.
        """
        self.area.convert_unit(unit)
        self.center.convert_unit(unit)    
        self.unit = unit  

    def copy(self) -> "DeltaFunctionComponent":
        """
        Return a deep copy of this component with independent parameters.
        """
        ModelCopy = DeltaFunctionComponent(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            unit=self.unit
        )
        ModelCopy.area.fixed = self.area.fixed
        ModelCopy.center.fixed = self.center.fixed
        return ModelCopy

    def __repr__(self):
        return f"DeltaFunctionComponent(name={self.name}, area={self.area}, center={self.center})"

class DHOComponent(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (float): Resonance frequency.
        width (float): Damping constant, approximately the HWHM of the peaks.
        area (float): Area of DHO.
    """

    def __init__(self, name='DHO', center=1.0, width=1.0, area=1.0,unit='meV'):
        # Validate inputs
        if not isinstance(area, (Number, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")
        if not isinstance(center, (Number, Parameter)):
            raise TypeError("center must be a number or an EasyScience Parameter.")
        if not isinstance(width, (Number, Parameter)):
            raise TypeError("width must be a number or an EasyScience Parameter.")
        if isinstance(width, Number):
            width=float(width)
        if isinstance(area, Number):
            area = float(area)
        if isinstance(center, Number):
            center = float(center)
        if width <= 0:
            raise ValueError("Width must be greater than 0 for DHO.")
        if area < 0:
            raise Warning("The area of the DHO with name {} is negative, which may not be physically meaningful.".format(name))
        
        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if isinstance(center, Number):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(width, Number):
            self.width = Parameter(name=name + ' width', value=width, unit=unit)
        else:
            self.width = width

        if isinstance(area, Number):
            self.area = Parameter(name=name + ' area', value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x):

        if self.width.value <= 0:
            raise ValueError("Width must be greater than 0 for DHO.")
        if self.area.value < 0:
            raise Warning("The area of the DHO with name {} is negative, which may not be physically meaningful.".format(self.name))
        return 2*self.area.value*self.center.value**2*self.width.value/np.pi/ (
            (x**2 - self.center.value**2) ** 2 + (2*self.width.value * x) ** 2
        )
    
    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]
    
    def copy(self) -> "DHOComponent":



        ModelCopy = DHOComponent(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit
        )
        ModelCopy.area.fixed = self.area.fixed
        ModelCopy.center.fixed = self.center.fixed
        ModelCopy.width.fixed = self.width.fixed
        return ModelCopy


    def __repr__(self):
        return f"DHOComponent(name={self.name}, area={self.area}, center={self.center}, width={self.width})"

class PolynomialComponent(ModelComponent):
    """
    Polynomial function component.

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(self, name='Polynomial', coefficients: list[float] = [0.0]):
        if not isinstance(coefficients,(list,tuple)):
            raise TypeError("coefficients must be a list or tuple of floats.")
        
        super().__init__(name=name)
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        self.coefficients = [
            Parameter(name=f"{name}_c{i}", value=coef)
            for i, coef in enumerate(coefficients)
        ]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x, dtype=float)
        for i, param in enumerate(self.coefficients):
            result += param.value * np.power(x, i)

        if any(result < 0):
            warnings.warn("The polynomial with name {} has negative values, which may not be physically meaningful.".format(self.name))
        return result

    def degree(self):
        return len(self.coefficients) - 1
    
    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return self.coefficients
    
    def copy(self) -> "PolynomialComponent":
        """
        Return a deep copy of this component with independent parameters.
        """

        ModelCopy = PolynomialComponent(
            name=self.name,
            coefficients=[param.value for param in self.coefficients]
        )
        for i, param in enumerate(ModelCopy.coefficients):
            param.fixed = self.coefficients[i].fixed
        return ModelCopy

    def __repr__(self):
        coeffs_str = ', '.join(f"{param.name}={param.value}" for param in self.coefficients)
        return f"PolynomialComponent(name={self.name}, coefficients=[{coeffs_str}])"
    
    def convert_unit(self, unit):
        raise ValueError("PolynomialComponent does not support unit conversion. Coefficients are dimensionless.")




class UserDefinedComponent(ModelComponent):
    """
    User-defined model component, defined via a custom function.

    Args:
        func (Callable): Function accepting (x, params) and returning np.ndarray.
        params (dict): Parameters passed to the function.
    """

    def __init__(self, name, func: Callable[[np.ndarray, Dict], np.ndarray], params: Dict):
        super().__init__(name=name)
        self.func = func
        self.params = params

    def evaluate(self, x):
        return self.func(x, self.params)
