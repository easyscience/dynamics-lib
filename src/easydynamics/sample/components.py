from abc import abstractmethod
from typing import Callable, Dict, Union, List, Tuple

import numpy as np
from scipy.special import voigt_profile

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

import warnings

from numbers import Number
from numbers import Real

import scipp as sc


class ModelComponent(ObjBase):
    """
    Abstract base class for all model components.
    """

    def __init__(self, name='ModelComponent'):
        super().__init__(name=name)
        self.unit=None  

    def fix_all_parameters(self):
        """Fix all parameters in the model component."""

        pars = self.get_parameters()
        if pars is None or len(pars) == 0:
            raise ValueError("No parameters found to fix.")
        else:
            for p in pars:
                p.fixed = True

    def fit_all_parameters(self):
        """Fit all parameters in the model component."""
        for p in self.get_parameters():
            p.fixed = False

    def get_parameter(self, parameter_name: str) -> Parameter:
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

    def set_parameter_value(self, parameter_name: str, value: float, unit: str = None):
        """
        Set the value of a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        if unit is not None:
            param.convert_unit(unit)
        param.value = value

    def set_parameter_bounds(self, parameter_name: str, min: Union[float,None] = None, max: Union[float, None] = None, unit: str = None):
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

    def fix_parameter(self, parameter_name: str):
        """
        Fix a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        param.fixed = True

    def free_parameter(self, parameter_name: str):
        """
        Free a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        param.fixed = False

    def convert_unit(self, unit: str):
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
    def evaluate(self, x: Union[float,np.ndarray,sc.Variable]) -> np.ndarray:
        """
        Evaluate the model component at input x.

        Args:
            x (Union[float, np.ndarray, sc.Variable]): Input values.

        Returns:
            np.ndarray: Evaluated function values.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[Parameter]:
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


class Gaussian(ModelComponent):
    """
    Gaussian function. Creates new EasyScience Parameters if floats are provided, otherwise uses the provided Parameters.

    Args:
        area (float): Area of the Gaussian. Has the same unit as the x axis
        center (float): Center of the Gaussian. If None, defaults to 0 and is fixed
        width (float): Standard deviation.
    """

    def __init__(self, 
                 name: str='Gaussian', 
                 area: Union[float,Parameter]=1.0, 
                 center: Union[float,Parameter,None]=None, 
                 width: Union[float,Parameter]=1.0, 
                 unit: str='meV'):

        # Validate inputs - throw errors before any Parameters are created
        if not isinstance(area, (Real, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")

        if center is not None and not isinstance(center, (Real, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")

        if not isinstance(width, (Real, Parameter)):
            raise TypeError("width must be a number or an EasyScience Parameter.")

        if isinstance(width, Real):
            width=float(width)
            if width <= 0:
                raise ValueError("The width of a Gaussian must be greater than zero.")

        if isinstance(area, Real):
            if area < 0:
                warnings.warn("The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)

        if isinstance(center, Real):
            center = float(center)
        
        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component

        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name= name+ ' center', value=0.0, unit=unit,fixed=True)
        elif isinstance(center, Real):
            self.center = Parameter(name=name+ ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(width, Real):
            self.width = Parameter(name=name+ ' width', value=width, unit=unit,min=0.0)
        else:
            self.width=width

        if isinstance(area, Real):
            self.area = Parameter(name=name+ ' area', value=area, unit=unit)
        else:
            self.area=area

    def evaluate(self, x: Union[float,np.ndarray,sc.Variable]) -> Union[float,np.ndarray]:
        if self.width.value <= 0:
            raise ValueError("The width of a Gaussian must be greater than zero.")
        if self.area.value < 0:
            warnings.warn("The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(self.name))

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(f"Input x has unit {x.unit}, but Gaussian component has unit {self.unit}. Converting Gaussian to {x.unit}.")
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return self.area.value * 1/(np.sqrt(2 * np.pi) * self.width.value) * np.exp(-0.5 * ((x_in - self.center.value) / self.width.value) ** 2)


    def get_parameters(self) -> List[Parameter]:
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]
    
    def copy(self) -> "Gaussian":
        """
        Return a deep copy of this component with independent parameters.
        """

        ModelCopy=Gaussian(
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
        return f"Gaussian(name={self.name}, area={self.area}, center={self.center}, width={self.width})"


class Lorentzian(ModelComponent):
    """
    Lorentzian function. Creates new EasyScience Parameters if floats are provided, otherwise uses the provided Parameters.

    Args:
        area (float or Parameter): Area of the Lorentzian.
        center (float or Parameter or None): Peak center. If None, defaults to 0 and is fixed.
        width (float or Parameter): Half Width at Half Maximum (HWHM)
    """

    def __init__(self, 
                 name: str = 'Lorentzian', 
                 area: Union[float, Parameter] = 1.0, 
                 center: Union[float, Parameter, None] = None, 
                 width: Union[float, Parameter] = 1.0, 
                 unit: str = 'meV'):

        
        # Validate inputs
        if not isinstance(area, (Real, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")

        if center is not None and not isinstance(center, (Real, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")

        if not isinstance(width, (Real, Parameter)):
            raise TypeError("width must be a number or an EasyScience Parameter.")

        if isinstance(width, Real):
            if width <= 0:
                raise ValueError("The width of a Lorentzian must be greater than zero.")
            width=float(width)

        if isinstance(area, Real):
            if area < 0:
                warnings.warn("The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)

        if isinstance(center, Real):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component

        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name=name + ' center', value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Real):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(width, Real):
            self.width = Parameter(name=name + ' width', value=width, unit=unit,min=0.0)
        else:
            self.width=width

        if isinstance(area, Real):
            self.area = Parameter(name=name + ' area', value=area, unit=unit)
        else:
            self.area=area

    def evaluate(self, x:Union[float,np.ndarray,sc.Variable]) -> Union[float,np.ndarray]:
        if self.width.value <= 0:
            raise ValueError("Width must be greater than 0 for Lorentzian.")
        if self.area.value < 0:
            warnings.warn("The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(self.name))

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(f"Input x has unit {x.unit}, but Lorentzian component has unit {self.unit}. Converting Lorentzian to {x.unit}.")
                self.convert_unit(x.unit.name)
        else:
            x_in = x    
        return self.area.value * (self.width.value/np.pi / ((x_in - self.center.value)**2 + self.width.value**2))


    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]
    
    def copy(self) -> "Lorentzian":

        ModelCopy =Lorentzian(
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
        return f"Lorentzian(name={self.name}, area={self.area}, center={self.center}, width={self.width})"


class Voigt(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (float): Center of the Voigt profile.
        Gwidth (float): Standard deviation of the Gaussian part.
        Lwidth (float): HWHM of the Lorentzian part.
        area (float): Total area under the curve.
    """

    def __init__(self, 
                 name: str = 'Voigt', 
                 area: Union[float, Parameter] = 1.0, 
                 center: Union[float, Parameter, None] = None, 
                 Gwidth: Union[float, Parameter] = 1.0, 
                 Lwidth: Union[float, Parameter] = 1.0, 
                 unit: str = 'meV'):
        
        # Validate inputs
        if not isinstance(area, (Real, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")

        if center is not None and not isinstance(center, (Real, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")
        
        if not isinstance(Gwidth, (Real, Parameter)):
            raise TypeError("Gwidth must be a number or an EasyScience Parameter.")
        
        if not isinstance(Lwidth, (Real, Parameter)):
            raise TypeError("Lwidth must be a number or an EasyScience Parameter.")
        
        if isinstance(Gwidth, Real):
            if Gwidth <= 0:
                raise ValueError("Gwidth must be greater than 0 for Voigt profile.")
            Gwidth=float(Gwidth)

        if isinstance(Lwidth, Real):
            if Lwidth <= 0:
                raise ValueError("Lwidth must be greater than 0 for Voigt profile.")
            Lwidth=float(Lwidth)

        if isinstance(area, Real):
            if area < 0:
                warnings.warn("The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)
        
        super().__init__(name=name)


        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name=name + ' center', value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Real):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(Gwidth, Real):
            self.Gwidth = Parameter(name=name + ' Gwidth', value=Gwidth, unit=unit,min=0.0)
        else:
            self.Gwidth=Gwidth

        if isinstance(Lwidth, Real):
            self.Lwidth = Parameter(name=name + ' Lwidth', value=Lwidth, unit=unit,min=0.0)
        else:
            self.Lwidth=Lwidth

        if isinstance(area, Real):
            self.area = Parameter(name=name + ' area', value=area, unit=unit)
        else:
            self.area=area

    def evaluate(self, x):
        if self.Gwidth.value <= 0:
            raise ValueError("Gwidth must be greater than 0 for Voigt profile.")
        if self.Lwidth.value <= 0:
            raise ValueError("Lwidth must be greater than 0 for Voigt profile.")
        if self.area.value < 0:
            warnings.warn("The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(self.name))

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(f"Input x has unit {x.unit}, but Voigt component has unit {self.unit}. Converting Voigt to {x.unit}.")
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return self.area.value * voigt_profile(x_in - self.center.value, self.Gwidth.value, self.Lwidth.value)

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.
        
        Args:
            unit (str): The new unit to convert to.
        """
        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.Gwidth.convert_unit(unit)
        self.Lwidth.convert_unit(unit)
        self.unit = unit

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.Gwidth, self.Lwidth]
    
    def copy(self) -> "Voigt":

        ModelCopy = Voigt(
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
        return f"Voigt(name={self.name}, area={self.area}, center={self.center}, Gwidth={self.Gwidth}, Lwidth={self.Lwidth})"


class DeltaFunction(ModelComponent):
    """
    Delta function. Evaluates to zero everywhere, except in convolutions, where it acts as an identity. This is handled in the ResolutionHandler.

    Args:
        center (float): Center of the delta function.
        area (float): Total area under the curve.
    """

    def __init__(self, 
                 name:str='DeltaFunction', 
                 center:Union[None, float, Parameter]=None, 
                 area:Union[float, Parameter]=1.0, 
                 unit='meV'):
        # Validate inputs
        if not isinstance(area, (Real, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")
        
        if center is not None and not isinstance(center, (Real, Parameter)):
            raise TypeError("center must be None, a number or an EasyScience Parameter.")
        
        if isinstance(area, Real):
            if area < 0:
                warnings.warn("The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(name))
            area = float(area)

        if isinstance(center, Real):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(name=name + ' center', value=0.0, unit=unit, fixed=True)
        elif isinstance(center, Real):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(area, Real):
            self.area = Parameter(name=name + ' area', value=area, unit=unit,min=0.0)
        else:
            self.area=area


    def evaluate(self, x):

        if self.area.value < 0:
            warnings.warn("The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(self.name))
        #TODO: Consider adding support for evaluation without resolution convolution
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

    def copy(self) -> "DeltaFunction":
        """
        Return a deep copy of this component with independent parameters.
        """
        ModelCopy = DeltaFunction(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            unit=self.unit
        )
        ModelCopy.area.fixed = self.area.fixed
        ModelCopy.center.fixed = self.center.fixed
        return ModelCopy

    def __repr__(self):
        return f"DeltaFunction(name={self.name}, area={self.area}, center={self.center})"

class DampedHarmonicOscillator(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (float): Resonance frequency.
        width (float): Damping constant, approximately the HWHM of the peaks.
        area (float): Area of DHO.
    """

    def __init__(self, 
                 name: str = 'DHO', 
                 center: Union[float, Parameter] = 1.0, 
                 width: Union[float, Parameter] = 1.0,
                 area: Union[float, Parameter] = 1.0, 
                 unit: str = 'meV'):
        # Validate inputs
        if not isinstance(area, (Real, Parameter)):
            raise TypeError("area must be a number or an EasyScience Parameter.")
        
        if not isinstance(center, (Real, Parameter)):
            raise TypeError("center must be a number or an EasyScience Parameter.")
        
        if not isinstance(width, (Real, Parameter)):
            raise TypeError("width must be a number or an EasyScience Parameter.")
        
        if isinstance(width, Real):
            width=float(width)

        if isinstance(area, Real):
            area = float(area)

        if isinstance(center, Real):
            center = float(center)

        if width <= 0:
            raise ValueError("Width of a Damped Harmonic Oscillator must be greater than 0.")
        if area < 0:
            raise Warning("The area of the Damped Harmonic Oscillator with name {} is negative, which may not be physically meaningful.".format(name))
        
        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if isinstance(center, Real):
            self.center = Parameter(name=name + ' center', value=center, unit=unit)
        else:
            self.center=center

        if isinstance(width, Real):
            self.width = Parameter(name=name + ' width', value=width, unit=unit,min=0.0)
        else:
            self.width = width

        if isinstance(area, Real):
            self.area = Parameter(name=name + ' area', value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[float,np.ndarray,sc.Variable]) -> Union[float,np.ndarray]:

        if self.width.value <= 0:
            raise ValueError("Width of a Damped Harmonic Oscillator must be greater than 0.")
        if self.area.value < 0:
            raise Warning("The area of the Damped Harmonic Oscillator with name {} is negative, which may not be physically meaningful.".format(self.name))
        
        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(f"Input x has unit {x.unit}, but DHO component has unit {self.unit}. Converting DHO to {x.unit}.")
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return 2*self.area.value*self.center.value**2*self.width.value/np.pi/ (
            (x_in**2 - self.center.value**2) ** 2 + (2*self.width.value * x_in) ** 2
        )
    
    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]
    
    def copy(self) -> "DampedHarmonicOscillator":
        """
        Return a deep copy of this component with independent parameters.
        """


        ModelCopy = DampedHarmonicOscillator(
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
        return f"DampedHarmonicOscillator(name={self.name}, area={self.area}, center={self.center}, width={self.width})"

class Polynomial(ModelComponent):
    """
    Polynomial function component. Supports units, but not conversion between units.

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(self, 
                 name: str='Polynomial', 
                 coefficients: Union[list[float],np.ndarray] = [0.0],
                 unit: str = 'meV'):
        if not isinstance(coefficients,(list,tuple,np.ndarray)):
            raise TypeError("coefficients must be a list, tuple or ndarray of floats.")
        
        super().__init__(name=name)
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        self.coefficients = [
        Parameter(
        name=f"{name}_c{i}",
        value=coef,    )
    for i, coef in enumerate(coefficients)
        ]
        self.unit = unit  

    def evaluate(self, x: Union[float,np.ndarray,sc.Variable]) -> np.ndarray:

        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                raise ValueError(f"Input x has unit {x.unit}, but DHO component has unit {self.unit}. Change the unit of the DHO and try again. ")
        else:
            x_in = x
        result = np.zeros_like(x_in, dtype=float)
        for i, param in enumerate(self.coefficients):
            result += param.value * np.power(x_in, i)

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
    
    def copy(self) -> "Polynomial":
        """
        Return a deep copy of this component with independent parameters.
        """

        ModelCopy = Polynomial(
            name=self.name,
            coefficients=[param.value for param in self.coefficients]
        )
        for i, param in enumerate(ModelCopy.coefficients):
            param.fixed = self.coefficients[i].fixed
        return ModelCopy

    def __repr__(self):
        coeffs_str = ', '.join(f"{param.name}={param.value}" for param in self.coefficients)
        return f"Polynomial(name={self.name}, coefficients=[{coeffs_str}])"
    
    def convert_unit(self, unit):
        raise NotImplementedError("Unit conversion is not implemented for Polynomial components. The automatic unit converter does not like powers of units. ")




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
