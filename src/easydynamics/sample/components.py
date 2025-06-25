from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Dict

from scipy.special import voigt_profile

from easyscience.variable import Parameter 

from easyscience.base_classes import ObjBase


#TODO: Allow specification of units for parameters in components
#TODO: Handle area and amplitude if user specifies area

class ModelComponent(ObjBase):
    """
    Abstract base class for all model components.
    """

    def __init__(self, name='ModelComponent'):
        super().__init__(name=name)

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

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class GaussianComponent(ModelComponent):
    """
    Gaussian function.

    Args:
        area (float): area of the Gaussian.
        center (float): Center of the Gaussian. If None, defaults to 0 and is fixed
        width (float): Standard deviation.
    """

    def __init__(self, name='Gaussian', area=1.0, center=None, width=1.0, unit='meV'):
        super().__init__(name=name)

        if center is None:
            self.center = Parameter(name= name+ 'center', value=0.0, unit=unit,fixed=True)
        else:
            self.center = Parameter(name=name+ 'center', value=center, unit=unit)

        self.width = Parameter(name=name+ 'width', value=width, unit=unit,min=0.0)

        self.area = Parameter(name=name+ 'area', value=area, unit=unit,min=0.0)

    def evaluate(self, x):
        #TODO: Handle units properly
        return self.area.value * 1/(np.sqrt(2 * np.pi) * self.width.value) * np.exp(-0.5 * ((x - self.center.value) / self.width.value) ** 2)

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]

    def __repr__(self):
        return f"GaussianComponent(name={self.name}, area={self.area}, center={self.center}, width={self.width})"


class LorentzianComponent(ModelComponent):
    """
    Lorentzian function.

    Args:
        area (float): Area of the Lorentzian.
        center (float): Peak center.
        width (float): HWHM (Half Width at Half Maximum).
    """

    def __init__(self, name='Lorentzian', area=1.0, center=None, width=1.0, unit='meV'):
        super().__init__(name=name)

        if center is None:
            self.center = Parameter(name=name + 'center', value=0.0, unit=unit, fixed=True)
        else:
            self.center = Parameter(name=name + 'center', value=center, unit=unit)
        self.width = Parameter(name=name + 'width', value=width, unit=unit,min=0.0)

        self.area = Parameter(name=name + 'area', value=area, unit=unit,min=0.0)

    def evaluate(self, x):
            #TODO: Handle units properly
        return self.area.value * (self.width.value/np.pi / ((x - self.center.value)**2 + self.width.value**2))


    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.width]

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
        super().__init__(name=name)
        if center is None:
            self.center = Parameter(name=name + 'center', value=0.0, unit=unit, fixed=True)
        else:
            self.center = Parameter(name=name + 'center', value=center, unit=unit)

        self.Gwidth = Parameter(name=name + 'Gwidth', value=Gwidth, unit=unit)
        self.Lwidth = Parameter(name=name + 'Lwidth', value=Lwidth, unit=unit)
        self.area = Parameter(name=name + 'area', value=area, unit=unit)

    def evaluate(self, x):
        return self.area.value * voigt_profile(x - self.center.value, self.Gwidth.value, self.Lwidth.value)    

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center, self.Gwidth, self.Lwidth]

    def __repr__(self):
        return f"VoigtComponent(name={self.name}, area={self.area}, center={self.center}, Gwidth={self.Gwidth}, Lwidth={self.Lwidth})"


class DHOComponent(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (float): Resonance frequency.
        width (float): Damping constant, approximately the HWHM of the peaks.
        area (float): Area of DHO.
    """

    def __init__(self, name='DHO', center=1.0, width=1.0, area=1.0,unit='meV'):
        super().__init__(name=name)
        self.center = Parameter(name=name + 'center', value=center, unit=unit)
        self.width = Parameter(name=name + 'width', value=width, unit=unit)
        self.area = Parameter(name=name + 'area', value=area, unit=unit)

    def evaluate(self, x):
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


    def __repr__(self):
        return f"DHOComponent(name={self.name}, area={self.area}, center={self.center}, width={self.width})"

class PolynomialComponent(ModelComponent):
    """
    Polynomial function component.

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(self, name='Polynomial', coefficients=(0.0,)):
        super().__init__(name=name)
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        self.coefficients = [
            Parameter(name=f"{name}_c{i}", value=coef, unit='meV')
            for i, coef in enumerate(coefficients)
        ]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x, dtype=float)
        for i, param in enumerate(self.coefficients):
            result += param.value * np.power(x, i)
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

    def __repr__(self):
        coeffs_str = ', '.join(f"{param.name}={param.value}" for param in self.coefficients)
        return f"PolynomialComponent(name={self.name}, coefficients=[{coeffs_str}])"



class DeltaFunctionComponent(ModelComponent):
    """
    Delta function.

    Args:
        center (float): Mean of the Gaussian.
        width (float): Standard deviation.
        amplitude (float): Peak height or
        area (float): Total area under the curve.
    """

    def __init__(self, name='DeltaFunction', center=None, area=1.0, unit='meV'):
        super().__init__(name=name)
        if center is None:
            self.center = Parameter(name=name + 'center', value=0.0, unit=unit, fixed=True)
        else:
            self.center = Parameter(name=name + 'center', value=center, unit=unit)
        self.area = Parameter(name=name + 'area', value=area, unit=unit,min=0.0)


    def evaluate(self, x):
        #TODO: Handle units properly. Also handle area if we want users to be able to plot it without resolution convolution
        return 0*x
    
    
    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.area, self.center]

    def __repr__(self):
        return f"DeltaFunctionComponent(name={self.name}, area={self.area}, center={self.center})"


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


def DetailedBalance(omega_meV, temperature_K):
    """
    Compute ω * (n + 1), where n is the Bose-Einstein occupation number.
    
    This expression arises in detailed balance factors in neutron and light scattering.

    Parameters
    ----------
    omega_meV : float or np.ndarray
        Energy transfer (ω) in meV.
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
