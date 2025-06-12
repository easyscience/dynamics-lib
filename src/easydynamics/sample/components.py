from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Dict

from scipy.special import voigt_profile

from easyscience.Objects.variable import Parameter 

from easyscience.Objects.ObjectClasses import BaseObj


#TODO: Allow specification of units for parameters in components
#TODO: Handle area and amplitude if user specifies area

class ModelComponent(BaseObj):
    """
    Abstract base class for all model components.
    """

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


class GaussianComponent(ModelComponent):
    """
    Gaussian function.

    Args:
        center (float): Mean of the Gaussian.
        width (float): Standard deviation.
        amplitude (float): Peak height or
        area (float): Total area under the curve.
    """

    def __init__(self, center=None, width=1.0, amplitude=None, area=None,unit='meV'):
        if center is None:
            self.center = Parameter(name='Gcenter', value=0.0, unit=unit,fixed=True)
        else:
            self.center = Parameter(name='Gcenter', value=center, unit=unit)

        self.width = Parameter(name='Gwidth', value=width, unit=unit)
            
        if amplitude is not None:
            self.amplitude = Parameter(name='Gamplitude', value=amplitude)
        elif area is not None:
            self.amplitude = Parameter(name='Gamplitude', value=area / (width * np.sqrt(2 * np.pi)))
        else:
            raise ValueError("Must provide either amplitude or area")

    def evaluate(self, x):
        #TODO: Handle units properly
        return self.amplitude.value * np.exp(-0.5 * ((x - self.center.value) / self.width.value) ** 2)

    @property
    def area(self):
        #TODO: Handle units properly
        return self.amplitude.value * self.width.value * np.sqrt(2 * np.pi)

    @area.setter
    def area(self, value):
        #TODO: Handle units properly
        self.amplitude.value = value / (self.width.value * np.sqrt(2 * np.pi))

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.center, self.width, self.amplitude]


class LorentzianComponent(ModelComponent):
    """
    Lorentzian function.

    Args:
        center (float): Peak center.
        width (float): HWHM (Half Width at Half Maximum).
        amplitude (float): Peak height or
        area (float): Total area under the curve.
    """

    def __init__(self, center=None, width=1.0, amplitude=None, area=None,unit='meV'):
        if center is None:
            self.center = Parameter(name='Lcenter', value=0.0, unit=unit,fixed=True)
        else:
            self.center = Parameter(name='Lcenter', value=center, unit=unit)
        self.width = Parameter(name='Lwidth', value=width, unit=unit)
            
        if amplitude is not None:
            self.amplitude = Parameter(name='Lamplitude', value=amplitude)
        elif area is not None:
            self.amplitude = Parameter(name='Lamplitude', value=area / (np.pi * self.width.value))
        else:
            raise ValueError("Must provide either amplitude or area")

    def evaluate(self, x):
            #TODO: Handle units properly
        return self.amplitude.value * (self.width.value**2 / ((x - self.center.value)**2 + self.width.value**2))

    @property
    def area(self):
        #TODO: Handle units properly
        return self.amplitude.value * np.pi * self.width.value

    @area.setter
    def area(self, value):
        #TODO: Handle units properly
        self.amplitude.value = value / (np.pi * self.width.value)

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.center, self.width, self.amplitude]


class VoigtComponent(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (float): Center of the Voigt profile.
        width (float): Standard deviation of the Gaussian part.
        gamma (float): HWHM of the Lorentzian part.
        area (float): Total area under the curve.
    """

    def __init__(self, center=None, Gwidth=1.0, Lwidth=1.0, area=1.0, unit='meV'):
        if center is None:
            self.center = Parameter(name='Vcenter', value=0.0, unit=unit,fixed=True)
        else:
            self.center = Parameter(name='Vcenter', value=center, unit=unit)
        self.Gwidth = Parameter(name='VGwidth', value=Gwidth, unit=unit)
        self.Lwidth = Parameter(name='VWidth', value=Lwidth, unit=unit)
        self.area = Parameter(name='Varea', value=area)

    def evaluate(self, x):
        return self.area.value * voigt_profile(x - self.center.value, self.Gwidth.value, self.Lwidth.value)    

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.center, self.Gwidth, self.Lwidth, self.area]


class DHOComponent(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (float): Resonance frequency.
        width (float): Damping constant, approximately the HWHM of the peaks.
        area (float): Area of DHO.
    """

    def __init__(self, center=1.0, width=1.0, area=1.0,unit='meV'):
        self.center = Parameter(name='center', value=center, unit=unit)
        self.width = Parameter(name='width', value=width, unit=unit)
        self.area = Parameter(name='area', value=area)

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
        return [self.center, self.width, self.area]


class PolynomialComponent(ModelComponent):
    """
    Polynomial function component.

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(self, coefficients=(0.0,)):
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        self.coefficients = [
            Parameter(name=f"c{i}", value=coef)
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



class DeltaFunctionComponent(ModelComponent):
    """
    Delta function.

    Args:
        center (float): Mean of the Gaussian.
        width (float): Standard deviation.
        amplitude (float): Peak height or
        area (float): Total area under the curve.
    """

    def __init__(self, center=None, area=1.0,unit='meV'):
        if center is None:
            self.center = Parameter(name='Dcenter', value=0.0, unit=unit,fixed=True)
        else:
            self.center = Parameter(name='Dcenter', value=center, unit=unit)
            self.area = Parameter(name='Darea', value=area, unit=unit)


    def evaluate(self, x):
        #TODO: Handle units properly
        return self.area.value if x==0 else 0
    
    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """ 
        return [self.center, self.area]




class UserDefinedComponent(ModelComponent):
    """
    User-defined model component, defined via a custom function.

    Args:
        func (Callable): Function accepting (x, params) and returning np.ndarray.
        params (dict): Parameters passed to the function.
    """

    def __init__(self, func: Callable[[np.ndarray, Dict], np.ndarray], params: Dict):
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
