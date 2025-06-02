from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Dict

from scipy.special import voigt_profile

from easyscience.Objects.variable import Parameter 

#TODO: Allow specification of units for parameters in components
#TODO: Handle area and amplitude if user specifies area

class ModelComponent(ABC):
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

    def __init__(self, center=0.0, width=1.0, amplitude=None, area=None,unit='meV'):
        self.center = Parameter(name='center', value=center, unit=unit)
        self.width = Parameter(name='width', value=width, unit=unit)
            
        if amplitude is not None:
            self.amplitude = Parameter(name='amplitude', value=amplitude)
        elif area is not None:
            self.amplitude = Parameter(name='amplitude', value=area / (width * np.sqrt(2 * np.pi)))
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


class LorentzianComponent(ModelComponent):
    """
    Lorentzian function.

    Args:
        center (float): Peak center.
        width (float): HWHM (Half Width at Half Maximum).
        amplitude (float): Peak height or
        area (float): Total area under the curve.
    """

    def __init__(self, center=0.0, width=1.0, amplitude=None, area=None,unit='meV'):
        self.center = Parameter(name='center', value=center, unit=unit)
        self.width = Parameter(name='width', value=width, unit=unit)
            
        if amplitude is not None:
            self.amplitude = Parameter(name='amplitude', value=amplitude)
        elif area is not None:
            self.amplitude = Parameter(name='amplitude', value=area / (np.pi * self.width.value))
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

class VoigtComponent(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (float): Center of the Voigt profile.
        width (float): Standard deviation of the Gaussian part.
        gamma (float): HWHM of the Lorentzian part.
        area (float): Total area under the curve.
    """

    def __init__(self, center=0.0, Gwidth=1.0, Lwidth=1.0, area=1.0, unit='meV'):
        self.center = Parameter(name='center', value=center, unit=unit)
        self.Gwidth = Parameter(name='Gwidth', value=Gwidth, unit=unit)
        self.Lwidth = Parameter(name='LWidth', value=Lwidth, unit=unit)
        self.area = Parameter(name='area', value=area)

    def evaluate(self, x):
        return self.area.value * voigt_profile(x - self.center.value, self.Gwidth.value, self.Lwidth.value)    



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


class PolynomialComponent(ModelComponent):
    """
    Polynomial function component.

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(self, coefficients=0.0):
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")
        self.coefficients = list(coefficients)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x, dtype=float)
        for i, coef in enumerate(self.coefficients):
            result += coef * np.power(x, i)
        return result

    def degree(self):
        return len(self.coefficients) - 1


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
