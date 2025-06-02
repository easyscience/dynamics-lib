from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Dict


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

    def __init__(self, center=0.0, width=1.0, amplitude=1.0, area=None):
        self.center = center
        self.width = width
        if amplitude is not None:
            self.amplitude = amplitude
        elif area is not None:
            self.amplitude = area / (width * np.sqrt(2 * np.pi))
        else:
            raise ValueError("Must provide either amplitude or area")

    def evaluate(self, x):
        return self.amplitude * np.exp(-0.5 * ((x - self.center) / self.width) ** 2)

    @property
    def area(self):
        return self.amplitude * self.width * np.sqrt(2 * np.pi)

    @area.setter
    def area(self, value):
        self.amplitude = value / (self.width * np.sqrt(2 * np.pi))


class LorentzianComponent(ModelComponent):
    """
    Lorentzian function.

    Args:
        center (float): Peak center.
        width (float): HWHM (Half Width at Half Maximum).
        amplitude (float): Peak height or
        area (float): Total area under the curve.
    """

    def __init__(self, center=0.0, width=1.0, amplitude=1.0, area=None):
        self.center = center
        self.width = width
        if amplitude is not None:
            self.amplitude = amplitude
        elif area is not None:
            self.amplitude = area / (np.pi * width)
        else:
            raise ValueError("Must provide either amplitude or area")

    def evaluate(self, x):
        return self.amplitude * (self.width**2 / ((x - self.center)**2 + self.width**2))

    @property
    def area(self):
        return self.amplitude * np.pi * self.width

    @area.setter
    def area(self, value):
        self.amplitude = value / (np.pi * self.width)


class DHOComponent(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (float): Resonance frequency.
        gamma (float): Damping constant.
        amplitude (float): Scaling factor.
    """

    def __init__(self, center=1.0, gamma=1.0, amplitude=1.0):
        self.center = center
        self.gamma = gamma
        self.amplitude = amplitude

    def evaluate(self, x):
        return self.amplitude * (self.gamma * x) / (
            (x**2 - self.center**2) ** 2 + (self.gamma * x) ** 2
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
