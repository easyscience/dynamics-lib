from abc import abstractmethod

from typing import Union, List, Optional


import numpy as np
from scipy.special import voigt_profile

from easyscience.variable import Parameter
from easyscience.base_classes import ObjBase

import warnings

import scipp as sc

Numeric = Union[float, int]


class ModelComponent(ObjBase):
    """
    Abstract base class for all model components.
    """

    def __init__(self, name="ModelComponent"):
        super().__init__(name=name)
        self.unit = None

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
            raise ValueError(
                f"Ambiguous parameter name '{parameter_name}' matches multiple parameters: {[p.name for p in matches]}"
            )
        else:
            raise ValueError(f"Parameter '{parameter_name}' not found.")

    def set_parameter_value(
        self, parameter_name: str, value: float, unit: Optional[str] = None
    ):
        """
        Set the value of a specific parameter by name.
        """
        param = self.get_parameter(parameter_name)
        if unit is not None:
            param.convert_unit(unit)
        param.value = value

    def set_parameter_bounds(
        self,
        parameter_name: str,
        min: Union[float, None] = None,
        max: Union[float, None] = None,
        unit: Optional[str] = None,
    ):
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

    @abstractmethod
    def evaluate(self, x: Union[Numeric, sc.Variable]) -> np.ndarray:
        """
        Evaluate the model component at input x.

        Args:
            x (Union[Numeric, sc.Variable]): Input values.

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
        area (Numeric or Parameter): Area of the Gaussian. Has the same unit as the x axis
        center (Numeric or Parameter or None): Center of the Gaussian. If None, defaults to 0 and is fixed
        width (Numeric or Parameter): Standard deviation.
    """

    def __init__(
        self,
        name: str = "Gaussian",
        area: Union[Numeric, Parameter] = 1.0,
        center: Union[Numeric, Parameter, None] = None,
        width: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs - throw errors before any Parameters are created
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(width, Numeric):
            if width <= 0:
                raise ValueError("The width of a Gaussian must be greater than zero.")
            width = float(width)

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )
            area = float(area)

        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component

        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        elif isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(width, Numeric):
            self.width = Parameter(
                name=name + " width", value=width, unit=unit, min=0.0
            )
        else:
            self.width = width

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.width.value <= 0:
            raise ValueError("The width of a Gaussian must be greater than zero.")
        if self.area.value < 0:
            warnings.warn(
                "The area of the Gaussian with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(
                    f"Input x has unit {x.unit}, but Gaussian component has unit {self.unit}. Converting Gaussian to {x.unit}."
                )
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return (
            self.area.value
            * 1
            / (np.sqrt(2 * np.pi) * self.width.value)
            * np.exp(-0.5 * ((x_in - self.center.value) / self.width.value) ** 2)
        )

    def get_parameters(self) -> List[Parameter]:
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center, self.width]

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

    def copy(self) -> "Gaussian":
        """
        Return a deep copy of this component with independent parameters.
        """

        model_copy = Gaussian(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit,
        )

        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.width.fixed = self.width.fixed
        return model_copy

    def __repr__(self):
        return f"Gaussian(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"


class Lorentzian(ModelComponent):
    """
    Lorentzian function. Creates new EasyScience Parameters if floats are provided, otherwise uses the provided Parameters.

    Args:
        area (Numeric or Parameter): Area of the Lorentzian.
        center (Numeric or Parameter or None): Peak center. If None, defaults to 0 and is fixed.
        width (Numeric or Parameter): Half Width at Half Maximum (HWHM)
    """

    def __init__(
        self,
        name: str = "Lorentzian",
        area: Union[Numeric, Parameter] = 1.0,
        center: Union[Numeric, Parameter, None] = None,
        width: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if isinstance(width, Numeric):
            if width <= 0:
                raise ValueError("The width of a Lorentzian must be greater than zero.")
            width = float(width)

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )
            area = float(area)

        if isinstance(center, Numeric):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component

        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        elif isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(width, Numeric):
            self.width = Parameter(
                name=name + " width", value=width, unit=unit, min=0.0
            )
        else:
            self.width = width

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.width.value <= 0:
            raise ValueError("The width of a Lorentzian must be greater than zero.")
        if self.area.value < 0:
            warnings.warn(
                "The area of the Lorentzian with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(
                    f"Input x has unit {x.unit}, but Lorentzian component has unit {self.unit}. Converting Lorentzian to {x.unit}."
                )
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return self.area.value * (
            self.width.value
            / np.pi
            / ((x_in - self.center.value) ** 2 + self.width.value**2)
        )

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center, self.width]

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

    def copy(self) -> "Lorentzian":
        model_copy = Lorentzian(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.width.fixed = self.width.fixed
        return model_copy

    def __repr__(self):
        return f"Lorentzian(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"


class Voigt(ModelComponent):
    """
    Voigt profile, a convolution of Gaussian and Lorentzian.

    Args:
        center (Numeric or Parameter or None): Center of the Voigt profile.
        gaussian_width (Numeric or Parameter): Standard deviation of the Gaussian part.
        lorentzian_width (Numeric or Parameter): Half width at half max (HWHM) of the Lorentzian part.
        area (Numeric or Parameter): Total area under the curve.
    """

    def __init__(
        self,
        name: str = "Voigt",
        area: Union[Numeric, Parameter] = 1.0,
        center: Union[Numeric, Parameter, None] = None,
        gaussian_width: Union[Numeric, Parameter] = 1.0,
        lorentzian_width: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(gaussian_width, (Numeric, Parameter)):
            raise TypeError("gaussian_width must be a number or a Parameter.")

        if not isinstance(lorentzian_width, (Numeric, Parameter)):
            raise TypeError("lorentzian_width must be a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(gaussian_width, Numeric):
            if gaussian_width <= 0:
                raise ValueError(
                    "The gaussian_width of a Voigt must be greater than zero."
                )
            gaussian_width = float(gaussian_width)

        if isinstance(lorentzian_width, Numeric):
            if lorentzian_width <= 0:
                raise ValueError(
                    "The lorentzian_width of a Voigt must be greater than zero."
                )
            lorentzian_width = float(lorentzian_width)

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )
            area = float(area)

        super().__init__(name=name)

        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        elif isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(gaussian_width, Numeric):
            self.gaussian_width = Parameter(
                name=name + " gaussian_width", value=gaussian_width, unit=unit, min=0.0
            )
        else:
            self.gaussian_width = gaussian_width

        if isinstance(lorentzian_width, Numeric):
            self.lorentzian_width = Parameter(
                name=name + " lorentzian_width",
                value=lorentzian_width,
                unit=unit,
                min=0.0,
            )
        else:
            self.lorentzian_width = lorentzian_width

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.gaussian_width.value <= 0:
            raise ValueError("The gaussian_width of a Voigt must be greater than zero.")
        if self.lorentzian_width.value <= 0:
            raise ValueError(
                "The lorentzian_width of a Voigt must be greater than zero."
            )
        if self.area.value < 0:
            warnings.warn(
                "The area of the Voigt profile with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(
                    f"Input x has unit {x.unit}, but Voigt component has unit {self.unit}. Converting Voigt to {x.unit}."
                )
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return self.area.value * voigt_profile(
            x_in - self.center.value,
            self.gaussian_width.value,
            self.lorentzian_width.value,
        )

    def convert_unit(self, unit: str):
        """
        Convert the unit of the Parameters in the component.

        Args:
            unit (str): The new unit to convert to.
        """
        self.area.convert_unit(unit)
        self.center.convert_unit(unit)
        self.gaussian_width.convert_unit(unit)
        self.lorentzian_width.convert_unit(unit)
        self.unit = unit

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center, self.gaussian_width, self.lorentzian_width]

    def copy(self) -> "Voigt":
        model_copy = Voigt(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            gaussian_width=self.gaussian_width.value,
            lorentzian_width=self.lorentzian_width.value,
            unit=self.unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.gaussian_width.fixed = self.gaussian_width.fixed
        model_copy.lorentzian_width.fixed = self.lorentzian_width.fixed

        return model_copy

    def __repr__(self):
        return f"Voigt(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center},\n gaussian_width = {self.gaussian_width},\n lorentzian_width = {self.lorentzian_width})"


class DeltaFunction(ModelComponent):
    """
    Delta function. Evaluates to zero everywhere, except in convolutions, where it acts as an identity. This is handled in the ResolutionHandler.

    Args:
        center (Numeric or Parameter or None): Center of the delta function. If None, defaults to 0 and is fixed.
        area (Numeric or Parameter): Total area under the curve.
    """

    def __init__(
        self,
        name: str = "DeltaFunction",
        center: Union[None, Numeric, Parameter] = None,
        area: Union[Numeric, Parameter] = 1.0,
        unit="meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if center is not None and not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be None, a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(area, Numeric):
            if area < 0:
                warnings.warn(
                    "The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )
            area = float(area)

        if isinstance(center, Numeric):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit
        # Create Parameters from floats, or set Parameters if already provided
        if center is None:
            self.center = Parameter(
                name=name + " center", value=0.0, unit=unit, fixed=True
            )
        elif isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x):
        if self.area.value < 0:
            warnings.warn(
                "The area of the Delta function with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )
        # TODO: Consider adding support for evaluation without resolution convolution
        return 0 * x

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
        model_copy = DeltaFunction(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            unit=self.unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        return model_copy

    def __repr__(self):
        return f"DeltaFunction(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center}"


class DampedHarmonicOscillator(ModelComponent):
    """
    Damped Harmonic Oscillator (DHO) component.

    Args:
        center (Numeric or Parameter): Resonance frequency, approximately the peak position.
        width (Numeric or Parameter): Damping constant, approximately the half width at half max (HWHM) of the peaks.
        area (Numeric or Parameter): Area under the curve.
    """

    def __init__(
        self,
        name: str = "DHO",
        center: Union[Numeric, Parameter] = 1.0,
        width: Union[Numeric, Parameter] = 1.0,
        area: Union[Numeric, Parameter] = 1.0,
        unit: str = "meV",
    ):
        # Validate inputs
        if not isinstance(area, (Numeric, Parameter)):
            raise TypeError("area must be a number or a Parameter.")

        if not isinstance(center, (Numeric, Parameter)):
            raise TypeError("center must be a number or a Parameter.")

        if not isinstance(width, (Numeric, Parameter)):
            raise TypeError("width must be a number or a Parameter.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        if isinstance(width, Numeric):
            width = float(width)
            if width <= 0:
                raise ValueError(
                    "The width of a DampedHarmonicOscillator must be greater than zero."
                )

        if isinstance(area, Numeric):
            area = float(area)
            if area < 0:
                warnings.warn(
                    "The area of the Damped Harmonic Oscillator with name {} is negative, which may not be physically meaningful.".format(
                        name
                    )
                )

        if isinstance(center, Numeric):
            center = float(center)

        super().__init__(name=name)
        self.unit = unit  # Set the unit for the component
        # Create Parameters from floats, or set Parameters if already provided
        if isinstance(center, Numeric):
            self.center = Parameter(name=name + " center", value=center, unit=unit)
        else:
            self.center = center

        if isinstance(width, Numeric):
            self.width = Parameter(
                name=name + " width", value=width, unit=unit, min=0.0
            )
        else:
            self.width = width

        if isinstance(area, Numeric):
            self.area = Parameter(name=name + " area", value=area, unit=unit)
        else:
            self.area = area

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> Union[float, np.ndarray]:
        if self.width.value <= 0:
            raise ValueError(
                "The width of a DampedHarmonicOscillator must be greater than zero."
            )
        if self.area.value < 0:
            warnings.warn(
                "The area of the DampedHarmonicOscillator with name {} is negative, which may not be physically meaningful.".format(
                    self.name
                )
            )

        # Handle units
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                warnings.warn(
                    f"Input x has unit {x.unit}, but DHO component has unit {self.unit}. Converting DHO to {x.unit}."
                )
                self.convert_unit(x.unit.name)
        else:
            x_in = x
        return (
            2
            * self.area.value
            * self.center.value**2
            * self.width.value
            / np.pi
            / (
                (x_in**2 - self.center.value**2) ** 2
                + (2 * self.width.value * x_in) ** 2
            )
        )

    def get_parameters(self):
        """
        Get all parameters from the model component.
        Returns:
        List[Parameter]: List of parameters in the component.
        """
        return [self.area, self.center, self.width]

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

    def copy(self) -> "DampedHarmonicOscillator":
        """
        Return a deep copy of this component with independent parameters.
        """

        model_copy = DampedHarmonicOscillator(
            name=self.name,
            area=self.area.value,
            center=self.center.value,
            width=self.width.value,
            unit=self.unit,
        )
        model_copy.area.fixed = self.area.fixed
        model_copy.center.fixed = self.center.fixed
        model_copy.width.fixed = self.width.fixed
        return model_copy

    def __repr__(self):
        return f"DampedHarmonicOscillator(name = {self.name}, unit = {self.unit},\n area = {self.area},\n center = {self.center},\n width = {self.width})"


class Polynomial(ModelComponent):
    """
    Polynomial function component. Supports units, but not conversion between units.

    Args:
        coefficients (list or tuple): Coefficients c0, c1, ..., cN
        representing f(x) = c0 + c1*x + c2*x^2 + ... + cN*x^N
    """

    def __init__(
        self,
        name: str = "Polynomial",
        coefficients: Union[list[float], np.ndarray] = [0.0],
        unit: str = "meV",
    ):
        if not isinstance(coefficients, (list, tuple, np.ndarray)):
            raise TypeError("coefficients must be a list, tuple or ndarray of floats.")

        if not all(isinstance(c, Numeric) for c in coefficients):
            raise TypeError("All coefficients must be numbers.")

        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")

        super().__init__(name=name)
        if not coefficients:
            raise ValueError("At least one coefficient must be provided.")

        self.coefficients = [
            Parameter(
                name=f"{name}_c{i}",
                value=coef,
            )
            for i, coef in enumerate(coefficients)
        ]
        self.unit = unit

    def evaluate(self, x: Union[Numeric, sc.Variable]) -> np.ndarray:
        if isinstance(x, sc.Variable):
            x_in = x.values
            if self.unit is not None and x.unit != self.unit:
                raise ValueError(
                    f"Input x has unit {x.unit}, but Polynomial component has unit {self.unit}. Change the unit of the Polynomial and try again. "
                )
        else:
            x_in = x
        result = np.zeros_like(x_in, dtype=float)
        for i, param in enumerate(self.coefficients):
            result += param.value * np.power(x_in, i)

        if any(result < 0):
            warnings.warn(
                "The Polynomial with name {} has negative values, which may not be physically meaningful.".format(
                    self.name
                )
            )
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

        model_copy = Polynomial(
            name=self.name, coefficients=[param.value for param in self.coefficients]
        )
        for i, param in enumerate(model_copy.coefficients):
            param.fixed = self.coefficients[i].fixed
        return model_copy

    def __repr__(self):
        coeffs_str = ", ".join(
            f"{param.name}={param.value}" for param in self.coefficients
        )
        return f"Polynomial(name = {self.name}, unit = {self.unit},\n coefficients = [{coeffs_str}])"

    def convert_unit(self, unit):
        raise NotImplementedError(
            "Unit conversion is not implemented for Polynomial components. The automatic unit converter does not like powers of units. "
        )


# from typing import Callable, Dict
# class UserDefinedComponent(ModelComponent):
#     """
#     User-defined model component, defined via a custom function.

#     Args:
#         func (Callable): Function accepting (x, params) and returning np.ndarray.
#         params (dict): Parameters passed to the function.
#     """

#     def __init__(
#         self, name, func: Callable[[np.ndarray, Dict], np.ndarray], params: Dict
#     ):
#         super().__init__(name=name)
#         self.func = func
#         self.params = params

#     def evaluate(self, x):
#         return self.func(x, self.params)
