from typing import Optional, Union

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.analytical_convolution import AnalyticalConvolution
from easydynamics.convolution.numerical_convolution import NumericalConvolution
from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.sample_model import (
    DeltaFunction,
    Gaussian,
    Lorentzian,
    SampleModel,
    Voigt,
)
from easydynamics.sample_model.components.model_component import ModelComponent

Numerical = Union[float, int]


class Convolution(NumericalConvolutionBase):
    """
    Convolution class that combines analytical and numerical convolution methods to efficiently perform convolutions
    of SampleModels with ResolutionModels.
    Supports analytical convolution for pairs of analytical model components (DeltaFunction, Gaussian, Lorentzian, Voigt),
    while using numerical convolution for other components.
    If temperature is provided, detailed balance correction is applied to the sample model. In this case, all convolutions
    are handled numerically.
    Includes optional upsampling and extended range to improve accuracy of the numerical convolutions. Also warns about
    numerical instabilities if peaks are very wide or very narrow.

    Args:
    energy : np.ndarray or scipp.Variable
        1D array of energy values where the convolution is evaluated.
    sample_model : SampleModel or ModelComponent
        The sample model to be convolved.
    resolution_model : SampleModel or ModelComponent
        The resolution model to convolve with.
    upsample_factor : int, optional
        The factor by which to upsample the input data before convolution. Default is 5.
    extension_factor : float, optional
        The factor by which to extend the input data range before convolution. Default is 0.2.
    temperature : Parameter, float, or None, optional
        The temperature to use for detailed balance correction. Default is None.
    temperature_unit : str or sc.Unit, optional
        The unit of the temperature parameter. Default is 'K'.
    energy_unit : str or sc.Unit, optional
        The unit of the energy. Default is 'meV'.
    normalize_detailed_balance : bool, optional
        Whether to normalize the detailed balance correction. Default is True.
    """

    def __init__(
        self,
        energy: Union[np.ndarray, sc.Variable],
        sample_model: Union[SampleModel, ModelComponent],
        resolution_model: Union[SampleModel, ModelComponent],
        upsample_factor: Optional[Numerical] = 5,
        extension_factor: Optional[float] = 0.2,
        temperature: Optional[Union[Parameter, float]] = None,
        temperature_unit: Optional[Union[str, sc.Unit]] = "K",
        energy_unit: Optional[Union[str, sc.Unit]] = "meV",
        normalize_detailed_balance: Optional[bool] = True,
    ):
        super().__init__(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
            temperature=temperature,
            temperature_unit=temperature_unit,
            energy_unit=energy_unit,
            normalize_detailed_balance=normalize_detailed_balance,
        )

        # Separate sample model components into pairs that can be handled analytically, delta functions, and the rest
        # Also initialize analytical and numerical convolvers based on sample model component
        self._separate_analytical_components()

    def convolution(
        self,
    ) -> np.ndarray:
        """
        Perform convolution using analytical convolutions where possible, and numerical convolutions for the remaining components.
        Returns:
            np.ndarray
                The convolved values evaluated at energy.
        """

        total = np.zeros_like(self.energy.values, dtype=float)

        # Analytical convolution
        if self._analytical_convolver is not None:
            total += self._analytical_convolver.convolution()

        # Numerical convolution
        if self._numerical_convolver is not None:
            total += self._numerical_convolver.convolution()

        # Delta function components (no convolution needed, and no detailed balancing)
        if self._delta_sample_model.components:
            for sample_component in self._delta_sample_model.components:
                total += sample_component.area.value * self._resolution_model.evaluate(
                    self.energy.values - sample_component.center.value
                )

        return total

    def _check_if_pair_is_analytic(
        self,
        sample_component: ModelComponent,
        resolution_component: ModelComponent,
    ) -> bool:
        """
        Check if the convolution of the given component pair can be handled analytically.

        Args:
            sample_component : ModelComponent
                The sample component to be convolved.
            resolution_component : ModelComponent
                The resolution component to convolve with.
        Returns:
            bool
                True if the component pair can be handled analytically, False otherwise.
        """

        if not isinstance(sample_component, ModelComponent):
            raise TypeError(
                f"`sample_component` is an instance of {type(sample_component).__name__}, but must be ModelComponent."
            )

        if not isinstance(resolution_component, ModelComponent):
            raise TypeError(
                f"`resolution_component` is an instance of {type(resolution_component).__name__}, but must be ModelComponent."
            )

        if isinstance(resolution_component, DeltaFunction):
            raise ValueError(
                "Resolution model contains delta functions. This is not supported."
            )

        analytical_types = (Gaussian, Lorentzian, Voigt)
        if isinstance(sample_component, analytical_types) and isinstance(
            resolution_component, analytical_types
        ):
            return True

        return False

    def _set_convolvers(self) -> None:
        """Initialize analytical and numerical convolvers based on sample model components."""

        if self._analytical_sample_model.components:
            self._analytical_convolver = AnalyticalConvolution(
                energy=self.energy,
                sample_model=self._analytical_sample_model,
                resolution_model=self._resolution_model,
            )
        else:
            self._analytical_convolver = None

        if self._numerical_sample_model.components:
            self._numerical_convolver = NumericalConvolution(
                energy=self.energy,
                sample_model=self._numerical_sample_model,
                resolution_model=self._resolution_model,
                upsample_factor=self._upsample_factor,
                extension_factor=self._extension_factor,
                temperature=self._temperature,
                temperature_unit=self._temperature_unit,
                normalize_detailed_balance=self._normalize_detailed_balance,
            )
        else:
            self._numerical_convolver = None

    def _separate_analytical_components(self) -> None:
        """ "    Separate sample model components into analytical pairs, delta functions, and the rest."""

        analytical_sample_model = SampleModel()
        delta_sample_model = SampleModel()
        numerical_sample_model = SampleModel()

        for sample_component in self._sample_model.components:
            # If delta function, put in delta sample model and go to the next component
            if isinstance(sample_component, DeltaFunction):
                delta_sample_model.add_component(sample_component)
                continue

            # If temperature is set, all other components go to numerical sample model
            if self.temperature is not None:
                numerical_sample_model.add_component(sample_component)
                continue

            pair_is_analytic = []
            for resolution_component in self._resolution_model.components:
                pair_is_analytic.append(
                    self._check_if_pair_is_analytic(
                        sample_component, resolution_component
                    )
                )
            # If all resolution components can be convolved analytically with this sample component, add it to analytical sample model
            if all(pair_is_analytic):
                analytical_sample_model.add_component(sample_component)
            else:
                numerical_sample_model.add_component(sample_component)

        self._analytical_sample_model = analytical_sample_model
        self._delta_sample_model = delta_sample_model
        self._numerical_sample_model = numerical_sample_model

        # Update convolvers
        self._set_convolvers()

    # Update some setters so the internal sample models are updated accordingly
    @NumericalConvolutionBase.sample_model.setter
    def sample_model(self, sample_model: Union[SampleModel, ModelComponent]) -> None:
        """Set the sample model and update internal sample models accordingly.

        Args:
            sample_model : SampleModel or ModelComponent
                The sample model to be convolved.

        Raises:
            TypeError: If sample_model is not a SampleModel or ModelComponent.
        """
        super(NumericalConvolutionBase).sample_model.sample_model = sample_model

        # Separate sample model components into pairs that can be handled analytically, delta functions, and the rest
        self._separate_analytical_components()

    @NumericalConvolutionBase.resolution_model.setter
    def resolution_model(
        self, resolution_model: Union[SampleModel, ModelComponent]
    ) -> None:
        """Set the resolution model and update internal sample models accordingly.

        Args:
            resolution_model : SampleModel or ModelComponent
                The resolution model to convolve with.
        Raises:
            TypeError: If resolution_model is not a SampleModel or ModelComponent.
        """
        super(
            NumericalConvolutionBase
        ).resolution_model.resolution_model = resolution_model

        # Separate sample model components into pairs that can be handled analytically, delta functions, and the rest
        self._separate_analytical_components()

    @NumericalConvolutionBase.temperature.setter
    def temperature(self, temperature: Optional[Union[Parameter, float]]) -> None:
        """Set the temperature and update internal sample models accordingly.

        Args:
            temperature : Parameter, float, or None
                The temperature to use for detailed balance correction.
        """
        super(NumericalConvolutionBase).temperature = temperature

        # Separate sample model components into pairs that can be handled analytically, delta functions, and the rest
        self._separate_analytical_components()
