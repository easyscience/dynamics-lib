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
    Convolution class that combines analytical and numerical convolution methods based on sample model components.

    Args:
    energy : np.ndarray
        1D array of energy values where the convolution is evaluated.
    sample_model : SampleModel or ModelComponent
        The sample model to be convolved.
    resolution_model : SampleModel or ModelComponent
        The resolution model to convolve with.
    offset_float : float, or None, optional
        The offset to apply to the input array.
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
        Whether to normalize the detailed balance factor. Default is True.
    """

    def __init__(
        self,
        energy: np.ndarray,
        sample_model: Union[SampleModel, ModelComponent],
        resolution_model: Union[SampleModel, ModelComponent],
        offset: Optional[Union[Numerical, Parameter]] = 0.0,
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
            offset=offset,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
            temperature=temperature,
            temperature_unit=temperature_unit,
            energy_unit=energy_unit,
            normalize_detailed_balance=normalize_detailed_balance,
        )

        # Separate sample model components into analytical pairs, delta functions, and the rest
        self._set_sample_models()
        # Initialize analytical and numerical convolvers based on sample model components
        self._set_convolvers()

    def convolution(
        self,
    ) -> np.ndarray:
        """
        Perform convolution using analytical method where possible, and numerical method for remaining components.
        """

        total = np.zeros_like(self.energy, dtype=float)

        # Analytical convolution
        if self._analytical_convolver is not None:
            total += self._analytical_convolver.convolution()

        # Numerical convolution
        if self._numerical_convolver is not None:
            total += self._numerical_convolver.convolution()

        # Delta function components (no convolution needed)
        if self._delta_sample_model.components:
            for sample_component in self._delta_sample_model.components:
                total += sample_component.area.value * self._resolution_model.evaluate(
                    self.energy - sample_component.center.value - self.offset.value
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
                energy_unit=self._energy_unit,
                sample_model=self._analytical_sample_model,
                resolution_model=self._resolution_model,
                offset=self.offset,
            )
        else:
            self._analytical_convolver = None

        if self._numerical_sample_model.components:
            self._numerical_convolver = NumericalConvolution(
                energy=self.energy,
                energy_unit=self._energy_unit,
                sample_model=self.numerical_sample_model,
                resolution_model=self.resolution_model,
                offset=self.offset,
                upsample_factor=self.upsample_factor,
                extension_factor=self.extension_factor,
                temperature=self.temperature,
                temperature_unit=self.temperature_unit,
                normalize_detailed_balance=self.normalize_detailed_balance,
            )
        else:
            self._numerical_convolver = None

    def _set_sample_models(self) -> None:
        """ "    Separate sample model components into analytical pairs, delta functions, and the rest."""

        analytical_sample_model = SampleModel()
        delta_sample_model = SampleModel()
        numerical_sample_model = SampleModel()
        for sample_component in self._sample_model.components:
            if isinstance(sample_component, DeltaFunction):
                delta_sample_model.add_component(sample_component)
                continue
            pair_is_analytic = []
            for resolution_component in self.resolution_model.components:
                pair_is_analytic.append(
                    self._check_if_pair_is_analytic(
                        sample_component, resolution_component
                    )
                )
            if all(pair_is_analytic) and self.temperature is None:
                analytical_sample_model.add_component(sample_component)
            else:
                numerical_sample_model.add_component(sample_component)

        self._analytical_sample_model = analytical_sample_model
        self._delta_sample_model = delta_sample_model
        self._numerical_sample_model = numerical_sample_model
