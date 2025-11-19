from typing import Optional, Union

import numpy as np
from easyscience.variable import Parameter
from scipy.special import voigt_profile

from easydynamics.convolution.convolution_base import ConvolutionBase
from easydynamics.sample_model import (
    DeltaFunction,
    Gaussian,
    Lorentzian,
    SampleModel,
    Voigt,
)
from easydynamics.sample_model.components.model_component import ModelComponent

Numerical = Union[float, int]

# TODO: update docstrings


class AnalyticalConvolution(ConvolutionBase):
    def __init__(
        self,
        energy: np.ndarray,
        energy_unit: str = "meV",
        sample_model: SampleModel = None,
        resolution_model: SampleModel = None,
        offset: Optional[Union[Numerical, Parameter]] = 0.0,
    ):
        super().__init__(
            energy=energy,
            sample_model=sample_model,
            resolution_model=resolution_model,
            energy_unit=energy_unit,
            offset=offset,
        )

    def convolution(
        self,
    ) -> np.ndarray:
        """
        Convolve sample with resolution analytically if possible. Accepts SampleModel or single ModelComponent for each.
        Possible analytical convolutions are any combination of delta functions, Gaussians, Lorentzians and Voigt profiles.

        Most validation happens in the main `convolution` function.

        Args:
            x : np.ndarray
                1D array of x values where the convolution is evaluated.
            sample_model : SampleModel or ModelComponent
                The sample model to be convolved.
            resolution_model : SampleModel or ModelComponent
                The resolution model to convolve with.
            self.offset.value : float
                The offset to apply to the convolution.
        Returns:
            np.ndarray
                The convolved values evaluated at x.

        Raises:
            ValueError
                If resolution_model contains delta functions.
            ValueError
                If component pair cannot be handled analytically.

        """

        # prepare list of components
        if isinstance(self.sample_model, SampleModel):
            sample_components = self.sample_model.components
        else:
            sample_components = [self.sample_model]

        if isinstance(self.resolution_model, SampleModel):
            resolution_components = self.resolution_model.components
        else:
            resolution_components = [self.resolution_model]

        total = np.zeros_like(self.energy, dtype=float)

        for sample_component in sample_components:
            # Go through resolution components, adding analytical contributions
            for resolution_component in resolution_components:
                contrib = self._calculate_analytic_pair(
                    sample_component=sample_component,
                    resolution_component=resolution_component,
                )
                total += contrib

        return total

    def _calculate_analytic_pair(
        self,
        sample_component: Union[ModelComponent, SampleModel],
        resolution_component: ModelComponent,
    ) -> np.ndarray:
        """
        Analytic convolution for component pair (sample_component, resolution_component).
        The convolution of two gaussian components results in another gaussian component with width sqrt(w1^2 + w2^2).
        The convolution of two lorentzian components results in another lorentzian component with width w1 + w2.
        The convolution of a gaussian and a lorentzian results in a voigt profile.
        The convolution of a gaussian and a voigt profile results in another voigt profile, with the lorentzian width unchanged and the gaussian widths summed in quadrature.
        The convolution of a lorentzian and a voigt profile results in another voigt profile, with the gaussian width unchanged and the lorentzian widths summed.
        The convolution of two voigt profiles results in another voigt profile, with the gaussian widths summed in quadrature and the lorentzian widths summed.
        The convolution of a delta function with any component or SampleModel results in the same component or SampleModel shifted by the delta center.
        All areas are multiplied.
        The output is shifted by self.offset.value.


        Args:
            sample_component : Union[ModelComponent, SampleModel]
                The sample component to be convolved.
            resolution_component : Union[ModelComponent, SampleModel]
                The resolution component to convolve with.

        Returns:
            np.ndarray: The convolution result

        Raises:
            ValueError: If the component pair cannot be handled analytically.
        """

        # Delta function + anything --> anything, shifted by delta center with area A1 * A2
        if isinstance(sample_component, DeltaFunction):
            return self._convolute_delta_any(
                sample_component,
                resolution_component,
            )

        # Gaussian + Gaussian --> Gaussian with width sqrt(w1^2 + w2^2) and area A1 * A2
        if isinstance(sample_component, Gaussian) and isinstance(
            resolution_component, Gaussian
        ):
            return self._convolute_gauss_gauss(
                sample_component,
                resolution_component,
            )

        # Gaussian + Lorentzian --> Voigt with area A1 * A2
        if (
            isinstance(sample_component, Gaussian)
            and isinstance(resolution_component, Lorentzian)
        ) or (
            isinstance(sample_component, Lorentzian)
            and isinstance(resolution_component, Gaussian)
        ):
            if isinstance(sample_component, Gaussian):
                gaussian, lorentzian = sample_component, resolution_component
            else:
                gaussian, lorentzian = resolution_component, sample_component

            return self._convolute_gauss_lorentz(
                gaussian,
                lorentzian,
            )

        # Gaussian + Voigt --> Voigt with area A1 * A2, Lorentzian width unchanged, Gaussian widths summed in quadrature
        if (
            isinstance(sample_component, Gaussian)
            and isinstance(resolution_component, Voigt)
        ) or (
            isinstance(sample_component, Voigt)
            and isinstance(resolution_component, Gaussian)
        ):
            if isinstance(sample_component, Gaussian):
                gaussian, voigt = sample_component, resolution_component
            else:
                gaussian, voigt = resolution_component, sample_component
            return self._convolute_gauss_voigt(
                gaussian,
                voigt,
            )

        # Lorentzian + Lorentzian --> Lorentzian with width w1 + w2 and area A1 * A2
        if isinstance(sample_component, Lorentzian) and isinstance(
            resolution_component, Lorentzian
        ):
            return self._convolute_lorentz_lorentz(
                sample_component,
                resolution_component,
            )

        #  Lorentzian + Voigt --> Voigt with area A1 * A2, Gaussian width unchanged, Lorentzian widths summed
        if (
            isinstance(sample_component, Lorentzian)
            and isinstance(resolution_component, Voigt)
        ) or (
            isinstance(sample_component, Voigt)
            and isinstance(resolution_component, Lorentzian)
        ):
            if isinstance(sample_component, Lorentzian):
                lorentzian, voigt = sample_component, resolution_component
            else:
                lorentzian, voigt = resolution_component, sample_component
            center = (voigt.center.value + lorentzian.center.value) + self.offset.value
            area = voigt.area.value * lorentzian.area.value
            g_width = voigt.g_width.value
            l_width = voigt.l_width.value + lorentzian.width.value

            return self._voigt_eval(center, g_width, l_width, area)

        # Voigt + Voigt --> Voigt with area A1 * A2, Gaussian widths summed in quadrature, Lorentzian widths summed
        if isinstance(sample_component, Voigt) and isinstance(
            resolution_component, Voigt
        ):
            return self.convolute_voigt_voigt(
                sample_component,
                resolution_component,
            )
        return ValueError(
            f"Analytical convolution not implemented for component pair: {type(sample_component).__name__}, {type(resolution_component).__name__}"
        )

    def _convolute_delta_any(
        self,
        sample_component: ModelComponent,
        resolution_model: Union[SampleModel, ModelComponent],
    ):
        """
        Convolution of delta function with any component or SampleModel results in the same component or SampleModel shifted by the delta center.
        The areas are multiplied.

        Args:
            sample_component : ModelComponent
                The sample component to be convolved.
            resolution_component : ModelComponent
                The resolution component to convolve with.
        Returns:
            np.ndarray
                The evaluated convolution values at self.energy.
        """
        return sample_component.area.value * resolution_model.evaluate(
            self.energy.values - sample_component.center.value - self.offset.value
        )

    def _convolute_gauss_gauss(
        self,
        sample_component: Gaussian,
        resolution_component: Gaussian,
    ) -> np.ndarray:
        """
        Convolution of two gaussian components results in another gaussian component with width sqrt(w1^2 + w2^2).
        The areas are multiplied.

        Args:
            sample_component : Gaussian
                The sample Gaussian component to be convolved.
            resolution_component : Gaussian
                The resolution Gaussian component to convolve with.

        Returns:
            np.ndarray

                The evaluated convolution values at self.energy.
        """

        width = np.sqrt(
            sample_component.width.value**2 + resolution_component.width.value**2
        )
        area = sample_component.area.value * resolution_component.area.value
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + self.offset.value

        return self._gaussian_eval(center, width, area)

    def _convolute_gauss_lorentz(
        self,
        sample_component: Gaussian,
        resolution_component: Lorentzian,
    ) -> np.ndarray:
        """
        Convolution of a Gaussian and a Lorentzian results in a Voigt profile.
        The areas are multiplied.

        Args:
            sample_component : Gaussian
                The sample Gaussian component to be convolved.
            resolution_component : Lorentzian
                The resolution Lorentzian component to convolve with.

        Returns:
            np.ndarray
                The evaluated convolution values at self.energy.
        """
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + self.offset.value
        area = sample_component.area.value * resolution_component.area.value

        return self._voigt_eval(
            center,
            sample_component.width.value,
            resolution_component.width.value,
            area,
        )

    def _convolute_gauss_voigt(
        self,
        sample_component: Gaussian,
        resolution_component: Voigt,
    ) -> np.ndarray:
        """
        Convolution of a Gaussian and a Voigt profile results in another Voigt profile.
        The Lorentzian width remains unchanged, while the Gaussian widths are summed in quadrature.
        The areas are multiplied.

        Args:
            sample_component : Gaussian
                The sample Gaussian component to be convolved.
            resolution_component : Voigt
                The resolution Voigt component to convolve with.

        Returns:
            np.ndarray
                The evaluated convolution values at self.energy.
        """
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + self.offset.value
        area = sample_component.area.value * resolution_component.area.value
        g_width = np.sqrt(
            sample_component.width.value**2 + resolution_component.g_width.value**2
        )
        l_width = resolution_component.l_width.value
        return self._voigt_eval(center, g_width, l_width, area)

    def _convolute_lorentz_lorentz(
        self,
        sample_component: Lorentzian,
        resolution_component: Lorentzian,
    ) -> np.ndarray:
        """
        Convolution of two Lorentzian components results in another Lorentzian component with width w1 + w2.
        The areas are multiplied.

        Args:
            sample_component : Lorentzian
                The sample Lorentzian component to be convolved.
            resolution_component : Lorentzian
                The resolution Lorentzian component to convolve with.
        Returns:
            np.ndarray
                The evaluated convolution values at self.energy.
        """
        width = sample_component.width.value + resolution_component.width.value
        area = sample_component.area.value * resolution_component.area.value
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + self.offset.value
        return self._lorentzian_eval(center, width, area)

    def _convolute_lorentz_voigt(
        self,
        sample_component: Lorentzian,
        resolution_component: Voigt,
    ) -> np.ndarray:
        """
        Convolution of a Lorentzian and a Voigt profile results in another Voigt profile.
        The Gaussian width remains unchanged, while the Lorentzian widths are summed.
        The areas are multiplied.
        Args:
            sample_component : Lorentzian
                The sample Lorentzian component to be convolved.
            resolution_component : Voigt
                The resolution Voigt component to convolve with.
        Returns:
            np.ndarray
                The evaluated convolution values at self.energy.
        """
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + self.offset.value
        area = sample_component.area.value * resolution_component.area.value
        g_width = resolution_component.g_width.value
        l_width = sample_component.width.value + resolution_component.l_width.value
        return self._voigt_eval(center, g_width, l_width, area)

    def convolute_voigt_voigt(
        self,
        sample_component: Voigt,
        resolution_component: Voigt,
    ) -> np.ndarray:
        """
        Convolution of two Voigt profiles results in another Voigt profile.
        The Gaussian widths are summed in quadrature, while the Lorentzian widths are summed.
        The areas are multiplied.
        Args:
            sample_component : Voigt
                The sample Voigt component to be convolved.
            resolution_component : Voigt
                The resolution Voigt component to convolve with.
        Returns:
            np.ndarray
                The evaluated convolution values at self.energy.
        """
        center = (
            sample_component.center.value + resolution_component.center.value
        ) + self.offset.value
        area = sample_component.area.value * resolution_component.area.value
        g_width = np.sqrt(
            sample_component.g_width.value**2 + resolution_component.g_width.value**2
        )
        l_width = sample_component.l_width.value + resolution_component.l_width.value
        return self._voigt_eval(center, g_width, l_width, area)

    def _gaussian_eval(self, center: float, width: float, area: float) -> np.ndarray:
        """
        Evaluate a Gaussian function. y = (area / (sqrt(2pi) * width)) * exp(-0.5 * ((x - center) / width)^2)
        All checks are handled in the calling function.

        Args:
            energy : np.ndarray
                1D array of energy values where the Gaussian is evaluated.
            center : float
                The center of the Gaussian.
            width : float
                The width (sigma) of the Gaussian.
            area : float
                The area under the Gaussian curve.
        Returns:
            np.ndarray
                The evaluated Gaussian values at self.energy.
        """
        return (
            area
            * 1
            / (np.sqrt(2 * np.pi) * width)
            * np.exp(-0.5 * ((self.energy.values - center) / width) ** 2)
        )

    def _lorentzian_eval(self, center: float, width: float, area: float) -> np.ndarray:
        """
        Evaluate a Lorentzian function. y = (area * width / pi) / ((x - center)^2 + width^2).
        All checks are handled in the calling function.

        Args:
            energy : np.ndarray
                1D array of energy values where the Lorentzian is evaluated.
            center : float
                The center of the Lorentzian.
            width : float
                The width (HWHM) of the Lorentzian.
            area : float
                The area under the Lorentzian.
        Returns:
            np.ndarray
                The evaluated Lorentzian values at self.energy.
        """
        return area * width / np.pi / ((self.energy.values - center) ** 2 + width**2)

    def _voigt_eval(
        self,
        center: float,
        g_width: float,
        l_width: float,
        area: float,
    ) -> np.ndarray:
        """
        Evaluate a Voigt profile function using scipy's voigt_profile.
        Args:
            energy : np.ndarray
                1D array of energy values where the Voigt profile is evaluated.
            center : float
                The center of the Voigt profile.
            g_width : float
                The Gaussian width (sigma) of the Voigt profile.
            l_width : float
                The Lorentzian width (HWHM) of the Voigt profile.
            area : float
                The area under the Voigt profile.
        Returns:
            np.ndarray
                The evaluated Voigt profile values at self.energy.
        """

        return area * voigt_profile(self.energy.values - center, g_width, l_width)
