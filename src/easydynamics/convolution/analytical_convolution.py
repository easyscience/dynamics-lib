# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import ClassVar

import numpy as np
from scipy.special import voigt_profile

from easydynamics.convolution.convolution_base import ConvolutionBase
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Voigt
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent


class AnalyticalConvolution(ConvolutionBase):
    """
    Analytical convolution of a ModelComponent or ComponentCollection with a ResolutionModel.

    Possible analytical convolutions are any combination of delta functions, Gaussians, Lorentzians
    and Voigt profiles.
    """

    # Mapping of supported component type pairs to convolution methods.
    # Delta functions are handled separately.
    _CONVOLUTIONS: ClassVar[dict[str, object]] = {
        ('Gaussian', 'Gaussian'): '_convolute_gaussian_gaussian',
        ('Gaussian', 'Lorentzian'): '_convolute_gaussian_lorentzian',
        ('Gaussian', 'Voigt'): '_convolute_gaussian_voigt',
        ('Lorentzian', 'Lorentzian'): '_convolute_lorentzian_lorentzian',
        ('Lorentzian', 'Voigt'): '_convolute_lorentzian_voigt',
        ('Voigt', 'Voigt'): '_convolute_voigt_voigt',
    }

    def convolution(
        self,
    ) -> np.ndarray:
        """
        Convolve sample with resolution analytically if possible.

        Accepts ComponentCollection or single ModelComponent for each. Possible analytical
        convolutions are any combination of delta functions, Gaussians, Lorentzians and Voigt
        profiles.

        Returns
        -------
        np.ndarray
            The convolution of the sample_components and resolution_components values evaluated at
            self.energy.
        """

        total = np.zeros_like(self.energy.values, dtype=float)

        for sample_component in self.sample_components:
            # Go through resolution components,
            # adding analytical contributions
            for resolution_component in self.resolution_components:
                contrib = self._convolute_analytic_pair(
                    sample_component=sample_component,
                    resolution_component=resolution_component,
                )
                total += contrib

        return total

    def _convolute_analytic_pair(
        self,
        sample_component: ModelComponent,
        resolution_component: ModelComponent,
    ) -> np.ndarray:
        r"""
        Analytic convolution for component pair (sample_component, resolution_component).

        The convolution of two Gaussian components results in another Gaussian component with width
        $\sqrt{w_1^2 + w_2^2}$.

        The convolution of two Lorentzian components results in another Lorentzian component with
        width $w_1 + w_2$.

        The convolution of a Gaussian and a Lorentzian results in a Voigt profile.

        The convolution of a Gaussian and a Voigt profile results in another Voigt profile, with
        the Lorentzian width unchanged and the Gaussian widths summed in quadrature.

        The convolution of a Lorentzian and a Voigt profile results in another Voigt profile, with
        the Gaussian width unchanged and the Lorentzian widths summed.

        The convolution of two Voigt profiles results in another Voigt profile, with the Gaussian
        widths summed in quadrature and the Lorentzian widths summed.

        The convolution of a delta function with any component or ComponentCollection results in
        the same component or ComponentCollection shifted by the delta center.

        All areas are multiplied in the convolution.

        Parameters
        ----------
        sample_component : ModelComponent
            The sample component to be convolved.
        resolution_component : ModelComponent
            The resolution component to convolve with.

        Raises
        ------
        ValueError
            If the component pair cannot be handled analytically.

        Returns
        -------
        np.ndarray
            The convolution result.
        """

        if isinstance(resolution_component, DeltaFunction):
            raise ValueError(
                'Analytical convolution with a delta function \
                    in the resolution model is not supported.'
            )

        # Delta function + anything -->
        # anything, shifted by delta center with area A1 * A2
        if isinstance(sample_component, DeltaFunction):
            return self._convolute_delta_any(
                sample_component,
                resolution_component,
            )

        pair = (type(sample_component).__name__, type(resolution_component).__name__)
        swapped = False

        if pair not in self._CONVOLUTIONS:
            # Try reversing the pair
            pair = (
                type(resolution_component).__name__,
                type(sample_component).__name__,
            )
            swapped = True

        func_name = self._CONVOLUTIONS.get(pair)

        if func_name is None:
            raise ValueError(
                f'Analytical convolution not supported for component pair: '
                f'{type(sample_component).__name__}, {type(resolution_component).__name__}'
            )

        # Call the corresponding method
        if swapped:
            return getattr(self, func_name)(resolution_component, sample_component)
        return getattr(self, func_name)(sample_component, resolution_component)

    def _convolute_delta_any(
        self,
        sample_component: DeltaFunction,
        resolution_components: ComponentCollection | ModelComponent,
    ) -> np.ndarray:
        """
        Convolution of delta function with any ModelComponent or ComponentCollection results in the
        same component or ComponentCollection shifted by the delta center. The areas are
        multiplied.

        Parameters
        ----------
        sample_component : DeltaFunction
            The sample component to be convolved.
        resolution_components : ComponentCollection | ModelComponent
            The resolution model to convolve with.

        Returns
        -------
        np.ndarray
            The evaluated convolution values at self.energy.
        """
        return sample_component.area.value * resolution_components.evaluate(
            self.energy_with_offset.values - sample_component.center.value
        )

    def _convolute_gaussian_gaussian(
        self,
        sample_component: Gaussian,
        resolution_component: Gaussian,
    ) -> np.ndarray:
        r"""
        Convolution of two Gaussian components results in another Gaussian component with width
        $\sqrt{w_1^2 + w_2^2}$. The areas are multiplied.

        Parameters
        ----------
        sample_component : Gaussian
            The sample Gaussian component to be convolved.
        resolution_component : Gaussian
            The resolution Gaussian component to convolve with.

        Returns
        -------
        np.ndarray
            The evaluated convolution values at self.energy.
        """

        width = np.sqrt(sample_component.width.value**2 + resolution_component.width.value**2)

        area = sample_component.area.value * resolution_component.area.value

        center = sample_component.center.value + resolution_component.center.value

        return self._gaussian_eval(area=area, center=center, width=width)

    def _convolute_gaussian_lorentzian(
        self,
        sample_component: Gaussian,
        resolution_component: Lorentzian,
    ) -> np.ndarray:
        """
        Convolution of a Gaussian and a Lorentzian results in a Voigt profile. The areas are
        multiplied.

        Parameters
        ----------
        sample_component : Gaussian
            The sample Gaussian component to be convolved.
        resolution_component : Lorentzian
            The resolution Lorentzian component to convolve with.

        Returns
        -------
        np.ndarray
            The evaluated convolution values at self.energy.
        """
        center = sample_component.center.value + resolution_component.center.value
        area = sample_component.area.value * resolution_component.area.value

        return self._voigt_eval(
            area=area,
            center=center,
            gaussian_width=sample_component.width.value,
            lorentzian_width=resolution_component.width.value,
        )

    def _convolute_gaussian_voigt(
        self,
        sample_component: Gaussian,
        resolution_component: Voigt,
    ) -> np.ndarray:
        """
        Convolution of a Gaussian and a Voigt profile results in another Voigt profile. The
        Lorentzian width remains unchanged, while the Gaussian widths are summed in quadrature. The
        areas are multiplied.

        Parameters
        ----------
        sample_component : Gaussian
            The sample Gaussian component to be convolved.
        resolution_component : Voigt
            The resolution Voigt component to convolve with.

        Returns
        -------
        np.ndarray
            The evaluated convolution values at self.energy.
        """
        area = sample_component.area.value * resolution_component.area.value

        center = sample_component.center.value + resolution_component.center.value

        gaussian_width = np.sqrt(
            sample_component.width.value**2 + resolution_component.gaussian_width.value**2
        )

        lorentzian_width = resolution_component.lorentzian_width.value

        return self._voigt_eval(
            area=area,
            center=center,
            gaussian_width=gaussian_width,
            lorentzian_width=lorentzian_width,
        )

    def _convolute_lorentzian_lorentzian(
        self,
        sample_component: Lorentzian,
        resolution_component: Lorentzian,
    ) -> np.ndarray:
        r"""
        Convolution of two Lorentzian components results in another Lorentzian component with width
        $w_1 + w_2$. The areas are multiplied.

        Parameters
        ----------
        sample_component : Lorentzian
            The sample Lorentzian component to be convolved.
        resolution_component : Lorentzian
            The resolution Lorentzian component to convolve with.

        Returns
        -------
        np.ndarray
            The evaluated convolution values at self.energy.
        """
        area = sample_component.area.value * resolution_component.area.value

        center = sample_component.center.value + resolution_component.center.value

        width = sample_component.width.value + resolution_component.width.value

        return self._lorentzian_eval(area=area, center=center, width=width)

    def _convolute_lorentzian_voigt(
        self,
        sample_component: Lorentzian,
        resolution_component: Voigt,
    ) -> np.ndarray:
        """
        Convolution of a Lorentzian and a Voigt profile results in another Voigt profile.

        The Gaussian width remains unchanged, while the Lorentzian widths are summed.

        The areas are multiplied.

        Parameters
        ----------
        sample_component : Lorentzian
            The sample Lorentzian component to be convolved.
        resolution_component : Voigt
            The resolution Voigt component to convolve with.

        Returns
        -------
        np.ndarray
            The evaluated convolution values at self.energy.
        """
        area = sample_component.area.value * resolution_component.area.value

        center = sample_component.center.value + resolution_component.center.value

        gaussian_width = resolution_component.gaussian_width.value

        lorentzian_width = (
            sample_component.width.value + resolution_component.lorentzian_width.value
        )

        return self._voigt_eval(
            area=area,
            center=center,
            gaussian_width=gaussian_width,
            lorentzian_width=lorentzian_width,
        )

    def _convolute_voigt_voigt(
        self,
        sample_component: Voigt,
        resolution_component: Voigt,
    ) -> np.ndarray:
        """
        Convolution of two Voigt profiles results in another Voigt profile.

        The Gaussian widths are summed in quadrature, while the Lorentzian widths are summed. The
        areas are multiplied.

        Parameters
        ----------
        sample_component : Voigt
            The sample Voigt component to be convolved.
        resolution_component : Voigt
            The resolution Voigt component to convolve with.

        Returns
        -------
        np.ndarray
            The evaluated convolution values at self.energy.
        """
        area = sample_component.area.value * resolution_component.area.value

        center = sample_component.center.value + resolution_component.center.value

        gaussian_width = np.sqrt(
            sample_component.gaussian_width.value**2 + resolution_component.gaussian_width.value**2
        )

        lorentzian_width = (
            sample_component.lorentzian_width.value + resolution_component.lorentzian_width.value
        )
        return self._voigt_eval(
            area=area,
            center=center,
            gaussian_width=gaussian_width,
            lorentzian_width=lorentzian_width,
        )

    def _gaussian_eval(
        self,
        area: float,
        center: float,
        width: float,
    ) -> np.ndarray:
        r"""
        Evaluate a Gaussian function.

        $$ I(x) = \frac{A}{\sigma \sqrt{2\pi}} \exp\left( -\frac{1}{2} \left(\frac{x -
        x_0}{\sigma}\right)^2 \right) $$

        where $A$ is the area, $x_0$ is the center, and $\sigma$ is the width.

        All checks are handled in the calling function.

        Parameters
        ----------
        area : float
            The area under the Gaussian curve.
        center : float
            The center of the Gaussian.
        width : float
            The width (sigma) of the Gaussian.

        Returns
        -------
        np.ndarray
            The evaluated Gaussian values at self.energy.
        """

        normalization = 1 / (np.sqrt(2 * np.pi) * width)
        exponent = -0.5 * ((self.energy_with_offset.values - center) / width) ** 2

        return area * normalization * np.exp(exponent)

    def _lorentzian_eval(self, area: float, center: float, width: float) -> np.ndarray:
        r"""
        Evaluate a Lorentzian function.

        $$ I(x) = \frac{A}{\\pi} \frac{\Gamma}{(x - x_0)^2 + \Gamma^2}, $$

        where $A$ is the area, $x_0$ is the center, and $\\Gamma$ is the half width at half maximum
        (HWHM).

        All checks are handled in the calling function.

        Parameters
        ----------
        area : float
            The area under the Lorentzian.
        center : float
            The center of the Lorentzian.
        width : float
            The width (HWHM) of the Lorentzian.

        Returns
        -------
        np.ndarray
            The evaluated Lorentzian values at self.energy.
        """

        normalization = width / np.pi
        denominator = (self.energy_with_offset.values - center) ** 2 + width**2

        return area * normalization / denominator

    def _voigt_eval(
        self,
        area: float,
        center: float,
        gaussian_width: float,
        lorentzian_width: float,
    ) -> np.ndarray:
        """
        Evaluate a Voigt profile function using scipy's voigt_profile.

        Parameters
        ----------
        area : float
            The area under the Voigt profile.
        center : float
            The center of the Voigt profile.
        gaussian_width : float
            The Gaussian width (sigma) of the Voigt profile.
        lorentzian_width : float
            The Lorentzian width (HWHM) of the Voigt profile.

        Returns
        -------
        np.ndarray
            The evaluated Voigt profile values at self.energy.
        """

        return area * voigt_profile(
            self.energy_with_offset.values - center, gaussian_width, lorentzian_width
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'display_name={self.display_name!r}, '
            f'unique_name={self.unique_name!r}, '
            f'x_unit={self.x_unit}, '
            f'energy_len={len(self.energy)})'
        )
