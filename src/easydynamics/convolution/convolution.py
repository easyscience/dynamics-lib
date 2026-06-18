# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import ClassVar

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.analytical_convolution import AnalyticalConvolution
from easydynamics.convolution.numerical_convolution import NumericalConvolution
from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Voigt
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.settings.convolution_settings import ConvolutionSettings
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings
from easydynamics.utils.utils import Numeric


class Convolution(NumericalConvolutionBase):
    """
    Convolution class that combines analytical and numerical convolution methods to efficiently
    perform convolutions of ComponentCollections with ResolutionComponents.

    Supports analytical convolution for pairs of analytical model components (DeltaFunction,
    Gaussian, Lorentzian, Voigt), while using numerical convolution for other components. If
    temperature is provided, detailed balance correction is applied to the sample model. In this
    case, all convolutions are handled numerically. Includes a setting to normalize the detailed
    balance correction. Includes optional upsampling and extended range to improve accuracy of the
    numerical convolutions. Also warns about numerical instabilities if peaks are very wide or very
    narrow.
    """

    # When these attributes are changed, the convolution plan
    # needs to be rebuilt
    _invalidate_plan_on_change: ClassVar[dict[str, object]] = {
        '_energy',
        '_energy_grid',
        '_sample_components',
        '_resolution_components',
        '_temperature',
        '_energy_unit',
        '_normalize_detailed_balance',
        '_detailed_balance_settings',
    }

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_components: ComponentCollection | ModelComponent,
        resolution_components: ComponentCollection | ModelComponent,
        energy_offset: Numeric | Parameter = 0.0,
        convolution_settings: ConvolutionSettings | None = None,
        temperature: Parameter | Numeric | None = None,
        temperature_unit: str | sc.Unit = 'K',
        detailed_balance_settings: DetailedBalanceSettings | None = None,
        unit: str | sc.Unit = 'meV',
        display_name: str | None = 'MyConvolution',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the Convolution class.

        Parameters
        ----------
        energy : np.ndarray | sc.Variable
            1D array of energy values where the convolution is evaluated.
        sample_components : ComponentCollection | ModelComponent
            The  sample components to be convolved.
        resolution_components : ComponentCollection | ModelComponent
            The resolution components to convolve with.
        energy_offset : Numeric | Parameter, default=0.0
            An energy offset to apply to the energy values before convolution.
        convolution_settings : ConvolutionSettings | None, default=None
            The settings for the convolution. If None, default settings will be used.
        temperature : Parameter | Numeric | None, default=None
            The temperature to use for detailed balance correction.
        temperature_unit : str | sc.Unit, default='K'
            The unit of the temperature parameter.
        detailed_balance_settings : DetailedBalanceSettings | None, default=None
            The settings for detailed balance. If None, default settings will be used.
        unit : str | sc.Unit, default='meV'
            The unit of the energy.
        display_name : str | None, default='MyConvolution'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        """

        self._reactions_enabled = False
        super().__init__(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy_offset=energy_offset,
            convolution_settings=convolution_settings,
            temperature=temperature,
            temperature_unit=temperature_unit,
            detailed_balance_settings=detailed_balance_settings,
            unit=unit,
            display_name=display_name,
            unique_name=unique_name,
        )

        self._reactions_enabled = True
        # Separate sample model components into pairs that can be
        # handled analytically, delta functions, and the rest
        # Also initialize analytical and numerical convolvers based on
        # sample model component
        self._build_convolution_plan()

    def convolution(
        self,
    ) -> np.ndarray:
        """
        Perform convolution using analytical convolutions where possible, and numerical
        convolutions for the remaining components.

        Returns
        -------
        np.ndarray
            The convolved values evaluated at energy.
        """
        if not self.convolution_settings.convolution_plan_is_valid:
            self._build_convolution_plan()
        total = np.zeros_like(self.energy.values, dtype=float)

        # Analytical convolution
        if self._analytical_convolver is not None:
            total += self._analytical_convolver.convolution()

        # Numerical convolution
        if self._numerical_convolver is not None:
            total += self._numerical_convolver.convolution()

        # Delta function components
        if self._delta_sample_components:
            total += self._convolve_delta_functions()

        return total

    def _convolve_delta_functions(self) -> np.ndarray:
        """
        Convolve delta function components of the sample model with the resolution components. No
        detailed balance correction is applied to delta functions.

        Returns
        -------
        np.ndarray
            The convolved values of the delta function c components evaluated at energy.
        """
        return sum(
            delta.area.value
            * self._resolution_components.evaluate(
                self.energy_with_offset.values - delta.center.value
            )
            for delta in self._delta_sample_components
        )

    def _check_if_pair_is_analytic(
        self,
        sample_component: ModelComponent,
        resolution_component: ModelComponent,
    ) -> bool:
        """
        Check if the convolution of the given component pair can be handled analytically.

        Parameters
        ----------
        sample_component : ModelComponent
            The sample component to be convolved.
        resolution_component : ModelComponent
            The resolution component to convolve with.

        Raises
        ------
        TypeError
            If the resolution component is a DeltaFunction.

        Returns
        -------
        bool
            True if the component pair can be handled analytically, False otherwise.
        """

        if isinstance(resolution_component, DeltaFunction):
            raise TypeError(
                'resolution components contains delta functions. This is not supported.'
            )

        analytical_types = (Gaussian, Lorentzian, Voigt)
        return bool(
            isinstance(sample_component, analytical_types)
            and isinstance(resolution_component, analytical_types)
        )

    def _build_convolution_plan(self) -> None:
        """
        Separate sample model components into analytical pairs, delta functions, and the rest.
        """

        analytical_sample_components = ComponentCollection()
        delta_sample_components = ComponentCollection()
        numerical_sample_components = ComponentCollection()

        for sample_component in self._sample_components:
            # If delta function, put in delta sample model and go to the
            # next component
            if isinstance(sample_component, DeltaFunction):
                delta_sample_components.append_component(sample_component)
                continue

            # If temperature is set, all other components go to
            # numerical sample model
            if (
                self.temperature is not None
                and self.detailed_balance_settings.use_detailed_balance
            ):
                numerical_sample_components.append_component(sample_component)
                continue

            # If temperature is not set, check if all
            # resolution components can be convolved analytically with
            # this sample component
            pair_is_analytic = [
                self._check_if_pair_is_analytic(sample_component, resolution_component)
                for resolution_component in self._resolution_components
            ]
            # If all resolution components can be convolved analytically
            # with this sample component, add it to analytical
            # sample model. If not, it goes to numerical sample model.
            if all(pair_is_analytic):
                analytical_sample_components.append_component(sample_component)
            else:
                numerical_sample_components.append_component(sample_component)

        self._analytical_sample_components = analytical_sample_components
        self._delta_sample_components = delta_sample_components
        self._numerical_sample_components = numerical_sample_components

        # Update convolvers
        self._set_convolvers()
        self.convolution_settings.convolution_plan_is_valid = True

    def _set_convolvers(self) -> None:
        """
        Initialize analytical and numerical convolvers based on sample model components.

        There is no delta function convolver, as delta functions are handled directly in the
        convolution method.
        """

        if self._analytical_sample_components:
            self._analytical_convolver = AnalyticalConvolution(
                energy=self.energy,
                energy_offset=self.energy_offset,
                sample_components=self._analytical_sample_components,
                resolution_components=self._resolution_components,
            )
        else:
            self._analytical_convolver = None

        if self._numerical_sample_components:
            self._numerical_convolver = NumericalConvolution(
                energy=self.energy,
                energy_offset=self.energy_offset,
                sample_components=self._numerical_sample_components,
                resolution_components=self._resolution_components,
                convolution_settings=self.convolution_settings,
                temperature=self.temperature,
                temperature_unit=self._temperature_unit,
                detailed_balance_settings=self.detailed_balance_settings,
                unit=self.unit,
            )
        else:
            self._numerical_convolver = None

    # Update some setters so the internal sample models are updated
    def __setattr__(self, name: str, value: any) -> None:
        """
        Custom setattr to invalidate convolution plan on relevant attribute changes, and build a
        new plan.

        The new plan is only built after initialization (when _reactions_enabled is True) to avoid
        issues during __init__.

        Parameters
        ----------
        name : str
            The name of the attribute to set.
        value : any
            The value to set the attribute to.
        """
        super().__setattr__(name, value)

        # Only rebuild the convolution plan if reactions are enabled, to
        # avoid issues during __init__
        if getattr(self, '_reactions_enabled', False) and name in self._invalidate_plan_on_change:
            self.convolution_settings.convolution_plan_is_valid = False
