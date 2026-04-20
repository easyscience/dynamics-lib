# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.convolution_base import ConvolutionBase
from easydynamics.convolution.convolution_settings import ConvolutionSettings
from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.detailed_balance_settings import DetailedBalanceSettings
from easydynamics.utils.utils import Numeric

# The thresholds are illustrated in
# performance_tests/convolution/convolution_width_thresholds.ipynb
LARGE_WIDTH_THRESHOLD = (
    0.1  # Threshold for large widths compared to span - warn if width > 10% of span
)
SMALL_WIDTH_THRESHOLD = (
    1.0  # Threshold for small widths compared to bin spacing - warn if width < dx
)


class NumericalConvolutionBase(ConvolutionBase):
    """
    Base class for numerical convolutions of sample and resolution models.

    Provides methods to handle upsampling, extension, and detailed balance correction. This base
    class has no convolution functionality.
    """

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
        Initialize the NumericalConvolutionBase.

        Parameters
        ----------
        energy : np.ndarray | sc.Variable
            1D array of energy values where the convolution is evaluated.
        sample_components : ComponentCollection | ModelComponent
            The components to be convolved.
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

        Raises
        ------
        TypeError
            If temperature is not None, a number, or a Parameter, or if temperature_unit is not a
            string or sc.Unit.
        """
        super().__init__(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            unit=unit,
            energy_offset=energy_offset,
            display_name=display_name,
            unique_name=unique_name,
        )

        if temperature is not None and not isinstance(temperature, (Numeric, Parameter)):
            raise TypeError('Temperature must be None, a number or a Parameter.')

        if not isinstance(temperature_unit, (str, sc.Unit)):
            raise TypeError('Temperature_unit must be a string or sc.Unit.')
        self._temperature_unit = temperature_unit
        self._temperature = None
        self.temperature = temperature

        if convolution_settings is None:
            convolution_settings = ConvolutionSettings()
        self._convolution_settings = convolution_settings

        if detailed_balance_settings is None:
            detailed_balance_settings = DetailedBalanceSettings()
        if not isinstance(detailed_balance_settings, DetailedBalanceSettings):
            raise TypeError(
                'detailed_balance_settings must be a DetailedBalanceSettings instance.'
            )
        self._detailed_balance_settings = detailed_balance_settings

        # Create a dense grid to improve accuracy.
        # When upsample_factor>1, we evaluate on this grid and
        # interpolate back to the original values at the end
        self._energy_grid = self._create_energy_grid()

    @property
    def convolution_settings(self) -> ConvolutionSettings:
        """
        Get the convolution settings.

        Returns
        -------
        ConvolutionSettings
            The convolution settings.
        """

        return self._convolution_settings

    @convolution_settings.setter
    def convolution_settings(self, settings: ConvolutionSettings) -> None:
        """
        Set the convolution settings and recreate the dense grid.

        Parameters
        ----------
        settings : ConvolutionSettings
            The new convolution settings.

        Raises
        ------
        TypeError
            If settings is not a ConvolutionSettings instance.
        """
        if not isinstance(settings, ConvolutionSettings):
            raise TypeError('settings must be a ConvolutionSettings instance.')
        self._convolution_settings = settings
        self._convolution_settings.convolution_plan_is_valid = False

    @ConvolutionBase.energy.setter
    def energy(self, energy: np.ndarray) -> None:
        """
        Set the energy array and recreate the dense grid.

        Parameters
        ----------
        energy : np.ndarray
            The new energy array.
        """
        ConvolutionBase.energy.fset(self, energy)
        self.convolution_settings.convolution_plan_is_valid = False

    @property
    def upsample_factor(self) -> Numeric | None:
        """
        Get the upsample factor.

        Returns
        -------
        Numeric | None
            The upsample factor.
        """

        return self.convolution_settings.upsample_factor

    @upsample_factor.setter
    def upsample_factor(self, factor: Numeric | None) -> None:
        """
        Set the upsample factor and recreate the dense grid.

        Parameters
        ----------
        factor : Numeric | None
            The new upsample factor.

        Raises
        ------
        TypeError
            If factor is not a number or None.
        ValueError
            If factor is not greater than 1.
        """
        if factor is None:
            self.convolution_settings.upsample_factor = factor
            return

        if not isinstance(factor, Numeric):
            raise TypeError('Upsample factor must be a numerical value or None.')
        factor = float(factor)
        if factor <= 1.0:
            raise ValueError('Upsample factor must be greater than 1.')

        self.convolution_settings.upsample_factor = factor

    @property
    def extension_factor(self) -> float:
        """
        Get the extension factor.

        The extension factor determines how much the energy range is extended on both sides before
        convolution. 0.2 means extending by 20% of the original energy span on each side

        Returns
        -------
        float
            The extension factor.
        """

        return self.convolution_settings.extension_factor

    @extension_factor.setter
    def extension_factor(self, factor: Numeric) -> None:
        """
        Set the extension factor and recreate the dense grid.

        The extension factor determines how much the energy range is extended on both sides before
        convolution. 0.2 means extending by 20% of the original energy span on each side.

        Parameters
        ----------
        factor : Numeric
            The new extension factor.

        Raises
        ------
        TypeError
            If factor is not a number.
        ValueError
            If factor is negative.
        """

        if not isinstance(factor, Numeric):
            raise TypeError('Extension factor must be a number.')
        if factor < 0.0:
            raise ValueError('Extension factor must be non-negative.')

        self.convolution_settings.extension_factor = float(factor)

    @property
    def temperature(self) -> Parameter | None:
        """
        Get the temperature.

        Returns
        -------
        Parameter | None
            The temperature parameter, or None if detailed balance correction is disabled.
        """

        return self._temperature

    @temperature.setter
    def temperature(self, temp: Parameter | Numeric | None) -> None:
        """
        Set the temperature.

        If None, disables detailed balance correction and removes the temperature parameter.

        Parameters
        ----------
        temp : Parameter | Numeric | None
            The temperature to set. The unit will be the same as the existing temperature parameter
            if it exists, otherwise 'K'.

        Raises
        ------
        TypeError
            If temp is not a Numeric, Parameter, or None.
        """

        if temp is None:
            self._temperature = None
        elif isinstance(temp, Numeric):
            if self._temperature is not None:
                self._temperature.value = float(temp)
            else:
                self._temperature = Parameter(
                    name='temperature',
                    value=float(temp),
                    unit=self._temperature_unit,
                    fixed=True,
                )
        elif isinstance(temp, Parameter):
            self._temperature = temp
        else:
            raise TypeError('Temperature must be None, a float or a Parameter.')

    @property
    def detailed_balance_settings(self) -> DetailedBalanceSettings:
        """
        Get the DetailedBalanceSettings of the Convolution.

        Returns
        -------
        DetailedBalanceSettings
            The DetailedBalanceSettings of the Convolution.
        """
        return self._detailed_balance_settings

    @detailed_balance_settings.setter
    def detailed_balance_settings(self, value: DetailedBalanceSettings) -> None:
        """
        Set the DetailedBalanceSettings of the Convolution.

        Parameters
        ----------
        value : DetailedBalanceSettings
            The DetailedBalanceSettings to set.

        Raises
        ------
        TypeError
            If value is not a DetailedBalanceSettings.
        """
        if not isinstance(value, DetailedBalanceSettings):
            raise TypeError('detailed_balance_settings must be a DetailedBalanceSettings')
        self._detailed_balance_settings = value

    def _create_energy_grid(
        self,
    ) -> EnergyGrid:
        """
        Create a dense grid by upsampling and extending the energy array.

        If upsample_factor is None, no upsampling or extension is performed. This dense grid is
        used for convolution to improve accuracy.

        Raises
        ------
        ValueError
            If energy array is not uniformly spaced when upsample_factor is None, or if energy
            array has less than 2 points.

        Returns
        -------
        EnergyGrid
            The dense grid created by upsampling and extending energy.
        """
        if self.upsample_factor is None:
            # Check if the array is uniformly spaced.
            energy_diff = np.diff(self.energy.values)
            is_uniform = np.allclose(energy_diff, energy_diff[0])
            if not is_uniform:
                raise ValueError(
                    'Input array `energy` must be uniformly spaced if upsample_factor is not given.'  # noqa: E501
                )
            energy_dense = self.energy.values

            energy_span_dense = self.energy.values.max() - self.energy.values.min()
        else:
            # Create an extended and upsampled energy grid
            energy_min, energy_max = self.energy.values.min(), self.energy.values.max()
            energy_span_original = energy_max - energy_min
            extra = self.extension_factor / 2 * energy_span_original
            extended_min = energy_min - extra
            extended_max = energy_max + extra
            num_points = round(len(self.energy.values) * self.upsample_factor)
            energy_dense = np.linspace(extended_min, extended_max, num_points)
            energy_span_dense = extended_max - extended_min

        if len(energy_dense) < 2:
            raise ValueError('Energy array must have at least two points.')
        energy_dense_step = energy_dense[1] - energy_dense[0]

        # Handle offset for even length of energy_dense in convolution.
        # The convolution of two arrays of length N is of length 2N-1.
        #  When using 'same' mode, only the central N points are kept,
        # so the output has the same length as the input.
        # However, if N is even, the center falls between two points,
        # leading to a half-bin offset.
        # For example, if N=4, the convolution has length 7, and when we
        # select the 4 central points we either get
        # indices [2,3,4,5] or [1,2,3,4], both of which are offset by
        # 0.5*dx from the true center at index 3.5.
        energy_even_length_offset = -0.5 * energy_dense_step if len(energy_dense) % 2 == 0 else 0.0

        # Handle the case when energy_dense is not symmetric around 0.
        # The resolution is still centered around zero (or close to it),
        # so it needs to be evaluated there.
        if not np.isclose(energy_dense.mean(), 0.0):
            energy_dense_centered = np.linspace(
                -0.5 * energy_span_dense, 0.5 * energy_span_dense, len(energy_dense)
            )
        else:
            energy_dense_centered = energy_dense

        energy_grid = EnergyGrid(
            energy_dense=energy_dense,
            energy_dense_centered=energy_dense_centered,
            energy_dense_step=energy_dense_step,
            energy_span_dense=energy_span_dense,
            energy_even_length_offset=energy_even_length_offset,
        )
        self._energy_grid = energy_grid
        self.convolution_settings.convolution_plan_is_valid = True
        return energy_grid

    def _check_width_thresholds(
        self,
        model: ComponentCollection | ModelComponent,
        model_name: str,
    ) -> None:
        """
        Helper function to check and warn if components are wide compared to the span of the data,
        or narrow compared to the spacing.

        In both cases, the convolution accuracy may be compromised.

        Parameters
        ----------
        model : ComponentCollection | ModelComponent
            The model to check.
        model_name : str
            A string indicating whether the model is a 'sample model' or 'resolution model' for
            warning messages.
        """

        # Handle ComponentCollection or ModelComponent
        components = model.components if isinstance(model, ComponentCollection) else [model]

        for comp in components:
            if hasattr(comp, 'width'):
                if comp.width.value > LARGE_WIDTH_THRESHOLD * self._energy_grid.energy_span_dense:
                    warnings.warn(
                        (
                            f"The width of the {model_name} component '{comp.unique_name}' "
                            f'({comp.width.value}) is large compared to the span of the input '
                            f'array ({self._energy_grid.energy_span_dense}). '
                            f'This may lead to inaccuracies in the convolution. '
                            f'Increase extension_factor to improve accuracy.'
                        ),
                        UserWarning,
                        stacklevel=3,
                    )
                if comp.width.value < SMALL_WIDTH_THRESHOLD * self._energy_grid.energy_dense_step:
                    warnings.warn(
                        (
                            f"The width of the {model_name} component '{comp.unique_name}' "
                            f'({comp.width.value}) is small compared to the spacing of the input '
                            f'array ({self._energy_grid.energy_dense_step}). '
                            f'This may lead to inaccuracies in the convolution. '
                            f'Increase upsample_factor to improve accuracy.'
                        ),
                        UserWarning,
                        stacklevel=3,
                    )

    def __repr__(self) -> str:
        """
        Return a string representation of the NumericalConvolutionBase.

        Returns
        -------
        str
            A string representation of the NumericalConvolutionBase.
        """
        return (
            f'{self.__class__.__name__}('
            f'energy=array of shape {self.energy.values.shape},\n '
            f'sample_components={self.sample_components!r}, \n'
            f'resolution_components={self.resolution_components!r},\n '
            f'unit={self.unit}, '
            f'upsample_factor={self.upsample_factor}, '
            f'extension_factor={self.extension_factor}, '
            f'temperature={self.temperature}, '
            f'detailed_balance={self.detailed_balance_settings!r})'
        )
