# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.convolution_base import ConvolutionBase
from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.settings.convolution_settings import ConvolutionSettings
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings
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
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
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
        x_unit : str | sc.Unit, default='meV'
            The unit of the energy axis.
        y_unit : str | sc.Unit, default='dimensionless'
            The unit of the model output (intensity).
        display_name : str | None, default='MyConvolution'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If sample_components or resolution_components is None, or if temperature is not
            None, a number, or a Parameter, or if temperature_unit is not a string or
            sc.Unit.
        """
        super().__init__(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            x_unit=x_unit,
            y_unit=y_unit,
            energy_offset=energy_offset,
            display_name=display_name,
            unique_name=unique_name,
        )

        # ConvolutionBase tolerates None collections, but numerical convolvers cannot
        # convolve without both models — fail early with a clear error.
        if self._sample_components is None:
            raise TypeError(
                'sample_components must be a ComponentCollection or ModelComponent, not None.'
            )
        if self._resolution_components is None:
            raise TypeError(
                'resolution_components must be a ComponentCollection or ModelComponent, not None.'
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
        self._mark_convolution_plan_current()

    def _convolution_plan_is_current(self) -> bool:
        """
        Check whether this convolver's plan is up to date.

        Plan validity is tracked per convolver so several convolvers can share one settings
        object: each convolver stores the plan versions of its ConvolutionSettings and
        DetailedBalanceSettings it last rebuilt against (None after a convolver-local
        invalidation such as a new energy grid), and the settings bump their versions
        whenever a knob changes. In addition, a snapshot of the component collections'
        mutation versions and the energy_offset binding is compared, so in-place mutations
        of a live collection (e.g. append_component) or rebinding the offset to a new
        Parameter also invalidate the plan.

        Returns
        -------
        bool
            True if the plan does not need to be rebuilt.
        """
        seen_version = getattr(self, '_plan_seen_version', None)
        if seen_version is None:
            return False
        if not self.convolution_settings._plan_valid_for(seen_version):  # ruff: ignore[private-member-access]
            return False
        seen_db_version = getattr(self, '_plan_seen_db_version', None)
        if not self.detailed_balance_settings._plan_valid_for(seen_db_version):  # ruff: ignore[private-member-access]
            return False
        return getattr(self, '_plan_seen_state', None) == self._plan_state_snapshot()

    def _mark_convolution_plan_current(self) -> None:
        """Record that this convolver's plan matches its current state and settings."""
        self._plan_seen_version = self.convolution_settings._plan_version  # ruff: ignore[private-member-access]
        self._plan_seen_db_version = self.detailed_balance_settings._plan_version  # ruff: ignore[private-member-access]
        self._plan_seen_state = self._plan_state_snapshot()

    def _plan_state_snapshot(self) -> tuple:
        """
        Snapshot the mutable state the convolution plan was built from.

        Captures the identity and mutation version of the sample and resolution collections
        (so both rebinding and in-place mutation are detected) and the identity of the
        energy_offset Parameter (so rebinding to a new Parameter invalidates the plan while
        numeric assignment mutating the shared Parameter does not).

        Returns
        -------
        tuple
            A comparable snapshot of the plan-relevant state.
        """
        return (
            id(self._sample_components),
            self._sample_components.version,
            id(self._resolution_components),
            self._resolution_components.version,
            id(self._energy_offset),
        )

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
        # Convolver-local invalidation: other convolvers sharing the new settings object are
        # unaffected.
        self._plan_seen_version = None

    @ConvolutionBase.energy.setter
    def energy(self, energy: np.ndarray) -> None:
        """
        Set the energy array and invalidate this convolver's plan.

        The dense grid is rebuilt lazily on the next convolution. Other convolvers sharing the same
        ConvolutionSettings are unaffected — a new energy array is a convolver-local change.

        Parameters
        ----------
        energy : np.ndarray
            The new energy array.
        """
        ConvolutionBase.energy.fset(self, energy)
        self._plan_seen_version = None

    def convert_x_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the energy axis, energy_offset, and all components to the specified unit, and
        invalidate this convolver's plan.

        The dense grid is rebuilt lazily on the next convolution. Other convolvers sharing the
        same ConvolutionSettings are unaffected.

        Parameters
        ----------
        unit : str | sc.Unit
            The unit of the energy.
        """
        super().convert_x_unit(unit)
        self._plan_seen_version = None

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
        Set the upsample factor.

        Parameters
        ----------
        factor : Numeric | None
            The new upsample factor.
        """
        self.convolution_settings.upsample_factor = factor

    @property
    def extension_factor(self) -> float | None:
        """
        Get the extension factor.

        The extension factor determines how much the energy range is extended on both sides before
        convolution. 0.2 means extending by 20% of the original energy span on each side

        Returns
        -------
        float | None
            The extension factor, or None if unset (only valid while upsample_factor is None).
        """

        return self.convolution_settings.extension_factor

    @extension_factor.setter
    def extension_factor(self, factor: Numeric | None) -> None:
        """
        Set the extension factor.

        The extension factor determines how much the energy range is extended on both sides before
        convolution. 0.2 means extending by 20% of the original energy span on each side. None is
        accepted but requires upsample_factor to be None as well before the next convolution.

        Parameters
        ----------
        factor : Numeric | None
            The new extension factor.
        """
        self.convolution_settings.extension_factor = factor

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
        # Convolver-local invalidation: other convolvers sharing the new settings object are
        # unaffected.
        self._plan_seen_version = None

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
        # Validate up front so both the upsampled and the non-upsampled path raise the same
        # clear error (a single point has no spacing, so no grid can be built from it).
        if len(self.energy.values) < 2:
            raise ValueError('Energy array must have at least two points.')

        if self.upsample_factor is None:
            # Check if the array is uniformly spaced.
            energy_diff = np.diff(self.energy.values)
            is_uniform = np.allclose(energy_diff, energy_diff[0])
            if not is_uniform:
                raise ValueError(
                    'Input array `energy` must be uniformly spaced if upsample_factor is not given.'  # ruff: ignore[line-too-long]
                )
            energy_dense = self.energy.values

            energy_span_dense = self.energy.values.max() - self.energy.values.min()
        else:
            if self.extension_factor is None:
                raise ValueError(
                    'extension_factor must be a number (not None) when upsample_factor is set.'
                )
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

        return EnergyGrid(
            energy_dense=energy_dense,
            energy_dense_centered=energy_dense_centered,
            energy_dense_step=energy_dense_step,
            energy_span_dense=energy_span_dense,
            energy_even_length_offset=energy_even_length_offset,
        )

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
        components = model if isinstance(model, ComponentCollection) else [model]

        # Cover plain-width components as well as Voigt-style components with separate
        # gaussian_width/lorentzian_width parameters.
        width_attribute_names = ('width', 'gaussian_width', 'lorentzian_width')

        for comp in components:
            for attribute_name in width_attribute_names:
                width_param = getattr(comp, attribute_name, None)
                if width_param is None:
                    continue
                width_label = attribute_name.replace('_', ' ')
                if width_param.value > LARGE_WIDTH_THRESHOLD * self._energy_grid.energy_span_dense:
                    warnings.warn(
                        (
                            f"The {width_label} of the {model_name} component "
                            f"'{comp.unique_name}' "
                            f'({width_param.value}) is large compared to the span of the input '
                            f'array ({self._energy_grid.energy_span_dense}). '
                            f'This may lead to inaccuracies in the convolution. '
                            f'Increase extension_factor to improve accuracy.'
                        ),
                        UserWarning,
                        stacklevel=3,
                    )
                if width_param.value < SMALL_WIDTH_THRESHOLD * self._energy_grid.energy_dense_step:
                    warnings.warn(
                        (
                            f"The {width_label} of the {model_name} component "
                            f"'{comp.unique_name}' "
                            f'({width_param.value}) is small compared to the spacing of the input '
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
            f'x_unit={self.x_unit}, y_unit={self.y_unit}, '
            f'upsample_factor={self.upsample_factor}, '
            f'extension_factor={self.extension_factor}, '
            f'temperature={self.temperature}, '
            f'detailed_balance={self.detailed_balance_settings!r})'
        )
