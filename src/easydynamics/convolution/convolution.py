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

Numerical = float | int


class Convolution(NumericalConvolutionBase):
    """Convolution class that combines analytical and numerical
    convolution methods to efficiently perform convolutions of
    ComponentCollections with ResolutionComponents. Supports analytical
    convolution for pairs of analytical model components (DeltaFunction,
    Gaussian, Lorentzian, Voigt), while using numerical convolution for
    other components. If temperature is provided, detailed balance
    correction is applied to the sample model. In this case, all
    convolutions are handled numerically. Includes a setting to
    normalize the detailed balance correction. Includes optional
    upsampling and extended range to improve accuracy of the numerical
    convolutions. Also warns about numerical instabilities if peaks are
    very wide or very narrow.

    Args:
    energy : np.ndarray or scipp.Variable
        1D array of energy values where the convolution is evaluated.
    sample_components : ComponentCollection or ModelComponent
        The sample components to be convolved.
    resolution_components : ComponentCollection or ModelComponent
        The resolution components to convolve with.
    upsample_factor : int, optional
        The factor by which to upsample the input data before
        convolution. Default is 5.
    extension_factor : float, optional
        The factor by which to extend the input data range before
        convolution. Default is 0.2.
    temperature : Parameter, float, or None, optional
        The temperature to use for detailed balance correction.
        Default is None.
    temperature_unit : str or sc.Unit, optional
        The unit of the temperature parameter. Default is 'K'.
    energy_unit : str or sc.Unit, optional
        The unit of the energy. Default is 'meV'.
    normalize_detailed_balance : bool, optional
        Whether to normalize the detailed balance correction.
        Default is True.
    """

    # When these attributes are changed, the convolution plan
    # needs to be rebuilt
    _invalidate_plan_on_change = {
        'energy',
        '_energy',
        '_energy_grid',
        '_sample_components',
        '_resolution_components',
        '_temperature',
        '_upsample_factor',
        '_extension_factor',
        '_energy_unit',
        '_normalize_detailed_balance',
    }

    def __init__(
        self,
        energy: np.ndarray | sc.Variable,
        sample_components: ComponentCollection | ModelComponent,
        resolution_components: ComponentCollection | ModelComponent,
        upsample_factor: Numerical = 5,
        extension_factor: Numerical = 0.2,
        temperature: Parameter | Numerical | None = None,
        temperature_unit: str | sc.Unit = 'K',
        energy_unit: str | sc.Unit = 'meV',
        normalize_detailed_balance: bool = True,
    ):
        self._convolution_plan_is_valid = False
        self._reactions_enabled = False
        super().__init__(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            upsample_factor=upsample_factor,
            extension_factor=extension_factor,
            temperature=temperature,
            temperature_unit=temperature_unit,
            energy_unit=energy_unit,
            normalize_detailed_balance=normalize_detailed_balance,
        )

        self._reactions_enabled = True
        # Separate sample model components into pairs that can be
        # handled analytically, delta functions, and the rest
        # Also initialize analytical and numerical convolvers based on s
        # ample model component
        self._build_convolution_plan()

    def convolution(
        self,
    ) -> np.ndarray:
        """Perform convolution using analytical convolutions where
        possible, and numerical convolutions for the remaining
        components.

        Returns:
            np.ndarray
                The convolved values evaluated at energy.
        """
        if not self._convolution_plan_is_valid:
            self._build_convolution_plan()
        total = np.zeros_like(self.energy.values, dtype=float)

        # Analytical convolution
        if self._analytical_convolver is not None:
            total += self._analytical_convolver.convolution()

        # Numerical convolution
        if self._numerical_convolver is not None:
            total += self._numerical_convolver.convolution()

        # Delta function components
        if self._delta_sample_components.components:
            total += self._convolve_delta_functions()

        return total

    def _convolve_delta_functions(self) -> np.ndarray:
        "Convolve delta function components of the sample model with"
        'the resolution components.'
        'No detailed balance correction is applied to delta functions.'
        return sum(
            delta.area.value
            * self._resolution_components.evaluate(self.energy.values - delta.center.value)
            for delta in self._delta_sample_components.components
        )

    def _check_if_pair_is_analytic(
        self,
        sample_component: ModelComponent,
        resolution_component: ModelComponent,
    ) -> bool:
        """Check if the convolution of the given component pair can be
        handled analytically.

        Args:
            sample_component : ModelComponent
                The sample component to be convolved.
            resolution_component : ModelComponent
                The resolution component to convolve with.
        Returns:
            bool
                True if the component pair can be handled analytically,
                False otherwise.
        """

        if not isinstance(sample_component, ModelComponent):
            raise TypeError(
                f'`sample_component` is an instance of {type(sample_component).__name__}, \
                but must be a ModelComponent.'
            )

        if not isinstance(resolution_component, ModelComponent):
            raise TypeError(
                f'`resolution_component` is an instance of {type(resolution_component).__name__}, \
                    but must be a ModelComponent.'
            )

        if isinstance(resolution_component, DeltaFunction):
            raise TypeError(
                'resolution components contains delta functions. This is not supported.'
            )

        analytical_types = (Gaussian, Lorentzian, Voigt)
        if isinstance(sample_component, analytical_types) and isinstance(
            resolution_component, analytical_types
        ):
            return True

        return False

    def _build_convolution_plan(self) -> None:
        """Separate sample model components into analytical pairs, delta
        functions, and the rest.
        """

        analytical_sample_components = ComponentCollection()
        delta_sample_components = ComponentCollection()
        numerical_sample_components = ComponentCollection()

        for sample_component in self._sample_components.components:
            # If delta function, put in delta sample model and go to the
            # next component
            if isinstance(sample_component, DeltaFunction):
                delta_sample_components.append_component(sample_component)
                continue

            # If temperature is set, all other components go to
            # numerical sample model
            if self.temperature is not None:
                numerical_sample_components.append_component(sample_component)
                continue

            # If temperature is not set, check if all
            # resolution components can be convolved analytically with
            # this sample component
            pair_is_analytic = []
            for resolution_component in self._resolution_components.components:
                pair_is_analytic.append(
                    self._check_if_pair_is_analytic(sample_component, resolution_component)
                )
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
        self._convolution_plan_is_valid = True

        # Update convolvers
        self._set_convolvers()

    def _set_convolvers(self) -> None:
        """Initialize analytical and numerical convolvers based on
        sample model components.

        There is no delta function convolver, as delta functions are
        handled directly in the convolution method.
        """

        if self._analytical_sample_components.components:
            self._analytical_convolver = AnalyticalConvolution(
                energy=self.energy,
                sample_components=self._analytical_sample_components,
                resolution_components=self._resolution_components,
            )
        else:
            self._analytical_convolver = None

        if self._numerical_sample_components.components:
            self._numerical_convolver = NumericalConvolution(
                energy=self.energy,
                sample_components=self._numerical_sample_components,
                resolution_components=self._resolution_components,
                upsample_factor=self.upsample_factor,
                extension_factor=self.extension_factor,
                temperature=self.temperature,
                temperature_unit=self._temperature_unit,
                normalize_detailed_balance=self.normalize_detailed_balance,
            )
        else:
            self._numerical_convolver = None

    # Update some setters so the internal sample models are updated
    def __setattr__(self, name, value):
        """Custom setattr to invalidate convolution plan on relevant
        attribute changes, and build a new plan.

        The new plan is only built after initialization (when
        _reactions_enabled is True) to avoid issues during __init__.
        """
        super().__setattr__(name, value)

        if name in self._invalidate_plan_on_change:
            self._convolution_plan_is_valid = False

        if getattr(self, '_reactions_enabled', False) and name in self._invalidate_plan_on_change:
            self._build_convolution_plan()
