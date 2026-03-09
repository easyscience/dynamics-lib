# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.fitting.fitter import Fitter as EasyScienceFitter
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from plopp.backends.matplotlib.figure import InteractiveFigure

from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.convolution.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent


class Analysis1d(AnalysisBase):
    """For analysing one-dimensional data, i.e. intensity as function of
    energy for a single Q index. Is used primarily in the Analysis
    class, but can also be used on its own for simpler analyses.

    Args:
        display_name (str): Display name of the analysis.
        unique_name (str or None): Unique name of the analysis. If None,
            a unique name is automatically generated.
        experiment (Experiment | None): The Experiment associated with
            this Analysis. If None, a default Experiment is created.
        sample_model (SampleModel | None): The SampleModel associated
            with this Analysis. If None, a default SampleModel is
            created.
        instrument_model (InstrumentModel | None): The InstrumentModel
            associated with this Analysis. If None, a default
            InstrumentModel is created.
        Q_index (int | None): The Q index to analyze. If None, the
            analysis will not be able to calculate or fit until a
            Q index is set.
        extra_parameters (Parameter | list[Parameter] | None): Extra
            parameters to be included in the analysis for advanced
            users. If None, no extra parameters are added.

    Attributes:
        experiment (Experiment): The Experiment associated with this
            Analysis.
        sample_model (SampleModel): The SampleModel associated with this
            Analysis.
        instrument_model (InstrumentModel): The InstrumentModel
            associated with this Analysis.
        Q (sc.Variable | None): The Q values from the associated
            Experiment, if available.
        energy (sc.Variable | None): The energy values from the
            associated Experiment, if available.
        temperature (Parameter | None): The temperature from the
            associated SampleModel, if available.
        Q_index (int | None): The Q index being analyzed.
        extra_parameters (list[Parameter]): The extra parameters
            included in this Analysis.
    """

    def __init__(
        self,
        display_name: str = 'MyAnalysis',
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        Q_index: int | None = None,
        extra_parameters: Parameter | list[Parameter] | None = None,
    ):
        """Initialize a Analysis1d.

        Args:
            display_name (str): Display name of the analysis.
            unique_name (str or None): Unique name of the analysis. If
                None, a unique name is automatically generated.
            experiment (Experiment | None): The Experiment associated
                with this Analysis. If None, a default Experiment is
                created.
            sample_model (SampleModel | None): The SampleModel
                associated with this Analysis. If None, a default
                SampleModel is created.
            instrument_model (InstrumentModel | None): The
                InstrumentModel associated with this Analysis. If None,
                a default InstrumentModel is created.
            Q_index (int | None): The Q index to analyze. If None, the
                analysis will not be able to calculate or fit until a
                Q index is set.
            extra_parameters (Parameter | list[Parameter] | None): Extra
                parameters to be included in the analysis for advanced
                users. If None, no extra parameters are added.
        """
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            extra_parameters=extra_parameters,
        )

        self._Q_index = self._verify_Q_index(Q_index)

        self._fit_result = None
        if self._Q_index is not None:
            self._convolver = self._create_convolver()
        else:
            self._convolver = None

    #############
    # Properties
    #############

    @property
    def Q_index(self) -> int | None:
        """Get the Q index associated with this Analysis.

        Returns:
            int | None: The Q index associated with this Analysis.
        """

        return self._Q_index

    @Q_index.setter
    def Q_index(self, value: int | None) -> None:
        """Set the Q index for single Q analysis.

        Args:
            index (int | None): The Q index.
        """

        self._Q_index = self._verify_Q_index(value)
        self._on_Q_index_changed()

    #############
    # Other methods
    #############

    def calculate(self) -> np.ndarray:
        """Calculate the model prediction for the chosen Q index. Makes
        sure the convolver is up to date before calculating.

        Returns:
            np.ndarray: The calculated model prediction.
        """

        self._convolver = self._create_convolver()

        return self._calculate()

    def _calculate(self) -> np.ndarray:
        """Calculate the model prediction for the chosen Q index. Does
        not check if the convolver is up to date.

        Returns:
            np.ndarray: The calculated model prediction.
        """

        sample_intensity = self._evaluate_sample()

        background_intensity = self._evaluate_background()

        sample_plus_background = sample_intensity + background_intensity

        return sample_plus_background

    def fit(self) -> FitResults:
        """Fit the model to the experimental data for the chosen Q
        index.

        The energy grid is fixed for the duration of the fit.
        Convolution objects are created once and reused during
        parameter optimization for performance reasons.

        Returns:
            FitResult: The result of the fit.

        Raises:
            ValueError: If no experiment is associated with this
                Analysis.

        Returns:
            FitResults: The result of the fit.
        """
        if self._experiment is None:
            raise ValueError('No experiment is associated with this Analysis.')

        # Create convolver once to reuse during fitting
        self._convolver = self._create_convolver()

        fitter = EasyScienceFitter(
            fit_object=self,
            fit_function=self.as_fit_function(),
        )

        x, y, weights = self.experiment._extract_x_y_weights_only_finite(
            Q_index=self._require_Q_index()
        )
        fit_result = fitter.fit(x=x, y=y, weights=weights)

        self._fit_result = fit_result

        return fit_result

    def as_fit_function(self, x=None, **kwargs) -> callable:
        """Return self._calculate as a fit function.

        The EasyScience fitter requires x as input, but
        self._calculate() already uses the correct energy from the
        experiment. So we ignore the x input and just return the
        calculated model.

        Args:
            x (Any): Ignored. The energy grid is taken from the
                experiment.
            kwargs (dict): Ignored. Included for compatibility with the
                EasyScience fitter.
        """

        def fit_function(x, **kwargs):
            return self._calculate()

        return fit_function

    def get_all_variables(self) -> list[DescriptorNumber]:
        """Get all variables used in the analysis.

        Returns:
            list[DescriptorNumber]: A list of all variables.
        """
        variables = self.sample_model.get_all_variables(Q_index=self.Q_index)

        variables.extend(self.instrument_model.get_all_variables(Q_index=self.Q_index))

        if self._extra_parameters:
            variables.extend(self._extra_parameters)

        return variables

    def plot_data_and_model(
        self,
        plot_components: bool = True,
        add_background=True,
        **kwargs,
    ) -> InteractiveFigure:
        """Plot the experimental data and the model prediction for the
        chosen Q index. Optionally also plot the individual components
        of the model.

        Uses Plopp for plotting: https://scipp.github.io/plopp/

        Args:
            plot_components (bool): Whether to plot the individual
                components of the model. Default is True.
            add_background (bool): Whether to add the background to the
                model prediction when plotting individual components.
            kwargs (dict): Keyword arguments to pass to the plotting
                function.

        Returns:
            InteractiveFigure: A plot of the data and model.
        """
        import plopp as pp

        if self.experiment.data is None:
            raise ValueError('No data to plot. Please load data first.')

        data = self.experiment.data['Q', self.Q_index]
        model_array = self._create_sample_scipp_array()

        component_dataset = self._create_components_dataset_single_Q(add_background=add_background)

        # Create a dataset containing the data, model, and individual
        # components for plotting.
        data_and_model = sc.Dataset({
            'Data': data,
            'Model': model_array,
        })

        data_and_model = sc.merge(data_and_model, component_dataset)
        plot_kwargs_defaults = {
            'title': self.display_name,
            'linestyle': {'Data': 'none', 'Model': '-'},
            'marker': {'Data': 'o', 'Model': 'none'},
            'color': {'Data': 'black', 'Model': 'red'},
            'markerfacecolor': {'Data': 'none', 'Model': 'none'},
        }

        if plot_components:
            for comp_name in component_dataset.keys():
                plot_kwargs_defaults['linestyle'][comp_name] = '--'
                plot_kwargs_defaults['marker'][comp_name] = None

        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)

        fig = pp.plot(
            data_and_model,
            **plot_kwargs_defaults,
        )
        return fig

    #############
    # Private methods: small utilities
    #############

    def _require_Q_index(self) -> int:
        """Get the Q index, ensuring it is set. Raises a ValueError if
        the Q index is not set.

        Returns:
            int: The Q index.

        Raises:
            ValueError: If the Q index is not set.
        """
        if self._Q_index is None:
            raise ValueError('Q_index must be set.')
        return self._Q_index

    def _on_Q_index_changed(self) -> None:
        """Handle changes to the Q index.

        This method is called whenever the Q index is changed. It
        updates the Convolution object for the new Q index.
        """
        self._convolver = self._create_convolver()

    #############
    # Private methods: evaluation
    #############

    def _evaluate_components(
        self,
        components: ComponentCollection | ModelComponent,
        convolver: Convolution | None = None,
        convolve: bool = True,
    ) -> np.ndarray:
        """Calculate the contribution of a set of components, optionally
        convolving with the resolution.

        If convolve is True and a
        Convolution object is provided (for full model evaluation), we
        use it to perform the convolution of the components with the
        resolution.
        If convolve is True but no Convolution object is
        provided, create a new Convolution object for the given
        components (for individual components).
        If convolve is False, evaluate the components directly without
        convolution (for background).

        Args:
            components (ComponentCollection | ModelComponent): The
                components to evaluate.
            convolver (Convolution | None): An optional Convolution
                object to use for convolution. If None, a new
                Convolution object will be created if convolve is True.
            convolve (bool): Whether to perform convolution with the
                resolution. Default is True.
        """

        Q_index = self._require_Q_index()
        energy = self.energy.values
        energy_offset = self.instrument_model.get_energy_offset_at_Q(Q_index)

        # If there are no components, return zero
        if isinstance(components, ComponentCollection) and components.is_empty:
            return np.zeros_like(energy)

        # No convolution
        if not convolve:
            return components.evaluate(energy - energy_offset.value)

        # If a convolver is provided, use it. This allows reusing the
        # same convolver for multiple evaluations during fitting for
        # performance reasons.
        if convolver is not None:
            return convolver.convolution()

        # If no convolver is provided, create a new one. This is for
        # evaluating individual components for plotting, where
        # performance is not important.

        # We don't create a convolver if the resolution is empty.
        resolution = self.instrument_model.resolution_model.get_component_collection(Q_index)
        if resolution.is_empty:
            return components.evaluate(energy - energy_offset.value)

        conv = Convolution(
            sample_components=components,
            resolution_components=resolution,
            energy=energy,
            temperature=self.temperature,
            energy_offset=energy_offset,
        )
        return conv.convolution()

    def _evaluate_sample(self) -> np.ndarray:
        """Evaluate the sample contribution for a given Q index.

        Assumes that self._convolver is up to date.

        Returns:
            np.ndarray: The evaluated sample contribution.
        """
        Q_index = self._require_Q_index()
        components = self.sample_model.get_component_collection(Q_index=Q_index)
        return self._evaluate_components(
            components=components,
            convolver=self._convolver,
            convolve=True,
        )

    def _evaluate_sample_component(
        self,
        component: ModelComponent,
    ) -> np.ndarray:
        """Evaluate a single sample component for the chosen Q index.

        Args:
            component (ModelComponent): The sample component to
                evaluate.

        Returns:
            np.ndarray: The evaluated sample component contribution.
        """
        return self._evaluate_components(
            components=component,
            convolver=None,
            convolve=True,
        )

    def _evaluate_background(self) -> np.ndarray:
        """Evaluate the background contribution for the chosen Q index.

        Returns:
            np.ndarray: The evaluated background contribution.
        """
        Q_index = self._require_Q_index()
        background_components = self.instrument_model.background_model.get_component_collection(
            Q_index=Q_index
        )
        return self._evaluate_components(
            components=background_components,
            convolver=None,
            convolve=False,
        )

    def _evaluate_background_component(
        self,
        component: ModelComponent,
    ) -> np.ndarray:
        """Evaluate a single background component for the chosen Q
        index.

        Args:
            component (ModelComponent): The background component to
                evaluate.

        Returns:
            np.ndarray: The evaluated background component contribution.
        """

        return self._evaluate_components(
            components=component,
            convolver=None,
            convolve=False,
        )

    def _create_convolver(self) -> Convolution | None:
        """Initialize and return a Convolution object for the chosen Q
        index. If the necessary components for convolution are not
        available, return None.

        Returns:
            Convolution | None: The initialized Convolution object or
                None if not available.
        """
        Q_index = self._require_Q_index()

        sample_components = self.sample_model.get_component_collection(Q_index)
        if sample_components.is_empty:
            return None

        resolution_components = self.instrument_model.resolution_model.get_component_collection(
            Q_index
        )
        if resolution_components.is_empty:
            return None
        energy = self.energy
        # TODO: allow convolution options to be set.
        convolver = Convolution(
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy=energy,
            temperature=self.temperature,
            energy_offset=self.instrument_model.get_energy_offset_at_Q(Q_index),
        )
        return convolver

    #############
    # Private methods: create scipp arrays for plotting
    #############

    def _create_component_scipp_array(
        self,
        component: ModelComponent,
        background: np.ndarray | None = None,
    ) -> sc.DataArray:
        """Create a scipp DataArray for a single component. Adds the
        background if it is not None.

        Args:
            component (ModelComponent): The component to evaluate
            background (np.ndarray | None): Optional background to add
                to the component.

        Returns:
            sc.DataArray: The model calculation of the component.
        """

        values = self._evaluate_sample_component(component=component)
        if background is not None:
            values += background
        return self._to_scipp_array(values=values)

    def _create_background_component_scipp_array(
        self,
        component: ModelComponent,
    ) -> sc.DataArray:
        """Create a scipp DataArray for a single background component.

        Args:
            component (ModelComponent): The component to evaluate.

        Returns:
            sc.DataArray: The model calculation of the component.
        """

        values = self._evaluate_background_component(component=component)
        return self._to_scipp_array(values=values)

    def _create_sample_scipp_array(self) -> sc.DataArray:
        """Create a scipp DataArray for the full sample model including
        background.

        Returns:
            sc.DataArray: The model calculation of the full sample
                model.
        """
        values = self._calculate()
        return self._to_scipp_array(values=values)

    def _create_components_dataset_single_Q(
        self,
        add_background: bool = True,
    ) -> dict[str, sc.DataArray]:
        """Create sc.DataArrays for all sample and background
        components.

        Args:
            add_background (bool): Whether to add background components.

        Returns:
            dict[str, sc.DataArray]: A dictionary of component names to
                their corresponding sc.DataArrays.
        """

        scipp_arrays = {}
        sample_components = self.sample_model.get_component_collection(
            Q_index=self.Q_index
        ).components

        background_components = self.instrument_model.background_model.get_component_collection(
            Q_index=self.Q_index
        ).components
        background = self._evaluate_background() if add_background else None
        for component in sample_components:
            scipp_arrays[component.display_name] = self._create_component_scipp_array(
                component=component, background=background
            )
        for component in background_components:
            scipp_arrays[component.display_name] = self._create_background_component_scipp_array(
                component=component
            )
        return sc.Dataset(scipp_arrays)

    def _to_scipp_array(self, values: np.ndarray) -> sc.DataArray:
        """Convert a numpy array of values to a sc.DataArray with the
        correct coordinates for energy and Q.

        Args:
            values (np.ndarray): The values to convert.

        Returns:
            sc.DataArray: The converted sc.DataArray.
        """

        return sc.DataArray(
            data=sc.array(dims=['energy'], values=values),
            coords={
                'energy': self.energy,
                'Q': self.Q[self.Q_index],
            },
        )
