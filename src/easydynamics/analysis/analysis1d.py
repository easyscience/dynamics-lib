# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import scipp as sc
from easyscience.fitting.fitter import Fitter as EasyScienceFitter
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from plopp.backends.matplotlib.figure import InteractiveFigure

from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.analysis.posterior_labels import ParameterLabels
from easydynamics.analysis.posterior_sampling import PosteriorSampler
from easydynamics.convolution.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.settings.convolution_settings import ConvolutionSettings
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings
from easydynamics.utils.detailed_balance import detailed_balance_factor
from easydynamics.utils.plotting import slicerplot_with_residuals
from easydynamics.utils.utils import verify_Q_index


class Analysis1d(AnalysisBase):
    """
    For analysing one-dimensional data, i.e. intensity as function of energy for a single Q index.

    Is used primarily in the Analysis class, but can also be used on its own for simpler analyses.

    Besides least-squares fitting with :meth:`fit`, the posterior distribution of the free
    parameters can be explored through :attr:`bayesian`; see
    :class:`~easydynamics.analysis.posterior_sampling.PosteriorSampler`.

    Examples
    --------
    **Fitting a single Q slice**

    Select a Q index with ``Q_index`` to fit only that slice of the dataset:
    ```python
    import pooch
    import easydynamics as edyn
    import easydynamics.sample_model as sm
    from easydynamics.analysis.analysis1d import Analysis1d

    file_path = pooch.retrieve(
        url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/master/docs/docs/tutorials/data/vanadium_data_example.h5',
        known_hash='16cc1b327c303feeb88fb9dda5390dc4880b62396b1793f98c6fef0b27c7b873',
    )
    experiment = edyn.Experiment('Vanadium')
    experiment.load_hdf5(filename=file_path)

    sample_model = sm.SampleModel(components=sm.DeltaFunction(area=1))
    resolution_model = sm.ResolutionModel(components=sm.Gaussian(width=0.1))
    background_model = sm.BackgroundModel(components=sm.Polynomial(coefficients=[0.001]))
    instrument_model = sm.InstrumentModel(
        resolution_model=resolution_model,
        background_model=background_model,
    )

    analysis = Analysis1d(
        display_name='Vanadium 1D Analysis',
        experiment=experiment,
        sample_model=sample_model,
        instrument_model=instrument_model,
        Q_index=5,
    )
    analysis.fit()
    analysis.plot_data_and_model(plot_residuals=True)
    ```
    """

    def __init__(
        self,
        display_name: str | None = 'MyAnalysis',
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        Q_index: int | None = None,
        convolution_settings: ConvolutionSettings | None = None,
        detailed_balance_settings: DetailedBalanceSettings | None = None,
        extra_parameters: Parameter | list[Parameter] | None = None,
    ) -> None:
        """
        Initialize a Analysis1d.

        Parameters
        ----------
        display_name : str | None, default='MyAnalysis'
            Display name of the analysis.
        unique_name : str | None, default=None
            Unique name of the analysis. If None, a unique name is automatically generated.
        experiment : Experiment | None, default=None
            The Experiment associated with this Analysis. If None, a default Experiment is created.
        sample_model : SampleModel | None, default=None
            The SampleModel associated with this Analysis. If None, a default SampleModel is
            created.
        instrument_model : InstrumentModel | None, default=None
            The InstrumentModel associated with this Analysis. If None, a default InstrumentModel
            is created.
        Q_index : int | None, default=None
            The Q index to analyze. If None, the analysis will not be able to calculate or fit
            until a Q index is set.
        convolution_settings : ConvolutionSettings | None, default=None
            The settings for the convolution. If None, default settings will be used.
        detailed_balance_settings : DetailedBalanceSettings | None, default=None
            The settings for detailed balance. If None, default settings will be used.
        extra_parameters : Parameter | list[Parameter] | None, default=None
            Extra parameters to be included in the analysis for advanced users. If None, no extra
            parameters are added.
        """
        # Initialize state read by observer callbacks (e.g. _on_experiment_changed) before
        # super().__init__ wires the sub-models and fires them.
        self._Q_index = None
        self._masked_energy = None
        self._fit_result = None
        self._convolver = None
        self._convolver_is_dirty = True
        self._fitter = None
        self._fitter_is_dirty = True
        self._bayesian = None

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            convolution_settings=convolution_settings,
            detailed_balance_settings=detailed_balance_settings,
            extra_parameters=extra_parameters,
        )

        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)
        self._Q_index = Q_index

        if self._Q_index is not None and self.experiment is not None:
            self._masked_energy = self.experiment.get_masked_energy(Q_index=self._Q_index)

    #############
    # Properties
    #############

    @property
    def Q_index(self) -> int | None:
        """
        Get the Q index associated with this Analysis.

        Returns
        -------
        int | None
            The Q index associated with this Analysis.
        """

        return self._Q_index

    @Q_index.setter
    def Q_index(self, value: int | None) -> None:
        """
        Set the Q index for single Q analysis.

        Parameters
        ----------
        value : int | None
            The Q index.
        """
        verify_Q_index(Q_index=value, Q=self.Q, allow_none=True)
        self._Q_index = value
        self._on_Q_index_changed()

    #############
    # Other methods
    #############

    def calculate(self, energy: sc.Variable | None = None) -> np.ndarray:
        """
        Calculate the model prediction for the chosen Q index.

        Creates a new convolver before calculating without touching the stored convolver.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for calculation. If None, the energy grid from the
            experiment is used.

        Returns
        -------
        np.ndarray
            The calculated model prediction.
        """
        energy = self._verify_energy(energy)
        convolver = self._create_convolver(energy=energy)
        return self._calculate(energy=energy, convolver=convolver)

    def _calculate(
        self, energy: sc.Variable | None = None, convolver: Convolution | None = None
    ) -> np.ndarray:
        """
        Calculate the model prediction for the chosen Q index.

        Does not check if the convolver is up to date.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for calculation. If None, the energy grid from the
            experiment is used.
        convolver : Convolution | None, default=None
            Optional convolver to use. If None, uses self._convolver.

        Returns
        -------
        np.ndarray
            The calculated model prediction.
        """
        if convolver is None:
            convolver = self._convolver
        Q_index = self._require_Q_index()
        sample = self._evaluate_with_convolution(
            self.sample_model.get_component_collection(Q_index),
            energy,
            convolver=convolver,
        )
        background = self._evaluate_direct(
            self.instrument_model.background_model.get_component_collection(Q_index),
            energy,
        )
        return sample + background

    def fit(self) -> FitResults:
        """
        Fit the model to the experimental data for the chosen Q index.

        The energy grid is fixed for the duration of the fit. Convolution objects are created once
        and reused during parameter optimization for performance reasons.

        Raises
        ------
        ValueError
            If no experiment is associated with this Analysis.

        Returns
        -------
        FitResults
            The result of the fit.
        """
        if self._experiment is None:
            raise ValueError('No experiment is associated with this Analysis.')

        self._prepare_for_sampling()

        x, y, weights = self._sampling_data()
        fit_result = self.fitter.fit(x=x, y=y, weights=weights)

        self._fit_result = fit_result

        return fit_result

    @property
    def fitter(self) -> EasyScienceFitter:
        """
        The EasyScience Fitter used for fitting and sampling, built on first use.

        Exposed so the minimizer, tolerance, and maximum evaluation count can be configured
        directly, e.g. ``analysis.fitter.switch_minimizer(AvailableMinimizers.Bumps)``.

        Returns
        -------
        EasyScienceFitter
            The cached Fitter.
        """
        if self._fitter_is_dirty or self._fitter is None:
            self._fitter = EasyScienceFitter(
                fit_object=self,
                fit_function=self.as_fit_function(),
            )
            self._fitter_is_dirty = False
        return self._fitter

    @property
    def bayesian(self) -> PosteriorSampler:
        """
        Bayesian posterior sampling for this Analysis, created on first use.

        Returns
        -------
        PosteriorSampler
            The sampler, which holds any chain that has been run.
        """
        if self._bayesian is None:
            self._bayesian = PosteriorSampler(
                analysis=self,
                sampling_data=self._sampling_data,
                chain_parameters=self._chain_parameters,
                parameter_labels=self._parameter_labels,
                prepare=self._prepare_for_sampling,
            )
        return self._bayesian

    def _invalidate_fitter(self) -> None:
        """Mark the Fitter, and the Sampler built from it, as needing a rebuild."""
        self._fitter_is_dirty = True
        self._invalidate_bayesian_sampler()

    def _invalidate_bayesian_sampler(self) -> None:
        """Mark the Sampler as needing a rebuild, the data having changed."""
        if self._bayesian is not None:
            self._bayesian.invalidate()

    #############
    # The contract PosteriorSampler relies on
    #############

    def _parameter_labels(self) -> ParameterLabels:
        """
        Get labels for the chain's parameters.

        A single Q index holds one copy of each parameter, so nothing needs qualifying.

        Returns
        -------
        ParameterLabels
            Labels over the current free parameters.
        """
        return ParameterLabels(self._chain_parameters())

    def _sampling_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the finite data for the chosen Q index, as used by both fitting and sampling.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The ``(x, y, weights)`` triple.
        """
        x, y, weights, _ = self.experiment.extract_x_y_weights_only_finite(
            Q_index=self._require_Q_index()
        )
        return x, y, weights

    def _chain_parameters(self) -> list[Parameter]:
        """
        Get the free parameters of this Analysis.

        Returns
        -------
        list[Parameter]
            The parameters that are free to vary, which are the ones the sampler explores.
        """
        return self.get_free_parameters()

    def _prepare_for_sampling(self) -> None:
        """
        Rebuild the convolver if anything it depends on has changed.

        The energy grid is fixed for the duration of a fit or a sampling run, so the convolution
        objects are built once here and reused for every model evaluation.
        """
        if (
            self.sample_model.component_collections_is_dirty
            or self.instrument_model.resolution_model.component_collections_is_dirty
        ):
            self._convolver_is_dirty = True

        self._ensure_convolver_current()

    def as_fit_function(
        self,
        _x: np.ndarray | sc.Variable | None = None,
        **kwargs: dict[str, Any],  # ruff: ignore[unused-method-argument]
    ) -> callable:
        """
        Return self._calculate as a fit function.

        The EasyScience fitter requires x as input, but self._calculate() already uses the correct
        energy from the experiment. So we ignore the x input and just return the calculated model.

        Parameters
        ----------
        _x : np.ndarray | sc.Variable | None, default=None
            Ignored. The energy grid is taken from the experiment.
        **kwargs : dict[str, Any]
            Ignored. Included for compatibility with the EasyScience fitter.

        Returns
        -------
        callable
            A function that can be used as a fit function in the EasyScience fitter, which returns
            the calculated model.
        """

        def fit_function(
            _x: np.ndarray | sc.Variable | None = None,
            **kwargs: dict[str, Any],  # ruff: ignore[unused-function-argument]
        ) -> np.ndarray:
            """Fit function."""
            return self._calculate()

        return fit_function

    def get_all_variables(self) -> list[DescriptorNumber]:
        """
        Get all variables used in the analysis.

        Returns
        -------
        list[DescriptorNumber]
            A list of all variables.
        """
        variables = self.sample_model.get_all_variables(Q_index=self.Q_index)

        variables.extend(self.instrument_model.get_all_variables(Q_index=self.Q_index))

        if self._extra_parameters:
            variables.extend(self._extra_parameters)

        return variables

    def plot_data_and_model(
        self,
        plot_components: bool = True,
        add_background: bool = True,
        plot_residuals: bool = False,
        energy: sc.Variable | None = None,
        **kwargs: dict[str, Any],
    ) -> InteractiveFigure:
        """
        Plot the experimental data and the model prediction for the chosen Q index. Optionally also
        plot the individual components of the model.

        Uses Plopp for plotting: https://scipp.github.io/plopp/

        Parameters
        ----------
        plot_components : bool, default=True
            Whether to plot the individual components of the model.
        add_background : bool, default=True
            Whether to add the background to the model prediction when plotting individual
            components.
        plot_residuals : bool, default=False
            Whether to plot the residuals (data - model).
        energy : sc.Variable | None, default=None
            Optional energy grid to use for plotting. If None, the energy grid from the experiment
            is used.
        **kwargs : dict[str, Any]
            Keyword arguments to pass to the plotting function.

        Returns
        -------
        InteractiveFigure
            A plot of the data and model.
        """
        import plopp as pp

        data_and_model = self.data_and_model_to_datagroup(
            energy=energy,
            add_background=add_background,
            include_components=plot_components,
            include_residuals=plot_residuals,
        )

        plot_kwargs_defaults = self._build_plot_style_defaults(data_and_model)
        plot_kwargs_defaults.update(kwargs)

        if plot_residuals:
            fig = slicerplot_with_residuals(
                data_and_model,
                residuals_key='Residuals',
                operation='sum',
                **plot_kwargs_defaults,
            )
        else:
            fig = pp.plot(
                data_and_model,
                **plot_kwargs_defaults,
            )
        fig.autoscale()
        return fig

    def data_and_model_to_datagroup(
        self,
        energy: sc.Variable | None = None,
        add_background: bool = True,
        include_components: bool = True,
        include_residuals: bool = False,
    ) -> sc.DataGroup:
        """
        Create a scipp DataGroup containing the experimental data, model calculation, and
        optionally the individual components.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for the model calculation. If None, the energy grid from
            the experiment is used.
        add_background : bool, default=True
            Whether to add the background to the model prediction when plotting individual
            components.
        include_components : bool, default=True
            Whether to include the individual components of the model in the DataGroup. If True,
            the DataGroup will include a DataArray for each component with the component's display
            name as the key
        include_residuals : bool, default=False
            Whether to include the residuals (data - model) in the DataGroup. If True, the
            DataGroup will include a DataArray with key 'Residuals' containing the residuals.

        Raises
        ------
        ValueError
            If no data is available in the experiment to include in the DataGroup. If no Q values
            are available in the experiment to create the DataGroup. If Q_index is not set to
            create the DataGroup.

        Returns
        -------
        sc.DataGroup
            A DataGroup containing the experimental data, model calculation, and optionally the
            individual components.
        """

        if self.experiment.binned_data is None:
            raise ValueError('No data to include in DataGroup. Please load data first.')

        if self.Q is None:
            raise ValueError(
                'No Q values available for creating DataGroup. Please check the experiment data.'
            )

        self._verify_bool(add_background, 'add_background')
        self._verify_bool(include_components, 'include_components')
        self._verify_bool(include_residuals, 'include_residuals')

        if self.Q_index is None:
            raise ValueError('Q_index must be set to create DataGroup.')

        energy = self._verify_energy(energy)

        if energy is None:
            energy = self._masked_energy

        data_and_model = {
            'Data': self.experiment.get_masked_binned_data(Q_index=self.Q_index),
            'Model': self._create_model_array(energy=energy),
        }

        if include_components:
            components = self._create_components_dataset_single_Q(
                add_background=add_background,
                energy=energy,
            )

            for key in components:
                data_and_model[key] = components[key]

        if include_residuals:
            residuals = self._create_residuals_array()
            data_and_model['Residuals'] = residuals

        return sc.DataGroup(data_and_model)

    def fix_energy_offset(self) -> None:
        """Fix the energy offset parameter for the current Q index."""
        self.instrument_model.fix_energy_offset(Q_index=self._require_Q_index())

    def free_energy_offset(self) -> None:
        """Free the energy offset parameter for the current Q index."""
        self.instrument_model.free_energy_offset(Q_index=self._require_Q_index())

    def rebin(self, dimensions: dict[str, int | sc.Variable]) -> None:
        """
        Rebin the experiment data along specified dimensions and update the analysis.

        Parameters
        ----------
        dimensions : dict[str, int | sc.Variable]
            A dictionary mapping dimension names to number of bins (int) or bin edges
            (sc.Variable).
        """
        self.experiment.rebin(dimensions)
        if self._Q_index is not None and self.experiment is not None:
            self._masked_energy = self.experiment.get_masked_energy(Q_index=self._Q_index)
        self._convolver_is_dirty = True
        self._invalidate_bayesian_sampler()

    def refresh_convolver(self, energy: sc.Variable | None = None) -> None:
        """Refresh the pre-built Convolution object for the current Q index."""
        self._convolver = self._create_convolver(energy=energy)
        self._convolver_is_dirty = False

    #############
    # Private methods: small utilities
    #############

    def _require_Q_index(self) -> int:
        """
        Get the Q index, ensuring it is set.

        Raises a ValueError if the Q index is not set.

        Raises
        ------
        ValueError
            If the Q index is not set.

        Returns
        -------
        int
            The Q index.
        """
        if self._Q_index is None:
            raise ValueError('Q_index must be set.')
        return self._Q_index

    def _on_Q_index_changed(self) -> None:
        """
        Handle changes to the Q index.

        This method is called whenever the Q index is changed. It updates the masked energy from
        the experiment for the new Q index and marks the convolver as dirty.
        """
        if self._Q_index is None:
            self._masked_energy = None
            self._convolver_is_dirty = True
            self._invalidate_bayesian_sampler()
            return
        masked_energy = self.experiment.get_masked_energy(Q_index=self._Q_index)
        self._masked_energy = masked_energy
        self._convolver_is_dirty = True
        # A different Q index means different data, and the Sampler binds its data at construction.
        self._invalidate_bayesian_sampler()

    def _on_experiment_changed(self) -> None:
        """Mark the convolver as dirty when the experiment changes."""
        super()._on_experiment_changed()
        # Refresh masked energy if Q_index is already set (i.e. post-init experiment swap).
        if self._Q_index is not None and self.experiment is not None:
            self._masked_energy = self.experiment.get_masked_energy(Q_index=self._Q_index)
        self._convolver_is_dirty = True
        self._invalidate_bayesian_sampler()

    def _on_sample_model_changed(self) -> None:
        """Mark the convolver as dirty when the sample model changes."""
        super()._on_sample_model_changed()
        self._convolver_is_dirty = True
        self._invalidate_fitter()

    def _on_instrument_model_changed(self) -> None:
        """Mark the convolver as dirty when the instrument model changes."""
        super()._on_instrument_model_changed()
        self._convolver_is_dirty = True
        self._invalidate_fitter()

    def _on_convolution_settings_changed(self) -> None:
        """Mark the convolver as dirty when the convolution settings change."""
        super()._on_convolution_settings_changed()
        self._convolver_is_dirty = True

    def _ensure_convolver_current(self) -> None:
        """Rebuild the convolver if any dependency has changed since it was last built."""
        if self._convolver_is_dirty:
            self._convolver = self._create_convolver()
            self._convolver_is_dirty = False

    def _calculate_energy_with_offset(
        self,
        energy: sc.Variable,
        energy_offset: Parameter,
    ) -> sc.Variable:
        """
        Calculate the energy grid with the energy offset applied.

        Parameters
        ----------
        energy : sc.Variable
            The energy grid to apply the offset to.
        energy_offset : Parameter
            The energy offset to apply.

        Returns
        -------
        sc.Variable
            The energy grid with the offset applied.
        """

        offset_value = sc.to_unit(energy_offset.full_value, energy.unit).value
        energy_with_offset = energy.copy()
        energy_with_offset.values = energy.values - offset_value
        return energy_with_offset

    #############
    # Private methods: evaluation
    #############

    def _evaluate_with_convolution(
        self,
        components: ComponentCollection | ModelComponent,
        energy: sc.Variable | None,
        convolver: Convolution | None = None,
    ) -> np.ndarray:
        """
        Evaluate sample components, applying convolution and detailed balance as appropriate.

        Uses the pre-built convolver when provided (fit path, for performance). If no convolver is
        given, creates a temporary one per call (plot path for individual components). Falls back
        to direct evaluation with detailed balance if there is no resolution model.

        Parameters
        ----------
        components : ComponentCollection | ModelComponent
            The sample components to evaluate.
        energy : sc.Variable | None
            Energy grid to use. If None, uses the masked energy from the experiment.
        convolver : Convolution | None, default=None
            Pre-built Convolution to reuse. If None, a new one is created if needed.

        Returns
        -------
        np.ndarray
            The evaluated sample contribution.
        """
        Q_index = self._require_Q_index()
        if energy is None:
            energy = self._masked_energy

        if isinstance(components, ComponentCollection) and components.is_empty:
            return np.zeros_like(energy.values)

        if convolver is not None:
            return convolver.convolution()

        energy_offset = self.instrument_model.get_energy_offset(Q_index)
        energy_with_offset = self._calculate_energy_with_offset(energy, energy_offset)
        resolution = self.instrument_model.resolution_model.get_component_collection(Q_index)

        if resolution.is_empty:
            result = components.evaluate(energy_with_offset)
            if (
                self.temperature is not None
                and self.detailed_balance_settings.use_detailed_balance
            ):
                result *= detailed_balance_factor(
                    energy=energy_with_offset,
                    temperature=self.temperature,
                    divide_by_temperature=self.detailed_balance_settings.normalize_detailed_balance,
                    energy_unit=self.x_unit,
                )
            return result

        return self._build_convolution(
            sample_components=components,
            resolution_components=resolution,
            energy=energy,
            energy_offset=energy_offset,
        ).convolution()

    def _evaluate_direct(
        self,
        components: ComponentCollection | ModelComponent,
        energy: sc.Variable | None,
    ) -> np.ndarray:
        """
        Evaluate background components directly — no convolution, no detailed balance factor.

        Parameters
        ----------
        components : ComponentCollection | ModelComponent
            The background components to evaluate.
        energy : sc.Variable | None
            Energy grid to use. If None, uses the masked energy from the experiment.

        Returns
        -------
        np.ndarray
            The evaluated background contribution.
        """
        Q_index = self._require_Q_index()
        if energy is None:
            energy = self._masked_energy

        if isinstance(components, ComponentCollection) and components.is_empty:
            return np.zeros_like(energy.values)

        energy_offset = self.instrument_model.get_energy_offset(Q_index)
        energy_with_offset = self._calculate_energy_with_offset(energy, energy_offset)
        return components.evaluate(energy_with_offset)

    def _create_convolver(
        self,
        energy: sc.Variable | None = None,
    ) -> Convolution | None:
        """
        Initialize and return a Convolution object for the chosen Q index. If the necessary
        components for convolution are not available, return None.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for convolution. If None, the energy grid from the
            experiment is used.

        Returns
        -------
        Convolution | None
            The initialized Convolution object or None if not available.
        """
        Q_index = self._require_Q_index()

        if energy is None:
            energy = self._masked_energy

        sample_components = self.sample_model.get_component_collection(Q_index)
        if sample_components.is_empty:
            return None

        resolution_components = self.instrument_model.resolution_model.get_component_collection(
            Q_index
        )
        if resolution_components.is_empty:
            return None

        return self._build_convolution(
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy=energy,
            energy_offset=self.instrument_model.get_energy_offset(Q_index),
        )

    def _build_convolution(
        self,
        sample_components: ComponentCollection | ModelComponent,
        resolution_components: ComponentCollection,
        energy: sc.Variable,
        energy_offset: Parameter,
    ) -> Convolution:
        return Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy_offset=energy_offset,
            convolution_settings=self.convolution_settings,
            temperature=self.temperature,
            detailed_balance_settings=self.detailed_balance_settings,
            x_unit=self.sample_model.x_unit,
            y_unit=self.sample_model.y_unit,
        )

    #############
    # Private methods: create scipp arrays for plotting
    #############

    def _create_model_array(self, energy: sc.Variable | None = None) -> sc.DataArray:
        """
        Create a scipp DataArray for the full sample model including background.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used.

        Returns
        -------
        sc.DataArray
            The model calculation of the full sample model.
        """
        values = self.calculate(energy=energy)
        return self._to_scipp_array(values=values, energy=energy)

    def _create_residuals_array(self) -> sc.DataArray:
        """
        Create a scipp DataArray for the residuals (data - model).

        Returns
        -------
        sc.DataArray
            The residuals (data - model).

        Raises
        ------
        ValueError
            If no data is available in the experiment to calculate residuals. If Q_index is not set
            to calculate residuals.
        """
        if self.Q_index is None:
            raise ValueError('Q_index must be set to calculate residuals.')

        data = self.experiment.get_masked_binned_data(Q_index=self.Q_index)
        model = self._create_model_array()
        return data.copy(deep=True) - model

    def _create_components_dataset_single_Q(
        self,
        add_background: bool = True,
        energy: sc.Variable | None = None,
    ) -> sc.Dataset:
        """
        Create sc.DataArrays for all sample and background components.

        Parameters
        ----------
        add_background : bool, default=True
            Whether to add the background to each sample component.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used.

        Returns
        -------
        sc.Dataset
            A Dataset of component names to their corresponding sc.DataArrays.
        """
        Q_index = self.Q_index
        if energy is None:
            energy = self._masked_energy

        background_components = self.instrument_model.background_model.get_component_collection(
            Q_index=Q_index
        )
        background_values = (
            self._evaluate_direct(background_components, energy) if add_background else None
        )

        result: dict[str, sc.DataArray] = {}
        for component in self.sample_model.get_component_collection(Q_index=Q_index):
            values = self._evaluate_with_convolution(component, energy)
            if background_values is not None:
                values = values + background_values
            result[component.display_name] = self._to_scipp_array(values, energy)

        for component in background_components:
            result[component.display_name] = self._to_scipp_array(
                self._evaluate_direct(component, energy), energy
            )

        return sc.Dataset(result)

    def _to_scipp_array(
        self,
        values: np.ndarray,
        energy: sc.Variable | None = None,
    ) -> sc.DataArray:
        """
        Convert a numpy array of values to a sc.DataArray with the correct coordinates for energy
        and Q.

        Parameters
        ----------
        values : np.ndarray
            The values to convert.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for the energy coordinate. If None, the energy grid from
            the experiment is used.

        Returns
        -------
        sc.DataArray
            The converted sc.DataArray.
        """

        if energy is None:
            energy = self._masked_energy
        return sc.DataArray(
            data=sc.array(dims=['energy'], values=values, unit=self.sample_model.y_unit),
            coords={
                'energy': energy,
                'Q': self.Q[self.Q_index],
            },
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'display_name={self.display_name!r}, '
            f'unique_name={self.unique_name!r}, '
            f'Q_index={self._Q_index})'
        )
