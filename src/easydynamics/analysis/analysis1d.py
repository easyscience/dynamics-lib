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


class Analysis1d(AnalysisBase):
    """
    For analysing one-dimensional data, i.e. intensity as function of energy for a single Q index.

    Is used primarily in the Analysis class, but can also be used on its own for simpler analyses.
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

        self._Q_index = self._verify_Q_index(Q_index)

        if self._Q_index is not None and self.experiment is not None:
            masked_energy = self.experiment.get_masked_energy(Q_index=self._Q_index)
            self._masked_energy = masked_energy
        else:
            self._masked_energy = None

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

        self._Q_index = self._verify_Q_index(value)
        self._on_Q_index_changed()

    #############
    # Other methods
    #############

    def calculate(self, energy: sc.Variable | None = None) -> np.ndarray:
        """
        Calculate the model prediction for the chosen Q index.

        Makes sure the convolver is up to date before calculating.

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
        self._convolver = self._create_convolver(energy=energy)

        return self._calculate(energy=energy)

    def _calculate(self, energy: sc.Variable | None = None) -> np.ndarray:
        """
        Calculate the model prediction for the chosen Q index.

        Does not check if the convolver is up to date.

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

        Q_index = self._require_Q_index()
        sample = self._evaluate_with_convolution(
            self.sample_model.get_component_collection(Q_index),
            energy,
            convolver=self._convolver,
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

        # Create convolver once to reuse during fitting
        self._convolver = self._create_convolver()

        fitter = EasyScienceFitter(
            fit_object=self,
            fit_function=self.as_fit_function(),
        )

        x, y, weights, _ = self.experiment._extract_x_y_weights_only_finite(  # noqa: SLF001
            Q_index=self._require_Q_index()
        )
        fit_result = fitter.fit(x=x, y=y, weights=weights)

        self._fit_result = fit_result

        return fit_result

    def as_fit_function(
        self,
        _x: np.ndarray | sc.Variable | None = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
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
            **kwargs: dict[str, Any],  # noqa: ARG001
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
        TypeError
            If add_background is not a boolean. If include_components is not a boolean.

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
            'Data': self.experiment.binned_data['Q', self.Q_index],
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

        This method is called whenever the Q index is changed. It updates the Convolution object
        for the new Q index and the masked energy from the experiment for the new Q index.
        """
        masked_energy = self.experiment.get_masked_energy(Q_index=self._Q_index)
        self._masked_energy = masked_energy
        self._convolver = self._create_convolver()

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

        Raises
        ------
        sc.UnitError
            If the energy and energy offset have incompatible units.

        Returns
        -------
        sc.Variable
            The energy grid with the offset applied.
        """

        if energy.unit != energy_offset.unit:
            try:
                energy_offset.convert_unit(str(energy.unit))
            except Exception as e:
                raise sc.UnitError(
                    f'Energy and energy offset must have compatible units. '
                    f'Got {energy.unit} and {energy_offset.unit}.'
                ) from e

        energy_with_offset = energy.copy(deep=True)
        energy_with_offset.values -= energy_offset.value
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

        Uses the pre-built convolver when provided (fit path, for performance). If no convolver
        is given, creates a temporary one per call (plot path for individual components). Falls
        back to direct evaluation with detailed balance if there is no resolution model.

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
                    energy_unit=self.unit,
                )
            return result

        return Convolution(
            energy=energy,
            sample_components=components,
            resolution_components=resolution,
            energy_offset=energy_offset,
            convolution_settings=self.convolution_settings,
            temperature=self.temperature,
            detailed_balance_settings=self.detailed_balance_settings,
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

        return Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy_offset=self.instrument_model.get_energy_offset(Q_index),
            convolution_settings=self.convolution_settings,
            temperature=self.temperature,
            detailed_balance_settings=self.detailed_balance_settings,
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

        data = self.experiment.binned_data['Q', self.Q_index]
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
            data=sc.array(dims=['energy'], values=values),
            coords={
                'energy': energy,
                'Q': self.Q[self.Q_index],
            },
        )
