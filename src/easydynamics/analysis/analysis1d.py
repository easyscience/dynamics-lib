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
        extra_parameters: Parameter | list[Parameter] | None = None,
    ) -> None:
        """
        Initialize a Analysis1d.

        Parameters
        ----------
        display_name : str | None, default='MyAnalysis'
            Display name of the analysis. By default, 'MyAnalysis'.
        unique_name : str | None, default=None
            Unique name of the analysis. If None, a unique name is automatically generated. By
            default, None.
        experiment : Experiment | None, default=None
            The Experiment associated with this Analysis. If None, a default Experiment is created.
            By default, None.
        sample_model : SampleModel | None, default=None
            The SampleModel associated with this Analysis. If None, a default SampleModel is
            created. By default, None.
        instrument_model : InstrumentModel | None, default=None
            The InstrumentModel associated with this Analysis. If None, a default InstrumentModel
            is created. By default, None.
        Q_index : int | None, default=None
            The Q index to analyze. If None, the analysis will not be able to calculate or fit
            until a Q index is set. By default, None.
        extra_parameters : Parameter | list[Parameter] | None, default=None
            Extra parameters to be included in the analysis for advanced users. If None, no extra
            parameters are added. By default, None.
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
            experiment is used. By default, None.

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
            experiment is used. By default, None.

        Returns
        -------
        np.ndarray
            The calculated model prediction.
        """

        sample_intensity = self._evaluate_sample(energy=energy)

        background_intensity = self._evaluate_background(energy=energy)

        return sample_intensity + background_intensity

    def fit(self) -> FitResults:
        """
        Fit the model to the experimental data for the chosen Q index.

        The energy grid is fixed for the duration of the fit. Convolution objects are created once
        and reused during parameter optimization for performance reasons.

        Raises
        ------
        ValueError :
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

        x, y, weights, _ = self.experiment._extract_x_y_weights_only_finite(
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
            Ignored. The energy grid is taken from the experiment. By default, None.
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
            Whether to plot the individual components of the model. By default, True.
        add_background : bool, default=True
            Whether to add the background to the model prediction when plotting individual
            components. By default, True.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for plotting. If None, the energy grid from the experiment
            is used. By default, None.
        **kwargs : dict[str, Any]
            Keyword arguments to pass to the plotting function.

        Raises
        ------
        ValueError :
            If no data is available to plot.

        Returns
        -------
        InteractiveFigure
            A plot of the data and model.
        """
        import plopp as pp

        if self.experiment.data is None:
            raise ValueError('No data to plot. Please load data first.')

        energy = self._verify_energy(energy)
        if energy is None:
            energy = self._masked_energy

        data = self.experiment.data['Q', self.Q_index]
        model_array = self._create_sample_scipp_array(energy=energy)

        component_dataset = self._create_components_dataset_single_Q(
            add_background=add_background, energy=energy
        )

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
            for comp_name in component_dataset:
                plot_kwargs_defaults['linestyle'][comp_name] = '--'
                plot_kwargs_defaults['marker'][comp_name] = None

        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)

        return pp.plot(
            data_and_model,
            **plot_kwargs_defaults,
        )

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
        ValueError :
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

    def _verify_energy(self, energy: sc.Variable | None) -> sc.Variable | None:
        """
        Verify that the provided energy is the correct type.

        Parameters
        ----------
        energy : sc.Variable | None
            The energy to verify.

        Raises
        ------
        TypeError :
            If energy is not a sc.Variable or None.

        Returns
        -------
        sc.Variable | None
            The verified energy, or None if no energy is provided.
        """

        if energy is not None and not isinstance(energy, sc.Variable):
            raise TypeError(f'Energy must be a sc.Variable or None. Got {type(energy)}.')
        return energy

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
        sc.UnitError :
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

    def _evaluate_components(
        self,
        components: ComponentCollection | ModelComponent,
        convolver: Convolution | None = None,
        convolve: bool = True,
        energy: sc.Variable | None = None,
    ) -> np.ndarray:
        """
        Calculate the contribution of a set of components, optionally convolving with the
        resolution.

        If convolve is True and a Convolution object is provided (for full model evaluation), we
        use it to perform the convolution of the components with the resolution. If convolve is
        True but no Convolution object is provided, create a new Convolution object for the given
        components (for individual components). If convolve is False, evaluate the components
        directly without convolution (for background).

        Parameters
        ----------
        components : ComponentCollection | ModelComponent
            The components to evaluate.
        convolver : Convolution | None, default=None
            An optional Convolution object to use for convolution. If None, a new Convolution
            object will be created if convolve is True. By default, None.
        convolve : bool, default=True
            Whether to perform convolution with the resolution. By default, True.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        np.ndarray
            The evaluated contribution of the components.
        """

        Q_index = self._require_Q_index()
        if energy is None:
            energy = self._masked_energy

        energy_offset = self.instrument_model.get_energy_offset(Q_index)
        energy_with_offset = self._calculate_energy_with_offset(
            energy=energy,
            energy_offset=energy_offset,
        )

        # If there are no components, return zero
        if isinstance(components, ComponentCollection) and components.is_empty:
            return np.zeros_like(energy.values)

        # No convolution
        if not convolve:
            return components.evaluate(energy_with_offset)

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
            return components.evaluate(energy_with_offset)

        conv = Convolution(
            sample_components=components,
            resolution_components=resolution,
            energy=energy,
            temperature=self.temperature,
            energy_offset=energy_offset,
        )
        return conv.convolution()

    def _evaluate_sample(
        self,
        energy: sc.Variable | None = None,
    ) -> np.ndarray:
        """
        Evaluate the sample contribution for a given Q index.

        Assumes that self._convolver is up to date.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        np.ndarray
            The evaluated sample contribution.
        """
        Q_index = self._require_Q_index()
        components = self.sample_model.get_component_collection(Q_index=Q_index)
        return self._evaluate_components(
            components=components,
            convolver=self._convolver,
            convolve=True,
            energy=energy,
        )

    def _evaluate_sample_component(
        self,
        component: ModelComponent,
        energy: sc.Variable | None = None,
    ) -> np.ndarray:
        """
        Evaluate a single sample component for the chosen Q index.

        Parameters
        ----------
        component : ModelComponent
            The sample component to evaluate.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        np.ndarray
            The evaluated sample component contribution.
        """
        return self._evaluate_components(
            components=component,
            convolver=None,
            convolve=True,
            energy=energy,
        )

    def _evaluate_background(self, energy: sc.Variable | None = None) -> np.ndarray:
        """
        Evaluate the background contribution for the chosen Q index.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        np.ndarray
            The evaluated background contribution.
        """
        Q_index = self._require_Q_index()
        background_components = self.instrument_model.background_model.get_component_collection(
            Q_index=Q_index
        )
        return self._evaluate_components(
            components=background_components,
            convolver=None,
            convolve=False,
            energy=energy,
        )

    def _evaluate_background_component(
        self,
        component: ModelComponent,
        energy: sc.Variable | None = None,
    ) -> np.ndarray:
        """
        Evaluate a single background component for the chosen Q index.

        Parameters
        ----------
        component : ModelComponent
            The background component to evaluate.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        np.ndarray
            The evaluated background component contribution.
        """

        return self._evaluate_components(
            components=component,
            convolver=None,
            convolve=False,
            energy=energy,
        )

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
            experiment is used. By default, None.

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

        # TODO: allow convolution options to be set.
        return Convolution(
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy=energy,
            temperature=self.temperature,
            energy_offset=self.instrument_model.get_energy_offset(Q_index),
        )

    #############
    # Private methods: create scipp arrays for plotting
    #############

    def _create_component_scipp_array(
        self,
        component: ModelComponent,
        background: np.ndarray | None = None,
        energy: sc.Variable | None = None,
    ) -> sc.DataArray:
        """
        Create a scipp DataArray for a single component.

        Adds the background if it is not None.

        Parameters
        ----------
        component : ModelComponent
            The component to evaluate.
        background : np.ndarray | None, default=None
            Optional background to add to the component. By default, None.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        sc.DataArray
            The model calculation of the component.
        """

        values = self._evaluate_sample_component(component=component, energy=energy)
        if background is not None:
            values += background
        return self._to_scipp_array(values=values, energy=energy)

    def _create_background_component_scipp_array(
        self,
        component: ModelComponent,
        energy: sc.Variable | None = None,
    ) -> sc.DataArray:
        """
        Create a scipp DataArray for a single background component.

        Parameters
        ----------
        component : ModelComponent
            The component to evaluate.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        sc.DataArray
            The model calculation of the component.
        """

        values = self._evaluate_background_component(
            component=component,
            energy=energy,
        )
        return self._to_scipp_array(values=values, energy=energy)

    def _create_sample_scipp_array(self, energy: sc.Variable | None = None) -> sc.DataArray:
        """
        Create a scipp DataArray for the full sample model including background.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        sc.DataArray
            The model calculation of the full sample model.
        """
        values = self.calculate(energy=energy)
        return self._to_scipp_array(values=values, energy=energy)

    def _create_components_dataset_single_Q(
        self,
        add_background: bool = True,
        energy: sc.Variable | None = None,
    ) -> dict[str, sc.DataArray]:
        """
        Create sc.DataArrays for all sample and background components.

        Parameters
        ----------
        add_background : bool, default=True
            Whether to add background components. By default, True.
        energy : sc.Variable | None, default=None
            Optional energy grid to use for evaluation. If None, the energy grid from the
            experiment is used. By default, None.

        Returns
        -------
        dict[str, sc.DataArray]
            A dictionary of component names to their corresponding sc.DataArrays.
        """
        scipp_arrays = {}
        sample_components = self.sample_model.get_component_collection(
            Q_index=self.Q_index
        ).components

        background_components = self.instrument_model.background_model.get_component_collection(
            Q_index=self.Q_index
        ).components

        if energy is None:
            energy = self._masked_energy

        background = self._evaluate_background(energy=energy) if add_background else None

        for component in sample_components:
            scipp_arrays[component.display_name] = self._create_component_scipp_array(
                component=component, background=background, energy=energy
            )
        for component in background_components:
            scipp_arrays[component.display_name] = self._create_background_component_scipp_array(
                component=component, energy=energy
            )
        return sc.Dataset(scipp_arrays)

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
            the experiment is used. By default, None.

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
