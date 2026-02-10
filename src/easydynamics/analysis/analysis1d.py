# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from inspect import Parameter

import numpy as np
import scipp as sc
from easyscience.fitting.fitter import Fitter as EasyScienceFitter
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.variable import DescriptorNumber

from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.convolution.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent


class Analysis1d(AnalysisBase):
    """For analysing data."""

    def __init__(
        self,
        display_name: str = "MyAnalysis",
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        Q_index: int | None = None,
        extra_parameters: Parameter | list[Parameter] | None = None,
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )

        if Q_index is not None:
            if (
                not isinstance(Q_index, int)
                or Q_index < 0
                or (self.Q is not None and Q_index >= len(self.Q))
            ):
                raise ValueError("Q_index must be a valid index for the Q values.")
        self._Q_index = Q_index

        self._fit_result = None

        self._convolver = self._create_convolver(Q_index=self.Q_index)

    #############
    # Properties
    #############

    @property
    def Q_index(self) -> int | None:
        """Get the Q index for single Q analysis."""
        return self._Q_index

    @Q_index.setter
    def Q_index(self, index: int | None) -> None:
        """Set the Q index for single Q analysis.

        Args:
            index (int | None): The Q index.
        """
        if index is not None:
            if (
                not isinstance(index, int)
                or index < 0
                or (self.Q is not None and index >= len(self.Q))
            ):
                raise ValueError("Q_index must be a valid index for the Q values.")
        self._Q_index = index
        self._on_Q_index_changed()

    #############
    # Other methods
    #############

    def calculate(self, energy: np.ndarray | sc.Variable | None = None) -> np.ndarray:
        """Calculate the model prediction for a given Q index.

        Args:
            energy (float): The energy value to calculate the model for.
        Returns:
            sc.DataArray: The calculated model prediction.
        """

        self._convolver = self._create_convolver(Q_index=self.Q_index, energy=energy)

        return self._calculate()

    def _calculate(self) -> np.ndarray:
        """Calculate the model prediction for a given Q index.

        Args:
            energy (float): The energy value to calculate the model for.
        Returns:
            sc.DataArray: The calculated model prediction.
        """

        sample_intensity = self._evaluate_sample()

        background_intensity = self._evaluate_background()

        sample_plus_background = sample_intensity + background_intensity

        return sample_plus_background

    def calculate_individual_components(
        self,
        energy: np.ndarray | sc.Variable | None = None,
    ) -> np.ndarray:
        """Calculate the model prediction for a given Q index.

        Args:
            energy (float): The energy value to calculate the model for.
        Returns:
            sc.DataArray: The calculated model prediction.
        """
        Q_index = self._require_Q_index()

        energy = self._handle_energy(energy)

        sample_components = self.sample_model.get_component_collection(Q_index)

        if sample_components.is_empty:
            sample_intensity = [np.zeros_like(energy)]
        else:
            sample_intensity = []
            for component in sample_components.components:
                component_intensity = self._evaluate_sample_component(
                    component=component,
                    energy=energy,
                )
                sample_intensity.append(component_intensity)

        # Background. Evaluate each background component separately to
        # get individual contributions.
        background_components = (
            self.instrument_model.background_model.get_component_collection(Q_index)
        )

        if background_components.is_empty:
            background_intensity = [np.zeros_like(energy)]
        else:
            background_intensity = []
            for component in background_components.components:
                component_intensity = self._evaluate_background_component(
                    component=component,
                    energy=energy,
                )
                background_intensity.append(component_intensity)

        return sample_intensity, background_intensity

    def fit(self) -> FitResults:
        """Fit the model to the experimental data for a given Q index.

        Args:
        Returns:
            FitResult: The result of the fit.

        Notes
        -----
        The energy grid is fixed for the duration of the fit.
        Convolution objects are created once and reused during
        parameter optimization for performance reasons.
        """
        if self._experiment is None:
            raise ValueError("No experiment is associated with this Analysis.")

        Q_index = self._require_Q_index()

        data = self.experiment.data["Q", Q_index]
        x = data.coords["energy"].values
        y = data.values
        e = data.variances**0.5

        self._convolver = self._create_convolver(Q_index=self.Q_index, energy=x)

        def fit_func(_):
            return self._calculate()

        fitter = EasyScienceFitter(
            fit_object=self,
            fit_function=fit_func,
        )

        # Perform the fit
        fit_result = fitter.fit(x=x, y=y, weights=1.0 / e)

        # Store result
        self._fit_result = fit_result

        return fit_result

    def plot_data_and_model(
        self,
        plot_individual_components: bool = True,
        add_background: bool = True,
    ) -> None:
        """Plot the experimental data and the model prediction.

        Args:
            plot_individual_components (bool): Whether to plot
            individual components. Default is True.
        """
        if not isinstance(plot_individual_components, bool):
            raise TypeError("plot_individual_components must be True or False.")

        import matplotlib.pyplot as plt

        Q_index = self._require_Q_index()
        if self.experiment is None or self.experiment.data is None:
            raise ValueError("Experiment data is not available for plotting.")
        data = self.experiment.data["Q", Q_index]
        energy = data.coords["energy"].values
        model = self.calculate(energy=energy)
        plt.figure()
        plt.errorbar(
            energy,
            data.values,
            yerr=data.variances**0.5,
            fmt="o",
            label="Data",
            color="black",
        )
        plt.plot(energy, model, label="Model", color="red")
        if plot_individual_components:
            sample_comps, background_comps = self.calculate_individual_components()
            if add_background:
                background = sum(background_comps)
                sample_comps = [comp + background for comp in sample_comps]
            for i, comp in enumerate(sample_comps):
                comp_name = (
                    self.sample_model.get_component_collection(Q_index)
                    .components[i]
                    .display_name
                )
                plt.plot(
                    energy,
                    comp,
                    label=comp_name,
                    linestyle="--",
                )
            for i, comp in enumerate(background_comps):
                comp_name = (
                    self.instrument_model.background_model.get_component_collection(
                        Q_index
                    )
                    .components[i]
                    .display_name
                )
                plt.plot(
                    energy,
                    comp,
                    label=comp_name,
                    linestyle=":",
                )
        plt.xlabel(f"Energy ({self.energy.unit})")
        plt.ylabel("Intensity (arb. units)")
        plt.title(f"Data and Model at Q index {Q_index}")
        plt.legend()
        plt.show()

    def get_all_variables(self) -> list[DescriptorNumber]:
        """Get all variables used in the analysis.

        Returns:
            List[Descriptor]: A list of all variables.
        """
        variables = self.sample_model.get_all_variables(Q_index=self.Q_index)

        variables.extend(self.instrument_model.get_all_variables(Q_index=self.Q_index))

        if self._extra_parameters:
            variables.extend(self._extra_parameters)

        return variables

    #############
    # Private methods
    #############

    def _require_Q_index(self) -> int:
        """
        Get the Q index for single Q analysis, ensuring it is set.
         Raises a ValueError if the Q index is not set.
        Returns:
            int: The Q index.
        """
        if self._Q_index is None:
            raise ValueError("Q_index must be set.")
        return self._Q_index

    def _handle_energy(
        self, energy: np.ndarray | sc.Variable | None
    ) -> np.ndarray | sc.Variable:
        """ "
        Handle the energy input for evaluation methods.

         If energy is None, use the energy values from the experiment.
         If energy is a sc.Variable, extract the values as a numpy array.
         If energy is already a numpy array, return it as is.

         Args:
             energy (np.ndarray | sc.Variable | None): The input energy values.
        Returns:
            np.ndarray: The energy values to use for evaluation.
        """
        # TODO: handle units properly

        if energy is None:
            energy = self.energy.values

        if isinstance(energy, np.ndarray):
            return energy

        if isinstance(energy, sc.Variable):
            return energy.values

        raise TypeError("Energy must be a numpy array, sc.Variable, or None.")

    def _on_Q_index_changed(self) -> None:
        """
        Handle changes to the Q index.

        This method is called whenever the Q index is changed. It updates
        the Convolution object for the new Q index.
        """
        self._convolver = self._create_convolver(Q_index=self.Q_index)

    def _evaluate_components(
        self,
        components: ComponentCollection | ModelComponent,
        energy: np.ndarray | sc.Variable | None = None,
        convolver: Convolution | None = None,
        convolve: bool = True,
        Q_index: int | None = None,
    ):
        """
        Calculate the contribution of a set of components, optionally
        convolving with the resolution.
            If convolve is True and a Convolution object is provided,
            use it to perform the convolution of the components with the
            resolution. If convolve is True but no Convolution object is
            provided, create a new Convolution object for the given
            components and energy. If convolve is False, evaluate the
            components directly without convolution.
        Args:
            components (ComponentCollection | ModelComponent):
                The components to evaluate.
            energy (np.ndarray | sc.Variable | None):
                The energy values to evaluate the components for. If
                None, the energy values from the experiment will be
                used.
            convolver (Convolution | None):
            An optional Convolution object to use for convolution.
            If None, a new Convolution object will be created if
            convolve is True.
            convolve (bool):
                Whether to perform convolution with the resolution.
                Default is True.
        """
        if Q_index is None:
            Q_index = self._require_Q_index()
        energy = self._handle_energy(energy)
        energy_offset = self.instrument_model.get_energy_offset_at_Q(Q_index).value

        # If there are no components, return zero
        if isinstance(components, ComponentCollection) and components.is_empty:
            return np.zeros_like(energy)

        # No convolution
        if not convolve:
            return components.evaluate(energy - energy_offset)

        resolution = self.instrument_model.resolution_model.get_component_collection(
            Q_index
        )
        if resolution.is_empty:
            return components.evaluate(energy - energy_offset)

        # Convolution For fitting we don't want to create a new
        # Convolution object at each iteration
        if convolver is not None:
            return convolver.convolution()

        # For evaluating individual components
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
        energy: np.ndarray | sc.Variable | None = None,
        Q_index: int | None = None,
    ):
        """
        Evaluate the sample contribution for a given Q index.

        If a Convolution object exists for the Q index, use it to
        perform the convolution of the sample components with the
        resolution components. If no Convolution object exists, evaluate
        the sample components directly without convolution.

        Args:
            energy (np.ndarray | sc.Variable | None): The energy values
            to evaluate the sample contribution for. If None, the energy
            values from the experiment will be used.
        Returns:
            np.ndarray: The evaluated sample contribution.
        """
        if Q_index is None:
            Q_index = self._require_Q_index()
        components = self.sample_model.get_component_collection(Q_index=Q_index)
        return self._evaluate_components(
            components=components,
            energy=energy,
            convolver=self._convolver,
            convolve=True,
        )

    def _evaluate_sample_component(
        self,
        component,
        energy: np.ndarray | sc.Variable | None = None,
    ):
        """
        Evaluate a single sample component for a given Q index.
        If a Convolution object exists for the Q index, use it to
        perform the convolution of the sample component with the
        resolution components. If no Convolution object exists, evaluate
        the sample component directly without convolution.
        Args:
            component: The sample component to evaluate.
            energy (np.ndarray | sc.Variable | None): The energy values
            to evaluate the sample component for. If None, the energy
            values from the experiment will be used.
        Returns:
            np.ndarray: The evaluated sample component contribution.
        """
        return self._evaluate_components(
            components=component,
            energy=energy,
            convolver=None,
            convolve=True,
        )

    def _evaluate_background(
        self,
        energy: np.ndarray | sc.Variable | None = None,
        Q_index: int | None = None,
    ):
        """
        Evaluate the background contribution for a given Q index.
         Evaluate each background component separately to get individual
         contributions. Args:
            energy (np.ndarray | sc.Variable | None): The energy values
            to evaluate the background contribution for. If None, the
            energy values from the experiment will be used.
        Returns:
            np.ndarray: The evaluated background contribution.
        """

        if Q_index is None:
            Q_index = self._require_Q_index()
        background_components = (
            self.instrument_model.background_model.get_component_collection(
                Q_index=Q_index
            )
        )
        return self._evaluate_components(
            components=background_components,
            energy=energy,
            convolver=None,
            convolve=False,
        )

    def _evaluate_background_component(
        self,
        component,
        energy: np.ndarray | sc.Variable | None = None,
    ):
        """
        Evaluate a single background component for a given Q index.
        Evaluate the background component directly without convolution.
        Args:
            component: The background component to evaluate.
            energy (np.ndarray | sc.Variable | None): The energy values
            to evaluate the background component for. If None, the energy
            values from the experiment will be used.
        Returns:
            np.ndarray: The evaluated background component contribution.
        """

        return self._evaluate_components(
            components=component,
            energy=energy,
            convolver=None,
            convolve=False,
        )

    def _create_convolver(
        self, Q_index: int, energy: np.ndarray | sc.Variable | None = None
    ) -> Convolution | None:
        """Initialize and return a Convolution object for the given Q
        index.
        """
        sample_components = self.sample_model.get_component_collection(Q_index)
        if sample_components.is_empty:
            return None

        resolution_components = (
            self.instrument_model.resolution_model.get_component_collection(Q_index)
        )
        if resolution_components.is_empty:
            return None
        if energy is None:
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
