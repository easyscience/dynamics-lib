# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from easyscience.fitting.fitter import Fitter as EasyScienceFitter
from easyscience.variable import DescriptorNumber

from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel


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

    #############
    # Other methods
    #############

    def calculate(self, energy: float | None = None) -> np.ndarray:
        """Calculate the model prediction for a given Q index.

        Args:
            energy (float): The energy value to calculate the model for.
        Returns:
            sc.DataArray: The calculated model prediction.
        """
        Q_index = self._require_Q_index()

        if energy is None:
            energy = self.energy.values

        # TODO: handle units properly

        energy_offset = self.instrument_model.get_energy_offset_at_Q(Q_index).value

        # Sample
        sample_components = self.sample_model.get_component_collection(Q_index)
        resolution_components = (
            self.instrument_model.resolution_model.get_component_collection(Q_index)
        )

        sample_intensity = self._evaluate_sample(
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy=energy,
            energy_offset=energy_offset,
        )

        # Background
        background_component_collection = (
            self.instrument_model.background_model.get_component_collection(Q_index)
        )
        background_intensity = self._evaluate_background(
            background_components=background_component_collection,
            energy=energy,
            energy_offset=energy_offset,
        )

        sample_plus_background = sample_intensity + background_intensity

        return sample_plus_background

    def calculate_individual_components(
        self,
        energy: float | None = None,
    ) -> np.ndarray:
        """Calculate the model prediction for a given Q index.

        Args:
            energy (float): The energy value to calculate the model for.
        Returns:
            sc.DataArray: The calculated model prediction.
        """
        Q_index = self._require_Q_index()

        if energy is None:
            energy = self.energy.values

        # TODO: handle units properly

        energy_offset = self.instrument_model.get_energy_offset_at_Q(Q_index).value

        # Sample. Convolve with resolution if resolution components are
        # present, otherwise just evaluate sample components one by one
        # to get individual contributions.
        sample_components = self.sample_model.get_component_collection(Q_index)

        resolution_components = (
            self.instrument_model.resolution_model.get_component_collection(Q_index)
        )

        if sample_components.is_empty:
            sample_intensity = [np.zeros_like(energy)]
        else:
            sample_intensity = []
            for component in sample_components.components:
                component_intensity = self._evaluate_sample_component(
                    component=component,
                    resolution_components=resolution_components,
                    energy=energy,
                    energy_offset=energy_offset,
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
                    energy_offset=energy_offset,
                )
                background_intensity.append(component_intensity)

        return sample_intensity, background_intensity

    def fit(self):
        """Fit the model to the experimental data for a given Q index.

        Args:
        Returns:
            FitResult: The result of the fit.
        """
        if self._experiment is None:
            raise ValueError("No experiment is associated with this Analysis.")

        Q_index = self._require_Q_index()

        data = self.experiment.data["Q", Q_index]
        x = data.coords["energy"].values
        y = data.values
        e = data.variances**0.5

        def fit_func(x_vals):
            return self.calculate(energy=x_vals)

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
                plt.plot(
                    energy,
                    comp,
                    label=f"Sample Component {i + 1}",
                    linestyle="--",
                )
            for i, comp in enumerate(background_comps):
                plt.plot(
                    energy,
                    comp,
                    label=f"Background Component {i + 1}",
                    linestyle=":",
                )
        plt.xlabel(f"Energy ({self.energy.unit})")
        plt.ylabel(f"Intensity ({self.sample_model.unit})")
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

        if self._extra_parameters != []:
            variables.extend(self._extra_parameters)

        return variables

    #############
    # Private methods
    #############
    def _evaluate_sample(
        self,
        sample_components,
        resolution_components,
        energy,
        energy_offset,
    ):
        if resolution_components.is_empty:
            return sample_components.evaluate(energy - energy_offset)
        convolver = self._convolvers[self._require_Q_index()]
        return convolver.convolution()

    def _evaluate_sample_component(
        self,
        component,
        resolution_components,
        energy,
        energy_offset,
    ):
        if resolution_components.is_empty:
            return component.evaluate(energy - energy_offset)
        convolver = Convolution(
            sample_components=component,
            resolution_components=resolution_components,
            energy=energy,
            temperature=self.temperature,
            energy_offset=energy_offset,
        )
        return convolver.convolution()

    def _evaluate_background(
        self,
        background_components,
        energy,
        energy_offset,
    ):
        if background_components.is_empty:
            return np.zeros_like(energy)
        return background_components.evaluate(energy - energy_offset)

    def _evaluate_background_component(
        self,
        component,
        energy,
        energy_offset,
    ):
        return component.evaluate(energy - energy_offset)

    def _require_Q_index(self) -> int:
        if self._Q_index is None:
            raise ValueError("Q_index must be set.")
        return self._Q_index
