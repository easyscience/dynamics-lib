# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from inspect import Parameter

import numpy as np
import scipp as sc
from easyscience.fitting.fitter import Fitter as EasyScienceFitter
from easyscience.variable import DescriptorNumber

from easydynamics.analysis.analysis_base import AnalysisBase
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

    def fit(self):
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
