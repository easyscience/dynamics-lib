# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import plopp as pp
import scipp as sc
from easyscience.base_classes.model_base import ModelBase as EasyScienceModelBase
from easyscience.fitting.fitter import Fitter as EasyScienceFitter
from easyscience.variable import Parameter

from easydynamics.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import BackgroundModel
from easydynamics.sample_model import ResolutionModel
from easydynamics.sample_model import SampleModel


class Analysis(EasyScienceModelBase):
    """For analysing data."""

    def __init__(
        self,
        display_name: str = "MyAnalysis",
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        resolution_model: ResolutionModel | None = None,
        background_model: BackgroundModel | None = None,
        energy_offset: None = None,
    ):

        super().__init__(display_name=display_name, unique_name=unique_name)

        if experiment is not None and not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an instance of Experiment or None.")

        self._experiment = experiment

        if sample_model is not None and not isinstance(sample_model, SampleModel):
            raise TypeError("sample_model must be an instance of SampleModel or None.")
        sample_model.Q = self.Q
        self._sample_model = sample_model

        if resolution_model is not None and not isinstance(
            resolution_model, ResolutionModel
        ):
            raise TypeError(
                "resolution_model must be an instance of ResolutionModel or None."
            )
        resolution_model.Q = self.Q
        self._resolution_model = resolution_model

        if background_model is not None and not isinstance(
            background_model, BackgroundModel
        ):
            raise TypeError(
                "background_model must be an instance of BackgroundModel or None."
            )
        background_model.Q = self.Q
        self._background_model = background_model

        self._convolvers = [None] * (len(self.Q) if self.Q is not None else 0)
        self._update_models()

    #############
    # Properties
    #############

    @property
    def experiment(self) -> Experiment | None:
        """The Experiment associated with this Analysis."""
        return self._experiment

    @experiment.setter
    def experiment(self, value: Experiment | None) -> None:
        if value is not None and not isinstance(value, Experiment):
            raise TypeError("experiment must be an instance of Experiment or None.")
        self._experiment = value
        self._update_models()

    @property
    def sample_model(self) -> SampleModel | None:
        """The SampleModel associated with this Analysis."""
        return self._sample_model

    @sample_model.setter
    def sample_model(self, value: SampleModel | None) -> None:
        if value is not None and not isinstance(value, SampleModel):
            raise TypeError("sample_model must be an instance of SampleModel or None.")
        self._sample_model = value
        self._update_models()

    @property
    def resolution_model(self) -> ResolutionModel | None:
        """The ResolutionModel associated with this Analysis."""
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, value: ResolutionModel | None) -> None:
        if value is not None and not isinstance(value, ResolutionModel):
            raise TypeError(
                "resolution_model must be an instance of ResolutionModel or None."
            )
        self._resolution_model = value
        self._update_models()

    @property
    def background_model(self) -> BackgroundModel | None:
        """The BackgroundModel associated with this Analysis."""
        return self._background_model

    @background_model.setter
    def background_model(self, value: BackgroundModel | None) -> None:
        if value is not None and not isinstance(value, BackgroundModel):
            raise TypeError(
                "background_model must be an instance of BackgroundModel or None."
            )
        self._background_model = value
        self._update_models()

    @property
    def Q(self) -> sc.Variable | None:
        """The Q values from the associated Experiment, if available."""
        if self.experiment is not None:
            return self.experiment.Q
        return None

    @Q.setter
    def Q(self, value) -> None:
        """Q is a read-only property derived from the Experiment."""
        raise AttributeError("Q is a read-only property derived from the Experiment.")

    @property
    def energy(self) -> sc.Variable | None:
        """The energy values from the associated Experiment, if
        available.
        """
        if self.experiment is not None:
            return self.experiment.energy
        return None

    @energy.setter
    def energy(self, value) -> None:
        """Energy is a read-only property derived from the
        Experiment.
        """
        raise AttributeError(
            "energy is a read-only property derived from the Experiment."
        )

    # TODO: make it use experiment temperature
    @property
    def temperature(self) -> Parameter | None:
        """The temperature from the associated Experiment, if
        available.
        """
        return None

    @temperature.setter
    def temperature(self, value) -> None:
        """Temperature is a read-only property derived from the
        Experiment.
        """
        raise AttributeError(
            "temperature is a read-only property derived from the Experiment."
        )

    # # TODO: make it use experiment temperature
    # @property def temperature(self) -> Parameter | None: """The
    # temperature from the associated Experiment, if available.""" if
    #     self.experiment is not None: return
    #     self.experiment.temperature return None

    # @temperature.setter def temperature(self, value) -> None:
    # """temperature is a read-only property derived from the
    #     Experiment.""" raise AttributeError( "temperature is a
    #     read-only property derived from the Experiment." )

    #############
    # Other methods
    #############

    def calculate(self, energy: float | None, Q_index: int) -> np.ndarray:
        """Calculate the model prediction for a given Q index.

        Args:
            energy (float): The energy value to calculate the model for.
            Q_index (int): The index of the Q value to calculate the
            model for.
        Returns:
            sc.DataArray: The calculated model prediction.
        """
        if energy is None:
            energy = self.energy

        if self.sample_model is None:
            sample_intensity = np.zeros_like(energy)
        else:
            if self.resolution_model is None:
                sample_intensity = self.sample_model._component_collections[
                    Q_index
                ].evaluate(energy)
            else:
                convolver = self._create_convolver(Q_index)
                sample_intensity = convolver.convolution()

        if self.background_model is None:
            background_intensity = np.zeros_like(energy)
        else:
            background_intensity = self.background_model._component_collections[
                Q_index
            ].evaluate(energy)

        sample_plus_background = sample_intensity + background_intensity

        return sample_plus_background

    def calculate_individual_components(
        self, Q_index: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Calculate the model prediction for a given Q index for each
        individual component.

        Args:
            Q_index (int): The index of the Q value to calculate the
            model for.
        Returns:
            list[np.ndarray]: The calculated model predictions for each
            individual component.
        """
        sample_results = []
        background_results = []

        if self.sample_model is not None:
            # Calculate sample components
            for component in self.sample_model._component_collections[
                Q_index
            ]._components:
                if self.resolution_model is None:
                    component_intensity = component.evaluate(self.energy)
                else:
                    convolver = Convolution(
                        sample_components=component,
                        resolution_components=self.resolution_model._component_collections[
                            Q_index
                        ],
                        energy=self.energy,
                        temperature=self.temperature,
                    )
                    component_intensity = convolver.convolution()
                sample_results.append(component_intensity)

        if self.background_model is not None:
            # Calculate background components
            for component in self.background_model._component_collections[
                Q_index
            ]._components:
                component_intensity = component.evaluate(self.energy)
                background_results.append(component_intensity)

        return sample_results, background_results

    def calculate_all_Q(self) -> list[np.ndarray]:
        """Calculate the model prediction for all Q indices.

        Returns:
            list[np.ndarray]: The calculated model predictions for all Q
            indices.
        """
        results = []
        for Q_index in range(len(self.Q)):
            result = self.calculate(Q_index)
            results.append(result)
        return results

    # def calculate_individual_components_all_Q(
    #     self,
    #     add_background: bool = True,
    # ) -> list[tuple[list[np.ndarray], list[np.ndarray]]]:
    #     """Calculate the model prediction for all Q indices for each
    #     individual component.

    #     Returns: list[tuple[list[np.ndarray], list[np.ndarray]]]: The
    #         calculated model predictions for each individual component
    #         at all Q indices. """ all_results = [] for Q_index in
    #         range(len(self.Q)): sample_results, background_results =
    #     self.calculate_individual_components( Q_index ) if
    #     add_background: sample_results = sample_results +
    #     background_results all_results.append((sample_results,
    #         background_results)) return all_results

    def calculate_single_component_all_Q(
        self,
        component_index: int,
    ) -> list[np.ndarray]:
        """Calculate the model prediction for all Q indices for a single
        component.

        Args:
            component_index (int): The index of the component
        Returns:
            list[np.ndarray]: The calculated model predictions for the
            specified component at all Q indices.
        """

        results = []
        for Q_index in range(len(self.Q)):
            if self.sample_model is not None:
                component = self.sample_model._component_collections[
                    Q_index
                ]._components[component_index]
                if self.resolution_model is None:
                    component_intensity = component.evaluate(self.energy)
                else:
                    convolver = Convolution(
                        sample_components=component,
                        resolution_components=self.resolution_model._component_collections[
                            Q_index
                        ],
                        energy=self.energy,
                        temperature=self.temperature,
                    )
                    component_intensity = convolver.convolution()
                results.append(component_intensity)
            else:
                results.append(np.zeros_like(self.energy))

        model_data_array = sc.DataArray(
            data=sc.array(dims=["Q", "energy"], values=results),
            coords={
                "Q": self.Q,
                "energy": self.energy,
            },
        )
        return model_data_array

    def fit(self, Q_index: int):
        """Fit the model to the experimental data for a given Q index.

        Args:
            Q_index (int): The index of the Q value to fit the model
            to.
        Returns:
            FitResult: The result of the fit.
        """
        if self._experiment is None:
            raise ValueError("No experiment is associated with this Analysis.")

        if not isinstance(Q_index, int) or Q_index < 0 or Q_index >= len(self.Q):
            raise ValueError("Q_index must be a valid index for the Q values.")

        data = self.experiment.data["Q", Q_index]
        x = data.coords["energy"].values
        y = data.values
        e = data.variances**0.5

        def fit_func(x_vals):
            return self.calculate_theory(energy=x_vals, Q_index=Q_index)

        fitter = EasyScienceFitter(
            fit_object=self,
            fit_function=fit_func,
        )

        # Perform the fit
        fit_result = fitter.fit(x=x, y=y, weights=1.0 / e)

        # Store result
        self.fit_result = fit_result

        return fit_result

    def plot_data_and_model(
        self,
        plot_individual_components: bool = True,
    ) -> None:
        """Plot the experimental data and the model prediction.

        Args:
            plot_individual_components (bool): Whether to plot
            individual components. Default is True.
        """
        if not isinstance(plot_individual_components, bool):
            raise TypeError("plot_individual_components must be True or False.")

        model_data_array = self._create_model_data_group(
            individual_components=plot_individual_components
        )
        if self.experiment is None or self.experiment.data is None:
            raise ValueError("Experiment data is not available for plotting.")

        from IPython.display import display

        fig = pp.slicer(
            {"Data": self.experiment.data, "Model": model_data_array},
            color={"Data": "black", "Model": "red"},
            linestyle={"Data": "none", "Model": "solid"},
            marker={"Data": "o", "Model": "None"},
        )
        display(fig)

    #############
    # Private methods
    #############

    def _update_models(self):
        """Update models based on the current experiment."""
        if self.experiment is None:
            return

        for Q_index in range(len(self.Q)):
            self._convolvers[Q_index] = self._create_convolver(Q_index)

    def _create_convolver(self, Q_index: int):
        """Initialize and return a Convolution object for the given Q
        index.
        """
        # Add checks of empty sample models etc

        sample_components = self.sample_model._component_collections[Q_index]
        resolution_components = self.resolution_model._component_collections[Q_index]
        energy = self.energy
        convolver = Convolution(
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy=energy,
            temperature=self.temperature,
        )
        return convolver

    def _create_model_data_group(self, individual_components=True) -> sc.DataArray:
        """Create a Scipp DataArray representing the model over all Q
        and energy values.
        """
        if self.Q is None or self.energy is None:
            raise ValueError("Q and energy must be defined in the experiment.")

        model_data = []
        for Q_index in range(len(self.Q)):
            model_at_Q = self.calculate(Q_index)
            model_data.append(model_at_Q)

        model_data_array = sc.DataArray(
            data=sc.array(dims=["Q", "energy"], values=model_data),
            coords={
                "Q": self.Q,
                "energy": self.energy,
            },
        )
        model_group = sc.DataGroup({"Model": model_data_array})

        #         if plot_individual_components: comps =
        #             ana.calculate_individual_components(E) for name,
        #             vals in comps.items(): if name not in
        #                 component_arrays: component_arrays[name] =
        #                     sc.zeros_like(data) csel =
        #                 component_arrays[name] for d, i in
        #                 zip(loop_dims, combo): csel = csel[d, i]
        #                     csel.values = vals fsel.values =
        #                 ana.calculate_theory(E)

        # # Build plot group
        # data_and_model = {"Data": self._experiment._data.data,
        # "Model": fit_total} if plot_individual_components and
        #     component_arrays: data_and_model.update(component_arrays)
        # data_and_model = sc.DataGroup(data_and_model)

        if individual_components:
            components = self.calculate_individual_components_all_Q()
            for Q_index, (sample_comps, background_comps) in enumerate(components):
                for samp_index, samp_comp in enumerate(sample_comps):
                    model_data_array[samp_comp.display_name] = sc.zeros_like(
                        model_data_array.data
                    )
                    model_data_array[samp_comp.display_name].data[
                        Q_index, :
                    ] = samp_comp
                for back_index, back_comp in enumerate(background_comps):
                    model_data_array[back_comp.display_name] = sc.zeros_like(
                        model_data_array.data
                    )
                    model_data_array[back_comp.display_name].data[
                        Q_index, :
                    ] = back_comp

        model_data_array = model_data_array + model_group  # WRONG BUT LINT
        return model_data_array

    # def _create_convolvers(
    #     self, energy: np.ndarray | sc.Variable | None = None
    # ) -> None:
    #     """Create Convolution objects for each Q value."""
    #     num_Q = len(self.Q) if self.Q is not None else 0
    #     self._convolvers = [
    #         self._create_convolver(i, energy=energy) for i in range(num_Q)
    #     ]
