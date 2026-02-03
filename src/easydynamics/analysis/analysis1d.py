# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase as EasyScienceModelBase
from easyscience.fitting.fitter import Fitter as EasyScienceFitter
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import BackgroundModel
from easydynamics.sample_model import ResolutionModel
from easydynamics.sample_model import SampleModel


class Analysis1d(EasyScienceModelBase):
    """For analysing data."""

    def __init__(
        self,
        display_name: str = 'MyAnalysis',
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        resolution_model: ResolutionModel | None = None,
        background_model: BackgroundModel | None = None,
        energy_offset: list[Parameter] | None = None,
        Q_index: int | None = None,
    ):
        super().__init__(display_name=display_name, unique_name=unique_name)

        if experiment is not None and not isinstance(experiment, Experiment):
            raise TypeError('experiment must be an instance of Experiment or None.')

        self._experiment = experiment

        if sample_model is not None and not isinstance(sample_model, SampleModel):
            raise TypeError('sample_model must be an instance of SampleModel or None.')
        sample_model.Q = self.Q
        self._sample_model = sample_model

        if resolution_model is not None and not isinstance(resolution_model, ResolutionModel):
            raise TypeError('resolution_model must be an instance of ResolutionModel or None.')
        resolution_model.Q = self.Q
        self._resolution_model = resolution_model

        if background_model is not None and not isinstance(background_model, BackgroundModel):
            raise TypeError('background_model must be an instance of BackgroundModel or None.')
        background_model.Q = self.Q
        self._background_model = background_model

        self._convolvers = [None] * (len(self.Q) if self.Q is not None else 0)
        self._update_models()

        if not isinstance(energy_offset, list) and energy_offset is not None:
            raise TypeError('energy_offset must be a list of Parameters or None.')

        if energy_offset is not None:
            if len(energy_offset) != len(self.Q):
                raise ValueError('energy_offset list length must match number of Q values.')
            for offset in energy_offset:
                if not isinstance(offset, Parameter):
                    raise TypeError('Each energy_offset must be an instance of Parameter.')
        else:
            energy_offset = [
                Parameter(name='energy_offset', value=0.0, unit=self.sample_model.unit)
                for _ in range(len(self.Q))
            ]
        self._energy_offset = energy_offset

        if Q_index is not None:
            if (
                not isinstance(Q_index, int)
                or Q_index < 0
                or (self.Q is not None and Q_index >= len(self.Q))
            ):
                raise ValueError('Q_index must be a valid index for the Q values.')
        self._Q_index = Q_index

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
            raise TypeError('experiment must be an instance of Experiment or None.')
        self._experiment = value
        self._update_models()

    @property
    def sample_model(self) -> SampleModel | None:
        """The SampleModel associated with this Analysis."""
        return self._sample_model

    @sample_model.setter
    def sample_model(self, value: SampleModel | None) -> None:
        if value is not None and not isinstance(value, SampleModel):
            raise TypeError('sample_model must be an instance of SampleModel or None.')
        self._sample_model = value
        self._update_models()

    @property
    def resolution_model(self) -> ResolutionModel | None:
        """The ResolutionModel associated with this Analysis."""
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, value: ResolutionModel | None) -> None:
        if value is not None and not isinstance(value, ResolutionModel):
            raise TypeError('resolution_model must be an instance of ResolutionModel or None.')
        self._resolution_model = value
        self._update_models()

    @property
    def background_model(self) -> BackgroundModel | None:
        """The BackgroundModel associated with this Analysis."""
        return self._background_model

    @background_model.setter
    def background_model(self, value: BackgroundModel | None) -> None:
        if value is not None and not isinstance(value, BackgroundModel):
            raise TypeError('background_model must be an instance of BackgroundModel or None.')
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
        raise AttributeError('Q is a read-only property derived from the Experiment.')

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
        raise AttributeError('energy is a read-only property derived from the Experiment.')

    @property
    def temperature(self) -> Parameter | None:
        """The temperature from the associated Experiment, if
        available.
        """
        return self.sample_model.temperature if self.sample_model is not None else None

    @temperature.setter
    def temperature(self, value) -> None:
        """Temperature is a read-only property derived from the
        Experiment.
        """
        raise AttributeError('temperature is a read-only property derived from the sample model.')

    @property
    def energy_offset(self) -> list[Parameter] | None:
        """Get the energy offsets for each Q value."""
        return self._energy_offset

    @energy_offset.setter
    def energy_offset(self, offsets: list[Parameter] | None) -> None:
        """Set the energy offsets for each Q value.

        Args:
            offsets (list[Parameter] | None): The list of energy
            offsets.
        Raises:
            TypeError: If offsets is not a list of Parameters or
            None.
        """
        if offsets is not None:
            if len(offsets) != len(self.Q):
                raise ValueError('energy_offset list length must match number of Q values.')
            for offset in offsets:
                if not isinstance(offset, Parameter):
                    raise TypeError('Each energy_offset must be an instance of Parameter.')
        self._energy_offset = offsets

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
                raise ValueError('Q_index must be a valid index for the Q values.')
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
        Q_index = self.Q_index
        if Q_index is None:
            raise ValueError('Q_index must be set to calculate the model.')

        if energy is None:
            energy = self.energy.values

        # TODO: handle units properly
        energy = energy - self.energy_offset[Q_index].value
        if self.sample_model is None:
            sample_intensity = np.zeros_like(energy)
        else:
            if self.resolution_model is None:
                sample_intensity = self.sample_model._component_collections[Q_index].evaluate(
                    energy
                )
            else:
                convolver = self._convolvers[Q_index]
                sample_intensity = convolver.convolution()

        if self.background_model is None:
            background_intensity = np.zeros_like(energy)
        else:
            background_intensity = self.background_model._component_collections[Q_index].evaluate(
                energy
            )

        sample_plus_background = sample_intensity + background_intensity

        return sample_plus_background

    def calculate_individual_components(
        self,
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
        Q_index = self.Q_index
        if Q_index is None:
            raise ValueError('Q_index must be set to calculate the model.')

        if self.sample_model is not None:
            # Calculate sample components
            for component in self.sample_model._component_collections[Q_index]._components:
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
            for component in self.background_model._component_collections[Q_index]._components:
                component_intensity = component.evaluate(self.energy)
                background_results.append(component_intensity)

        return sample_results, background_results

    def fit(self):
        """Fit the model to the experimental data for a given Q index.

        Args:
        Returns:
            FitResult: The result of the fit.
        """
        if self._experiment is None:
            raise ValueError('No experiment is associated with this Analysis.')

        Q_index = self.Q_index
        if Q_index is None:
            raise ValueError('Q_index must be set to perform the fit.')

        data = self.experiment.data['Q', Q_index]
        x = data.coords['energy'].values
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
            raise TypeError('plot_individual_components must be True or False.')

        import matplotlib.pyplot as plt

        Q_index = self.Q_index
        if Q_index is None:
            raise ValueError('Q_index must be set to plot the data and model.')
        if self.experiment is None or self.experiment.data is None:
            raise ValueError('Experiment data is not available for plotting.')
        data = self.experiment.data['Q', Q_index]
        energy = data.coords['energy'].values
        model = self.calculate(energy=energy)
        plt.figure()
        plt.errorbar(
            energy,
            data.values,
            yerr=data.variances**0.5,
            fmt='o',
            label='Data',
            color='black',
        )
        plt.plot(energy, model, label='Model', color='red')
        if plot_individual_components:
            sample_comps, background_comps = self.calculate_individual_components()
            for i, comp in enumerate(sample_comps):
                plt.plot(
                    energy,
                    comp,
                    label=f'Sample Component {i + 1}',
                    linestyle='--',
                )
            for i, comp in enumerate(background_comps):
                plt.plot(
                    energy,
                    comp,
                    label=f'Background Component {i + 1}',
                    linestyle=':',
                )
        plt.xlabel(f'Energy ({self.energy.unit})')
        plt.ylabel(f'Intensity ({self.sample_model.unit})')
        plt.title(f'Data and Model at Q index {Q_index}')
        plt.legend()
        plt.show()
        # model_data_array = self._create_model_data_group(
        #     individual_components=plot_individual_components ) if
        # self.experiment is None or self.experiment.data is None: raise
        # ValueError("Experiment data is not available for plotting.")

        # from IPython.display import display

        # fig = pp.slicer(
        #     {"Data": self.experiment.data, "Model": model_data_array},
        #     color={"Data": "black", "Model": "red"},
        #     linestyle={"Data": "none", "Model": "solid"},
        #     marker={"Data": "o", "Model": "None"},
        # )
        # display(fig)

    def get_all_variables(self) -> list[DescriptorNumber]:
        """Get all variables used in the analysis.

        Returns:
            List[Descriptor]: A list of all variables.
        """
        variables = []
        if self.sample_model is not None:
            variables.extend(
                self.sample_model._component_collections[self.Q_index].get_all_variables()
            )
        if self.resolution_model is not None:
            variables.extend(
                self.resolution_model._component_collections[self.Q_index].get_all_variables()
            )
        if self.background_model is not None:
            variables.extend(
                self.background_model._component_collections[self.Q_index].get_all_variables()
            )
        variables.append(self.energy_offset[self.Q_index])
        # TODO temperature and diffusion
        return variables

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
        if self.sample_model is None or self.resolution_model is None:
            raise ValueError('Both sample_model and resolution_model must be defined.')

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
            raise ValueError('Q and energy must be defined in the experiment.')

        model_data = []
        for Q_index in range(len(self.Q)):
            model_at_Q = self.calculate(Q_index)
            model_data.append(model_at_Q)

        model_data_array = sc.DataArray(
            data=sc.array(dims=['Q', 'energy'], values=model_data),
            coords={
                'Q': self.Q,
                'energy': self.energy,
            },
        )
        model_group = sc.DataGroup({'Model': model_data_array})

        if individual_components:
            components = self.calculate_individual_components_all_Q()
            for Q_index, (sample_comps, background_comps) in enumerate(components):
                for samp_index, samp_comp in enumerate(sample_comps):
                    model_data_array[samp_comp.display_name] = sc.zeros_like(model_data_array.data)
                    model_data_array[samp_comp.display_name].data[Q_index, :] = samp_comp
                for back_index, back_comp in enumerate(background_comps):
                    model_data_array[back_comp.display_name] = sc.zeros_like(model_data_array.data)
                    model_data_array[back_comp.display_name].data[Q_index, :] = back_comp

        model_data_array = model_data_array + model_group  # WRONG BUT LINT
        return model_data_array
