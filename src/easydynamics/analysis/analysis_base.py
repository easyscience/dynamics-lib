# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase as EasyScienceModelBase
from easyscience.variable import Parameter

from easydynamics.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent


class AnalysisBase(EasyScienceModelBase):
    """For analysing data."""

    def __init__(
        self,
        display_name: str = "MyAnalysis",
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        extra_parameters: Parameter | list[Parameter] | None = None,
    ):
        super().__init__(display_name=display_name, unique_name=unique_name)

        if experiment is None:
            self._experiment = Experiment()
        elif isinstance(experiment, Experiment):
            self._experiment = experiment
        else:
            raise TypeError("experiment must be an instance of Experiment or None.")

        if sample_model is None:
            self._sample_model = SampleModel()
        elif isinstance(sample_model, SampleModel):
            self._sample_model = sample_model
        else:
            raise TypeError("sample_model must be an instance of SampleModel or None.")

        if instrument_model is None:
            self._instrument_model = InstrumentModel()
        elif isinstance(instrument_model, InstrumentModel):
            self._instrument_model = instrument_model
        else:
            raise TypeError(
                "instrument_model must be an instance of InstrumentModel or None."
            )

        if extra_parameters is not None:
            if isinstance(extra_parameters, Parameter):
                self._extra_parameters = [extra_parameters]
            elif isinstance(extra_parameters, list) and all(
                isinstance(p, Parameter) for p in extra_parameters
            ):
                self._extra_parameters = extra_parameters
            else:
                raise TypeError(
                    "extra_parameters must be a Parameter or a list of Parameters."
                )
        else:
            self._extra_parameters = []

        self._on_experiment_changed()

    #############
    # Properties
    #############

    @property
    def experiment(self) -> Experiment | None:
        """The Experiment associated with this Analysis."""
        return self._experiment

    @experiment.setter
    def experiment(self, value: Experiment) -> None:
        if not isinstance(value, Experiment):
            raise TypeError("experiment must be an instance of Experiment")
        self._experiment = value
        self._on_experiment_changed()

    @property
    def sample_model(self) -> SampleModel:
        """The SampleModel associated with this Analysis."""
        return self._sample_model

    @sample_model.setter
    def sample_model(self, value: SampleModel) -> None:
        if not isinstance(value, SampleModel):
            raise TypeError("sample_model must be an instance of SampleModel")
        self._sample_model = value
        self._on_sample_model_changed()

    @property
    def instrument_model(self) -> InstrumentModel:
        """The InstrumentModel associated with this Analysis."""
        return self._instrument_model

    @instrument_model.setter
    def instrument_model(self, value: InstrumentModel) -> None:
        if not isinstance(value, InstrumentModel):
            raise TypeError("instrument_model must be an instance of InstrumentModel")
        self._instrument_model = value
        self._on_instrument_model_changed()

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

    @property
    def temperature(self) -> Parameter | None:
        """
        The temperature from the associated SampleModel, if available.
        """
        return self.sample_model.temperature if self.sample_model is not None else None

    @temperature.setter
    def temperature(self, value) -> None:
        """
        Temperature is a read-only property derived from the
        SampleModel.
        """
        raise AttributeError(
            "temperature is a read-only property derived from the sample model."
        )

    #############
    # Other methods
    #############

    #############
    # Private methods
    #############

    def _on_experiment_changed(self) -> None:
        self._sample_model.Q = self.Q
        self._instrument_model.Q = self.Q

    def _on_sample_model_changed(self) -> None:
        self._sample_model.Q = self.Q

    def _on_instrument_model_changed(self) -> None:
        self._instrument_model.Q = self.Q

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

    #############
    # Dunder methods
    #############

    def __repr__(self) -> str:
        return f"AnalysisBase(display_name={self.display_name}, unique_name={self.unique_name})"
