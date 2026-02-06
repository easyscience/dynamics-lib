# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import scipp as sc
from easyscience.base_classes.model_base import ModelBase as EasyScienceModelBase
from easyscience.variable import Parameter

from easydynamics.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel


class Analysis1Base(EasyScienceModelBase):
    """For analysing data."""

    def __init__(
        self,
        display_name: str = "MyAnalysis",
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
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

    def _on_experiment_changed(self):
        pass

    def _on_sample_model_changed(self):
        pass

    def _on_instrument_model_changed(self):
        pass

    # def _update_models(self):
    #     """Update models based on the current experiment."""
    #     if self.experiment is None:
    #         return

    #     for Q_index in range(len(self.Q)):
    #         self._convolvers[Q_index] = self._create_convolver(Q_index)

    def _create_convolver(self, Q_index: int):
        """Initialize and return a Convolution object for the given Q
        index.
        """
        sample_components = self.sample_model._component_collections[Q_index]
        if sample_components == []:
            raise ValueError(f"Sample model has no components at Q index {Q_index}.")

        resolution_components = (
            self.instrument_model.resolution_model._component_collections[Q_index]
        )
        if resolution_components == []:
            raise ValueError(
                f"Resolution model has no components at Q index {Q_index}."
            )

        energy = self.energy
        convolver = Convolution(
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy=energy,
            temperature=self.temperature,
        )
        return convolver
