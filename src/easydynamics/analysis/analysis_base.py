# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase as EasyScienceModelBase
from easyscience.variable import Parameter

from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel


class AnalysisBase(EasyScienceModelBase):
    """Base class for analysis in EasyDynamics. This class is not meant
    to be used directly.

    An Analysis consists of an Experiment, a SampleModel, and an
    InstrumentModel. The Experiment contains the data to be fitted, the
    SampleModel contains the model for the sample, and the
    InstrumentModel contains the model for the instrument, including
    background and resolution

    Args:
        display_name (str): Display name of the analysis.
        unique_name (str or None): Unique name of the analysis. If None,
            a unique name is automatically generated.
        experiment (Experiment | None): The Experiment associated with
            this Analysis. If None, a default Experiment is created.
        sample_model (SampleModel | None): The SampleModel associated
            with this Analysis. If None, a default SampleModel is
            created.
        instrument_model (InstrumentModel | None): The InstrumentModel
            associated with this Analysis. If None, a default
            InstrumentModel is created.
        extra_parameters (Parameter | list[Parameter] | None):
        Extra parameters to be included in the analysis for advanced
            users. If None, no extra parameters are added.

    Attributes:
        experiment (Experiment): The Experiment associated with this
            Analysis.
        sample_model (SampleModel): The SampleModel associated with this
            Analysis.
        instrument_model (InstrumentModel): The InstrumentModel
            associated with this Analysis.
        Q (sc.Variable | None): The Q values from the associated
            Experiment, if available.
        energy (sc.Variable | None): The energy values from the
            associated Experiment, if available.
        temperature (Parameter | None): The temperature from the
            associated SampleModel, if available.
        extra_parameters (list[Parameter]): The extra parameters
        included in this Analysis.
    """

    def __init__(
        self,
        display_name: str = 'MyAnalysis',
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
            raise TypeError('experiment must be an instance of Experiment or None.')

        if sample_model is None:
            self._sample_model = SampleModel()
        elif isinstance(sample_model, SampleModel):
            self._sample_model = sample_model
        else:
            raise TypeError('sample_model must be an instance of SampleModel or None.')

        if instrument_model is None:
            self._instrument_model = InstrumentModel()
        elif isinstance(instrument_model, InstrumentModel):
            self._instrument_model = instrument_model
        else:
            raise TypeError('instrument_model must be an instance of InstrumentModel or None.')

        if extra_parameters is not None:
            if isinstance(extra_parameters, Parameter):
                self._extra_parameters = [extra_parameters]
            elif isinstance(extra_parameters, list) and all(
                isinstance(p, Parameter) for p in extra_parameters
            ):
                self._extra_parameters = extra_parameters
            else:
                raise TypeError('extra_parameters must be a Parameter or a list of Parameters.')
        else:
            self._extra_parameters = []

        self._on_experiment_changed()

    #############
    # Properties
    #############

    @property
    def experiment(self) -> Experiment:
        """Get the Experiment associated with this Analysis.

        Returns:
            Experiment: The Experiment associated with this Analysis.
        """

        return self._experiment

    @experiment.setter
    def experiment(self, value: Experiment) -> None:
        """Set the Experiment for this Analysis.

        Raises:
            TypeError: if value is not an Experiment.
        """

        if not isinstance(value, Experiment):
            raise TypeError('experiment must be an instance of Experiment')
        self._experiment = value
        self._on_experiment_changed()

    @property
    def sample_model(self) -> SampleModel:
        """Get the SampleModel associated with this Analysis.

        Returns:
            SampleModel: The SampleModel associated with this Analysis.
        """

        return self._sample_model

    @sample_model.setter
    def sample_model(self, value: SampleModel) -> None:
        """Set the SampleModel for this Analysis.

        Raises:
            TypeError: if value is not a SampleModel.
        """
        if not isinstance(value, SampleModel):
            raise TypeError('sample_model must be an instance of SampleModel')
        self._sample_model = value
        self._on_sample_model_changed()

    @property
    def instrument_model(self) -> InstrumentModel:
        """Get the InstrumentModel associated with this Analysis.

        Returns:
            InstrumentModel: The InstrumentModel associated with this
                Analysis.
        """
        return self._instrument_model

    @instrument_model.setter
    def instrument_model(self, value: InstrumentModel) -> None:
        """Set the InstrumentModel for this Analysis.

        Raises:
            TypeError: if value is not an InstrumentModel.
        """
        if not isinstance(value, InstrumentModel):
            raise TypeError('instrument_model must be an instance of InstrumentModel')
        self._instrument_model = value
        self._on_instrument_model_changed()

    @property
    def Q(self) -> sc.Variable | None:
        """Get the Q values from the associated Experiment, if
        available.

        Returns:
            sc.Variable: The Q values from the associated Experiment,
                if available.
            None: If the Experiment does not have any data.
        """
        return self.experiment.Q

    @Q.setter
    def Q(self, value) -> None:
        """Q cannot be set, as it is a read-only property derived from
        the Experiment.

        Raises:
            AttributeError: If trying to set Q.
        """
        raise AttributeError('Q is a read-only property derived from the Experiment.')

    @property
    def energy(self) -> sc.Variable | None:
        """Get the energy values from the associated Experiment, if
        available.

        Returns:
            sc.Variable: The energy values from the associated
            Experiment, if available.
            None: If the Experiment does not have any data.
        """

        return self.experiment.energy

    @energy.setter
    def energy(self, value) -> None:
        """Energy cannot be set, as it is a read-only property derived
        from the Experiment.

        Raises:
            AttributeError: If trying to set energy.
        """

        raise AttributeError('energy is a read-only property derived from the Experiment.')

    @property
    def temperature(self) -> Parameter | None:
        """Get the temperature from the associated SampleModel, if
        available.

        Returns:
            Parameter: The temperature from the associated SampleModel,
            if available.
            None: If the SampleModel does not have a temperature.
        """

        return self.sample_model.temperature

    @temperature.setter
    def temperature(self, value) -> None:
        """Temperature cannot be set, as it is a read-only property
        derived from the SampleModel.

        Raises:
            AttributeError: If trying to set temperature.
        """

        raise AttributeError('temperature is a read-only property derived from the SampleModel.')

    @property
    def extra_parameters(self) -> list[Parameter]:
        """Get the extra parameters included in this Analysis.

        Returns:
            list[Parameter]: The extra parameters included in this
                Analysis.
        """
        return self._extra_parameters

    @extra_parameters.setter
    def extra_parameters(self, value: Parameter | list[Parameter]) -> None:
        """Set the extra parameters for this Analysis.

        Args:
            value (Parameter | list[Parameter]): The extra parameters to
            include in this Analysis.

        Raises:
            TypeError: If value is not a Parameter or a list of
            Parameters.
        """
        if isinstance(value, Parameter):
            self._extra_parameters = [value]
        elif isinstance(value, list) and all(isinstance(p, Parameter) for p in value):
            self._extra_parameters = value
        else:
            raise TypeError('extra_parameters must be a Parameter or a list of Parameters.')

    #############
    # Other methods
    #############

    #############
    # Private methods
    #############

    def _on_experiment_changed(self) -> None:
        """Update the Q values in the sample and instrument models when
        the experiment changes.
        """
        self.sample_model.Q = self.Q
        self.instrument_model.Q = self.Q

    def _on_sample_model_changed(self) -> None:
        """Update the Q values in the sample model when the sample model
        changes.
        """
        self.sample_model.Q = self.Q

    def _on_instrument_model_changed(self) -> None:
        """Update the Q values in the instrument model when the
        instrument model changes.
        """
        self.instrument_model.Q = self.Q

    def _verify_Q_index(self, Q_index: int | None) -> int | None:
        """Verify that the Q index is valid.

        Args:
            Q_index (int | None): The Q index to verify.

        Returns:
            int | None: The verified Q index.

        Raises:
            IndexError: If the Q index is not valid.
        """
        if Q_index is not None:
            if (
                not isinstance(Q_index, int)
                or Q_index < 0
                or (self.Q is not None and Q_index >= len(self.Q))
            ):
                raise IndexError('Q_index must be a valid index for the Q values.')
        return Q_index

    def _extract_x_y_weights_from_experiment(
        self, Q_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract the x, y, and weights arrays from the experiment for
        the given Q index.

        Args:
            Q_index (int): The Q index to extract the data for.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The x, y, and
                weights arrays extracted from the experiment for the
                given Q index.
        """
        data = self.experiment.data['Q', Q_index]
        x = data.coords['energy'].values
        y = data.values
        e = data.variances**0.5
        if np.any(e == 0):
            raise ValueError('Cannot compute weights: some variances are zero.')
        weights = 1.0 / e
        return x, y, weights

    #############
    # Dunder methods
    #############

    def __repr__(self) -> str:
        """Return a string representation of the Analysis.

        Returns:
            str: A string representation of the Analysis.
        """
        return f' {self.__class__.__name__}  (display_name={self.display_name}, \
        unique_name={self.unique_name})'
