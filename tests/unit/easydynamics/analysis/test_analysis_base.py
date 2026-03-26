# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np
import pytest
from easyscience.variable import Parameter

from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel


class TestAnalysisBase:
    @pytest.fixture
    def analysis_base(self):
        experiment = Experiment()
        sample_model = SampleModel()
        instrument_model = InstrumentModel()
        analysis_base = AnalysisBase(
            display_name='TestAnalysis',
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )
        return analysis_base

    def test_init(self, analysis_base):
        # WHEN THEN

        # EXPECT
        assert analysis_base.display_name == 'TestAnalysis'
        assert isinstance(analysis_base.experiment, Experiment)
        assert isinstance(analysis_base.sample_model, SampleModel)
        assert isinstance(analysis_base.instrument_model, InstrumentModel)
        assert analysis_base._extra_parameters == []

    def test_init_extra_parameter(self):
        extra_parameter = Parameter(name='param1', value=1.0)
        analysis = AnalysisBase(extra_parameters=extra_parameter)
        assert analysis._extra_parameters == [extra_parameter]

    def test_init_extra_parameters(self):
        extra_parameters = [
            Parameter(name='param1', value=1.0),
            Parameter(name='param2', value=2.0),
        ]
        analysis = AnalysisBase(extra_parameters=extra_parameters)
        assert analysis._extra_parameters == extra_parameters

    def test_init_calls_on_experiment_changed(self):
        with patch.object(AnalysisBase, '_on_experiment_changed') as mock_on_experiment_changed:
            AnalysisBase()
            mock_on_experiment_changed.assert_called_once()

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            (
                {'experiment': 123},
                TypeError,
                'experiment must be an instance of Experiment',
            ),
            (
                {'sample_model': 'not a model'},
                TypeError,
                'sample_model must be an instance of SampleModel',
            ),
            (
                {'instrument_model': 'not a model'},
                TypeError,
                'instrument_model must be an instance of InstrumentModel',
            ),
            (
                {'extra_parameters': 123},
                TypeError,
                'extra_parameters must be a Parameter or a list of Parameters.',
            ),
            (
                {'extra_parameters': [123]},
                TypeError,
                'extra_parameters must be a Parameter or a list of Parameters.',
            ),
        ],
        ids=[
            'invalid experiment',
            'invalid sample_model',
            'invalid instrument_model',
            'invalid extra_parameters',
            'invalid extra_parameters list',
        ],
    )
    def test_init_invalid_inputs(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            AnalysisBase(**kwargs)

    def test_experiment_setter_calls_on_experiment_changed(self, analysis_base):
        with patch.object(analysis_base, '_on_experiment_changed') as mock_on_experiment_changed:
            new_experiment = Experiment()
            analysis_base.experiment = new_experiment
            mock_on_experiment_changed.assert_called_once()

    def test_experiment_setter_invalid_type(self, analysis_base):
        with pytest.raises(TypeError, match='experiment must be an instance of Experiment'):
            analysis_base.experiment = 'not an experiment'

    def test_experiment_setter_valid(self, analysis_base):
        new_experiment = Experiment()
        analysis_base.experiment = new_experiment
        assert analysis_base.experiment == new_experiment

    def test_sample_model_setter_invalid_type(self, analysis_base):
        with pytest.raises(TypeError, match='sample_model must be an instance of SampleModel'):
            analysis_base.sample_model = 'not a sample model'

    def test_sample_model_setter_valid(self, analysis_base):
        new_sample_model = SampleModel()
        analysis_base.sample_model = new_sample_model
        assert analysis_base.sample_model == new_sample_model

    def test_sample_model_setter_calls_on_sample_model_changed(self, analysis_base):
        with patch.object(
            analysis_base, '_on_sample_model_changed'
        ) as mock_on_sample_model_changed:
            new_sample_model = SampleModel()
            analysis_base.sample_model = new_sample_model
            mock_on_sample_model_changed.assert_called_once()

    def test_instrument_model_setter_invalid_type(self, analysis_base):
        with pytest.raises(
            TypeError, match='instrument_model must be an instance of InstrumentModel'
        ):
            analysis_base.instrument_model = 'not an instrument model'

    def test_instrument_model_setter_valid(self, analysis_base):
        new_instrument_model = InstrumentModel()
        analysis_base.instrument_model = new_instrument_model
        assert analysis_base.instrument_model == new_instrument_model

    def test_instrument_model_setter_calls_on_instrument_model_changed(self, analysis_base):
        with patch.object(
            analysis_base, '_on_instrument_model_changed'
        ) as mock_on_instrument_model_changed:
            new_instrument_model = InstrumentModel()
            analysis_base.instrument_model = new_instrument_model
            mock_on_instrument_model_changed.assert_called_once()

    def test_Q_property(self, analysis_base):
        # Create a mock Q value
        fake_Q = [1, 2, 3]

        # Patch the 'experiment' attribute's Q property
        with patch.object(
            type(analysis_base.experiment), 'Q', new_callable=PropertyMock
        ) as mock_Q:
            mock_Q.return_value = fake_Q
            result = analysis_base.Q  # Access the property
            assert result == fake_Q
            mock_Q.assert_called_once()

    def test_Q_setter_raises(self, analysis_base):
        with pytest.raises(
            AttributeError,
            match='Q is a read-only property derived from the Experiment.',
        ):
            analysis_base.Q = [1, 2, 3]

    def test_energy_property(self, analysis_base):
        # Create a mock energy value
        fake_energy = [10, 20, 30]

        # Patch the 'experiment' attribute's energy property
        with patch.object(
            type(analysis_base.experiment), 'energy', new_callable=PropertyMock
        ) as mock_energy:
            mock_energy.return_value = fake_energy
            result = analysis_base.energy  # Access the property
            assert result == fake_energy
            mock_energy.assert_called_once()

    def test_energy_setter_raises(self, analysis_base):
        with pytest.raises(
            AttributeError,
            match='energy is a read-only property derived from the Experiment.',
        ):
            analysis_base.energy = [10, 20, 30]

    def test_temperature_property_no_temperature(self, analysis_base):
        # Patch the 'experiment' attribute's temperature property to
        # return None
        with patch.object(
            type(analysis_base.sample_model), 'temperature', new_callable=PropertyMock
        ) as mock_temperature:
            mock_temperature.return_value = None
            result = analysis_base.temperature  # Access the property
            assert result is None
            mock_temperature.assert_called_once()

    def test_temperature_property(self, analysis_base):
        # Create a mock temperature value
        fake_temperature = 300

        # Patch the 'sample_model' attribute's temperature property
        with patch.object(
            type(analysis_base.sample_model), 'temperature', new_callable=PropertyMock
        ) as mock_temperature:
            mock_temperature.return_value = fake_temperature
            result = analysis_base.temperature  # Access the property
            assert result == fake_temperature
            mock_temperature.assert_called_once()

    def test_temperature_setter_raises(self, analysis_base):
        with pytest.raises(
            AttributeError,
            match='temperature is a read-only property',
        ):
            analysis_base.temperature = 300

    @pytest.mark.parametrize(
        'extra_parameters',
        [
            Parameter(name='param1', value=1.0),
            [
                Parameter(name='param1', value=1.0),
                Parameter(name='param2', value=2.0),
            ],
        ],
        ids=[
            'single parameter',
            'list of parameters',
        ],
    )
    def test_extra_parameters_property(self, analysis_base, extra_parameters):
        # WHEN
        analysis_base.extra_parameters = extra_parameters

        # THEN
        analysis_base.extra_parameters = extra_parameters

        # EXPECT
        expected = (
            [extra_parameters] if isinstance(extra_parameters, Parameter) else extra_parameters
        )

        assert analysis_base.extra_parameters == expected

    @pytest.mark.parametrize(
        'invalid_extra_parameters',
        [
            'not a parameter',
            [Parameter(name='param1', value=1.0), 'not a parameter'],
        ],
        ids=[
            'single invalid parameter',
            'list with invalid parameter',
        ],
    )
    def test_extra_parameters_setter_invalid_type(self, analysis_base, invalid_extra_parameters):
        with pytest.raises(
            TypeError,
            match='extra_parameters must be',
        ):
            analysis_base.extra_parameters = invalid_extra_parameters

    #############
    # Other methods
    #############

    def test_normalize_resolution_calls_instrument_model(self, analysis_base):
        with patch.object(
            analysis_base.instrument_model, 'normalize_resolution'
        ) as mock_normalize_resolution:
            analysis_base.normalize_resolution()
            mock_normalize_resolution.assert_called_once()

    #############
    # Private methods
    #############

    def test_on_experiment_changed_updates_Q(self, analysis_base):
        # WHEN
        fake_Q = [1, 2, 3]

        # Patch the Q property of analysis_base
        with patch.object(
            type(analysis_base.experiment), 'Q', new_callable=PropertyMock
        ) as mock_Q:
            mock_Q.return_value = fake_Q

            # THEN
            analysis_base._on_experiment_changed()

            # EXPECT
            # assert that the Q attribute was set
            np.testing.assert_array_equal(analysis_base.Q, fake_Q)
            np.testing.assert_array_equal(analysis_base.sample_model.Q, fake_Q)
            np.testing.assert_array_equal(analysis_base.instrument_model.Q, fake_Q)

    def test_on_sample_model_changed_updates_Q(self, analysis_base):
        # WHEN
        fake_Q = [1, 2, 3]

        # Patch the Q property of analysis_base
        with patch.object(
            type(analysis_base.experiment), 'Q', new_callable=PropertyMock
        ) as mock_Q:
            mock_Q.return_value = fake_Q

            # THEN
            analysis_base._on_sample_model_changed()

            # EXPECT
            np.testing.assert_array_equal(analysis_base.sample_model.Q, fake_Q)

    def test_on_instrument_model_changed_updates_Q(self, analysis_base):
        fake_Q = [1, 2, 3]

        # Patch the Q property of analysis_base
        with patch.object(
            type(analysis_base.experiment), 'Q', new_callable=PropertyMock
        ) as mock_Q:
            mock_Q.return_value = fake_Q

            analysis_base._on_instrument_model_changed()
            np.testing.assert_array_equal(analysis_base.instrument_model.Q, fake_Q)

    def test_verify_Q_index_valid(self, analysis_base):
        # WHEN
        valid_Q_index = 0

        # THEN
        result = analysis_base._verify_Q_index(valid_Q_index)

        # EXPECT
        assert result == valid_Q_index

    def test_verify_Q_index_invalid(self, analysis_base):
        # WHEN
        invalid_Q_index = -1

        # THEN / EXPECT
        with pytest.raises(IndexError, match='Q_index must be a valid index'):
            analysis_base._verify_Q_index(invalid_Q_index)

    def test_repr(self, analysis_base):
        # WHEN
        repr_str = repr(analysis_base)

        # THEN EXPECT
        assert 'AnalysisBase' in repr_str
        assert 'display_name=TestAnalysis' in repr_str
        assert 'unique_name=' in repr_str
