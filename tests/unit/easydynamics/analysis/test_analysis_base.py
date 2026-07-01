# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.gaussian import Gaussian
from easydynamics.settings.convolution_settings import ConvolutionSettings
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings


class TestAnalysisBase:
    @pytest.fixture
    def analysis_base(self):
        experiment = Experiment()
        sample_model = SampleModel()
        instrument_model = InstrumentModel()
        return AnalysisBase(
            display_name='TestAnalysis',
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )

    @pytest.fixture
    def analysis_base_with_components(self):
        Q = sc.array(dims=['Q'], values=[1, 2], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})

        experiment = Experiment(data=data_array)

        comp1 = Gaussian(name='Gaussian1', area=1, width=2, center=3)
        comp2 = Gaussian(name='Gaussian2', area=4, width=5, center=6)
        sample_model = SampleModel()
        sample_model.append_component(comp1)
        sample_model.append_component(comp2)
        instrument_model = InstrumentModel()
        return AnalysisBase(
            display_name='TestAnalysis',
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )

    def test_init(self, analysis_base):
        # WHEN THEN

        # EXPECT
        assert analysis_base.display_name == 'TestAnalysis'
        assert isinstance(analysis_base.experiment, Experiment)
        assert isinstance(analysis_base.sample_model, SampleModel)
        assert isinstance(analysis_base.instrument_model, InstrumentModel)
        assert analysis_base._extra_parameters == []

    def test_init_convolution_settings(self):
        # WHEN
        convolution_settings = ConvolutionSettings(upsample_factor=2, extension_factor=0.5)

        # THEN
        analysis = AnalysisBase(convolution_settings=convolution_settings)

        # EXPECT
        assert analysis.convolution_settings is convolution_settings

    def test_init_detailed_balance_settings(self):
        # WHEN
        detailed_balance_settings = DetailedBalanceSettings(
            use_detailed_balance=False,
            normalize_detailed_balance=False,
        )

        # THEN
        analysis = AnalysisBase(detailed_balance_settings=detailed_balance_settings)

        # EXPECT
        assert analysis.detailed_balance_settings is detailed_balance_settings

    def test_init_extra_parameter(self):
        # WHEN
        extra_parameter = Parameter(name='param1', value=1.0)

        # THEN
        analysis = AnalysisBase(extra_parameters=extra_parameter)
        # EXPECT
        assert analysis._extra_parameters == [extra_parameter]

    def test_init_extra_parameters(self):
        # WHEN
        extra_parameters = [
            Parameter(name='param1', value=1.0),
            Parameter(name='param2', value=2.0),
        ]

        # THEN
        analysis = AnalysisBase(extra_parameters=extra_parameters)
        # EXPECT
        assert analysis._extra_parameters == extra_parameters

    def test_init_calls_on_experiment_changed(self):
        # WHEN THEN EXPECT
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
                {'convolution_settings': 'not convolution settings'},
                TypeError,
                'convolution_settings must be an instance of ConvolutionSettings',
            ),
            (
                {'detailed_balance_settings': 'not detailed balance settings'},
                TypeError,
                'detailed_balance_settings must be an instance of DetailedBalanceSettings',
            ),
            (
                {'extra_parameters': 123},
                TypeError,
                'extra_parameters must be a Parameter, a list of Parameters, or None.',
            ),
            (
                {'extra_parameters': [123]},
                TypeError,
                'extra_parameters must be a Parameter, a list of Parameters, or None.',
            ),
        ],
        ids=[
            'invalid experiment',
            'invalid sample_model',
            'invalid instrument_model',
            'invalid convolution_settings',
            'invalid detailed_balance_settings',
            'invalid extra_parameters',
            'invalid extra_parameters list',
        ],
    )
    def test_init_invalid_inputs(self, kwargs, expected_exception, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            AnalysisBase(**kwargs)

    def test_experiment_setter_calls_on_experiment_changed(self, analysis_base):
        # WHEN THEN EXPECT
        with patch.object(analysis_base, '_on_experiment_changed') as mock_on_experiment_changed:
            new_experiment = Experiment()
            analysis_base.experiment = new_experiment
            mock_on_experiment_changed.assert_called_once()

    def test_experiment_setter_invalid_type(self, analysis_base):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='experiment must be an instance of Experiment'):
            analysis_base.experiment = 'not an experiment'

    def test_experiment_setter_valid(self, analysis_base):
        # WHEN
        new_experiment = Experiment()
        analysis_base.experiment = new_experiment
        # THEN EXPECT
        assert analysis_base.experiment == new_experiment

    def test_sample_model_setter_invalid_type(self, analysis_base):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='sample_model must be an instance of SampleModel'):
            analysis_base.sample_model = 'not a sample model'

    def test_sample_model_setter_valid(self, analysis_base):
        # WHEN
        new_sample_model = SampleModel()

        # THEN
        analysis_base.sample_model = new_sample_model
        # EXPECT
        assert analysis_base.sample_model == new_sample_model

    def test_sample_model_setter_calls_on_sample_model_changed(self, analysis_base):
        # WHEN
        new_sample_model = SampleModel()
        with patch.object(
            analysis_base, '_on_sample_model_changed'
        ) as mock_on_sample_model_changed:
            # THEN
            analysis_base.sample_model = new_sample_model

            # EXPECT
            mock_on_sample_model_changed.assert_called_once()

    def test_instrument_model_setter_invalid_type(self, analysis_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError, match='instrument_model must be an instance of InstrumentModel'
        ):
            analysis_base.instrument_model = 'not an instrument model'

    def test_instrument_model_setter_valid(self, analysis_base):
        # WHEN
        new_instrument_model = InstrumentModel()

        # THEN
        analysis_base.instrument_model = new_instrument_model

        # EXPECT
        assert analysis_base.instrument_model == new_instrument_model

    def test_instrument_model_setter_calls_on_instrument_model_changed(self, analysis_base):
        # WHEN
        new_instrument_model = InstrumentModel()
        with patch.object(
            analysis_base, '_on_instrument_model_changed'
        ) as mock_on_instrument_model_changed:
            # THEN
            analysis_base.instrument_model = new_instrument_model

            # EXPECT
            mock_on_instrument_model_changed.assert_called_once()

    def test_Q_property(self, analysis_base):
        # WHEN
        fake_Q = [1, 2, 3]
        with patch.object(
            type(analysis_base.experiment), 'Q', new_callable=PropertyMock
        ) as mock_Q:
            mock_Q.return_value = fake_Q
            # THEN
            result = analysis_base.Q
            # EXPECT
            assert result == fake_Q
            mock_Q.assert_called_once()

    def test_Q_setter_raises(self, analysis_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match=r'Q is a read-only property derived from the Experiment.',
        ):
            analysis_base.Q = [1, 2, 3]

    def test_energy_property(self, analysis_base):
        # WHEN
        fake_energy = [10, 20, 30]
        with patch.object(
            type(analysis_base.experiment), 'energy', new_callable=PropertyMock
        ) as mock_energy:
            mock_energy.return_value = fake_energy
            # THEN
            result = analysis_base.energy
            # EXPECT
            assert result == fake_energy
            mock_energy.assert_called_once()

    def test_energy_setter_raises(self, analysis_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match=r'energy is a read-only property derived from the Experiment.',
        ):
            analysis_base.energy = [10, 20, 30]

    def test_temperature_property_no_temperature(self, analysis_base):
        # WHEN
        with patch.object(
            type(analysis_base.sample_model), 'temperature', new_callable=PropertyMock
        ) as mock_temperature:
            mock_temperature.return_value = None
            # THEN
            result = analysis_base.temperature
            # EXPECT
            assert result is None
            mock_temperature.assert_called_once()

    def test_temperature_property(self, analysis_base):
        # WHEN
        fake_temperature = 300
        with patch.object(
            type(analysis_base.sample_model), 'temperature', new_callable=PropertyMock
        ) as mock_temperature:
            mock_temperature.return_value = fake_temperature
            # THEN
            result = analysis_base.temperature
            # EXPECT
            assert result == fake_temperature
            mock_temperature.assert_called_once()

    def test_temperature_setter_raises(self, analysis_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match='temperature is a read-only property',
        ):
            analysis_base.temperature = 300

    def test_convolution_settings_property(self, analysis_base):
        # WHEN
        convolution_settings = ConvolutionSettings(
            upsample_factor=2,
            extension_factor=0.5,
        )

        # THEN
        analysis_base.convolution_settings = convolution_settings

        # EXPECT
        assert analysis_base.convolution_settings is convolution_settings

    def test_convolution_settings_setter_invalid_type(self, analysis_base):

        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='convolution_settings must be an instance of ConvolutionSettings',
        ):
            analysis_base.convolution_settings = 'not convolution settings'

    def test_convolution_settings_calls_on_convolution_settings_changed(self, analysis_base):
        # WHEN
        convolution_settings = ConvolutionSettings(
            upsample_factor=2,
            extension_factor=0.5,
        )
        with patch.object(
            analysis_base, '_on_convolution_settings_changed'
        ) as mock_on_convolution_settings_changed:
            # THEN
            analysis_base.convolution_settings = convolution_settings

            # EXPECT
            mock_on_convolution_settings_changed.assert_called_once()

    def test_detailed_balance_settings_property(self, analysis_base):
        # WHEN
        new_settings = DetailedBalanceSettings(
            use_detailed_balance=False, normalize_detailed_balance=False
        )

        # THEN
        analysis_base.detailed_balance_settings = new_settings

        # EXPECT
        assert analysis_base.detailed_balance_settings is new_settings

    def test_detailed_balance_settings_setter_invalid(self, analysis_base):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='detailed_balance_settings must be a DetailedBalanceSettings',
        ):
            analysis_base.detailed_balance_settings = 'invalid_settings'

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
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='extra_parameters must be',
        ):
            analysis_base.extra_parameters = invalid_extra_parameters

    #############
    # Other methods
    #############

    def test_normalize_resolution_calls_instrument_model(self, analysis_base):
        # WHEN THEN EXPECT
        with patch.object(
            analysis_base.instrument_model, 'normalize_resolution'
        ) as mock_normalize_resolution:
            analysis_base.normalize_resolution()
            mock_normalize_resolution.assert_called_once()

    def test_get_parameters_near_bounds_no_bounds(self, analysis_base_with_components):
        # WHEN THEN
        near_bounds = analysis_base_with_components.get_parameters_near_bounds()

        # EXPECT
        assert isinstance(near_bounds, list)
        assert len(near_bounds) == 0

    def test_get_parameters_near_bounds_at_bounds(self, analysis_base_with_components):
        # WHEN
        components = analysis_base_with_components.sample_model.get_component_collection(Q_index=0)
        components[0].area.min = 1.0
        components[1].center.max = 6.0

        # THEN
        near_bounds = analysis_base_with_components.get_parameters_near_bounds()

        # EXPECT
        assert isinstance(near_bounds, list)
        assert len(near_bounds) == 2
        assert components[0].area in near_bounds
        assert components[1].center in near_bounds

    def test_get_parameters_near_bounds_with_tolerances(self, analysis_base_with_components):
        # WHEN
        components = analysis_base_with_components.sample_model.get_component_collection(Q_index=0)
        components[0].area.min = 0.99999
        components[1].center.max = 6.00001

        # THEN
        near_bounds = analysis_base_with_components.get_parameters_near_bounds()

        # EXPECT
        assert isinstance(near_bounds, list)
        assert len(near_bounds) == 2
        assert components[0].area in near_bounds
        assert components[1].center in near_bounds

    @pytest.mark.parametrize(
        'rtol, atol, expected_param, expected_error',
        [
            ('1e-5', 1e-8, 'rtol', TypeError),  # str
            (None, 1e-8, 'rtol', TypeError),  # None
            (1e-5, '1e-8', 'atol', TypeError),  # str
            (1e-5, None, 'atol', TypeError),  # None
            (-1e-5, 1e-8, 'rtol', ValueError),  # negative rtol
            (1e-5, -1e-8, 'atol', ValueError),  # negative atol
        ],
        ids=[
            'rtol as string',
            'rtol as None',
            'atol as string',
            'atol as None',
            'rtol negative',
            'atol negative',
        ],
    )
    def test_get_parameters_near_bounds_errors(
        self, analysis_base_with_components, rtol, atol, expected_param, expected_error
    ):
        # WHEN THEN
        with pytest.raises(expected_error) as exc:
            analysis_base_with_components.get_parameters_near_bounds(
                rtol=rtol,
                atol=atol,
            )
        # EXPECT
        assert expected_param in str(exc.value)

    def test_not_finite_parameters(self, analysis_base_with_components):
        # WHEN
        components = analysis_base_with_components.sample_model.get_component_collection(Q_index=0)
        components[0].area.value = np.inf
        components[1].center.value = np.nan

        # THEN
        near_bounds = analysis_base_with_components.get_parameters_near_bounds()

        # EXPECT
        assert isinstance(near_bounds, list)
        assert len(near_bounds) == 2
        assert components[0].area in near_bounds
        assert components[1].center in near_bounds

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
        # WHEN
        fake_Q = [1, 2, 3]
        with patch.object(
            type(analysis_base.experiment), 'Q', new_callable=PropertyMock
        ) as mock_Q:
            mock_Q.return_value = fake_Q
            # THEN
            analysis_base._on_instrument_model_changed()
            # EXPECT
            np.testing.assert_array_equal(analysis_base.instrument_model.Q, fake_Q)

    def test_verify_Q_index_valid(self, analysis_base_with_components):
        # WHEN
        valid_Q_index = 0

        # THEN
        result = analysis_base_with_components._verify_Q_index(valid_Q_index)

        # EXPECT
        assert result == valid_Q_index

    def test_verify_Q_index_invalid(self, analysis_base):
        # WHEN
        invalid_Q_index = -1

        # THEN / EXPECT
        with pytest.raises(IndexError, match='Q_index must be an integer'):
            analysis_base._verify_Q_index(invalid_Q_index)

    def test_verify_Q_index_invalid_when_Q_is_none(self, analysis_base):
        # WHEN
        positive_Q_index = 0

        # THEN / EXPECT
        with pytest.raises(IndexError, match='Q_index must be an integer'):
            analysis_base._verify_Q_index(positive_Q_index)

    def test_repr(self, analysis_base):
        # WHEN
        repr_str = repr(analysis_base)

        # THEN EXPECT
        assert 'AnalysisBase' in repr_str
        assert "display_name='TestAnalysis'" in repr_str
        assert 'unique_name=' in repr_str
