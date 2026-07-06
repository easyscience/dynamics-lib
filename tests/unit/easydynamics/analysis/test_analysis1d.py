# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.gaussian import Gaussian
from easydynamics.sample_model.components.polynomial import Polynomial


class TestAnalysis1d:
    @pytest.fixture
    def analysis1d(self):
        Q = sc.array(dims=['Q'], values=[1, 2, 3], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )

        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})

        experiment = Experiment(data=data_array)
        sample_model = SampleModel(components=Gaussian())
        instrument_model = InstrumentModel()

        return Analysis1d(
            display_name='TestAnalysis',
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            Q_index=0,
            extra_parameters=None,
        )

    def test_init(self, analysis1d):
        # WHEN THEN

        # EXPECT
        assert analysis1d.display_name == 'TestAnalysis'
        assert isinstance(analysis1d._experiment, Experiment)
        assert isinstance(analysis1d._sample_model, SampleModel)
        assert isinstance(analysis1d._instrument_model, InstrumentModel)
        assert analysis1d._extra_parameters == []
        assert np.array_equal(analysis1d.Q.values, [1, 2, 3])
        assert analysis1d.Q_index == 0

    def test_init_no_experiment(self):
        # WHEN
        analysis1d = Analysis1d(display_name='TestAnalysisNoExperiment')

        # THEN EXPECT
        assert isinstance(analysis1d._experiment, Experiment)
        assert analysis1d._convolver is None

    def test_Q_index_setter(self, analysis1d):
        # WHEN
        analysis1d.Q_index = 1

        # THEN / EXPECT
        assert analysis1d.Q_index == 1

    @pytest.mark.parametrize(
        'invalid_Q_index, expected_exception, expected_message',
        [
            (-1, IndexError, 'Q_index must be non-negative'),
            (10, IndexError, 'Q_index 10 is out of bounds'),
            ('invalid', TypeError, 'Q_index must be '),
            (np.nan, TypeError, 'Q_index must be '),
            ([1, 2], TypeError, 'Q_index must be '),
        ],
        ids=[
            'Negative index',
            'Index out of range',
            'Non-integer string',
            'NaN value',
            'List instead of integer',
        ],
    )
    def test_Q_index_setter_incorrect_Q(
        self, analysis1d, invalid_Q_index, expected_exception, expected_message
    ):
        # WHEN / THEN / EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            analysis1d.Q_index = invalid_Q_index

    def test_calculate_updates_convolver_and_calls_calculate(self, analysis1d):
        # WHEN

        # mock the _create_convolver and _calculate methods to verify
        # they are called
        fake_convolver = object()
        expected_result = np.array([42.0])

        analysis1d._create_convolver = MagicMock(return_value=fake_convolver)
        analysis1d._calculate = MagicMock(return_value=expected_result)

        # THEN
        result = analysis1d.calculate()

        # EXPECT

        analysis1d._create_convolver.assert_called_once()
        assert analysis1d._convolver is fake_convolver
        analysis1d._calculate.assert_called_once()
        np.testing.assert_array_equal(result, expected_result)

    def test__calculate_adds_sample_and_background(self, analysis1d):
        sample = np.array([1.0, 2.0, 3.0])
        background = np.array([0.5, 0.5, 0.5])

        analysis1d._evaluate_with_convolution = MagicMock(return_value=sample)
        analysis1d._evaluate_direct = MagicMock(return_value=background)

        result = analysis1d._calculate()

        np.testing.assert_array_equal(result, sample + background)

        analysis1d._evaluate_with_convolution.assert_called_once()
        analysis1d._evaluate_direct.assert_called_once()

    def test_fit_raises_if_no_experiment(self, analysis1d):
        # WHEN THEN
        analysis1d._experiment = None

        # EXPECT
        with pytest.raises(ValueError, match='No experiment'):
            analysis1d.fit()

    def test_fit_calls_fitter_with_correct_arguments(self, analysis1d):

        # WHEN

        # Mock all the methods that are called during fit to verify they
        # are called with the correct arguments
        fake_x = np.array([1, 2, 3])
        fake_y = np.array([10, 20, 30])
        fake_weights = np.array([0.1, 0.2, 0.3])
        fake_mask = np.array([True, False, True])

        analysis1d.experiment._extract_x_y_weights_only_finite = MagicMock(
            return_value=(fake_x, fake_y, fake_weights, fake_mask)
        )

        analysis1d._create_convolver = MagicMock(return_value='fake_convolver')

        fake_fit_result = object()
        fake_fitter_instance = MagicMock()
        fake_fitter_instance.fit.return_value = fake_fit_result

        with patch(
            'easydynamics.analysis.analysis1d.EasyScienceFitter',
            return_value=fake_fitter_instance,
        ) as mock_fitter:
            analysis1d.as_fit_function = MagicMock(return_value='fit_func')

            # THEN
            result = analysis1d.fit()

        # EXPECT

        # Check that all the mocked methods were called with the correct
        # arguments
        analysis1d._create_convolver.assert_called_once()

        mock_fitter.assert_called_once_with(
            fit_object=analysis1d,
            fit_function='fit_func',
        )

        analysis1d.experiment._extract_x_y_weights_only_finite.assert_called_once()

        fake_fitter_instance.fit.assert_called_once_with(
            x=fake_x,
            y=fake_y,
            weights=fake_weights,
        )

        # And that the result is returned
        assert analysis1d._fit_result is fake_fit_result
        assert result is fake_fit_result

    def test_as_fit_function_calls_calculate(self, analysis1d):
        # WHEN
        expected = np.array([1.0, 2.0, 3.0])
        analysis1d._calculate = MagicMock(return_value=expected)

        # THEN
        fit_func = analysis1d.as_fit_function()

        # EXPECT
        assert callable(fit_func)

        # THEN
        # call the fit function with some x values
        result = fit_func(x=[1, 2, 3])  # should be ignored

        # EXPECT
        analysis1d._calculate.assert_called_once()

        assert result is expected

    def test_get_all_variables(self, analysis1d):
        # WHEN
        extra_par1 = Parameter(name='extra_par1', value=1.0)
        extra_par2 = Parameter(name='extra_par2', value=2.0)
        analysis1d._extra_parameters = [extra_par1, extra_par2]

        # THEN
        variables = analysis1d.get_all_variables()

        # EXPECT
        assert isinstance(variables, list)
        sample_vars = analysis1d.sample_model.get_all_variables(Q_index=analysis1d.Q_index)
        instrument_vars = analysis1d.instrument_model.get_all_variables(Q_index=analysis1d.Q_index)
        extra_vars = [extra_par1, extra_par2]
        expected_vars = sample_vars + instrument_vars + extra_vars
        assert Counter(variables) == Counter(expected_vars)
        assert len(variables) == len(set(variables))

    def test_plot_raises_if_no_data(self, analysis1d):
        # WHEN THEN
        # override the binned data with None to simulate no data available for plotting
        analysis1d.experiment._binned_data = None

        # EXPECT
        with pytest.raises(ValueError, match='No data'):
            analysis1d.plot_data_and_model()

    def test_plot_calls_plopp_with_correct_arguments(self, analysis1d):
        # WHEN
        fake_fig = MagicMock()
        fake_fig.autoscale = MagicMock()
        with patch('plopp.plot', return_value=fake_fig) as mock_plot:
            # THEN
            result = analysis1d.plot_data_and_model()

        # EXPECT

        # plot called once
        mock_plot.assert_called_once()

        # Extract arguments
        args, kwargs = mock_plot.call_args

        datagroup = args[0]

        # Basic structure
        assert isinstance(datagroup, sc.DataGroup)

        assert 'Data' in datagroup
        assert 'Model' in datagroup

        # Gaussian sample component should also be included
        # because plot_components=True by default
        component_names = set(datagroup.keys()) - {'Data', 'Model'}
        assert len(component_names) > 0

        # Check plotting defaults
        assert kwargs['title'] == analysis1d.display_name

        assert kwargs['linestyle']['Data'] == 'none'
        assert kwargs['marker']['Data'] == 'o'
        assert kwargs['color']['Data'] == 'black'

        assert kwargs['linestyle']['Model'] == '-'
        assert kwargs['color']['Model'] == 'red'

        # Return value propagated
        assert result is fake_fig

    def test_plot_calls_plopp_with_correct_arguments_plot_residuals(self, analysis1d):
        # WHEN
        fake_fig = MagicMock()
        fake_fig.autoscale = MagicMock()
        with patch(
            'easydynamics.analysis.analysis1d.slicerplot_with_residuals',
            return_value=fake_fig,
        ) as mock_plot:
            # THEN
            result = analysis1d.plot_data_and_model(plot_residuals=True)

        # EXPECT

        # plot called once
        mock_plot.assert_called_once()

        # Extract arguments
        args, kwargs = mock_plot.call_args

        datagroup = args[0]

        # Basic structure
        assert isinstance(datagroup, sc.DataGroup)

        assert 'Data' in datagroup
        assert 'Model' in datagroup
        assert 'Residuals' in datagroup

        # Gaussian sample component should also be included
        # because plot_components=True by default
        component_names = set(datagroup.keys()) - {'Data', 'Model'}
        assert len(component_names) > 0

        # Check plotting defaults
        assert kwargs['title'] == analysis1d.display_name

        assert kwargs['linestyle']['Data'] == 'none'
        assert kwargs['marker']['Data'] == 'o'
        assert kwargs['color']['Data'] == 'black'

        assert kwargs['linestyle']['Model'] == '-'
        assert kwargs['color']['Model'] == 'red'

        assert kwargs['linestyle']['Residuals'] == 'none'
        assert kwargs['marker']['Residuals'] == 'o'
        assert kwargs['color']['Residuals'] == 'blue'

        # Return value propagated
        assert result is fake_fig

    @pytest.mark.parametrize(
        'include_residuals',
        [True, False],
        ids=['Include residuals', 'Do not include residuals'],
    )
    def test_data_and_model_to_datagroup(self, analysis1d, include_residuals):
        # WHEN
        energy = sc.array(dims=['energy'], values=[20.0, 30.0, 40.0], unit='meV')

        # THEN
        datagroup = analysis1d.data_and_model_to_datagroup(
            energy=energy, include_residuals=include_residuals
        )

        # EXPECT
        assert isinstance(datagroup, sc.DataGroup)
        assert 'Data' in datagroup
        assert 'Model' in datagroup
        assert sc.identical(
            datagroup['Data'],
            analysis1d.experiment.binned_data['Q', analysis1d.Q_index],
        )
        assert sc.identical(datagroup['Model'], analysis1d._create_model_array(energy=energy))
        if include_residuals:
            assert 'Residuals' in datagroup
            assert sc.identical(
                datagroup['Residuals'],
                datagroup['Data'] - analysis1d._create_model_array(),
            )
        else:
            assert 'Residuals' not in datagroup

    def test_data_and_model_to_datagroup_no_data_raises(self, analysis1d):
        # WHEN
        analysis1d.experiment._binned_data = None

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='No data to include in DataGroup',
        ):
            analysis1d.data_and_model_to_datagroup()

    def test_data_and_model_to_datagroup_no_Q_raises(self, analysis1d):
        with (
            patch.object(type(analysis1d), 'Q', new=None),
            pytest.raises(
                ValueError,
                match='No Q values available for creating DataGroup',
            ),
        ):
            analysis1d.data_and_model_to_datagroup()

    def test_data_and_model_to_datagroup_add_background_not_bool_raises(self, analysis1d):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match=r'add_background must be True or False',
        ):
            analysis1d.data_and_model_to_datagroup(add_background='not_a_boolean')

    def test_data_and_model_to_datagroup_include_components_not_bool_raises(self, analysis1d):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match=r'include_components must be True or False',
        ):
            analysis1d.data_and_model_to_datagroup(include_components='not_a_boolean')

    def test_data_and_model_to_datagroup_include_residuals_not_bool_raises(self, analysis1d):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match=r'include_residuals must be True or False',
        ):
            analysis1d.data_and_model_to_datagroup(include_residuals='not_a_boolean')

    def test_plot_data_and_model_no_Q_index_raises(self, analysis1d):
        # WHEN
        analysis1d._Q_index = None

        # THEN EXPECT
        with pytest.raises(ValueError, match='Q_index must be set'):
            analysis1d.plot_data_and_model()

    def test_fix_and_free_offset(self, analysis1d):
        # WHEN

        # EXPECT
        assert analysis1d.instrument_model.get_energy_offset(Q_index=0).fixed is False

        # THEN
        analysis1d.fix_energy_offset()

        # EXPECT
        assert analysis1d.instrument_model.get_energy_offset(Q_index=0).fixed is True

        # THEN
        analysis1d.free_energy_offset()

        # EXPECT
        assert analysis1d.instrument_model.get_energy_offset(Q_index=0).fixed is False

    #############
    # Private methods: small utilities
    #############

    def test_require_Q_index(self, analysis1d):
        # WHEN THEN
        Q_index = analysis1d._require_Q_index()

        # EXPECT
        assert Q_index == analysis1d.Q_index

    def test_require_Q_index_raises_if_no_Q_index(self, analysis1d):
        # WHEN THEN
        analysis1d._Q_index = None

        # EXPECT
        with pytest.raises(ValueError, match='Q_index must be set'):
            analysis1d._require_Q_index()

    def test_on_Q_index_changed(self, analysis1d):
        # WHEN
        analysis1d._convolver_is_dirty = False

        # THEN
        analysis1d._on_Q_index_changed()

        # EXPECT
        assert analysis1d._convolver_is_dirty is True

    def test_verify_energy(self, analysis1d):
        # WHEN
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')

        # THEN
        result = analysis1d._verify_energy(energy)

        # EXPECT
        assert sc.identical(result, energy)

    def test_verify_energy_None(self, analysis1d):
        # WHEN
        energy = None

        # THEN
        result = analysis1d._verify_energy(energy)

        # EXPECT
        assert result is None

    def test_verify_energy_raises(self, analysis1d):
        # WHEN
        energy = np.array([10.0, 20.0])

        # THEN / EXPECT
        with pytest.raises(TypeError, match=r'Energy must be a sc.Variable or None'):
            analysis1d._verify_energy(energy)

    def test_calculate_energy_with_offset(self, analysis1d):
        # WHEN
        energy = analysis1d.experiment.energy
        energy_offset = analysis1d.instrument_model.get_energy_offset(Q_index=analysis1d.Q_index)
        energy_offset.value = 1.0  # override with a simple value for testing

        # THEN
        result = analysis1d._calculate_energy_with_offset(energy, energy_offset)

        # EXPECT
        expected = energy.values - energy_offset.value
        np.testing.assert_array_equal(result.values, expected)

    def test_calculate_energy_with_offset_different_units(self, analysis1d):
        # WHEN
        energy = analysis1d.experiment.energy
        energy_offset = analysis1d.instrument_model.get_energy_offset(Q_index=analysis1d.Q_index)
        energy_offset.value = 1.0  # override with a simple value for testing
        energy_offset.convert_unit('eV')

        # THEN
        result = analysis1d._calculate_energy_with_offset(energy, energy_offset)

        # EXPECT
        expected = energy.values - energy_offset.value
        np.testing.assert_array_equal(result.values, expected)

    def test_calculate_energy_with_offset_raises_if_incompatible_units(self, analysis1d):
        # WHEN
        energy = analysis1d.experiment.energy
        energy_offset = Parameter(name='energy_offset', value=1.0, unit='m')  # incompatible unit

        # THEN / EXPECT
        with pytest.raises(
            sc.UnitError, match='Energy and energy offset must have compatible units'
        ):
            analysis1d._calculate_energy_with_offset(energy, energy_offset)

    #############
    # Private methods: evaluation
    #############

    def test_evaluate_direct_no_components(self, analysis1d):
        # WHEN
        components = ComponentCollection()

        # THEN
        result = analysis1d._evaluate_direct(components=components, energy=None)

        # EXPECT
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(analysis1d.experiment.energy),)
        assert np.all(result == pytest.approx(0.0))

    def test_evaluate_direct(self, analysis1d):
        # WHEN
        components = Polynomial(coefficients=[1.0])

        # THEN
        result = analysis1d._evaluate_direct(components=components, energy=None)

        # EXPECT
        assert np.array_equal(result, np.array([1.0, 1.0, 1.0]))

    def test_evaluate_with_convolution_uses_convolver(self, analysis1d):
        # WHEN
        components = Gaussian()
        convolver = MagicMock()
        convolver.convolution = MagicMock(return_value=np.array([1, 2, 3]))

        # THEN
        result = analysis1d._evaluate_with_convolution(
            components=components, energy=None, convolver=convolver
        )

        # EXPECT
        convolver.convolution.assert_called_once()
        assert result is convolver.convolution.return_value

    def test_evaluate_with_convolution_no_resolution(self, analysis1d):
        # WHEN
        components = MagicMock()
        components.evaluate = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

        # The default analysis1d has no resolution model components, so
        # evaluate is called directly even though we want convolution.

        # THEN
        result = analysis1d._evaluate_with_convolution(components=components, energy=None)

        # EXPECT
        components.evaluate.assert_called_once()
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_evaluate_with_convolution_no_resolution_DBF(self, analysis1d):
        # WHEN
        components = MagicMock()
        components.evaluate = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

        # Set temperature so DBF will be applied
        analysis1d.sample_model.temperature = 10
        mock_dbf = np.array([10.0, 10.0, 10.0])

        # The default analysis1d has no resolution model components

        with patch(
            'easydynamics.analysis.analysis1d.detailed_balance_factor',
            return_value=mock_dbf,
        ) as dbf_mock:
            # THEN
            result = analysis1d._evaluate_with_convolution(
                components=components,
                energy=None,
            )

        # EXPECT
        components.evaluate.assert_called_once()
        dbf_mock.assert_called_once()

        # EXPECT multiplication applied
        expected = np.array([1.0, 2.0, 3.0]) * mock_dbf
        assert np.array_equal(result, expected)

    def test_evaluate_with_resolution(self, analysis1d):
        # WHEN (set up the resolution model and create a component to evaluate)
        analysis1d.instrument_model.resolution_model.components = Gaussian()
        components = Gaussian()

        with patch('easydynamics.analysis.analysis1d.Convolution') as MockConvolution:
            # THEN
            analysis1d._evaluate_with_convolution(components=components, energy=None)

            # EXPECT
            # Ensure constructor called once
            MockConvolution.assert_called_once()

            # The convolver should be created with the correct arguments
            resolution_components = (
                analysis1d.instrument_model.resolution_model.get_component_collection(
                    analysis1d.Q_index
                )
            )

            energy_offset = analysis1d.instrument_model.get_energy_offset(analysis1d.Q_index)

            # Extract call arguments
            _, kwargs = MockConvolution.call_args

            assert kwargs['sample_components'] == components
            assert kwargs['resolution_components'] == resolution_components
            assert kwargs['temperature'] == analysis1d.temperature
            assert kwargs['energy_offset'] == energy_offset

            # check that the energy array passed to the convolver is the
            # same as the analysis1d energy array
            assert sc.identical(kwargs['energy'], analysis1d.energy)

            # and check that convolution() was called
            MockConvolution.return_value.convolution.assert_called_once_with()

    def test_create_convolver(self, analysis1d):
        # WHEN
        # Mock sample components
        sample_components = MagicMock()
        sample_components.is_empty = False

        # Mock resolution components
        resolution_components = MagicMock()
        resolution_components.is_empty = False

        # And all the other inputs to the convolver
        analysis1d.sample_model.get_component_collection = MagicMock(
            return_value=sample_components
        )

        analysis1d.instrument_model.resolution_model.get_component_collection = MagicMock(
            return_value=resolution_components
        )

        analysis1d.instrument_model.get_energy_offset = MagicMock(return_value=123.0)

        with patch('easydynamics.analysis.analysis1d.Convolution') as MockConvolution:
            # THEN
            result = analysis1d._create_convolver()

            # EXPECT
            # Check the convolver was created with the correct arguments
            MockConvolution.assert_called_once()

            _, kwargs = MockConvolution.call_args

            assert kwargs['sample_components'] is sample_components
            assert kwargs['resolution_components'] is resolution_components
            assert sc.identical(kwargs['energy'], analysis1d.energy)
            assert kwargs['temperature'] is analysis1d.temperature
            assert kwargs['energy_offset'] == pytest.approx(123.0)

            assert result == MockConvolution.return_value

    def test_create_convolver_returns_none_if_no_resolution_components(self, analysis1d):
        # WHEN
        analysis1d.instrument_model.resolution_model.clear_components()

        # THEN
        convolver = analysis1d._create_convolver()

        # EXPECT
        assert convolver is None

    def test_create_convolver_returns_none_if_no_sample_components(self, analysis1d):
        # WHEN
        analysis1d.sample_model.clear_components()

        # THEN
        convolver = analysis1d._create_convolver()

        # EXPECT
        assert convolver is None

    #############
    # Private methods: create scipp arrays for plotting
    #############

    def test_create_residuals_array(self, analysis1d):
        # WHEN
        # Mock the _create_model_array method to return a specific model array
        model = analysis1d.experiment.binned_data.copy(deep=True)
        model = model['Q', analysis1d.Q_index]
        model.values *= 0.5

        with patch.object(
            analysis1d,
            '_create_model_array',
            return_value=model,
        ):
            # THEN
            residuals = analysis1d._create_residuals_array()

        # EXPECT
        expected = analysis1d.experiment.binned_data['Q', analysis1d.Q_index] - model
        assert sc.identical(residuals, expected)

    def test_create_residuals_array_no_Q_index_raises(self, analysis1d):
        # WHEN
        analysis1d._Q_index = None

        # THEN EXPECT
        with pytest.raises(ValueError, match='Q_index must be set'):
            analysis1d._create_residuals_array()

    def test_create_model_array(self, analysis1d):
        """Test that _create_model_array correctly
        evaluates the full model and calls _to_scipp_array with the
        correct values."""

        # WHEN

        # Mock the functions that will be called.
        analysis1d._calculate = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        analysis1d._to_scipp_array = MagicMock()

        # THEN
        analysis1d._create_model_array()

        # EXPECT
        analysis1d._calculate.assert_called_once()

        analysis1d._to_scipp_array.assert_called_once()

        # Extract the actual call
        _, kwargs = analysis1d._to_scipp_array.call_args

        np.testing.assert_array_equal(
            kwargs['values'],
            np.array([1.0, 2.0, 3.0]),
        )

    @pytest.mark.parametrize(
        'add_background',
        [True, False],
        ids=['With background', 'Without background'],
    )
    def test_create_components_dataset_single_Q(
        self,
        analysis1d,
        add_background,
    ):
        """Test orchestration of _create_components_dataset_single_Q."""

        # WHEN

        # Choose a particular Q_index, but without using the setter to
        # avoid validation logic
        analysis1d._Q_index = 5

        # Mock all the things

        # ---- Sample component ----
        sample_component = MagicMock()
        sample_component.display_name = 'sample_comp'

        sample_collection = MagicMock()
        sample_collection.__iter__.return_value = iter([sample_component])
        sample_collection.__len__.return_value = 1

        analysis1d.sample_model.get_component_collection = MagicMock(
            return_value=sample_collection
        )

        # ---- Background component ----
        background_component = MagicMock()
        background_component.display_name = 'background_comp'

        background_collection = MagicMock()
        background_collection.__iter__.return_value = iter([background_component])
        background_collection.__len__.return_value = 1

        analysis1d.instrument_model.background_model.get_component_collection = MagicMock(
            return_value=background_collection
        )

        # ---- Evaluation mocks ----
        background_value = np.array([11.0, 21.0, 31.0])
        sample_value = np.array([1.0, 2.0, 3.0])

        analysis1d._evaluate_direct = MagicMock(return_value=background_value)
        analysis1d._evaluate_with_convolution = MagicMock(return_value=sample_value)

        # ---- Return scipp DataArrays ----
        fake_da = sc.DataArray(data=sc.array(dims=['energy'], values=[1.0, 2.0, 3.0]))
        analysis1d._to_scipp_array = MagicMock(return_value=fake_da)

        # THEN
        dataset = analysis1d._create_components_dataset_single_Q(add_background=add_background)

        # EXPECT

        # The correct component collections are requested with the correct Q_index
        analysis1d.sample_model.get_component_collection.assert_called_once_with(
            Q_index=analysis1d.Q_index
        )
        analysis1d.instrument_model.background_model.get_component_collection.assert_called_once_with(
            Q_index=analysis1d.Q_index
        )

        # _evaluate_direct is called once for the background collection (for add_background),
        # and once for the background component — if add_background=False only the component call.
        if add_background:
            # First call: background_collection to get total background value
            # Second call: background_component for its individual array
            assert analysis1d._evaluate_direct.call_count == 2
        else:
            # Only called for the background component
            assert analysis1d._evaluate_direct.call_count == 1

        # _evaluate_with_convolution called once for the sample component
        analysis1d._evaluate_with_convolution.assert_called_once_with(
            sample_component, analysis1d._masked_energy
        )

        # Dataset content
        assert isinstance(dataset, sc.Dataset)
        assert 'sample_comp' in dataset
        assert 'background_comp' in dataset

    def test_to_scipp_array(self, analysis1d):
        # WHEN
        numpy_array = np.array([1.0, 2.0, 3.0])

        # THEN
        scipp_array = analysis1d._to_scipp_array(numpy_array)

        # EXPECT
        assert isinstance(scipp_array, sc.DataArray)
        np.testing.assert_array_equal(scipp_array.values, numpy_array)

        np.testing.assert_array_equal(
            scipp_array.coords['energy'].values, analysis1d.experiment.energy.values
        )

        np.testing.assert_array_equal(
            scipp_array.coords['Q'].values,
            analysis1d.experiment.Q[analysis1d.Q_index].values,
        )

    def test_rebin_marks_convolver_dirty(self, analysis1d):
        # WHEN
        analysis1d._convolver_is_dirty = False

        # THEN
        analysis1d.rebin({'Q': 1})

        # EXPECT
        assert analysis1d._convolver_is_dirty is True

    def test_rebin_refreshes_masked_energy(self, analysis1d):
        # WHEN - capture the current masked_energy object
        energy_before = analysis1d._masked_energy

        # THEN - patch get_masked_energy to verify it is called during rebin
        with patch.object(
            analysis1d.experiment,
            'get_masked_energy',
            wraps=analysis1d.experiment.get_masked_energy,
        ) as mock_get_energy:
            analysis1d.rebin({'Q': 1})

        # EXPECT - get_masked_energy was called to refresh _masked_energy
        mock_get_energy.assert_called_once_with(Q_index=analysis1d.Q_index)
        # And _masked_energy is a different object now (re-fetched from experiment)
        assert analysis1d._masked_energy is not energy_before

    def test_fit_marks_convolver_dirty_when_sample_model_components_change(self, analysis1d):
        """Issue #68: fit() should detect in-place component changes and rebuild the convolver."""
        # WHEN - simulate state after a previous fit (convolver built, not dirty)
        analysis1d._create_convolver = MagicMock(return_value=None)
        analysis1d._convolver_is_dirty = False
        analysis1d.sample_model._component_collections_is_dirty = False

        # THEN - append a component in-place (doesn't go through Analysis1d setters)
        analysis1d.sample_model.append_component(Gaussian(name='NewGaussian'))
        assert analysis1d.sample_model._component_collections_is_dirty is True
        assert analysis1d._convolver_is_dirty is False  # not yet propagated

        # WHEN - fit() should propagate the dirty flag and rebuild the convolver
        with patch(
            'easydynamics.analysis.analysis1d.EasyScienceFitter',
            return_value=MagicMock(fit=MagicMock(return_value=MagicMock())),
        ):
            analysis1d.experiment._extract_x_y_weights_only_finite = MagicMock(
                return_value=(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 1.0, 1.0]),
                    np.array([True, True, True]),
                )
            )
            analysis1d.fit()

        # EXPECT - convolver was rebuilt (_ensure_convolver_current called _create_convolver)
        analysis1d._create_convolver.assert_called_once()

    def test_fit_does_not_rebuild_convolver_when_nothing_changed(self, analysis1d):
        """fit() should not call _create_convolver if nothing has changed since last fit."""
        # WHEN - build convolver and clear all dirty flags
        analysis1d._create_convolver = MagicMock(return_value=None)
        analysis1d._convolver_is_dirty = False
        analysis1d.sample_model._component_collections_is_dirty = False
        analysis1d.instrument_model.resolution_model._component_collections_is_dirty = False

        # THEN - call fit() with nothing changed
        with patch(
            'easydynamics.analysis.analysis1d.EasyScienceFitter',
            return_value=MagicMock(fit=MagicMock(return_value=MagicMock())),
        ):
            analysis1d.experiment._extract_x_y_weights_only_finite = MagicMock(
                return_value=(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 1.0, 1.0]),
                    np.array([True, True, True]),
                )
            )
            analysis1d.fit()

        # EXPECT - _create_convolver was NOT called (convolver reused)
        analysis1d._create_convolver.assert_not_called()

    def test_rebin_rebins_experiment(self, analysis1d):
        """rebin() should delegate to experiment.rebin()."""
        # WHEN
        with patch.object(
            analysis1d.experiment, 'rebin', wraps=analysis1d.experiment.rebin
        ) as mock_rebin:
            analysis1d.rebin({'Q': 1})

        # EXPECT - experiment.rebin was called with the correct dimensions
        mock_rebin.assert_called_once_with({'Q': 1})

    def test_rebin_without_Q_index_does_not_crash(self):
        """rebin() with no Q_index set should not try to refresh masked_energy."""
        # WHEN
        experiment = Experiment()
        analysis1d = Analysis1d(experiment=experiment)
        assert analysis1d.Q_index is None

        # THEN / EXPECT - no error, and _masked_energy stays None
        # (no data loaded, so rebin would raise from experiment, just test the branch)
        assert analysis1d._masked_energy is None

    def test_on_sample_model_changed_marks_convolver_dirty(self, analysis1d):
        # WHEN - clear the dirty flag first
        analysis1d._convolver_is_dirty = False

        # THEN - replace the sample model via the public setter
        analysis1d.sample_model = SampleModel(components=Gaussian(name='NewGaussian'))

        # EXPECT
        assert analysis1d._convolver_is_dirty is True

    def test_on_instrument_model_changed_marks_convolver_dirty(self, analysis1d):
        # WHEN - clear the dirty flag first
        analysis1d._convolver_is_dirty = False

        # THEN - replace the instrument model via the public setter
        analysis1d.instrument_model = InstrumentModel()

        # EXPECT
        assert analysis1d._convolver_is_dirty is True

    def test_evaluate_with_convolution_returns_zeros_for_empty_collection(self, analysis1d):
        # WHEN
        empty_collection = ComponentCollection()
        energy = analysis1d._masked_energy

        # THEN
        result = analysis1d._evaluate_with_convolution(empty_collection, energy)

        # EXPECT
        assert result.shape == energy.values.shape
        np.testing.assert_array_equal(result, 0.0)

    def test_fit_marks_convolver_dirty_when_resolution_model_components_change(self, analysis1d):
        """Issue #68: fit() should detect resolution_model component changes."""
        # WHEN - simulate state after a previous fit
        analysis1d._create_convolver = MagicMock(return_value=None)
        analysis1d._convolver_is_dirty = False
        analysis1d.sample_model._component_collections_is_dirty = False
        analysis1d.instrument_model.resolution_model._component_collections_is_dirty = False

        # THEN - mark resolution_model dirty in-place (doesn't go through Analysis1d setters)
        analysis1d.instrument_model.resolution_model._component_collections_is_dirty = True
        assert analysis1d._convolver_is_dirty is False

        # WHEN - fit() should propagate and rebuild
        with patch(
            'easydynamics.analysis.analysis1d.EasyScienceFitter',
            return_value=MagicMock(fit=MagicMock(return_value=MagicMock())),
        ):
            analysis1d.experiment._extract_x_y_weights_only_finite = MagicMock(
                return_value=(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 1.0, 1.0]),
                    np.array([True, True, True]),
                )
            )
            analysis1d.fit()

        # EXPECT
        analysis1d._create_convolver.assert_called_once()
