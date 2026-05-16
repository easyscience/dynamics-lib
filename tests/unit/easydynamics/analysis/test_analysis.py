# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc

from easydynamics.analysis.analysis import Analysis
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.gaussian import Gaussian
from easydynamics.settings.convolution_settings import ConvolutionSettings


class TestAnalysis:
    @pytest.fixture
    def analysis(self):
        Q = sc.array(dims=['Q'], values=[1, 2, 3], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )

        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})

        experiment = Experiment(data=data_array)
        sample_model = SampleModel(
            components=Gaussian(name='GaussianName'), display_name='Gaussian'
        )
        instrument_model = InstrumentModel()

        return Analysis(
            display_name='TestAnalysis',
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            extra_parameters=None,
        )

    @pytest.fixture
    def analysis_single_Q(self):
        Q = sc.array(dims=['Q'], values=[1, 2, 3], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )

        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})

        experiment = Experiment(data=data_array)
        experiment.rebin({'Q': 1})
        sample_model = SampleModel(
            components=Gaussian(name='GaussianName'), display_name='Gaussian'
        )
        instrument_model = InstrumentModel()

        return Analysis(
            display_name='TestAnalysis',
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            extra_parameters=None,
        )

    def test_init(self, analysis):
        # WHEN THEN

        # EXPECT
        assert analysis.display_name == 'TestAnalysis'
        assert isinstance(analysis._experiment, Experiment)
        assert isinstance(analysis._sample_model, SampleModel)
        assert isinstance(analysis._instrument_model, InstrumentModel)
        assert analysis._extra_parameters == []
        assert np.array_equal(analysis.Q.values, [1, 2, 3])
        assert len(analysis.analysis_list) == 3

    def test_init_raises_with_invalid_experiment(self):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='experiment must be an instance of Experiment',
        ):
            Analysis(experiment='invalid_experiment')

    def test_analysis_list_contains_all_Q_indices(self, analysis):
        # WHEN THEN

        # EXPECT
        assert len(analysis.analysis_list) == 3
        for i in range(3):
            assert analysis.analysis_list[i].Q_index == i

    def test_analysis_list_setter_raises(self, analysis):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            AttributeError,
            match='analysis_list is read-only',
        ):
            analysis.analysis_list = 'invalid_analysis_list'

    def test_calculate_with_Q_index(self, analysis):
        # WHEN
        analysis.analysis_list[1].calculate = MagicMock(return_value=np.array([4.0, 5.0, 6.0]))

        # THEN
        result = analysis.calculate(Q_index=1)

        # EXPECT
        analysis.analysis_list[1].calculate.assert_called_once()
        _, kwargs = analysis.analysis_list[1].calculate.call_args
        assert sc.identical(kwargs['energy'], analysis.energy)
        np.testing.assert_array_equal(result, np.array([4.0, 5.0, 6.0]))

    def test_calculate_with_Q_index_and_energy(self, analysis):
        # WHEN
        analysis.analysis_list[1].calculate = MagicMock(return_value=np.array([4.0, 5.0, 6.0]))
        energy = sc.array(dims=['energy'], values=[20.0, 30.0, 40.0], unit='meV')

        # THEN
        result = analysis.calculate(Q_index=1, energy=energy)

        # EXPECT
        analysis.analysis_list[1].calculate.assert_called_once()
        _, kwargs = analysis.analysis_list[1].calculate.call_args
        assert kwargs['energy'] is energy
        np.testing.assert_array_equal(result, np.array([4.0, 5.0, 6.0]))

    def test_calculate_without_Q_index(self, analysis):
        # WHEN
        for i in range(3):
            analysis.analysis_list[i].calculate = MagicMock(
                return_value=np.array([1.0, 2.0, 3.0]) + i
            )

        # THEN
        result = analysis.calculate()

        # EXPECT
        for i in range(3):
            analysis.analysis_list[i].calculate.assert_called_once()
            np.testing.assert_array_equal(result[i], np.array([1.0, 2.0, 3.0]) + i)

    def test_calculate_with_invalid_Q_index(self, analysis):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            IndexError,
            match='must be a valid index',
        ):
            analysis.calculate(Q_index=3)

    def test_fit_no_Q_values_raises(self, analysis):
        # WHEN
        analysis.experiment = Experiment()

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='No Q values available for fitting',
        ):
            analysis.fit()

    def test_fit_fit_method_independent_with_Q_index(self, analysis):
        # WHEN
        analysis.analysis_list[1].fit = MagicMock(return_value='fit_result_Q1')

        # THEN
        result = analysis.fit(fit_method='independent', Q_index=1)

        # EXPECT
        analysis.analysis_list[1].fit.assert_called_once()
        assert result == 'fit_result_Q1'

    def test_fit_fit_method_independent_without_Q_index(self, analysis):
        # WHEN
        for i in range(3):
            analysis.analysis_list[i].fit = MagicMock(return_value=f'fit_result_Q{i}')

        # THEN
        result = analysis.fit(fit_method='independent')

        # EXPECT
        for i in range(3):
            analysis.analysis_list[i].fit.assert_called_once()
            assert result[i] == f'fit_result_Q{i}'

    def test_fit_fit_method_simultaneous(self, analysis):
        # WHEN
        analysis._fit_all_Q_simultaneously = MagicMock(return_value='simultaneous_fit_result')

        # THEN
        result = analysis.fit(fit_method='simultaneous')

        # EXPECT
        analysis._fit_all_Q_simultaneously.assert_called_once()
        assert result == 'simultaneous_fit_result'

    def test_fit_with_invalid_fit_method(self, analysis):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            ValueError,
            match=r"Invalid fit method. Choose 'independent' or 'simultaneous'.",
        ):
            analysis.fit(fit_method='invalid_fit_method')

    def test_plot_data_and_model_not_in_notebook_raises(self, analysis):
        # WHEN / THEN / EXPECT
        with (
            patch('easydynamics.analysis.analysis._in_notebook', return_value=False),
            pytest.raises(
                RuntimeError,
                match=r'can only be used in a Jupyter notebook environment',
            ),
        ):
            analysis.plot_data_and_model()

    def test_plot_data_and_model_Q_index(self, analysis):

        # WHEN
        analysis.analysis_list[1].plot_data_and_model = MagicMock(return_value='plot_Q1')

        kwargs = {
            'marker': {'amplitude': 'x', 'width': 's'},
            'title': 'My Plot',
        }

        # THEN
        result = analysis.plot_data_and_model(
            Q_index=1, plot_components=True, add_background=True, **kwargs
        )

        # EXPECT
        analysis.analysis_list[1].plot_data_and_model.assert_called_once_with(
            plot_components=True, add_background=True, energy=None, **kwargs
        )
        assert result == 'plot_Q1'

    def test_plot_data_and_model_no_data_raises(self, analysis):
        # WHEN
        analysis.experiment = Experiment()

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='No data to plot',
        ):
            analysis.plot_data_and_model()

    def test_plot_data_and_model_invalid_plot_components_raises(self, analysis):
        # WHEN / THEN / EXPECT

        with (
            patch('easydynamics.analysis.analysis._in_notebook', return_value=True),
            pytest.raises(
                TypeError,
                match='plot_components must be True or False',
            ),
        ):
            analysis.plot_data_and_model(plot_components='not_a_boolean')

    def test_plot_data_and_model_invalid_add_background_raises(self, analysis):
        # WHEN / THEN / EXPECT
        with (
            patch('easydynamics.analysis.analysis._in_notebook', return_value=True),
            pytest.raises(
                TypeError,
                match='add_background must be True or False',
            ),
        ):
            analysis.plot_data_and_model(add_background='not_a_boolean')

    @pytest.mark.parametrize(
        'plot_components, expect_components',
        [
            (False, False),
            (True, True),
        ],
        ids=[
            'No components',
            'With components',
        ],
    )
    def test_plot_data_and_model(
        self,
        analysis,
        plot_components,
        expect_components,
    ):
        # WHEN
        fake_widget = MagicMock()
        fake_widget.slider_toggler = MagicMock()

        fake_fig = MagicMock()
        fake_fig.bottom_bar = [MagicMock()]
        fake_fig.bottom_bar[0].controls = {'test': fake_widget}

        with (
            patch('plopp.slicer', return_value=fake_fig) as mock_slicer,
            patch('easydynamics.analysis.analysis._in_notebook', return_value=True),
        ):
            # THEN
            fig = analysis.plot_data_and_model(plot_components=plot_components)

        # EXPECT
        assert fig is fake_fig

        mock_slicer.assert_called_once()

        args, kwargs = mock_slicer.call_args

        datagroup = args[0]

        assert isinstance(datagroup, sc.DataGroup)

        assert 'Data' in datagroup
        assert 'Model' in datagroup

        component_keys = set(datagroup.keys()) - {'Data', 'Model'}

        if expect_components:
            assert len(component_keys) > 0

            for key in component_keys:
                assert kwargs['linestyle'][key] == '--'
                assert kwargs['marker'][key] is None
        else:
            assert component_keys == set()

        # Default plotting kwargs
        assert kwargs['title'] == analysis.display_name

        assert kwargs['linestyle']['Data'] == 'none'
        assert kwargs['linestyle']['Model'] == '-'

        assert kwargs['marker']['Data'] == 'o'
        assert kwargs['marker']['Model'] is None

        assert kwargs['color']['Data'] == 'black'
        assert kwargs['color']['Model'] == 'red'

        assert kwargs['markerfacecolor']['Data'] == 'none'
        assert kwargs['markerfacecolor']['Model'] == 'none'

        assert kwargs['keep'] == 'energy'

        # Widget toggler updated
        assert fake_widget.slider_toggler.value == '-o-'

    def test_data_and_model_to_datagroup(self, analysis):
        # WHEN
        energy = sc.array(dims=['energy'], values=[20.0, 30.0, 40.0], unit='meV')
        datagroup = analysis.data_and_model_to_datagroup(energy=energy)

        # EXPECT
        assert isinstance(datagroup, sc.DataGroup)
        assert 'Data' in datagroup
        assert 'Model' in datagroup
        assert sc.identical(datagroup['Data'], analysis.experiment.binned_data)
        assert sc.identical(datagroup['Model'], analysis._create_model_array(energy=energy))

    def test_data_and_model_to_datagroup_no_data_raises(self, analysis):
        # WHEN
        analysis.experiment._binned_data = None

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='No data to include in DataGroup',
        ):
            analysis.data_and_model_to_datagroup()

    def test_data_and_model_to_datagroup_no_Q_raises(self, analysis):
        with (
            patch.object(type(analysis), 'Q', new=None),
            pytest.raises(
                ValueError,
                match='No Q values available for creating DataGroup',
            ),
        ):
            analysis.data_and_model_to_datagroup()

    def test_data_and_model_to_datagroup_add_background_not_bool_raises(self, analysis):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match=r'add_background must be True or False',
        ):
            analysis.data_and_model_to_datagroup(add_background='not_a_boolean')

    def test_data_and_model_to_datagroup_include_components_not_bool_raises(self, analysis):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match=r'include_components must be True or False',
        ):
            analysis.data_and_model_to_datagroup(include_components='not_a_boolean')

    def test_parameters_to_dataset(self, analysis):
        # WHEN
        analysis.sample_model.append_component(
            Gaussian(name='Gaussian2Name', display_name='Gaussian2', area=0.5)
        )
        # THEN
        parameters_dataset = analysis.parameters_to_dataset()

        # EXPECT
        assert isinstance(parameters_dataset, sc.Dataset)
        parameter_names = [
            'GaussianName area',
            'GaussianName center',
            'GaussianName width',
            'Gaussian2Name area',
            'Gaussian2Name center',
            'Gaussian2Name width',
            'energy_offset',
        ]
        for parameter_name in parameter_names:
            assert parameter_name in parameters_dataset
            assert 'Q' in parameters_dataset[parameter_name].dims

    def test_parameters_to_dataset_different_units(self, analysis):

        # WHEN
        analysis.sample_model.append_component(
            Gaussian(name='Gaussian2Name', display_name='Gaussian2', area=0.5)
        )

        # Convert the unit of a component to eV.
        analysis.sample_model.get_component_collection(Q_index=1)[0].convert_unit('eV')

        # THEN
        parameters_dataset = analysis.parameters_to_dataset()

        # EXPECT
        assert isinstance(parameters_dataset, sc.Dataset)
        parameter_names = [
            'GaussianName area',
            'GaussianName center',
            'GaussianName width',
            'Gaussian2Name area',
            'Gaussian2Name center',
            'Gaussian2Name width',
            'energy_offset',
        ]
        for parameter_name in parameter_names:
            assert parameter_name in parameters_dataset
            assert 'Q' in parameters_dataset[parameter_name].dims

    @pytest.mark.parametrize(
        'parameter_names',
        [
            123,  # not str or list
            ['parameter_name', 123],  # list contains non-string
            {'a': 1},  # completely wrong type
        ],
        ids=[
            'not_string_or_list',
            'list_contains_non_string',
            'wrong_container_type',
        ],
    )
    def test_plot_parameters_raises_with_invalid_parameter_names(self, analysis, parameter_names):

        with pytest.raises(
            TypeError,
            match='names must be a string or a list of strings',
        ):
            analysis.plot_parameters(names=parameter_names)

    def test_plot_parameters_raises_with_nonexistent_parameter_names(self, analysis):
        with pytest.raises(
            ValueError,
            match='not found in dataset',
        ):
            analysis.plot_parameters(names='nonexistent_parameter')

    def test_plot_parameters(self, analysis):

        # WHEN

        # Mock all the methods that are called.
        fake_fig = object()
        user_kwargs = {
            'title': 'My Plot',
            'marker': {'amplitude': 'x', 'width': 's'},
        }

        fake_dataset = {
            'amplitude': object(),
            'width': object(),
        }

        analysis.parameters_to_dataset = MagicMock(return_value=fake_dataset)

        with patch('plopp.plot', return_value=fake_fig) as mock_plot:
            # THEN
            result = analysis.plot_parameters(**user_kwargs)

        # EXPECT
        mock_plot.assert_called_once()

        # Inspect arguments
        args, kwargs = mock_plot.call_args

        dataset_passed = args[0]

        assert dataset_passed == fake_dataset

        # Check default kwargs
        assert 'linestyle' in kwargs
        assert kwargs['linestyle'] == {
            'amplitude': 'none',
            'width': 'none',
        }

        assert 'markerfacecolor' in kwargs
        assert kwargs['markerfacecolor'] == {
            'amplitude': 'none',
            'width': 'none',
        }

        # Check that user kwargs override defaults
        assert kwargs['marker'] == user_kwargs['marker']
        assert kwargs['title'] == 'My Plot'

        # and that we return the figure
        assert result is fake_fig

    def test_fix_and_free_energy_offset(self, analysis):
        # EXPECT
        offsets = analysis.instrument_model.get_energy_offset()
        for offset in offsets:
            assert offset.fixed is False

        # THEN
        analysis.fix_energy_offset()

        # EXPECT
        for offset in offsets:
            assert offset.fixed is True

        # THEN
        analysis.free_energy_offset()

        # EXPECT
        for offset in offsets:
            assert offset.fixed is False

        # THEN
        analysis.fix_energy_offset(Q_index=1)

        # EXPECT
        for i, offset in enumerate(offsets):
            if i == 1:
                assert offset.fixed is True
            else:
                assert offset.fixed is False

        # THEN
        analysis.free_energy_offset(Q_index=1)

        # EXPECT
        for offset in offsets:
            assert offset.fixed is False

    def test_on_experiment_changed_similar_Q(self, analysis):
        # WHEN
        # Create a new experiment.
        Q = sc.array(dims=['Q'], values=[1, 2, 3], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[20.0, 30.0, 40.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )

        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})

        new_experiment = Experiment(data=data_array)

        # THEN (this calls _on_experiment_changed internally)
        analysis.experiment = new_experiment

        # EXPECT
        assert np.array_equal(analysis.Q.values, [1, 2, 3])
        assert len(analysis.analysis_list) == 3
        for analysis1d in analysis.analysis_list:
            assert analysis1d.experiment is new_experiment

    def test_on_experiment_changed_different_Q_raises(self, analysis):
        # WHEN
        # Create a new experiment with different Q values.
        Q = sc.array(dims=['Q'], values=[4, 5, 6], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[20.0, 30.0, 40.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )

        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})

        new_experiment = Experiment(data=data_array)

        # THEN / EXPECT
        with pytest.raises(
            ValueError,
            match='New Q values are not similar to the old ones',
        ):
            analysis.experiment = new_experiment

    def test_on_sample_model_changed(self, analysis):
        # WHEN
        # Create a new sample model.
        new_sample_model = SampleModel()

        # THEN (this call _on_sample_model_changed internally)
        analysis.sample_model = new_sample_model

        # EXPECT
        assert analysis.sample_model is new_sample_model
        for analysis1d in analysis.analysis_list:
            assert analysis1d.sample_model is new_sample_model

    def test_on_instrument_model_changed(self, analysis):
        # WHEN
        # Create a new instrument model.
        new_instrument_model = InstrumentModel()

        # THEN (this call _on_instrument_model_changed internally)
        analysis.instrument_model = new_instrument_model

        # EXPECT
        assert analysis.instrument_model is new_instrument_model
        for analysis1d in analysis.analysis_list:
            assert analysis1d.instrument_model is new_instrument_model

    def test_on_convolution_settings_changed(self, analysis):
        # WHEN
        new_convolution_settings = ConvolutionSettings()

        # THEN (this calls _on_convolution_settings_changed internally)
        analysis.convolution_settings = new_convolution_settings

        # EXPECT
        assert analysis.convolution_settings is new_convolution_settings
        for analysis1d in analysis.analysis_list:
            assert analysis1d.convolution_settings is new_convolution_settings

    def test_fit_single_Q_valid(self, analysis):
        # WHEN
        analysis.analysis_list[1].fit = MagicMock(return_value='fit_result_Q1')

        # THEN
        result = analysis._fit_single_Q(Q_index=1)

        # EXPECT
        analysis.analysis_list[1].fit.assert_called_once()
        assert result == 'fit_result_Q1'

    def test_fit_single_Q_invalid_Q_index(self, analysis):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            IndexError,
            match='must be a valid index',
        ):
            analysis._fit_single_Q(Q_index=3)

    def test_fit_all_Q_independently(self, analysis):
        # WHEN
        for i in range(3):
            analysis.analysis_list[i].fit = MagicMock(return_value=f'fit_result_Q{i}')

        # THEN
        result = analysis._fit_all_Q_independently()

        # EXPECT
        for i in range(3):
            analysis.analysis_list[i].fit.assert_called_once()
            assert result[i] == f'fit_result_Q{i}'

    def test_fit_all_Q_simultaneously(self, analysis):
        # WHEN
        # Mock the MultiFitter and its fit method

        fake_fit_result = object()

        fake_fitter_instance = MagicMock()
        fake_fitter_instance.fit.return_value = fake_fit_result

        # Also mock the get_fit_functions method to return a list of fit
        # functions for each Q index
        analysis.get_fit_functions = MagicMock(
            return_value=['fit_function_Q0', 'fit_function_Q1', 'fit_function_Q2']
        )
        with patch(
            'easydynamics.analysis.analysis.MultiFitter',
            return_value=fake_fitter_instance,
        ) as mock_fitter:
            result = analysis._fit_all_Q_simultaneously()

        # EXPECT
        # Check that the correct objects are passed to the MultiFitter
        expected_fit_objects = analysis.analysis_list
        expected_fit_functions = analysis.get_fit_functions()
        mock_fitter.assert_called_once()
        _, kwargs = mock_fitter.call_args
        assert kwargs['fit_objects'] == expected_fit_objects
        assert kwargs['fit_functions'] == expected_fit_functions

        # And check that the correct x, y, and weights arrays are passed
        # to the fit method of the MultiFitter
        expected_xs = []
        expected_ys = []
        expected_ws = []
        for analysis1d in analysis.analysis_list:
            data = analysis1d.experiment.data['Q', analysis1d.Q_index]

            expected_xs.append(data.coords['energy'].values)
            expected_ys.append(data.values)
            expected_ws.append(1.0 / np.sqrt(data.variances))
        fake_fitter_instance.fit.assert_called_once()

        _, kwargs = fake_fitter_instance.fit.call_args
        np.testing.assert_array_equal(kwargs['x'], expected_xs)
        np.testing.assert_array_equal(kwargs['y'], expected_ys)
        np.testing.assert_array_equal(kwargs['weights'], expected_ws)

        # And that the result from the fit method is returned
        assert result == fake_fit_result

    def test_get_fit_functions(self, analysis):
        # WHEN

        # THEN
        fit_functions = analysis.get_fit_functions()

        # EXPECT
        assert isinstance(fit_functions, list)
        assert len(fit_functions) == len(analysis.analysis_list)
        for fit_function in fit_functions:
            assert callable(fit_function)

    def test_create_model_array(self, analysis):
        # WHEN
        analysis.calculate = MagicMock(
            return_value=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        )

        # THEN
        model_array = analysis._create_model_array()

        # EXPECT
        analysis.calculate.assert_called_once()
        assert isinstance(model_array, sc.DataArray)
        assert 'Q' in model_array.dims
        assert 'energy' in model_array.dims
        assert sc.identical(model_array.coords['Q'], analysis.Q)
        assert sc.identical(
            model_array.coords['energy'], analysis.experiment.data.coords['energy']
        )
        np.testing.assert_array_equal(
            model_array.values,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )

    def test_create_components_dataset_raises(self, analysis):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='add_background must be True or False',
        ):
            analysis._create_components_dataset(add_background='123')

    def test_create_components_dataset(self, analysis):
        # WHEN
        # Add another component so that there are two components
        analysis.sample_model.append_component(Gaussian(name='Gaussian2', area=0.5))

        # THEN
        components_dataset = analysis._create_components_dataset(add_background=True)

        # THEN EXPECT
        assert isinstance(components_dataset, sc.Dataset)
        component_names = [comp.name for comp in analysis.sample_model.components]
        for component_name in component_names:
            assert component_name in components_dataset
            assert 'Q' in components_dataset[component_name].dims
            assert 'energy' in components_dataset[component_name].dims
        assert components_dataset.sizes['Q'] == analysis.Q.sizes['Q']

    def test_create_components_dataset_single_Q(self, analysis_single_Q):
        # WHEN

        # Add another component so that there are two components
        analysis_single_Q.sample_model.append_component(Gaussian(name='Gaussian2', area=0.5))

        # THEN
        components_dataset = analysis_single_Q._create_components_dataset(add_background=True)

        # THEN EXPECT
        assert isinstance(components_dataset, sc.Dataset)
        component_names = [comp.name for comp in analysis_single_Q.sample_model.components]
        for component_name in component_names:
            assert component_name in components_dataset
            assert 'Q' in components_dataset[component_name].dims
            assert 'energy' in components_dataset[component_name].dims

        assert components_dataset.coords['Q'].dims == ('Q',)
        assert components_dataset.sizes['Q'] == 1
        assert components_dataset.coords['Q'].ndim == 1
