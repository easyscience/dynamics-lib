# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.fitting import AvailableMinimizers
from easyscience.variable import Parameter

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import ResolutionModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.gaussian import Gaussian
from easydynamics.sample_model.components.polynomial import Polynomial
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings

# The per-consumer convolver staleness tracking relies on the ModelBase.state_version contract;
# until it lands, the conservative fallback rebuilds on every prepare, so 'no rebuild' tests
# cannot pass.
HAS_STATE_VERSION = hasattr(SampleModel, 'state_version')


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
        # calculate() passes the convolver directly to _calculate without storing it on self
        _, call_kwargs = analysis1d._calculate.call_args
        assert call_kwargs['convolver'] is fake_convolver
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

    #############
    # The cached fitter
    #############

    @pytest.fixture
    def fittable_analysis1d(self):
        # The analysis1d fixture holds three points against three free parameters, which leaves
        # a fit with no degrees of freedom. This one has a curve to land on.
        energy_values = np.linspace(-5.0, 5.0, 20)
        intensity = 3.0 * np.exp(-0.5 * (energy_values / 1.2) ** 2)
        data = sc.array(
            dims=['Q', 'energy'],
            values=intensity[None, :],
            variances=np.full_like(intensity, 0.01)[None, :],
        )
        experiment = Experiment(
            data=sc.DataArray(
                data=data,
                coords={
                    'Q': sc.array(dims=['Q'], values=[1.0], unit='1/Angstrom'),
                    'energy': sc.array(dims=['energy'], values=energy_values, unit='meV'),
                },
            )
        )
        analysis = Analysis1d(
            display_name='TestFittable',
            experiment=experiment,
            sample_model=SampleModel(components=Gaussian(area=3.0, width=1.2, center=0.0)),
            instrument_model=InstrumentModel(),
            Q_index=0,
        )
        analysis.instrument_model.fix_energy_offset(Q_index=0)
        return analysis

    def test_fitter_is_built_lazily_and_cached(self, analysis1d):
        # THEN
        fitter = analysis1d.fitter

        # EXPECT
        assert fitter is analysis1d.fitter
        assert fitter.fit_object is analysis1d

    def test_fitter_is_rebuilt_when_the_sample_model_changes(self, analysis1d):
        # WHEN
        original = analysis1d.fitter

        # THEN
        analysis1d.sample_model = SampleModel(components=Gaussian(area=1.0))

        # EXPECT
        assert analysis1d.fitter is not original

    def test_minimizer_can_be_switched_through_the_fitter(self, analysis1d):
        # THEN
        analysis1d.fitter.switch_minimizer(AvailableMinimizers.Bumps)

        # EXPECT
        assert analysis1d.fitter.minimizer.enum == AvailableMinimizers.Bumps

    def test_fit_uses_the_persistent_fitter(self, fittable_analysis1d):
        # THEN
        result = fittable_analysis1d.fit()

        # EXPECT
        assert result is fittable_analysis1d._fit_result
        assert np.isfinite(result.reduced_chi2)

    #############
    # Fitting
    #############

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

        analysis1d.experiment.extract_x_y_weights_only_finite = MagicMock(
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

        analysis1d.experiment.extract_x_y_weights_only_finite.assert_called_once()

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

        # THEN residuals cannot be computed on a custom grid, so they are omitted with a warning
        if include_residuals:
            with pytest.warns(UserWarning, match='omitted'):
                datagroup = analysis1d.data_and_model_to_datagroup(
                    energy=energy, include_residuals=include_residuals
                )
        else:
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
        assert 'Residuals' not in datagroup

    def test_data_and_model_to_datagroup_residuals_on_experiment_grid(self, analysis1d):
        # WHEN no custom energy grid is given

        # THEN
        datagroup = analysis1d.data_and_model_to_datagroup(include_residuals=True)

        # EXPECT residuals present and consistent with the data and model on the same grid
        assert 'Residuals' in datagroup
        assert sc.identical(
            datagroup['Residuals'],
            datagroup['Data'] - analysis1d._create_model_array(),
        )

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
        energy_offset.value = 1.0  # set to 1.0 in original unit (meV)
        energy_offset.convert_unit('eV')  # now 0.001 eV, still represents 1 meV

        # THEN
        result = analysis1d._calculate_energy_with_offset(energy, energy_offset)

        # EXPECT: offset must be converted to energy's unit before subtraction
        offset_in_energy_unit = sc.to_unit(
            sc.scalar(energy_offset.value, unit=str(energy_offset.unit)),
            str(energy.unit),
        ).value
        expected = energy.values - offset_in_energy_unit
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_calculate_energy_with_offset_raises_if_incompatible_units(self, analysis1d):
        # WHEN
        energy = analysis1d.experiment.energy
        energy_offset = Parameter(name='energy_offset', value=1.0, unit='m')  # incompatible unit

        # THEN / EXPECT
        with pytest.raises(sc.UnitError):
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

    @pytest.fixture
    def analysis1d_counts(self):
        "Analysis1d with data and all models in counts, as loaded from a real experiment"
        Q = sc.array(dims=['Q'], values=[1.0], unit='1/Angstrom')
        energy = sc.linspace('energy', -5.0, 5.0, num=20, unit='meV')
        values = np.ones((1, 20))
        data_array = sc.DataArray(
            data=sc.array(dims=['Q', 'energy'], values=values, variances=values, unit='counts'),
            coords={'Q': Q, 'energy': energy},
        )

        return Analysis1d(
            experiment=Experiment(data=data_array),
            sample_model=SampleModel(y_unit='counts', components=Gaussian(y_unit='counts')),
            instrument_model=InstrumentModel(
                resolution_model=ResolutionModel(components=Gaussian(width=0.5))
            ),
            Q_index=0,
        )

    def test_model_array_has_sample_model_y_unit(self, analysis1d_counts):
        "Regression: model arrays must carry the sample model y_unit, not dimensionless"
        # WHEN THEN
        model_array = analysis1d_counts._create_model_array()

        # EXPECT
        assert model_array.unit == sc.Unit('counts')

    def test_residuals_and_datagroup_with_counts_data(self, analysis1d_counts):
        "Regression: residuals (data - model) must work when the data is not dimensionless"
        # WHEN THEN
        datagroup = analysis1d_counts.data_and_model_to_datagroup(include_residuals=True)

        # EXPECT
        assert datagroup['Model'].unit == sc.Unit('counts')
        assert datagroup['Residuals'].unit == sc.Unit('counts')

    def test_convolver_gets_sample_model_units(self, analysis1d_counts):
        "Regression: the convolver must be built with the sample model's units"
        # WHEN THEN
        convolver = analysis1d_counts._create_convolver()

        # EXPECT
        assert convolver.x_unit == analysis1d_counts.sample_model.x_unit
        assert convolver.y_unit == 'counts'

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
            analysis1d.experiment.extract_x_y_weights_only_finite = MagicMock(
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

    @pytest.mark.skipif(not HAS_STATE_VERSION, reason='pending ModelBase.state_version contract')
    def test_fit_does_not_rebuild_convolver_when_nothing_changed(self, analysis1d):
        """fit() should not call _create_convolver if nothing has changed since last fit."""
        # WHEN - a first fit has built the convolver against the current model state
        analysis1d._create_convolver = MagicMock(return_value=None)

        # THEN - fit once to sync, then fit again with nothing changed
        with patch(
            'easydynamics.analysis.analysis1d.EasyScienceFitter',
            return_value=MagicMock(fit=MagicMock(return_value=MagicMock())),
        ):
            analysis1d.experiment.extract_x_y_weights_only_finite = MagicMock(
                return_value=(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 2.0, 3.0]),
                    np.array([1.0, 1.0, 1.0]),
                    np.array([True, True, True]),
                )
            )
            analysis1d.fit()
            analysis1d._create_convolver.reset_mock()
            analysis1d.fit()

        # EXPECT - _create_convolver was NOT called again (convolver reused)
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
            analysis1d.experiment.extract_x_y_weights_only_finite = MagicMock(
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

    #############
    # Convolver staleness across analyses sharing a model
    #############

    @pytest.fixture
    def sibling_analyses(self):
        """Two Analysis1d objects sharing one SampleModel and one InstrumentModel."""
        Q = sc.array(dims=['Q'], values=[1.0, 2.0], unit='1/Angstrom')
        energy = sc.linspace('energy', -5.0, 5.0, num=11, unit='meV')
        values = np.ones((2, 11))
        data_array = sc.DataArray(
            data=sc.array(dims=['Q', 'energy'], values=values, variances=values),
            coords={'Q': Q, 'energy': energy},
        )
        experiment = Experiment(data=data_array)
        sample_model = SampleModel(components=Gaussian())
        instrument_model = InstrumentModel(
            resolution_model=ResolutionModel(components=Gaussian(width=0.5))
        )
        return [
            Analysis1d(
                display_name=f'Sibling{q_index}',
                experiment=experiment,
                sample_model=sample_model,
                instrument_model=instrument_model,
                Q_index=q_index,
            )
            for q_index in (0, 1)
        ]

    def test_in_place_model_edit_rebuilds_the_convolvers_of_all_siblings(self, sibling_analyses):
        """Regression: the first sibling to prepare must not consume the staleness signal."""
        # WHEN both siblings have built their convolvers against the shared model
        first, second = sibling_analyses
        first._prepare_for_sampling()
        second._prepare_for_sampling()
        first_convolver = first._convolver
        second_convolver = second._convolver
        assert first_convolver is not None
        assert second_convolver is not None

        # THEN the shared model is edited in place (not through any Analysis1d setter)
        first.sample_model.append_component(Gaussian(name='ExtraGaussian'))
        first._prepare_for_sampling()
        second._prepare_for_sampling()

        # EXPECT both siblings rebuilt their convolvers, not only the first one to prepare
        assert first._convolver is not first_convolver
        assert second._convolver is not second_convolver

    def test_in_place_resolution_edit_rebuilds_the_convolvers_of_all_siblings(
        self, sibling_analyses
    ):
        # WHEN both siblings have built their convolvers against the shared resolution model
        first, second = sibling_analyses
        first._prepare_for_sampling()
        second._prepare_for_sampling()
        first_convolver = first._convolver
        second_convolver = second._convolver

        # THEN the shared resolution model is edited in place
        first.instrument_model.resolution_model.append_component(Gaussian(name='ExtraResolution'))
        first._prepare_for_sampling()
        second._prepare_for_sampling()

        # EXPECT both siblings rebuilt their convolvers
        assert first._convolver is not first_convolver
        assert second._convolver is not second_convolver

    @pytest.mark.skipif(not HAS_STATE_VERSION, reason='pending ModelBase.state_version contract')
    def test_prepare_does_not_rebuild_when_the_models_are_unchanged(self, sibling_analyses):
        # WHEN a convolver has been built against the current model state
        first, _ = sibling_analyses
        first._prepare_for_sampling()
        convolver = first._convolver

        # THEN preparing again with nothing changed
        first._prepare_for_sampling()

        # EXPECT the convolver is reused, not rebuilt
        assert first._convolver is convolver

    #############
    # Detailed balance settings
    #############

    def test_detailed_balance_settings_change_marks_convolver_dirty(self, analysis1d):
        # WHEN
        analysis1d._convolver_is_dirty = False

        # THEN a new settings object is assigned
        analysis1d.detailed_balance_settings = DetailedBalanceSettings(use_detailed_balance=False)

        # EXPECT
        assert analysis1d._convolver_is_dirty is True

    #############
    # The bayesian sampler (the Analysis1d side of the contract)
    #############

    def test_bayesian_returns_the_cached_sampler(self, analysis1d):
        # THEN
        sampler = analysis1d.bayesian

        # EXPECT the same object on second access
        assert sampler is analysis1d.bayesian

    def test_bayesian_is_invalidated_when_the_Q_index_changes(self, analysis1d):
        # WHEN
        sampler = analysis1d.bayesian

        # THEN a different Q index means different data
        with patch.object(sampler, 'invalidate') as mock_invalidate:
            analysis1d.Q_index = 1

        # EXPECT
        mock_invalidate.assert_called_once()

    def test_bayesian_is_invalidated_when_the_experiment_changes(self, analysis1d):
        # WHEN
        sampler = analysis1d.bayesian
        new_experiment = Experiment(data=analysis1d.experiment.data.copy(deep=True))

        # THEN
        with patch.object(sampler, 'invalidate') as mock_invalidate:
            analysis1d.experiment = new_experiment

        # EXPECT
        mock_invalidate.assert_called_once()

    def test_bayesian_is_invalidated_when_the_sample_model_changes(self, analysis1d):
        # WHEN
        sampler = analysis1d.bayesian

        # THEN
        with patch.object(sampler, 'invalidate') as mock_invalidate:
            analysis1d.sample_model = SampleModel(components=Gaussian())

        # EXPECT
        mock_invalidate.assert_called_once()

    def test_bayesian_is_invalidated_on_rebin(self, analysis1d):
        # WHEN
        sampler = analysis1d.bayesian

        # THEN rebinning changes the data the sampler was bound to
        with patch.object(sampler, 'invalidate') as mock_invalidate:
            analysis1d.rebin({'Q': 1})

        # EXPECT
        mock_invalidate.assert_called_once()

    #############
    # Regression tests
    #############

    @pytest.fixture
    def analysis1d_with_nan(self):
        """analysis1d fixture whose data contains a NaN at the second energy point."""
        Q = sc.array(dims=['Q'], values=[1.0], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[1.0, float('nan'), 3.0]],
            variances=[[0.1, 0.2, 0.3]],
        )
        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})
        experiment = Experiment(data=data_array)
        sample_model = SampleModel(components=Gaussian())
        instrument_model = InstrumentModel()
        return Analysis1d(
            display_name='TestNaN',
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            Q_index=0,
        )

    def test_create_residuals_array_with_nan_data_does_not_crash(self, analysis1d_with_nan):
        # Before the fix, residuals subtracted a 2-point model from 3-point data
        # (including NaN) which caused a dimension mismatch crash.
        # WHEN

        # THEN
        result = analysis1d_with_nan._create_residuals_array()

        # EXPECT
        assert isinstance(result, sc.DataArray)
        # Only the 2 finite energy points survive the mask.
        assert result.sizes['energy'] == 2

    def test_data_and_model_to_datagroup_with_nan_excludes_nan_from_data(
        self, analysis1d_with_nan
    ):
        # Before the fix, 'Data' contained the full 3-point grid (including NaN)
        # and computing Residuals crashed on the dimension mismatch.
        # WHEN

        # THEN
        datagroup = analysis1d_with_nan.data_and_model_to_datagroup(include_residuals=True)

        # EXPECT
        assert isinstance(datagroup, sc.DataGroup)
        # 'Data' must contain only the 2 finite points.
        assert datagroup['Data'].sizes['energy'] == 2
        # Residuals must be present and have matching size.
        assert 'Residuals' in datagroup
        assert datagroup['Residuals'].sizes['energy'] == 2

    def test_repr(self, analysis1d):
        repr_str = repr(analysis1d)
        assert 'Analysis1d' in repr_str
        assert 'display_name=' in repr_str
        assert 'Q_index=' in repr_str

    #############
    # Change handlers
    #############

    @staticmethod
    def _coverage_analysis1d():
        Q = sc.array(dims=['Q'], values=[1, 2, 3], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )
        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})
        return Analysis1d(
            display_name='CoverageAnalysis',
            experiment=Experiment(data=data_array),
            sample_model=SampleModel(components=Gaussian()),
            instrument_model=InstrumentModel(),
            Q_index=0,
        )

    def test_on_Q_index_changed_with_none_clears_masked_energy(self):
        # WHEN an analysis whose Q_index has been cleared
        analysis1d = self._coverage_analysis1d()
        analysis1d._Q_index = None

        # THEN the Q-index-changed handler runs
        analysis1d._on_Q_index_changed()

        # EXPECT masked energy cleared and convolver marked dirty
        assert analysis1d._masked_energy is None
        assert analysis1d._convolver_is_dirty is True

    def test_on_experiment_changed_refreshes_masked_energy_when_Q_index_set(self):
        # WHEN an analysis with a Q_index already set
        analysis1d = self._coverage_analysis1d()

        # THEN the experiment-changed handler runs
        analysis1d._on_experiment_changed()

        # EXPECT masked energy refreshed and convolver marked dirty
        assert analysis1d._masked_energy is not None
        assert analysis1d._convolver_is_dirty is True
