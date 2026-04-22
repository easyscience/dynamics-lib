# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment


class TestExperiment:
    @pytest.fixture
    def experiment(self):
        Q = sc.linspace('Q', 0.5, 1.5, num=10, unit='1/Angstrom')
        energy = sc.linspace('energy', -5, 5, num=11, unit='meV')
        values = sc.array(dims=['Q', 'energy'], values=np.ones((10, 11)))
        data = sc.DataArray(data=values, coords={'Q': Q, 'energy': energy})

        return Experiment(display_name='test_experiment', data=data)

    @pytest.fixture
    def experiment_with_data(self):
        "Fixture that provides an Experiment with data for testing methods that require data"
        Q = sc.array(dims=['Q'], values=[1, 2, 3], unit='1/Angstrom')
        energy = sc.array(dims=['energy'], values=[10.0, 20.0, 30.0], unit='meV')
        data = sc.array(
            dims=['Q', 'energy'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )

        data_array = sc.DataArray(data=data, coords={'Q': Q, 'energy': energy})

        return Experiment(data=data_array)

    ##############
    # test init
    ##############

    def test_init_array(self, experiment):
        "Test initialization with a Scipp DataArray"
        # WHEN THEN EXPECT
        assert experiment.display_name == 'test_experiment'
        assert isinstance(experiment._data, sc.DataArray)
        assert 'Q' in experiment._data.dims
        assert 'energy' in experiment._data.dims
        assert experiment._data.sizes['Q'] == 10
        assert experiment._data.sizes['energy'] == 11
        assert sc.identical(
            experiment._data.data,
            sc.array(dims=['Q', 'energy'], values=np.ones((10, 11))),
        )

    def test_init_string(self, tmp_path):
        "Test initialization with a filename string,"
        'should load the file'
        # WHEN
        Q = sc.linspace('Q', 0.5, 1.5, num=10, unit='1/Angstrom')
        energy = sc.linspace('energy', -5, 5, num=11, unit='meV')
        values = sc.array(dims=['Q', 'energy'], values=np.ones((10, 11)))
        data = sc.DataArray(data=values, coords={'Q': Q, 'energy': energy})

        filename = tmp_path / 'test_experiment.h5'
        sc.io.save_hdf5(data, filename)

        # THEN
        experiment = Experiment(display_name='loaded_experiment', data=str(filename))

        # EXPECT
        assert experiment.display_name == 'loaded_experiment'
        assert isinstance(experiment._data, sc.DataArray)
        assert 'Q' in experiment._data.dims
        assert 'energy' in experiment._data.dims
        assert experiment._data.sizes['Q'] == 10
        assert experiment._data.sizes['energy'] == 11
        assert sc.identical(
            experiment._data.data,
            sc.array(dims=['Q', 'energy'], values=np.ones((10, 11))),
        )

    def test_init_no_data(self):
        "Test initialization with no data"
        # WHEN
        experiment = Experiment(display_name='empty_experiment')

        # THEN EXPECT
        assert experiment.display_name == 'empty_experiment'
        assert experiment._data is None
        assert experiment.energy is None
        assert experiment.Q is None

    def test_init_invalid_data(self):
        "Test initialization with invalid data type"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError):
            Experiment(data=123)

    ##############
    # test data manipulation
    ##############

    def test_load_hdf5(self, tmp_path, experiment):
        "Test loading data from an HDF5 file."
        'First use scipp to save data to a file, '
        'then load it using the method.'
        # WHEN
        # First create a file to load from
        filename = tmp_path / 'test.h5'
        data_to_save = experiment.data
        sc.io.save_hdf5(data_to_save, filename)

        # THEN
        new_experiment = Experiment(display_name='new_experiment')
        new_experiment.load_hdf5(str(filename), display_name='loaded_data')
        loaded_data = new_experiment.data

        # EXPECT
        assert sc.identical(data_to_save, loaded_data)
        assert new_experiment.display_name == 'loaded_data'

    def test_load_hdf5_invalid_name_raises(self, experiment):
        "Test loading data from an HDF5 file,"
        'giving the Experiment an invalid name'
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError):
            experiment.load_hdf5('some_file.h5', display_name=123)

    def test_load_hdf5_invalid_filename_raises(self, experiment):
        "Test loading data from an HDF5 file with an invalid filename"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError, match='must be a string'):
            experiment.load_hdf5(123)

    def test_load_hdf5_invalid_file_raises(self, experiment):
        "Test loading data from a non-existent HDF5 file"
        # WHEN / THEN EXPECT

        with pytest.raises(OSError):
            experiment.load_hdf5('non_existent_file.h5')

    def test_save_hdf5(self, tmp_path, experiment):
        "Test saving data to an HDF5 file. Load the saved file"
        'using scipp and compare to the original data.'
        # WHEN THEN
        filename = tmp_path / 'saved_data.h5'
        experiment.save_hdf5(str(filename))

        # EXPECT
        loaded_data = sc.io.load_hdf5(str(filename))
        original_data = experiment.data
        assert sc.identical(original_data, loaded_data)

    def test_save_hdf5_default_filename(self, tmp_path, experiment, monkeypatch):
        "Test saving data to an HDF5 file with default filename"
        # WHEN
        monkeypatch.chdir(tmp_path)

        # THEN
        experiment.save_hdf5()

        # EXPECT
        expected_filename = tmp_path / f'{experiment.unique_name}.h5'
        loaded_data = sc.io.load_hdf5(str(expected_filename))
        original_data = experiment.data
        assert sc.identical(original_data, loaded_data)

    def test_save_hdf5_no_data_raises(self):
        "Test saving data to an HDF5 file when no data is present"
        'in the experiment'
        # WHEN
        experiment = Experiment()

        # THEN EXPECT
        with pytest.raises(ValueError):
            experiment.save_hdf5('should_fail.h5')

    def test_save_hdf5_invalid_filename_raises(self, experiment):
        "Test saving data to an HDF5 file with an invalid filename"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError, match='must be a string'):
            experiment.save_hdf5(123)

    def test_remove_data(self, experiment):
        "Test removing data from the experiment"
        # WHEN
        experiment.remove_data()

        # THEN EXPECT
        assert experiment._data is None

    @pytest.mark.parametrize(
        'new_Q_bins, new_energy_bins',
        [
            (
                sc.linspace('Q', 0.5, 1.5, num=7, unit='1/Angstrom'),
                sc.linspace('energy', -5, 5, num=8, unit='meV'),
            ),
            (
                6,
                7,
            ),
            (
                6.0,
                7.0,
            ),
            (
                sc.linspace('Q', 0.5, 1.5, num=7, unit='1/Angstrom'),
                7,
            ),
        ],
        ids=['sc_bins', 'integers_bins', 'float_bins', 'mixed_bins'],
    )
    def test_rebin(self, experiment, new_Q_bins, new_energy_bins):
        "Test rebinning data in the experiment"
        # WHEN

        # THEN
        experiment.rebin({'Q': new_Q_bins, 'energy': new_energy_bins})

        # EXPECT
        rebinned_data = experiment.binned_data
        assert rebinned_data.sizes['Q'] == 6
        assert rebinned_data.sizes['energy'] == 7

    def test_rebin_no_data_raises(self):
        "Test rebinning data when no data is present"
        # WHEN
        experiment = Experiment()

        # THEN EXPECT
        with pytest.raises(ValueError):
            experiment.rebin({'Q': 6, 'energy': 7})

    def test_rebin_invalid_dimensions_raises(self, experiment):
        "Test rebinning data with invalid dimensions"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError):
            experiment.rebin('invalid_dimensions')

    def test_rebin_invalid_dimension_name_raises(self, experiment):
        "Test rebinning data with invalid dimension name"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError, match='Dimension keys must be strings'):
            experiment.rebin({123: 6, 'energy': 7})

    def test_rebin_dimension_not_in_data_raises(self, experiment):
        "Test rebinning data with a dimension not in the data"
        # WHEN / THEN EXPECT
        with pytest.raises(KeyError, match="Dimension 'time' not a valid"):
            experiment.rebin({'time': 6, 'energy': 7})

    def test_rebin_invalid_bin_values_raises(self, experiment):
        "Test rebinning data with invalid bin values"
        # WHEN / THEN EXPECT
        with pytest.raises(
            TypeError,
            match='Dimension values must be integers or',
        ):
            experiment.rebin({'Q': [0.5, 1.0, 1.5], 'energy': 7})

    ##############
    # test setters and getters
    ##############

    def test_data_setter_raises_type_error(self, experiment):
        "Test setting data to an invalid type raises TypeError"
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            experiment.data = 123

    def test_binned_data_setter_raises(self, experiment):
        "Test that setting binned data raises AttributeError"
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            experiment.binned_data = experiment.binned_data

    def test_energy_setter_raises(self, experiment):
        "Test that setting energy data raises AttributeError"
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            experiment.energy = experiment.energy

    def test_Q_setter_raises(self, experiment):
        "Test that setting Q data raises AttributeError"
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            experiment.Q = experiment.Q

    def test_get_masked_energy(self, experiment_with_data):
        "Test getting masked energy"
        # WHEN
        Q_index = 0
        invalid_data = experiment_with_data._data.copy()
        invalid_data.data.values[Q_index][0] = np.inf
        invalid_data.data.variances[Q_index][1] = np.nan

        experiment_with_data.data = invalid_data

        # THEN
        masked_energy = experiment_with_data.get_masked_energy(Q_index=Q_index)

        # EXPECT
        assert len(masked_energy) == 1
        assert masked_energy.values == pytest.approx(30.0)

    def test_get_masked_energy_no_data_returns_None(self):
        "Test getting masked energy returns zero when no data is present"

        experiment = Experiment()
        # WHEN THEN
        masked_energy = experiment.get_masked_energy(Q_index=0)

        # EXPECT
        assert masked_energy is None

    @pytest.mark.parametrize(
        'Q_index',
        [-1, 100, 'not an index'],
        ids=['negative_index', 'out_of_bounds_index', 'invalid_type'],
    )
    def test_get_masked_energy_invalid_Q_index_raises(self, experiment_with_data, Q_index):
        "Test getting masked energy raises IndexError when Q index is invalid"
        # WHEN THEN EXPECT
        with pytest.raises(IndexError):
            experiment_with_data.get_masked_energy(Q_index=Q_index)

    ##############
    # test plotting
    ##############

    @pytest.mark.parametrize(
        'transpose_axes, expected_dims',
        [
            (False, ['Q', 'energy']),
            (True, ['energy', 'Q']),
        ],
        ids=['no_transpose', 'transpose'],
    )
    def test_plot_data_success(self, experiment, transpose_axes, expected_dims):
        "Test plotting data successfully when in notebook environment"
        # WHEN
        with (
            patch(f'{Experiment.__module__}._in_notebook', return_value=True),
            patch('plopp.plot') as mock_plot,
        ):
            mock_fig = MagicMock()
            mock_plot.return_value = mock_fig

            # THEN
            result = experiment.plot_data(transpose_axes=transpose_axes)

            # EXPECT
            mock_plot.assert_called_once()
            args, kwargs = mock_plot.call_args
            assert sc.identical(args[0], experiment.data.transpose(dims=expected_dims))
            assert kwargs['title'] == f'{experiment.display_name}'
            assert result == mock_fig

    def test_plot_data_slicer_success(self, experiment):
        "Test plotting data successfully when in notebook environment"
        # WHEN
        with (
            patch(f'{Experiment.__module__}._in_notebook', return_value=True),
            patch('plopp.slicer') as mock_plot,
        ):
            mock_fig = MagicMock()
            mock_plot.return_value = mock_fig

            # THEN
            result = experiment.plot_data(slicer=True)

            # EXPECT
            mock_plot.assert_called_once()
            args, kwargs = mock_plot.call_args
            assert sc.identical(args[0], experiment.data)
            assert kwargs['title'] == f'{experiment.display_name}'
            assert kwargs['keep'] == 'energy'
            assert result == mock_fig

    def test_plot_data_no_data_raises(self):
        "Test plotting data raises ValueError when no data is present"
        # WHEN
        experiment = Experiment()

        # THEN EXPECT
        with pytest.raises(ValueError, match='No data to plot'):
            experiment.plot_data()

    def test_plot_data_not_in_notebook_raises(self, experiment):
        "Test plotting data raises RuntimeError"
        'when not in notebook environment'
        # WHEN
        with (
            patch(f'{Experiment.__module__}._in_notebook', return_value=False),
            pytest.raises(
                RuntimeError,
                match='plot_data\\(\\) can only be used in a Jupyter notebook environment',
            ),
        ):
            experiment.plot_data()

    def test_plot_data_invalid_slicer_type_raises(self, experiment):
        "Test plotting data raises TypeError when slicer argument is invalid"
        # WHEN THEN EXPECT

        with (
            patch(f'{Experiment.__module__}._in_notebook', return_value=True),
            pytest.raises(TypeError, match='slicer must be True or False'),
        ):
            experiment.plot_data(slicer='not_a_boolean')

    def test_plot_data_invalid_transpose_type_raises(self, experiment):
        "Test plotting data raises TypeError when transpose argument is invalid"
        # WHEN THEN EXPECT
        with (
            patch(f'{Experiment.__module__}._in_notebook', return_value=True),
            pytest.raises(TypeError, match='transpose_axes must be True or False'),
        ):
            experiment.plot_data(transpose_axes='not_a_boolean')

    ##############
    # test private methods
    ##############

    def test_validate_coordinates(self, experiment):
        "Test that _validate_coordinates does not raise for valid data"
        # WHEN / THEN EXPECT
        experiment._validate_coordinates(experiment._data)

    def test_validate_coordinates_raises_missing_Q(self, experiment):
        "Test that _validate_coordinates raises ValueError when Q coord"
        'is missing'
        # WHEN
        invalid_data = experiment._data.copy()
        invalid_data.coords.pop('Q')

        # THEN EXPECT
        with pytest.raises(ValueError, match='missing required coordinate'):
            experiment._validate_coordinates(invalid_data)

    def test_validate_coordinates_raises_missing_energy(self, experiment):
        "Test that _validate_coordinates raises ValueError when energy"
        'coord is missing'
        # WHEN
        invalid_data = experiment._data.copy()
        invalid_data.coords.pop('energy')

        # THEN EXPECT
        with pytest.raises(ValueError, match='missing required coordinate'):
            experiment._validate_coordinates(invalid_data)

    def test_validate_coordinates_raises_not_DataArray(self):
        "Test that _validate_coordinates raises TypeError when data is"
        'not a Scipp DataArray'
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='must be a'):
            Experiment()._validate_coordinates('not_a_data_array')

    def test_convert_to_bin_centers(self, experiment):
        "Test that _convert_to_bin_centers converts edges to centers"
        # WHEN
        Q_edges = sc.linspace('Q', 0.0, 2.0, num=11, unit='1/Angstrom')
        energy_edges = sc.linspace('energy', -6, 6, num=13, unit='meV')
        values = sc.array(dims=['Q', 'energy'], values=np.ones((10, 12)))
        binned_data = sc.DataArray(data=values, coords={'Q': Q_edges, 'energy': energy_edges})

        # THEN
        experiment._data = binned_data  # Set data to avoid warnings
        converted_data = experiment._convert_to_bin_centers(binned_data)

        # EXPECT
        expected_Q = 0.5 * (Q_edges[:-1] + Q_edges[1:])
        expected_energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])

        assert sc.identical(converted_data.coords['Q'], expected_Q)
        assert sc.identical(converted_data.coords['energy'], expected_energy)
        assert sc.identical(converted_data.data, binned_data.data)

    def test_extract_x_y_var(self, experiment_with_data):
        # WHEN

        Q_index = 0

        # THEN
        x, y, var = experiment_with_data._extract_x_y_var(Q_index=Q_index)

        # EXPECT
        assert np.array_equal(x, experiment_with_data.energy.values)
        assert np.array_equal(y, experiment_with_data.data.values[Q_index])
        assert np.array_equal(
            var,
            experiment_with_data.data.variances[Q_index],
        )

    def test_extract_x_y_weights_only_finite_zero_variances(self, experiment_with_data):
        "Test that _extract_x_y_weights_only_finite raises ValueError when variances contain zeros"
        # WHEN
        Q_index = 0
        invalid_data = experiment_with_data._data.copy()
        invalid_data.data.variances[Q_index] = 0  # Set variances to zero
        # throw in a nan for good measure
        invalid_data.data.variances[Q_index][0] = np.nan

        # THEN EXPECT
        with pytest.raises(ValueError, match='Cannot compute weights: some variances are zero'):
            Experiment(data=invalid_data)._extract_x_y_weights_only_finite(Q_index=Q_index)

    def test_extract_x_y_weights_only_finite(self, experiment_with_data):
        "Test that _extract_x_y_weights_only_finite only returns finite values"
        # WHEN
        Q_index = 0
        invalid_data = experiment_with_data._data.copy()
        invalid_data.data.values[Q_index][0] = np.inf
        invalid_data.data.variances[Q_index][1] = np.nan

        # THEN
        x, y, weights, mask = Experiment(data=invalid_data)._extract_x_y_weights_only_finite(
            Q_index=Q_index
        )

        # EXPECT
        assert np.isfinite(x).all()
        assert np.isfinite(y).all()
        assert np.isfinite(weights).all()
        assert weights[0] == pytest.approx(
            1.0 / (experiment_with_data.data.variances[Q_index][2] ** 0.5)
        )
        assert len(x) == len(y) == len(weights) == 1  # 2 values should be removed
        # Mask should indicate which values were removed
        assert np.array_equal(mask, [False, False, True])

    def test_extract_x_y_weights_only_finite_zero_variance(self, experiment_with_data):
        "Test getting x y and weights when variances are None"
        # WHEN
        Q_index = 0
        data = experiment_with_data._data.copy()
        data.variances = None

        experiment_with_data.data = data

        # THEN
        x, y, weights, mask = experiment_with_data._extract_x_y_weights_only_finite(
            Q_index=Q_index
        )

        # EXPECT
        assert np.array_equal(x, experiment_with_data.energy.values)
        assert np.array_equal(y, experiment_with_data.data.values[Q_index])
        assert np.array_equal(weights, np.ones_like(y))
        assert np.array_equal(mask, np.isfinite(y) & np.isfinite(x))

    ##############
    # test dunder methods
    ##############

    def test_repr(self, experiment):
        # WHEN
        repr_str = repr(experiment)

        # THEN EXPECT
        assert repr_str == f'Experiment `{experiment.unique_name}` with data: {experiment._data}'

    def test_copy_experiment(self, experiment):
        "Test copying an Experiment object."
        'The copied object should have the same attributes '
        'but be a different object in memory.'
        # WHEN
        copied_experiment = copy(experiment)

        # THEN EXPECT
        assert copied_experiment.display_name == experiment.display_name
        assert sc.identical(copied_experiment.data, experiment.data)
        assert copied_experiment is not experiment
        assert copied_experiment.data is not experiment.data
