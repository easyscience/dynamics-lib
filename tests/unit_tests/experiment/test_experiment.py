from copy import copy

import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment


class TestExperiment:
    @pytest.fixture
    def experiment(self):
        Q = sc.linspace("Q", 0.5, 1.5, num=10, unit="1/Angstrom")
        energy = sc.linspace("energy", -5, 5, num=11, unit="meV")
        values = sc.array(dims=["Q", "energy"], values=np.ones((10, 11)))
        data = sc.DataArray(data=values, coords={"Q": Q, "energy": energy})

        experiment = Experiment(name="test_experiment", data=data)
        return experiment

    def test_init_array(self, experiment):
        "Test initialization with a Scipp DataArray"
        # WHEN THEN EXPECT
        assert experiment.name == "test_experiment"
        assert isinstance(experiment._data, sc.DataArray)
        assert "Q" in experiment._data.dims
        assert "energy" in experiment._data.dims
        assert experiment._data.sizes["Q"] == 10
        assert experiment._data.sizes["energy"] == 11
        assert sc.identical(
            experiment._data.data,
            sc.array(dims=["Q", "energy"], values=np.ones((10, 11))),
        )

    def test_init_string(self, tmp_path):
        "Test initialization with a filename string - should load the file"
        # WHEN
        Q = sc.linspace("Q", 0.5, 1.5, num=10, unit="1/Angstrom")
        energy = sc.linspace("energy", -5, 5, num=11, unit="meV")
        values = sc.array(dims=["Q", "energy"], values=np.ones((10, 11)))
        data = sc.DataArray(data=values, coords={"Q": Q, "energy": energy})

        filename = tmp_path / "test_experiment.h5"
        sc.io.save_hdf5(data, filename)

        # THEN
        experiment = Experiment(name="loaded_experiment", data=str(filename))

        # EXPECT
        assert experiment.name == "loaded_experiment"
        assert isinstance(experiment._data, sc.DataArray)
        assert "Q" in experiment._data.dims
        assert "energy" in experiment._data.dims
        assert experiment._data.sizes["Q"] == 10
        assert experiment._data.sizes["energy"] == 11
        assert sc.identical(
            experiment._data.data,
            sc.array(dims=["Q", "energy"], values=np.ones((10, 11))),
        )

    def test_init_no_data(self):
        "Test initialization with no data"
        # WHEN
        experiment = Experiment(name="empty_experiment")

        # THEN EXPECT
        assert experiment.name == "empty_experiment"
        assert experiment._data is None

    def test_init_invalid_data(self):
        "Test initialization with invalid data type"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError):
            Experiment(name="invalid_experiment", data=123)

    def test_load_hdf5(self, tmp_path, experiment):
        "Test loading data from an HDF5 file. First use scipp to save data to a file, then load it using the method."
        # WHEN
        # First create a file to load from
        filename = tmp_path / "test.h5"
        data_to_save = experiment.data
        sc.io.save_hdf5(data_to_save, filename)

        # THEN
        new_experiment = Experiment("new_experiment")
        new_experiment.load_hdf5(str(filename), name="loaded_data")
        loaded_data = new_experiment.data

        # EXPECT
        assert sc.identical(data_to_save, loaded_data)
        assert new_experiment.name == "loaded_data"

    def test_load_hdf5_invalid_name_raises(self, experiment):
        "Test loading data from an HDF5 file, giving the Experiment an invalid name"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError):
            experiment.load_hdf5("some_file.h5", name=123)

    def test_load_hdf5_invalid_filename_raises(self, experiment):
        "Test loading data from an HDF5 file with an invalid filename"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError):
            experiment.load_hdf5(123)

    def test_load_hdf5_invalid_file_raises(self, experiment):
        "Test loading data from a non-existent HDF5 file"
        # WHEN / THEN EXPECT
        with pytest.raises(OSError):
            experiment.load_hdf5("non_existent_file.h5")

    def test_save_hdf5(self, tmp_path, experiment):
        "Test saving data to an HDF5 file. Load the saved file using scipp and compare to the original data."
        # WHEN THEN
        filename = tmp_path / "saved_data.h5"
        experiment.save_hdf5(str(filename))

        # EXPECT
        loaded_data = sc.io.load_hdf5(str(filename))
        original_data = experiment.data
        assert sc.identical(original_data, loaded_data)

    def test_save_hdf5_no_data_raises(self):
        "Test saving data to an HDF5 file when no data is present in the experiment"
        # WHEN
        experiment = Experiment(name="no_data_experiment")

        # THEN EXPECT
        with pytest.raises(ValueError):
            experiment.save_hdf5("should_fail.h5")

    def test_save_hdf5_invalid_filename_raises(self, experiment):
        "Test saving data to an HDF5 file with an invalid filename"
        # WHEN / THEN EXPECT
        with pytest.raises(TypeError):
            experiment.save_hdf5(123)

    def test_remove_data(self, experiment):
        "Test removing data from the experiment"
        # WHEN
        experiment.remove_data()

        # THEN EXPECT
        assert experiment._data is None

    def test_repr(self, experiment):
        # WHEN
        repr_str = repr(experiment)

        # THEN EXPECT
        assert (
            repr_str == f"Experiment `{experiment.name}` with data: {experiment._data}"
        )

    def test_copy_experiment(self, experiment):
        "Test copying an Experiment object. The copied object should have the same attributes but be a different object in memory."
        # WHEN
        copied_experiment = copy(experiment)

        # THEN EXPECT
        assert copied_experiment.name == experiment.name
        assert sc.identical(copied_experiment.data, experiment.data)
        assert copied_experiment is not experiment
        assert copied_experiment.data is not experiment.data
