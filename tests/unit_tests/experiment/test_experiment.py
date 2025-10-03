import numpy as np
import pytest
import scipp as sc

from easydynamics.experiment import Experiment


class TestExperiment:
    @pytest.fixture
    def experiment(self):
        experiment = Experiment("test_experiment")
        name = "test_data"
        Q = sc.linspace("Q", 0.5, 1.5, num=10, unit="1/Angstrom")
        energy = sc.linspace("energy", -5, 5, num=11, unit="meV")
        values = sc.array(dims=["Q", "energy"], values=np.ones((10, 11)))
        data = sc.DataArray(data=values, coords={"Q": Q, "energy": energy})
        experiment.append_data(data, name)
        return experiment

    def test_get_data(self, experiment):
        # WHEN
        data = experiment.get_data("test_data")

        # THEN EXPECT
        assert isinstance(data, sc.DataArray)
        assert "Q" in data.dims
        assert "energy" in data.dims
        assert data.sizes["Q"] == 10
        assert data.sizes["energy"] == 11
        assert sc.identical(
            data.data, sc.array(dims=["Q", "energy"], values=np.ones((10, 11)))
        )

    def test_remove_all_data(self, experiment):
        # WHEN
        experiment.remove_all_data()
        data = experiment.get_data()

        # THEN EXPECT
        assert data == {}

    def test_get_all_data(self, experiment):
        # WHEN
        data = experiment.get_data()

        # THEN EXPECT
        assert isinstance(data, dict)
        assert "test_data" in data
        assert isinstance(data["test_data"], sc.DataArray)

    def test_load_hdf5(self, tmp_path, experiment):
        # WHEN
        filename = tmp_path / "test.h5"
        data_to_save = experiment.get_data("test_data")
        sc.io.save_hdf5(data_to_save, filename)

        # THEN
        new_experiment = Experiment("new_experiment")
        new_experiment.load_hdf5(str(filename), "loaded_data")
        loaded_data = new_experiment.get_data("loaded_data")

        # EXPECT
        assert sc.identical(data_to_save, loaded_data)

    def test_save_hdf5(self, tmp_path, experiment):
        # WHEN THEN
        filename = tmp_path / "saved_data.h5"
        experiment.save_hdf5("test_data", str(filename))

        # EXPECT
        loaded_data = sc.io.load_hdf5(str(filename))
        original_data = experiment.get_data("test_data")
        assert sc.identical(original_data, loaded_data)

    def test_save_all_hdf5(self, tmp_path, experiment):
        # WHEN THEN
        folder = tmp_path / "data_folder"
        experiment.save_all_hdf5(str(folder))

        # EXPECT
        import os

        files = os.listdir(folder)
        assert "test_data.h5" in files
        loaded_data = sc.io.load_hdf5(str(folder / "test_data.h5"))
        original_data = experiment.get_data("test_data")
        assert sc.identical(original_data, loaded_data)

    def test_append_data(self, experiment):
        # WHEN
        name = "new_data"
        Q = sc.linspace("Q", 1.0, 2.0, num=5, unit="1/Angstrom")
        energy = sc.linspace("energy", -10, 10, num=6, unit="meV")
        values = sc.array(dims=["Q", "energy"], values=np.ones((5, 6)) * 2)
        new_data = sc.DataArray(data=values, coords={"Q": Q, "energy": energy})

        # THEN
        experiment.append_data(new_data, name)

        # EXPECT
        assert experiment._data.keys() == {"test_data", "new_data"}
        data = experiment.get_data(name)
        assert isinstance(data, sc.DataArray)
        assert "Q" in data.dims
        assert "energy" in data.dims
        assert data.sizes["Q"] == 5
        assert data.sizes["energy"] == 6
        assert sc.identical(
            data.data, sc.array(dims=["Q", "energy"], values=np.ones((5, 6)) * 2)
        )

    # Test helpful and dunder methods

    def test_items(self, experiment):
        items = list(experiment.items())
        assert len(items) == 1
        name, data = items[0]
        assert name == "test_data"
        assert isinstance(data, sc.DataArray)

    def test_values(self, experiment):
        values = list(experiment.values())
        assert len(values) == 1
        data = values[0]
        assert isinstance(data, sc.DataArray)

    def test_keys(self, experiment):
        keys = list(experiment.keys())
        assert keys == ["test_data"]

    def test_dunder_methods(self, experiment):
        # __getitem__
        data = experiment["test_data"]
        assert isinstance(data, sc.DataArray)

        # __setitem__
        Q = sc.linspace("Q", 2.0, 3.0, num=4, unit="1/Angstrom")
        energy = sc.linspace("energy", -15, 15, num=5, unit="meV")
        values = sc.array(dims=["Q", "energy"], values=np.ones((4, 5)) * 3)
        set_data = sc.DataArray(data=values, coords={"Q": Q, "energy": energy})
        experiment["set_data"] = set_data
        assert "set_data" in experiment._data

        # __delitem__
        del experiment["set_data"]
        assert "set_data" not in experiment._data

        # __contains__
        assert "test_data" in experiment
        assert "non_existent" not in experiment

        # __repr__ and __str__
        repr_str = repr(experiment)
        str_str = str(experiment)
        assert "Experiment(name = test_experiment" in repr_str
        assert repr_str == str_str

        # Final check of data integrity
        data = experiment.get_data("test_data")
        assert isinstance(data, sc.DataArray)
        assert "Q" in data.dims
        assert "energy" in data.dims
        assert data.sizes["Q"] == 10
        assert data.sizes["energy"] == 11
        assert sc.identical(
            data.data, sc.array(dims=["Q", "energy"], values=np.ones((10, 11)))
        )
