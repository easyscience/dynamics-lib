import pytest

from easydynamics.experiment import Experiment
import scipp as sc
import numpy as np


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
        experiment.plot_data(name)
        return experiment

    def test_get_data(self, experiment):
        data = experiment.get_data("test_data")
        assert isinstance(data, sc.DataArray)
        assert "Q" in data.dims
        assert "energy" in data.dims
        assert data.sizes["Q"] == 10
        assert data.sizes["energy"] == 11
        assert sc.identical(
            data.data, sc.array(dims=["Q", "energy"], values=np.ones((10, 11)))
        )
