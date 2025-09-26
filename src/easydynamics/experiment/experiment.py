from easyscience.job.experiment import ExperimentBase

import scipp as sc
import plopp as pp

from IPython.display import display


class Experiment(ExperimentBase):
    def __init__(self, name="MyExperiment"):
        """
        Initialize the Experiment class.
        """
        super().__init__(name)
        self._data = None
        self._data = {}  # store data as {name: DataArray}

    def load_hdf5(self, filename: str, name: str):
        """
        Load data from an HDF5 file.

        Args:
            file_path (str): Path to the data file.
        """
        self.append_data(sc.io.load_hdf5(filename), name)

        # TODO: Add checks of dimensions etc.

    def save_hdf5(self, name: str, filename: str):
        """Save a single dataset to HDF5."""
        sc.io.to_hdf5(self._data[name], filename)

    def save_all_hdf5(self, folder: str):
        """Save all datasets to individual HDF5 files in a folder."""
        import os

        os.makedirs(folder, exist_ok=True)
        for name, data in self._data.items():
            sc.io.to_hdf5(data, os.path.join(folder, f"{name}.h5"))

    def append_data(self, new_data: sc.DataArray, name: str):
        """Append data with a name."""
        self._data[name] = new_data

    def get_data(self, name: str = None):
        """Return the stored data. If name is None, return the full dict."""
        if name is None:
            return self._data
        return self._data[name]

    def remove_all_data(self):
        self._data = {}

    def plot_data(self, name: str = None):
        """Plot all datasets."""
        if not self._data:
            raise ValueError("No data to plot. Please load data first.")

        if name:
            data = self._data.get(name)
            if data is None:
                raise ValueError(f"No data found for name: {name}")
            fig = pp.plot(data.transpose(), title=f"{name}")
            display(fig)
            return

        for name, data in self._data.items():
            fig = pp.plot(data.transpose(), title=f"{name}")
            display(fig)

    # Helpful methods
    def items(self):
        """Return (name, data) pairs, like dict.items()."""
        return self._data.items()

    def values(self):
        """Return all DataArrays, like dict.values()."""
        return self._data.values()

    def keys(self):
        """Return all dataset names, like dict.keys()."""
        return self._data.keys()

    # Dunder methods
    def __getitem__(self, key: str):
        """Allow dictionary-style access: my_exp['vanadium']"""
        return self._data[key]

    def __setitem__(self, key: str, value: sc.DataArray):
        """Allow dictionary-style setting: my_exp['vanadium'] = data"""
        self._data[key] = value

    def __delitem__(self, key: str):
        """Allow dictionary-style deletion: del my_exp['vanadium']"""
        del self._data[key]

    def __contains__(self, key: str):
        """Allow use of 'in' keyword: 'vanadium' in my_exp"""
        return key in self._data

    def __repr__(self):
        return f"Experiment(name = {self.name}, datasets={list(self._data.keys())})"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        """Iterate over dataset names."""
        return iter(self._data)
