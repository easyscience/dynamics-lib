from easyscience.job.experiment import ExperimentBase

import scipp as sc
import plopp as pp


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
        if not isinstance(filename, str):
            raise TypeError(f"Filename must be a string, not {type(filename).__name__}")

        if not isinstance(name, str):
            raise TypeError(f"Dataset name must be a string, not {type(name).__name__}")

        # TODO: Add checks of dimensions etc. I'm not yet sure what dimensions I want to allow, so for now I trust myself.

        self.append_data(sc.io.load_hdf5(filename), name)

    def save_hdf5(self, name: str, filename: str):
        """Save a single dataset to HDF5.

        Args:
            name (str): Name of the dataset to save.
            filename (str): Path to the output HDF5 file.
        """

        if not isinstance(name, str):
            raise TypeError(f"Dataset name must be a string, not {type(name).__name__}")

        if not isinstance(filename, str):
            raise TypeError(f"Filename must be a string, not {type(filename).__name__}")

        if name not in self._data:
            raise KeyError(
                f"No dataset named '{name}' in Experiment {self.name}. "
                f"Available datasets: {list(self._data.keys())}"
            )

        sc.io.save_hdf5(self._data[name], filename)

    def save_all_hdf5(self, folder: str):
        """Save all datasets to individual HDF5 files in a folder.

        Args:
            folder (str): Path to the output folder.
        """
        if not isinstance(folder, str):
            raise TypeError(f"Folder must be a string, not {type(folder).__name__}")

        if not self._data:
            raise ValueError("No data to save. Please load data first.")

        import os

        os.makedirs(folder, exist_ok=True)
        for name, data in self._data.items():
            sc.io.save_hdf5(data, os.path.join(folder, f"{name}.h5"))

    def append_data(self, new_data: sc.DataArray, name: str):
        """Append data with a name."""

        if not isinstance(name, str):
            raise TypeError(f"Dataset name must be a string, not {type(name).__name__}")

        if not isinstance(new_data, sc.DataArray):
            raise TypeError(
                f"Data must be a scipp.DataArray, not {type(new_data).__name__}"
            )
        self._data[name] = new_data

    def get_data(self, name: str = None):
        """Return the stored data. If name is None, return the full dict."""
        if name is None:
            return self._data
        if not isinstance(name, str):
            raise TypeError(f"Dataset name must be a string, not {type(name).__name__}")

        if name not in self._data:
            raise KeyError(
                f"No dataset named '{name}' in Experiment {self.name}. "
                f"Available datasets: {list(self._data.keys())}"
            )
        return self._data[name]

    def remove_all_data(self):
        """Remove all stored data."""
        self._data = {}

    def plot_data(self, name: str = None):
        """Plot all datasets."""
        if not self._data:
            raise ValueError("No data to plot. Please load data first.")

        if not self._in_notebook():
            raise RuntimeError(
                "plot_data() can only be used in a Jupyter notebook environment."
            )

        from IPython.display import display

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

    @staticmethod
    def _in_notebook():
        try:
            from IPython import get_ipython

            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or JupyterLab
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal IPython
            else:
                return False
        except (NameError, ImportError):
            return False  # Standard Python (no IPython)

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
        if key not in self._data:
            raise KeyError(
                f"No dataset named '{key}' in Experiment {self.name}. "
                f"Available datasets: {list(self._data.keys())}"
            )
        return self._data[key]

    def __setitem__(self, key: str, value: sc.DataArray):
        """Allow dictionary-style setting: my_exp['vanadium'] = data"""
        if not isinstance(key, str):
            raise TypeError(f"Dataset name must be a string, not {type(key).__name__}")
        if not isinstance(value, sc.DataArray):
            raise TypeError(
                f"Value must be a scipp.DataArray, not {type(value).__name__}"
            )
        if key in self._data:
            raise ValueError(
                f"Dataset '{key}' already exists. "
                f"Use a different name or remove it first."
            )
        self._data[key] = value

    def __delitem__(self, key: str):
        """Allow dictionary-style deletion: del my_exp['vanadium']"""
        if key not in self._data:
            raise KeyError(
                f"No dataset named '{key}' in Experiment {self.name}. "
                f"Available datasets: {list(self._data.keys())}"
            )
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
