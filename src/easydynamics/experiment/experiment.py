from typing import Optional, Union

import plopp as pp
import scipp as sc
from easyscience.job.experiment import ExperimentBase
from scipp.io import load_hdf5 as sc_load_hdf5
from scipp.io import save_hdf5 as sc_save_hdf5


class Experiment(ExperimentBase):
    """
    Holds data from an experiment as a sc.DataArray along with metadata.
    """

    def __init__(
        self,
        name: str,
        data: Optional[Union[sc.DataArray, str]] = None,
        *args,
        **kwargs,
    ):
        super(Experiment, self).__init__(name, *args, **kwargs)

        if isinstance(data, str):
            self.load_hdf5(filename=data)

        elif isinstance(data, sc.DataArray):
            self._data = data

    def load_hdf5(self, filename: str):
        """
        Load data from an HDF5 file.

        Args:
            file_path (str): Path to the data file.
            name (str): Name to assign to the loaded dataset.
        """
        if not isinstance(filename, str):
            raise TypeError(f"Filename must be a string, not {type(filename).__name__}")

        # TODO: Add checks of dimensions etc. I'm not yet sure what dimensions I want to allow, so for now I trust myself.

        self._data = sc_load_hdf5(filename)

    def save_hdf5(self, filename: Optional[str] = None):
        """Save the dataset to HDF5.

        Args:
            filename (str): Path to the output HDF5 file.
        """

        if filename is None:
            filename = f"{self.name}.h5"

        if not isinstance(filename, str):
            raise TypeError(f"Filename must be a string, not {type(filename).__name__}")

        if self._data is None:
            raise ValueError("No data to save.")

        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        sc_save_hdf5(self._data, filename)

    def plot_data(self):
        """Plot the dataset using plopp."""

        if self._data is None:
            raise ValueError("No data to plot. Please load data first.")

        if not self._in_notebook():
            raise RuntimeError(
                "plot_data() can only be used in a Jupyter notebook environment."
            )

        from IPython.display import display

        fig = pp.plot(self._data.transpose(), title=f"{self.name}")
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
