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
        super().__init__(name, *args, **kwargs)

        if data is None:
            self._data: Optional[sc.DataArray] = None
        elif isinstance(data, str):
            self.load_hdf5(filename=data)
        elif isinstance(data, sc.DataArray):
            self._data = data
        else:
            raise TypeError(
                f"Data must be a sc.DataArray or a filename string, not {type(data).__name__}"
            )

    def load_hdf5(self, filename: str, name: Optional[str] = None):
        """
        Load data from an HDF5 file.

        Args:
            filename (str): Path to the HDF5 file.
        """
        if not isinstance(filename, str):
            raise TypeError(f"Filename must be a string, not {type(filename).__name__}")

        if name is not None:
            if not isinstance(name, str):
                raise TypeError(f"Name must be a string, not {type(name).__name__}")
            self.name = name

        # TODO: Add checks of dimensions etc. I'm not yet sure what dimensions I want to allow, so for now I trust myself.
        loaded_data = sc_load_hdf5(filename)
        if not isinstance(loaded_data, sc.DataArray):
            raise TypeError(
                f"Loaded data must be a sc.DataArray, not {type(loaded_data).__name__}"
            )
        self._data = loaded_data

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

        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        sc_save_hdf5(self._data, filename)

    def remove_data(self):
        """Remove the dataset from the experiment."""
        self._data = None

    @property
    def data(self) -> Optional[sc.DataArray]:
        """Get the dataset associated with this experiment."""
        return self._data

    @data.setter
    def data(self, value: sc.DataArray):
        """Set the dataset associated with this experiment."""
        if not isinstance(value, sc.DataArray):
            raise TypeError(f"Data must be a sc.DataArray, not {type(value).__name__}")
        self._data = value

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

    def __repr__(self) -> str:
        return f"Experiment `{self.name}` with data: {self._data}"

    def __copy__(self) -> "Experiment":
        """Return a copy of the object."""
        temp = self.as_dict(skip=["unique_name"])
        new_obj = self.__class__.from_dict(temp)
        new_obj.data = self.data.copy() if self.data is not None else None
        return new_obj
