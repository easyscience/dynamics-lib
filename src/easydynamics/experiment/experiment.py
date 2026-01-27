from typing import Optional

import plopp as pp
import scipp as sc

# from easyscience.job.experiment import ExperimentBase
from easyscience.base_classes.new_base import NewBase
from scipp.io import load_hdf5 as sc_load_hdf5
from scipp.io import save_hdf5 as sc_save_hdf5


class Experiment(NewBase):
    """Holds data from an experiment as a sc.DataArray along with
    metadata.
    """

    def __init__(
        self,
        display_name: str = 'MyExperiment',
        unique_name: str | None = None,
        data: sc.DataArray | str | None = None,
    ):
        super().__init__(display_name, unique_name=unique_name)

        if data is None:
            self._data: Optional[sc.DataArray] = None
        elif isinstance(data, str):
            self.load_hdf5(filename=data)
        elif isinstance(data, sc.DataArray):
            self._data = data
        else:
            raise TypeError(
                f'Data must be a sc.DataArray or a filename string, not {type(data).__name__}'
            )

    ###########
    # Properties
    ###########

    @property
    def data(self) -> sc.DataArray | None:
        """Get the dataset associated with this experiment."""
        return self._data

    @data.setter
    def data(self, value: sc.DataArray):
        """Set the dataset associated with this experiment."""
        if not isinstance(value, sc.DataArray):
            raise TypeError(f'Data must be a sc.DataArray, not {type(value).__name__}')
        self._data = value

    @property
    def Q(self) -> sc.Variable:
        """Get the Q values from the dataset."""
        if self._data is None:
            Warning('No data loaded.', UserWarning)
        if 'Q' not in self._data.coords:
            raise ValueError("Data does not contain 'Q' coordinate.")
        return self._data.coords['Q']

    @Q.setter
    def Q(self, value: sc.Variable):
        """Set the Q values for the dataset."""
        raise AttributeError('Q is a read-only property derived from the data.')

    property

    def energy(self) -> sc.Variable:
        """Get the energy values from the dataset."""
        if self._data is None:
            Warning('No data loaded.', UserWarning)
        if 'energy' not in self._data.coords:
            raise ValueError("Data does not contain 'energy' coordinate.")
        return self._data.coords['energy']

    @energy.setter
    def energy(self, value: sc.Variable):
        """Set the energy values for the dataset."""
        raise AttributeError('energy is a read-only property derived from the data.')

    ###########
    # Handle data
    ###########

    def load_hdf5(self, filename: str, display_name: str | None = None):
        """Load data from an HDF5 file.

        Args:
            filename (str ): Path to the HDF5 file.
            display_name (str | None): Optional display name for the
            experiment.
        """
        if not isinstance(filename, str):
            raise TypeError(f'Filename must be a string, not {type(filename).__name__}')

        if display_name is not None:
            if not isinstance(display_name, str):
                raise TypeError(
                    f'Display name must be a string, not {type(display_name).__name__}'
                )
            self.name = display_name

        # TODO: Add checks of dimensions etc.
        # I'm not yet sure what dimensions I want to allow,
        # so for now I trust that the data is valid.
        loaded_data = sc_load_hdf5(filename)
        if not isinstance(loaded_data, sc.DataArray):
            raise TypeError(
                f'Loaded data must be a sc.DataArray, not {type(loaded_data).__name__}'
            )
        self._data = loaded_data

    def save_hdf5(self, filename: str | None = None):
        """Save the dataset to HDF5.

        Args:
            filename (str | None): Path to the output HDF5 file.
        """

        if filename is None:
            filename = f'{self.name}.h5'

        if not isinstance(filename, str):
            raise TypeError(f'Filename must be a string, not {type(filename).__name__}')

        if self._data is None:
            raise ValueError('No data to save.')

        import os

        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        sc_save_hdf5(self._data, filename)

    def remove_data(self):
        """Remove the dataset from the experiment."""
        self._data = None

    def rebin(self, dim: str, bins: sc.Variable):
        #   def rebin(self, dimensions: dict[str, Numeric]) -> None:
        # """
        raise NotImplementedError('Binning not yet implemented.')

    ###########
    # other methods
    ###########

    def plot_data(self):
        """Plot the dataset using plopp."""

        if self._data is None:
            raise ValueError('No data to plot. Please load data first.')

        if not self._in_notebook():
            raise RuntimeError('plot_data() can only be used in a Jupyter notebook environment.')

        from IPython.display import display

        fig = pp.plot(self._data.transpose(), title=f'{self.name}')
        display(fig)

    ###########
    # private methods
    ###########

    @staticmethod
    def _in_notebook():
        try:
            from IPython import get_ipython

            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or JupyterLab
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal IPython
            else:
                return False
        except (NameError, ImportError):
            return False  # Standard Python (no IPython)

    def _validate_coordinates(self):
        """Validate that required coordinates are present in the data.

        Raises:
            ValueError: If required coordinates are missing.
        """
        if self._data is None:
            raise ValueError('No data loaded to validate.')

        required_coords = ['Q', 'energy']
        for coord in required_coords:
            if coord not in self._data.coords:
                raise ValueError(f"Data is missing required coordinate: '{coord}'")

    ########
    # dunder methods
    ###########

    def __repr__(self) -> str:
        return f'Experiment `{self.name}` with data: {self._data}'

    def __copy__(self) -> 'Experiment':
        """Return a copy of the object."""
        temp = self.as_dict(skip=['unique_name'])
        new_obj = self.__class__.from_dict(temp)
        new_obj.data = self.data.copy() if self.data is not None else None
        return new_obj
