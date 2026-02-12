import os

import plopp as pp
import scipp as sc
from easyscience.base_classes.new_base import NewBase
from scipp.io import load_hdf5 as sc_load_hdf5
from scipp.io import save_hdf5 as sc_save_hdf5

from easydynamics.utils.utils import _in_notebook


class Experiment(NewBase):
    """Holds data from an experiment as a sc.DataArray along with
    metadata.

    This is a minimal implementation that will be extended in the
    future.
    """

    def __init__(
        self,
        display_name: str = 'MyExperiment',
        unique_name: str | None = None,
        data: sc.DataArray | str | None = None,
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        if data is None:
            self._data = None
        elif isinstance(data, str):
            self.load_hdf5(filename=data)
        elif isinstance(data, sc.DataArray):
            self._validate_coordinates(data)
            self._data = data
        else:
            raise TypeError(
                f'Data must be a sc.DataArray or a filename string, not {type(data).__name__}'
            )

        self._binned_data = (
            self._convert_to_bin_centers(self._data) if self._data is not None else None
        )

    ###########
    # Properties
    ###########

    @property
    def data(self) -> sc.DataArray | None:
        """Get the dataset associated with this experiment."""
        return self._data

    @data.setter
    def data(self, value: sc.DataArray) -> None:
        """Set the dataset associated with this experiment."""
        if not isinstance(value, sc.DataArray):
            raise TypeError(f'Data must be a sc.DataArray, not {type(value).__name__}')
        self._validate_coordinates(value)
        self._data = value
        self._binned_data = (
            self._convert_to_bin_centers(self._data) if self._data is not None else None
        )

    @property
    def binned_data(self) -> sc.DataArray | None:
        """Get the binned dataset associated with this experiment."""
        return self._binned_data

    @binned_data.setter
    def binned_data(self, value: sc.DataArray) -> None:
        """Set the binned dataset associated with this experiment."""
        raise AttributeError('binned_data is a read-only property. Use rebin() to rebin the data')

    @property
    def Q(self) -> sc.Variable | None:
        """Get the Q values from the dataset."""
        if self._data is None:
            return None
        return self._binned_data.coords['Q']

    @Q.setter
    def Q(self, value: sc.Variable) -> None:
        """Set the Q values for the dataset."""
        raise AttributeError('Q is a read-only property derived from the data.')

    @property
    def energy(self) -> sc.Variable | None:
        """Get the energy values from the dataset."""
        if self._data is None:
            return None
        return self._binned_data.coords['energy']

    @energy.setter
    def energy(self, value: sc.Variable) -> None:
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
            self.display_name = display_name

        loaded_data = sc_load_hdf5(filename)
        if not isinstance(loaded_data, sc.DataArray):
            raise TypeError(
                f'Loaded data must be a sc.DataArray, not {type(loaded_data).__name__}'
            )
        self._validate_coordinates(loaded_data)
        self.data = loaded_data

    def save_hdf5(self, filename: str | None = None):
        """Save the dataset to HDF5.

        Args:
            filename (str | None): Path to the output HDF5 file.
        """

        if filename is None:
            filename = f'{self.unique_name}.h5'

        if not isinstance(filename, str):
            raise TypeError(f'Filename must be a string, not {type(filename).__name__}')

        if self._data is None:
            raise ValueError('No data to save.')

        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        sc_save_hdf5(self._data, filename)

    def remove_data(self):
        """Remove the dataset from the experiment."""
        self._data = None
        self._binned_data = None

    def rebin(self, dimensions: dict[str, int | sc.Variable]) -> None:
        """Rebin the dataset along specified dimensions.

        Args:
            dimensions (dict[str, int | sc.Variable]): A dictionary
            mapping dimension names to number of bins (int) or bin edges
            (sc.Variable).
        Raises:
            TypeError: If dimensions is not a dictionary or if
            keys/values are of incorrect types. KeyError: If a specified
            dimension is not in the dataset.
        """

        if not isinstance(dimensions, dict):
            raise TypeError(
                'dimensions must be a dictionary mapping dimension names '
                'to number of bins or bin values as sc.Variable.'
            )
        if self._data is None:
            raise ValueError('No data to rebin. Please load data first.')
        binned_data = self._data.copy()
        dim_copy = dimensions.copy()
        for dim, value in dim_copy.items():
            if not isinstance(dim, str):
                raise TypeError(
                    f'Dimension keys must be strings. Got {type(dim)} for {dim} instead.'
                )
            if dim not in self._data.dims:
                raise KeyError(
                    f"Dimension '{dim}' not a valid dimension for rebinning. "
                    f'Should be one of {self._data.dims}.'
                )
            if isinstance(value, float) and value.is_integer():  # I allow eg. 2.0 as well as 2
                value = int(value)
                # This line can be removed when scipp resize support
                # resizing with coordinates
                dimensions[dim] = value
            if not (isinstance(value, int) or isinstance(value, sc.Variable)):
                raise TypeError(
                    f'Dimension values must be integers or sc.Variable. '
                    f"Got {type(value)} for dimension '{dim}' instead."
                )
            binned_data = binned_data.bin({dim: value})

        binned_data = binned_data.bins.mean()
        binned_data = self._convert_to_bin_centers(binned_data)
        self._binned_data = binned_data

    ###########
    # other methods
    ###########

    def plot_data(self, slicer=False, **kwargs) -> None:
        """Plot the dataset using plopp."""

        if self._binned_data is None:
            raise ValueError('No data to plot. Please load data first.')

        if not _in_notebook():
            raise RuntimeError('plot_data() can only be used in a Jupyter notebook environment.')

        from IPython.display import display

        plot_kwargs_defaults = {
            'title': self.display_name,
        }
        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)
        if slicer:
            fig = pp.slicer(
                self._binned_data,
                **plot_kwargs_defaults,
            )
        else:
            fig = pp.plot(
                self._binned_data.transpose(dims=['energy', 'Q']),
                **plot_kwargs_defaults,
            )
        display(fig)

    ###########
    # private methods
    ###########

    @staticmethod
    def _validate_coordinates(data: sc.DataArray) -> None:
        """Validate that required coordinates are present in the data.

        Raises:
            ValueError: If required coordinates are missing.
        """
        if not isinstance(data, sc.DataArray):
            raise TypeError('Data must be a sc.DataArray.')

        required_coords = ['Q', 'energy']
        for coord in required_coords:
            if coord not in data.coords:
                raise ValueError(f"Data is missing required coordinate: '{coord}'")

    def _convert_to_bin_centers(self, data: sc.DataArray) -> sc.DataArray:
        """Convert the coordinates of the data to bin centers.

        Args:
            data (sc.DataArray): The data to check.

        Returns:
            sc.DataArray: The data with coordinates at bin centers.
        """
        for dim in data.dims:
            coord = data.coords[dim]
            if coord.ndim == 1 and coord.size == data.sizes[dim] + 1:
                # Coordinate is at bin edges, convert to bin centers
                data = data.assign_coords({dim: sc.midpoints(coord)})
        return data

    ########
    # dunder methods
    ###########

    def __repr__(self) -> str:
        return f'Experiment `{self.unique_name}` with data: {self._data}'

    def __copy__(self) -> 'Experiment':
        """Return a copy of the object."""
        temp = self.to_dict(skip=['unique_name'])
        new_obj = self.__class__.from_dict(temp)
        new_obj.data = self.data.copy() if self.data is not None else None
        return new_obj
