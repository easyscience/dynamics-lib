# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from pathlib import Path

import numpy as np
import plopp as pp
import scipp as sc
from plopp.backends.matplotlib.figure import InteractiveFigure
from scipp.io import load_hdf5 as sc_load_hdf5
from scipp.io import save_hdf5 as sc_save_hdf5

from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.utils.utils import _in_notebook
from easydynamics.utils.utils import verify_Q_index


class Experiment(EasyDynamicsBase):
    """
    Holds data from an experiment as a sc.DataArray along with metadata.

    This is a minimal implementation that will be extended in the future.

    Examples
    --------
    **Loading experimental data from the repository**

    Use ``pooch`` to download the example vanadium dataset directly from the repository:
    ```python
    import pooch
    import easydynamics as edyn

    file_path = pooch.retrieve(
        url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/master/docs/docs/tutorials/data/vanadium_data_example.h5',
        known_hash='16cc1b327c303feeb88fb9dda5390dc4880b62396b1793f98c6fef0b27c7b873',
    )

    experiment = edyn.Experiment(display_name='Vanadium')
    experiment.load_hdf5(filename=file_path)
    ```

    **Rebinning and plotting**

    After loading, rebin to reduce the number of bins along each dimension before plotting:
    ```python
    experiment.rebin({'Q': 5, 'energy': 50})
    experiment.plot_data()
    ```
    """

    def __init__(
        self,
        display_name: str | None = 'MyExperiment',
        unique_name: str | None = None,
        data: sc.DataArray | str | None = None,
    ) -> None:
        """
        Initialize the Experiment object.

        Parameters
        ----------
        display_name : str | None, default='MyExperiment'
            Display name of the experiment.
        unique_name : str | None, default=None
            Unique name of the experiment. If None, a unique name will be generated. None.
        data : sc.DataArray | str | None, default=None
            Dataset associated with the experiment. Can be a sc.DataArray or a filename string to
            load from. If None, no data is loaded.

        Raises
        ------
        TypeError
            If data is not a sc.DataArray, a string, or None.
        """
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
        """
        Get the dataset associated with this experiment.

        Returns
        -------
        sc.DataArray | None
            The dataset associated with this experiment, or None if no data is loaded.
        """
        return self._data

    @data.setter
    def data(self, value: sc.DataArray) -> None:
        """
        Set the dataset associated with this experiment.

        Parameters
        ----------
        value : sc.DataArray
            The new dataset to associate with this experiment.

        Raises
        ------
        TypeError
            If the value is not a sc.DataArray.
        """
        if not isinstance(value, sc.DataArray):
            raise TypeError(f'Data must be a sc.DataArray, not {type(value).__name__}')
        self._validate_coordinates(value)
        self._data = value
        self._binned_data = (
            self._convert_to_bin_centers(self._data) if self._data is not None else None
        )

    @property
    def binned_data(self) -> sc.DataArray | None:
        """
        Get the binned dataset associated with this experiment.

        Returns
        -------
        sc.DataArray | None
            The binned dataset associated with this experiment, or None if no data is loaded.
        """
        return self._binned_data

    @binned_data.setter
    def binned_data(self, _value: sc.DataArray) -> None:
        """
        Set the binned dataset associated with this experiment.

        Read- only property. Use rebin() to rebin the data instead.

        Parameters
        ----------
        _value : sc.DataArray
            The new binned dataset to associate with this experiment (ignored).

        Raises
        ------
        AttributeError
            Always, since binned_data is read-only.
        """
        raise AttributeError('binned_data is a read-only property. Use rebin() to rebin the data')

    @property
    def Q(self) -> sc.Variable | None:
        """
        Get the Q values from the dataset.

        Returns
        -------
        sc.Variable | None
            The Q values from the dataset, or None if no data is loaded.
        """
        if self.binned_data is None:
            return None
        return self.binned_data.coords['Q']

    @Q.setter
    def Q(self, _value: sc.Variable) -> None:
        """
        Set the Q values for the dataset.

        Q is a read-only property derived from the data, so this setter raises an error.

        Parameters
        ----------
        _value : sc.Variable
            The new Q values to set (ignored).

        Raises
        ------
        AttributeError
            Always, since Q is read-only.
        """
        raise AttributeError('Q is a read-only property derived from the data.')

    @property
    def energy(self) -> sc.Variable | None:
        """
        Get the energy values from the dataset.

        Returns
        -------
        sc.Variable | None
            The energy values from the dataset, or None if no data is loaded.
        """
        if self.binned_data is None:
            return None
        return self.binned_data.coords['energy']

    @energy.setter
    def energy(self, _value: sc.Variable) -> None:
        """
        Set the energy values for the dataset.

        Energy is a read-only property derived from the data, so this setter raises an error.

        Parameters
        ----------
        _value : sc.Variable
            The new energy values to set (ignored).

        Raises
        ------
        AttributeError
            Always, since energy is read-only.
        """
        raise AttributeError('energy is a read-only property derived from the data.')

    def get_masked_energy(
        self, Q_index: int, mask: sc.Variable | None = None
    ) -> sc.Variable | None:
        """
        Get the energy values from the dataset, removing points where the y values or variances are
        NaN or Inf for the given Q index.

        Parameters
        ----------
        Q_index : int
            The Q index to get the masked energy values for.
        mask : sc.Variable | None, default=None
            Optional precomputed finite-energy mask (as returned by
            :meth:`get_finite_energy_mask`), so callers that already extracted the data do not pay
            for a second extraction. If None, the mask is computed.

        Returns
        -------
        sc.Variable | None
            The masked energy values from the dataset, or None if no data is loaded.
        """
        if mask is None:
            mask = self.get_finite_energy_mask(Q_index=Q_index)
        if mask is None:
            return None

        return self.binned_data.coords['energy'][mask]

    def get_finite_energy_mask(self, Q_index: int) -> sc.Variable | None:
        """
        Get a boolean scipp Variable selecting energy points with finite intensity at the given Q.

        Parameters
        ----------
        Q_index : int
            The Q index to get the mask for.

        Returns
        -------
        sc.Variable | None
            Boolean scipp Variable of length n_energy with dim ``'energy'``, or None if no data is
            loaded.
        """
        if self.binned_data is None:
            return None

        verify_Q_index(Q_index, self.Q)

        _, _, _, mask = self.extract_x_y_weights_only_finite(Q_index=Q_index)
        return sc.array(dims=['energy'], values=mask)

    def get_masked_binned_data(self, Q_index: int) -> sc.DataArray | None:
        """
        Get the binned data for a single Q slice with non-finite points masked out.

        Parameters
        ----------
        Q_index : int
            The Q index to extract.

        Returns
        -------
        sc.DataArray | None
            The binned data for the given Q index with NaN/Inf points removed, or None if no data
            is loaded.
        """
        mask_var = self.get_finite_energy_mask(Q_index=Q_index)
        if mask_var is None:
            return None

        return self.binned_data['Q', Q_index][mask_var]

    ###########
    # Handle data
    ###########

    def load_hdf5(self, filename: str, display_name: str | None = None) -> None:
        """
        Load data from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.
        display_name : str | None, default=None
            Optional display name for the experiment.

        Raises
        ------
        TypeError
            If filename is not a string or if display_name is not a string or None or if the loaded
            data is not a sc.DataArray.
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

    def save_hdf5(self, filename: str | None = None) -> None:
        """
        Save the dataset to HDF5.

        Parameters
        ----------
        filename : str | None, default=None
            Path to the output HDF5 file. If None, the file will be named after the unique_name of
            the experiment with a .h5 extension.

        Raises
        ------
        TypeError
            If filename is not a string or None.
        ValueError
            If there is no data to save.
        """

        if filename is None:
            filename = f'{self.unique_name}.h5'

        if not isinstance(filename, str):
            raise TypeError(f'Filename must be a string, not {type(filename).__name__}')

        if self._data is None:
            raise ValueError('No data to save.')

        path = Path(filename)
        path.parent.mkdir(exist_ok=True, parents=True)
        sc_save_hdf5(self._data, filename)

    def remove_data(self) -> None:
        """Remove the dataset from the experiment."""
        self._data = None
        self._binned_data = None

    def rebin(self, dimensions: dict[str, int | sc.Variable]) -> None:
        """
        Rebin the dataset along specified dimensions.

        Parameters
        ----------
        dimensions : dict[str, int | sc.Variable]
            A dictionary mapping dimension names to number of bins (int) or bin edges
            (sc.Variable).

        Raises
        ------
        TypeError
            If dimensions is not a dictionary or if keys/values are of incorrect types.
        ValueError
            If there is no data to rebin.
        KeyError
            If a specified dimension is not in the dataset.
        """

        if not isinstance(dimensions, dict):
            raise TypeError(
                'dimensions must be a dictionary mapping dimension names '
                'to number of bins or bin values as sc.Variable.'
            )
        if self._data is None:
            raise ValueError('No data to rebin. Please load data first.')
        # sc.bin cannot handle multi-dimensional dense data with bin-edge
        # coordinates, so convert them to bin centers first.
        binned_data = self._convert_to_bin_centers(self._data.copy())
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
            if not (isinstance(value, (int, sc.Variable))):
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

    def plot_data(
        self,
        slicer: bool = False,
        transpose_axes: bool = False,
        **kwargs: dict,
    ) -> InteractiveFigure:
        """
        Plot the dataset using plopp: https://scipp.github.io/plopp/.

        Parameters
        ----------
        slicer : bool, default=False
            If True, use plopp's slicer instead of plot.
        transpose_axes : bool, default=False
            If True, transpose the data to have dimensions in the order (energy, Q) before
            plotting, so that energy is on the x-axis. This only applies when slicer=False.
        **kwargs : dict
            Additional keyword arguments to pass to plopp.

        Returns
        -------
        InteractiveFigure
            A plot of the data and model.

        Raises
        ------
        ValueError
            If there is no data to plot.
        RuntimeError
            If not in a Jupyter notebook environment.
        TypeError
            If slicer or transpose_axes are not True or False.
        """

        if self.binned_data is None:
            raise ValueError('No data to plot. Please load data first.')

        if not _in_notebook():
            raise RuntimeError('plot_data() can only be used in a Jupyter notebook environment.')

        if not isinstance(slicer, bool):
            raise TypeError(f'slicer must be True or False, not {type(slicer).__name__}')

        if not isinstance(transpose_axes, bool):
            raise TypeError(
                f'transpose_axes must be True or False, not {type(transpose_axes).__name__}'
            )

        plot_kwargs_defaults = {
            'title': self.display_name,
        }

        if slicer:
            plot_kwargs_defaults['keep'] = 'energy'

        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)
        if slicer:
            fig = pp.slicer(
                self.binned_data,
                **plot_kwargs_defaults,
            )
            for widget in fig.bottom_bar[0].controls.values():
                widget.slider_toggler.value = '-o-'
            fig.autoscale()

        else:
            if transpose_axes:
                data_to_plot = self.binned_data.transpose(dims=['energy', 'Q'])
            else:
                data_to_plot = self.binned_data.transpose(dims=['Q', 'energy'])
            fig = pp.plot(
                data_to_plot,
                **plot_kwargs_defaults,
            )
        return fig

    ###########
    # private methods
    ###########

    @staticmethod
    def _validate_coordinates(data: sc.DataArray) -> None:
        """
        Validate that required coordinates are present in the data.

        Parameters
        ----------
        data : sc.DataArray
            The data to validate.

        Raises
        ------
        TypeError
            If data is not a sc.DataArray.
        ValueError
            If required coordinates are missing.
        """
        if not isinstance(data, sc.DataArray):
            raise TypeError('Data must be a sc.DataArray.')

        required_coords = ['Q', 'energy']
        for coord in required_coords:
            if coord not in data.coords:
                raise ValueError(f"Data is missing required coordinate: '{coord}'")

    def _convert_to_bin_centers(self, data: sc.DataArray) -> sc.DataArray:
        """
        Convert the coordinates of the data to bin centers.

        Parameters
        ----------
        data : sc.DataArray
            The data to convert.

        Returns
        -------
        sc.DataArray
            The data with coordinates at bin centers.
        """
        for dim in data.dims:
            coord = data.coords[dim]
            if coord.ndim == 1 and coord.size == data.sizes[dim] + 1:
                # Coordinate is at bin edges, convert to bin centers
                data = data.assign_coords({dim: sc.midpoints(coord)})
        return data

    def _extract_x_y_var(self, Q_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the x, y, and weights arrays from the experiment for the given Q index.

        Parameters
        ----------
        Q_index : int
            The Q index to extract the data for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The x, y, and variances arrays extracted from the experiment for the given Q index.
        """
        data = self.binned_data['Q', Q_index]
        x = data.coords['energy'].values
        y = data.values
        var = data.variances
        return x, y, var

    @property
    def has_variances(self) -> bool:
        """
        Whether the data carries variances.

        When it does not, :meth:`extract_x_y_weights_only_finite` falls back to all-ones weights,
        which are placeholders for the fit rather than measured uncertainties.

        Returns
        -------
        bool
            True when there is data and it has variances.
        """
        return self._binned_data is not None and self._binned_data.variances is not None

    def extract_x_y_weights_only_finite(
        self, Q_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the x, y, and weights arrays from the experiment for the given Q index, removing
        any NaN and Inf values.

        Parameters
        ----------
        Q_index : int
            The Q index to extract the data for.

        Raises
        ------
        ValueError
            If any variances are zero after removing NaNs and Infs, since this would lead to
            infinite weights.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The x, y, weights, and mask arrays extracted from the experiment for the given Q index,
            with NaNs and Infs removed.
        """
        x, y, var = self._extract_x_y_var(Q_index)

        if var is None:
            mask = np.isfinite(y) & np.isfinite(x)
            # If variances are not provided, set them to 1 to make all
            # weights be 1
            var = np.ones_like(y)
        else:
            mask = np.isfinite(y) & np.isfinite(var) & np.isfinite(x)

        x = x[mask]
        y = y[mask]
        var = var[mask]

        if np.any(var == 0):
            raise ValueError('Cannot compute weights: some variances are zero.')

        weights = 1.0 / var**0.5

        return x, y, weights, mask

    ########
    # dunder methods
    ###########

    def __repr__(self) -> str:
        """
        Return a string representation of the Experiment object.

        Returns
        -------
        str
            A string representation of the Experiment object.
        """

        return f'Experiment `{self.unique_name}` with data: {self._data}'

    def __copy__(self) -> 'Experiment':
        """
        Return a copy of the object.

        Returns
        -------
        'Experiment'
            A copy of the Experiment object.
        """
        temp = self.to_dict(skip=['unique_name'])
        new_obj = self.__class__.from_dict(temp)
        new_obj.data = self.data.copy() if self.data is not None else None
        return new_obj
