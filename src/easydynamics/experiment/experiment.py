from easyscience.job.experiment import ExperimentBase

import scipp as sc


class Experiment(ExperimentBase):
    def __init__(self, name="MyExperiment"):
        """
        Initialize the Experiment class.
        """
        super().__init__(name)
        self._data = None

    def load_csv(self, filename: str):
        """
        Load data from a CSV file.

        Args:
            file_path (str): Path to the data file.
        """
        self.append_data(sc.io.load_csv(filename))

        # TODO: Add checks of dimensions etc.

    def load_hdf5(self, filename: str):
        """
        Load data from an HDF5 file.

        Args:
            file_path (str): Path to the data file.
        """
        self.append_data(sc.io.load_hdf5(filename))

        # TODO: Add checks of dimensions etc.

    def append_data(self, new_data: sc.DataArray):
        """
        Append new data to the existing data.

        Args:
            new_data (sc.DataArray): New data to append.
        """
        if self._data is None:
            self._data = [new_data]
        else:
            self._data.append(new_data)

    def get_data(self):
        """
        Get the stored data.

        Returns:
            : The experimental data.
        """
        return self._data

    def remove_all_data(self):
        """
        Remove the stored data.
        """
        self._data = None

    def remove_outliers(self):
        """
        Remove outliers from the data.

        Placeholder.

        """
        raise NotImplementedError("Outlier removal is not implemented yet.")

    def plot_data(self):
        """ "
        Plot the data
        """

        if self._data is None:
            raise ValueError("No data to plot. Please load data first.")

        for i, data in enumerate(self._data):
            sc.plot(data, title=f"Data {i + 1}")
