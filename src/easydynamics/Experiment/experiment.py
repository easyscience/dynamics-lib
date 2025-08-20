
from easyscience.job.experiment import ExperimentBase

from easydynamics.experiment .data import Data

import numpy as np
import scipp as sc

class Experiment(ExperimentBase):

    def __init__ (self,name="MyExperiment"):
        """
        Initialize the Experiment class.
        """
        super().__init__(name)
        self._data = None

    def set_data(self, data: Data):
        """ Set the experimental data.
        Args:
            data (Data): The experimental data to be used in the analysis.
        """
        
        if not isinstance(data, Data):
            raise TypeError("Data must be an instance of Data.")
        self._data = data

    def extract_xye_data(self, data):
        """
        Extract x, y, and e data from the experiment.
        
        Returns:
            tuple: A tuple containing x, y, and e data.
        """

        if isinstance(data, Data):
            data = data.get_data()

        if isinstance(data, sc.DataArray):
            x = data.coords['energy'].values
            y = data.values
            e = np.sqrt(data.variances)

        return x, y, e
