
from easyscience.job.experiment import ExperimentBase

from easyscience.variable import Parameter

from easydynamics.experiment .data import Data
from easydynamics.sample import SampleModel

import numpy as np
import scipp as sc

class Experiment(ExperimentBase):

    def __init__ (self,name):
        """
        Initialize the Experiment class.
        """
        super().__init__(name)
        self._data = None
        self._resolution_model = None
        self._background_model = None
        self.offset=Parameter(name='offset', value=0.0, unit='meV')

    def set_background_model(self, background:SampleModel):
        """ Set the background model for the experiment.
        Args:
            background (SampleModel): The background model to be used in the experiment.
        """
        if not isinstance(background, SampleModel):
            raise TypeError("Background model must be an instance of SampleModel.")
        self._background_model = background

    def set_resolution_model(self, resolution:SampleModel):
        """        Set the resolution model for the experiment.
        Args:
            resolution (SampleModel): The resolution model to be used in the experiment.
        """
        # TODO: allow resolution to be DataArray or SampleModel

        if resolution is not None and not isinstance(resolution, SampleModel):
            raise TypeError("Resolution model must be an instance of SampleModel.")
        self._resolution_model = resolution

        if self._resolution_model is not None:
            self.normalize_resolution()

    def normalize_resolution(self):

        self._resolution_model.normalize_area()

    def set_data(self, data: Data):
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


    def set_offset(self, offset: float):
        # TODO: handle units properly
        
        self.offset.value= offset

    def fix_offset(self, fix: bool = True):
    
        self.offset.fixed = fix