from easyscience.job.experiment import ExperimentBase


import numpy as np
import scipp as sc


class Experiment(ExperimentBase):
    def __init__(self, name="MyExperiment"):
        """
        Initialize the Experiment class.
        """
        super().__init__(name)
        self._data = None
