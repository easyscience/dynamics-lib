
from easyscience.job.experiment import ExperimentBase

from easydynamics.Experiment import Data
from easydynamics.sample import SampleModel


class Experiment(ExperimentBase):

    def __init__ (self):
        """
        Initialize the Experiment class.
        """
        super().__init__()
        self.data = Data()
        self.ResolutionModel = None
        self.BackgroundModel = None

    def set_background_model(self, background:SampleModel):
        """ Set the background model for the experiment.
        Args:
            background (SampleModel): The background model to be used in the experiment.
        """
        # TODO: handle offset more elegantly
        if not isinstance(background, SampleModel):
            raise TypeError("Background model must be an instance of SampleModel.")
        self.BackgroundModel = background
        self.BackgroundModel.offset.value = 0.0  # Ensure sample model has an offset of 0
        self.BackgroundModel.fix_offset(True)  # Fix the offset to avoid fitting it

    def set_resolution_model(self, resolution:SampleModel):
        """        Set the resolution model for the experiment.
        Args:
            resolution (SampleModel): The resolution model to be used in the experiment.
        """
        # TODO: handle offset more elegantly
        # TODO: allow resolution to be DataArray or SampleModel

        if not isinstance(resolution, SampleModel):
            raise TypeError("Resolution model must be an instance of SampleModel.")
        self.ResolutionModel = resolution
        self.ResolutionModel.offset.value = 0.0  # Ensure resolution model has an offset of 0
        self.ResolutionModel.fix_offset(True)  # Fix the offset to avoid fitting it

    