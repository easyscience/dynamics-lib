from easyscience.job.job import JobBase

from easydynamics.sample import SampleModel
from easydynamics.experiment import Experiment
from easydynamics.analysis import Analysis
from easydynamics.experiment.data import Data


class Job(JobBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.name = name
        self._theory = None
        self._resolution_model = None
        self._background_model = None
        self._experiment = None
        self._analysis = []
        self._summary = None
        self._info = None


    def set_theory(self, theory):
        self._theory = theory

    def set_experiment(self, experiment):
        self._experiment = experiment   


    def set_background_model(self, background:SampleModel):
        """ Set the model for the background.
        Args:
            background (SampleModel): The background model.
        """
        if not isinstance(background, SampleModel):
            raise TypeError("Background model must be an instance of SampleModel.")
        self._background_model = background

    def set_resolution_model(self, resolution:SampleModel):
        """        Set the resolution model for the experiment. The resolution will be normalised to have area 1.
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
        """ Normalize the resolution model to have an area of 1.
        """
        self._resolution_model.normalize_area()        

    def set_analysis(self, analysis):
        self._analysis.append(analysis)
        if self._experiment is not None:
            self._analysis[-1].set_experiment(self._experiment)
        if self._theory is not None:
            self._analysis[-1].set_theory(self._theory)

    def fit(self):
        if self._analysis is None:
            raise RuntimeError("Analysis is not set in Job.")

        for i in range(len(self._analysis)):
            self._analysis[i].fit()
        # return self._analysis.fit()

    def generate_analysis_for_cuts(self):
        for i in range(self._experiment._data.data.sizes['Q']):
            this_analysis=Analysis()
            this_analysis.set_theory(self._theory.copy())

            this_analysis.set_background_model(self._background_model.copy())
            this_analysis.set_resolution_model(self._resolution_model.copy())

            this_experiment=Experiment()
            this_data=Data()
            this_data.append(self._experiment._data.data['Q',i])
            this_experiment.set_data(this_data)

            this_analysis.set_experiment(this_experiment)
            self._analysis.append(this_analysis)


    
    @property
    def analysis(self):
        return self._analysis
    
    def calculate_theory(self, x):
        return self._analysis.calculate_theory(x,_experiment=self._experiment, theory=self._theory)
    
    def experiment(self):
        return self._experiment
    
    def theoretical_model(self):
        return self._theory
    
    def get_fit_parameters(self):
        return self._analysis.get_fit_parameters()
    
    def get_parameters(self):
        return self._analysis.get_parameters()


