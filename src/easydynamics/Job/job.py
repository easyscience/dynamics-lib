from easyscience.job.job import JobBase

from easydynamics.sample import SampleModel
from easydynamics.experiment import Experiment
from easydynamics.analysis import Analysis
from easydynamics.experiment.data import Data

import scipp as sc
import plopp as pp


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

            if self._background_model is not None:
                this_analysis.set_background_model(self._background_model.copy())
            if self._resolution_model is not None:
                this_analysis.set_resolution_model(self._resolution_model.copy())

            this_experiment=Experiment()
            this_data=Data()
            this_data.append(self._experiment._data.data['Q',i])
            this_experiment.set_data(this_data)

            this_analysis.set_experiment(this_experiment)
            self._analysis.append(this_analysis)

    def plot_data_and_model(self):

        model = sc.zeros_like(self._experiment._data.data)

        # data['Q',i]
        for i in range(len(self._analysis)):
            model['Q',i].values = self._analysis[i].calculate_theory(model['Q',i].coords['energy'].values)


        data_and_fit = sc.DataGroup({'Data': self._experiment._data.data,
                                    'Fit': model})


        INTENSITY_MIN_VANADIUM=0.0
        INTENSITY_MAX_VANADIUM=0.06

        ENERGY_MIN_VANADIUM = -0.02 * sc.Unit('meV')
        ENERGY_MAX_VANADIUM = 0.02 * sc.Unit('meV')
        plot=pp.slicer(data_and_fit['energy',ENERGY_MIN_VANADIUM:ENERGY_MAX_VANADIUM],
                vmin=INTENSITY_MIN_VANADIUM,vmax=INTENSITY_MAX_VANADIUM,
                    keep=['energy'],
            linestyle=         {'Data': 'none',    'Fit': '-'},
            marker=            {'Data': 'o',       'Fit':'none'},
            markerfacecolor=   {'Data': 'none',    'Fit':'red'},
            color=             {'Data': 'black',   'Fit':'red'})

        return plot


    
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


