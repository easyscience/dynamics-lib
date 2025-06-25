from easyscience.job.job import JobBase


class Job(JobBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.name = name
        self._theory = None
        self._experiment = None
        self._analysis = None
        self._summary = None
        self._info = None

    def set_theory(self, theory):
        self._theory = theory

    def set_experiment(self, experiment):
        self._experiment = experiment   

    def set_analysis(self, analysis):
        self._analysis = analysis
        if self._experiment is not None:
            self._analysis.set_experiment(self._experiment)
        if self._theory is not None:
            self._analysis.set_theory(self._theory)

    def fit(self):
        if self._analysis is None:
            raise RuntimeError("Analysis is not set in Job.")
        return self._analysis.fit()    
    
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


