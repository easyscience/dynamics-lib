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
    

    def fit(self):
        if self._analysis is None:
            raise RuntimeError("Analysis is not set in Job.")
        return self._analysis.fit(self._experiment,self._theory)    

    # 'analysis', 'calculate_theory', 'experiment', 'fit', 'theoretical_model'