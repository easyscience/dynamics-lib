from easyscience.job.job import JobBase


class Job(JobBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.Experiment= None
        self.Analysis = None
        self.SampleModel = None

