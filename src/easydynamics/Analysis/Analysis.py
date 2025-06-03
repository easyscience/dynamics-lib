from easyscience.Objects.job.analysis import AnalysisBase
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import FitResults
from easyscience.fitting.multi_fitter import MultiFitter as EasyScienceMultiFitter

from easyscience.Objects.variable import Parameter

from easydynamics.resolution import ResolutionHandler

import numpy as np

import scipp as sc


class Analysis(AnalysisBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._SampleModel= None
        self._ResolutionModel = None
        self._BackgroundModel = None

    def calculate_theory(self,
                        x: np.ndarray) -> np.ndarray:
        """
        Calculate the theoretical model by convolving the sample model with the resolution model
        and adding the background model.
        """
        MyResolutionHandler=ResolutionHandler()
        y=MyResolutionHandler.convolve(x, self._SampleModel, self._ResolutionModel)+ self._BackgroundModel.evaluate(x)
        return y
    
    def fit(self):

        x, y, e = self._data

        # Define the function to be minimized
        def fit_func(x_vals):
            return self.calculate_theory(x_vals)

        # Wrap into EasyScience MultiFitter
        multi_fitter = EasyScienceMultiFitter(
            fit_objects=[self],
            fit_functions=[fit_func],
        )

# quad = BaseObj(name='quad', a=a, b=b, c=c)
# f = Fitter(quad, math_model)

        # Perform the fit
        fit_result = multi_fitter.fit(x=[x], y=[y], weights=[1.0 / e])

        # Store result
        self.fit_result = fit_result

        return fit_result

    def switch_minimizer(self, minimizer: AvailableMinimizers) -> None:
        """
        Switch the minimizer for the fitting.

        :param minimizer: Minimizer to be switched to
        """
        self.easy_science_multi_fitter.switch_minimizer(minimizer)

        
    def set_sample_model(self, sample_model):
        """
        Set the sample model for the analysis.

        Args:
            sample_model (SampleModel): The sample model to be used in the analysis.
        """
        self._SampleModel = sample_model

    def set_resolution_model(self, resolution_model):
        """
        Set the resolution model for the analysis.

        Args:
            resolution_model (SampleModel): The resolution model to be used in the analysis.
        """
        self._ResolutionModel = resolution_model 

    def set_background_model(self, background_model):   
        """
        Set the background model for the analysis.

        Args:
            background_model (SampleModel): The background model to be used in the analysis.
        """
        self._BackgroundModel = background_model

    def set_data(self,data):
        """
        Set the data for the analysis.

        Args:
            data (scipp DataArray or np.ndarray): The data to be used in the analysis.
        """
        #TODO: handle data properly
        if isinstance(data, sc.DataArray):
            x= data.coords['energy'].values
            y = data.values
            e= np.sqrt(data.variances) 

            data=[x, y, e]
            
        self._data = data

    def get_data(self):
        """
        Get the data used in the analysis.

        Returns:
            data (scipp DataArray or np.ndarray): The data used in the analysis.
        """
        return self._data

 
    def get_fit_parameters(self):
        def collect_parameters(obj):
            found = []
            if isinstance(obj, Parameter):
                found.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    found.extend(collect_parameters(v))
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    found.extend(collect_parameters(item))
            elif hasattr(obj, '__dict__'):
                for v in vars(obj).values():
                    found.extend(collect_parameters(v))
            return found

        params = []
        for model in [self._SampleModel, self._ResolutionModel, self._BackgroundModel]:
            if model is not None:
                for comp in model.components:
                    params.extend(collect_parameters(comp))
        return params
            


