from easyscience.job.analysis import AnalysisBase
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import FitResults
from easyscience.fitting.fitter import Fitter as EasyScienceFitter

from easyscience.variable import Parameter

from easydynamics.resolution import ResolutionHandler

import numpy as np

import scipp as sc

import matplotlib.pyplot as plt

class Analysis(AnalysisBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._SampleModel= None
        self._ResolutionModel = None
        self._BackgroundModel = None


    def plot_data_and_model(self,plot_individual_components=False):
        """
        Plot the data and the fit result.
        """



        # Plotting using matplotlib


        fig= plt.figure(figsize=(10, 6))
        x, y, e = self._data
        plt.errorbar(x, y, yerr=e, label='Data', color='black', marker='o', linestyle='None',markerfacecolor='none')


        fit_y = self.calculate_theory(x)
        plt.plot(x, fit_y, label='Fit', color='red')

        if plot_individual_components:
            # Plot individual components of the sample model. Need to handle resolution
            for comp in self._SampleModel.components.values():
                comp_y = comp.evaluate(x-self._SampleModel.offset.value)
                plt.plot(x, comp_y, label=f'Component: {comp.name}', linestyle='--')


        plt.xlabel('Energy (meV)') #TODO: Handle units properly
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
        return fig
        


    def calculate_theory(self,
                        x: np.ndarray) -> np.ndarray:
        """
        Calculate the theoretical model by convolving the sample model with the resolution model
        and adding the background model.
        """
        
        if self._ResolutionModel is None:
            y= self._SampleModel.evaluate(x)
        else:
            MyResolutionHandler=ResolutionHandler()
            y= MyResolutionHandler.numerical_convolve(x, self._SampleModel, self._ResolutionModel)

        if self._BackgroundModel is not None:
            y += self._BackgroundModel.evaluate(x)

        return y
    
    def fit(self):

        x, y, e = self._data

        def fit_func(x_vals):
            return self.calculate_theory(x_vals)

        # multi_fitter = EasyScienceMultiFitter(
        #     fit_objects=[self],
        #     fit_functions=[fit_func],
        # )


        # # Perform the fit
        # fit_result = multi_fitter.fit(x=[x], y=[y], weights=[1.0 / e])


        fitter = EasyScienceFitter(
        fit_object=self,
        fit_function=fit_func,
        )


        # Perform the fit
        fit_result = fitter.fit(x=x, y=y, weights=1.0 / e)

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
        #TODO: handle offset more elegantly
        self._SampleModel = sample_model

    def set_resolution_model(self, resolution_model):
        raise NotImplementedError ("Resolution model setting is not implemented yet.")

    def set_background_model(self, background_model):   
        raise NotImplementedError ("Background model setting is not implemented yet.")

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

 
    def get_parameters(self):
        """
        Get all parameters from the sample, resolution, and background models.
        Returns:
            List[Parameter]: A list of all parameters from the models.
        """ 
        params= []
        for model in [self._SampleModel, self._ResolutionModel, self._BackgroundModel]:
            if model is not None:
                params.extend(model.get_parameters())
        return params   
    
    def get_fit_parameters(self):
        """
        Get all fit parameters from the sample, resolution, and background models that are not fixed.
        Returns:
            List[Parameter]: A list of all fit parameters from the models that are not fixed.
        """
        params= self.get_parameters()
        return [param for param in params if not getattr(param, 'fixed', False)]





        # def collect_parameters(obj):
        #     found = []
        #     if isinstance(obj, Parameter):
        #         found.append(obj)
        #     elif isinstance(obj, dict):
        #         for v in obj.values():
        #             found.extend(collect_parameters(v))
        #     elif isinstance(obj, (list, tuple, set)):
        #         for item in obj:
        #             found.extend(collect_parameters(item))
        #     elif hasattr(obj, '__dict__'):
        #         for v in vars(obj).values():
        #             found.extend(collect_parameters(v))
        #     return found

        # params = []
        # for model in [self._SampleModel, self._ResolutionModel, self._BackgroundModel]:
        #     if model is not None:
        #         for comp in model.components:
        #             params.extend(collect_parameters(comp))
        # return params
            


