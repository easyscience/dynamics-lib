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
        self._theory= None
        self._experiment= None


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
            for comp in self._theory.components.values():
                comp_y = comp.evaluate(x-self._theory.offset.value)
                plt.plot(x, comp_y, label=f'Component: {comp.name}', linestyle='--')


        plt.xlabel('Energy (meV)') #TODO: Handle units properly
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
        return fig
        


    # def calculate_theory(self,
    #                     x: np.ndarray) -> np.ndarray:
    #     """
    #     Calculate the theoretical model by convolving the sample model with the resolution model
    #     and adding the background model.
    #     """
        
    #     if self._resolution_model is None:
    #         y= self._theory.evaluate(x)
    #     else:
    #         resolution_handler=ResolutionHandler()
    #         y= resolution_handler.numerical_convolve(x, self._theory, self._resolution_model)

    #     if self._background_model is not None:
    #         y += self._background_model.evaluate(x)

    #     return y

    def calculate_theory(self, x, experiment, theory) -> np.ndarray:
        """
        Calculate the theoretical model by convolving the sample model with the resolution model
        and adding the background model.
        """
        
        if experiment._resolution_model is None:
            y = self.theory.evaluate(x)
        else:
            resolution_handler = ResolutionHandler()
            y = resolution_handler.numerical_convolve(x, theory, experiment._resolution_model)

        if experiment._background_model is not None:
            y += experiment._background_model.evaluate(x)

        return y


    def fit(self, experiment, theory):

        x, y, e = experiment.extract_xye_data(experiment._data)

        def fit_func(x_vals):
            return self.calculate_theory(x_vals, experiment, theory)

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

 
    def get_parameters(self):
        """
        Get all parameters from the sample, resolution, and background models.
        Returns:
            List[Parameter]: A list of all parameters from the models.
        """ 
        params= []
        for model in [self._theory, self._resolution_model, self._background_model]:
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


