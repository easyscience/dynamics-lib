from easyscience.job.analysis import AnalysisBase
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import FitResults
from easyscience.fitting.fitter import Fitter as EasyScienceFitter

from easyscience.variable import Parameter

from easydynamics.resolution import ResolutionHandler

from easydynamics.sample.components import DeltaFunctionComponent


import numpy as np

import scipp as sc

import matplotlib.pyplot as plt

class Analysis(AnalysisBase):
    def __init__(self, name: str, interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._theory= None
        self._experiment= None

    def plot_data_and_model(self, plot_individual_components: bool = False):
        """
        Plot the experimental data and the theoretical fit.

        Args:
            plot_individual_components (bool): If True, plots individual components of the theory model.
        """
        if self._experiment is None or self._theory is None:
            raise RuntimeError("Experiment and theory must be set before plotting.")

        if self._experiment._data is None:
            raise RuntimeError("No data has been set in the experiment.")

        # Extract data
        x, y, e = self._experiment.extract_xye_data(self._experiment._data)

        # Start plot
        fig = plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=e, label='Data', color='black', marker='o', linestyle='None', markerfacecolor='none')

        # Compute and plot fit
        fit_y = self.calculate_theory(x)
        plt.plot(x, fit_y, label='Fit', color='red')

        # Plot individual components, shifted by offset
        #TODO: handle resolution convolution
        if plot_individual_components:
            for comp in self._theory.components.values():
                # comp_y = comp.evaluate(x - shift)

                if self._experiment._resolution_model is None:
                    y = comp.evaluate(x- self._experiment.offset.value)
                else:
                    resolution_handler = ResolutionHandler()
                    y = resolution_handler.numerical_convolve(x, comp, self._experiment._resolution_model, self._experiment.offset)
                    # If detailed balance is used, calculate the detailed balance factor. TODO: This should be handled before convolution.
                    if self._theory.use_detailed_balance and self._theory._temperature.value >= 0 and not isinstance(comp, DeltaFunctionComponent):
                        y*=self._theory.detailed_balance_factor(x- self._experiment.offset.value, self._theory._temperature.value)

                plt.plot(x, y, label=f'Component: {comp.name}', linestyle='--')

        # Labels and legend
        plt.xlabel('Energy (meV)')  # TODO: Handle units 
        plt.ylabel('Intensity')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return fig

    def set_theory(self, theory):
        self._theory = theory

    def set_experiment(self, experiment):
        self._experiment = experiment   


    def calculate_theory(self, x) -> np.ndarray:
        """
        Calculate the theoretical model by convolving the sample model with the resolution model
        and adding the background model.
        """

        if self._experiment._resolution_model is None:
            y = self._theory.evaluate(x- self._experiment.offset.value)
        else:
            resolution_handler = ResolutionHandler()
            y = resolution_handler.numerical_convolve(x, self._theory, self._experiment._resolution_model, self._experiment.offset)

        if self._experiment._background_model is not None:
            y += self._experiment._background_model.evaluate(x)

        return y


    def fit(self):

        x, y, e = self._experiment.extract_xye_data(self._experiment._data)

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

 
    def get_parameters(self):
        """
        Get all parameters from the theory, resolution, background models, and experiment offset.
        
        Returns:
            List[Parameter]: A list of all parameters.
        """
        params = []

        if self._theory is not None:
            params.extend(self._theory.get_parameters())

        if self._experiment is not None:
            if self._experiment._resolution_model is not None:
                params.extend(self._experiment._resolution_model.get_parameters())
            if self._experiment._background_model is not None:
                params.extend(self._experiment._background_model.get_parameters())
            if hasattr(self._experiment, "offset"):
                params.append(self._experiment.offset)

        return params

    def get_fit_parameters(self):
        """
        Get all fit parameters from the theory, resolution, background models, and experiment offset,
        filtering out fixed parameters.

        Returns:
            List[Parameter]: A list of unfixed fit parameters.
        """
        return [param for param in self.get_parameters() if not getattr(param, 'fixed', False)]
