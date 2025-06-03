from easyscience.Objects.job.analysis import AnalysisBase


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
    
    def fit():
        raise NotImplementedError("fit not implemented")
        
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


            



    # @abstractmethod
    # def calculate_theory(self,
    #                      x: Union[xr.DataArray, np.ndarray],
    #                      **kwargs) -> np.ndarray:
    #     raise NotImplementedError("calculate_theory not implemented")

    # @abstractmethod
    # def fit(self,
    #         x: Union[xr.DataArray, np.ndarray],
    #         y: Union[xr.DataArray, np.ndarray],
    #         e: Union[xr.DataArray, np.ndarray],
    #         **kwargs) -> None:
    #     raise NotImplementedError("fit not implemented")

    # @property
    # def calculator(self) -> str:
    #     if self._calculator is None:
    #         self._calculator = self.interface.current_interface_name
    #     return self._calculator

    # @calculator.setter
    # def calculator(self, value) -> None:
    #     # TODO: check if the calculator is available for the given JobType
    #     self.interface.switch(value, fitter=self._fitter)

    # @property
    # def minimizer(self) -> MinimizerBase:
    #     return self._minimizer

    # @minimizer.setter
    # def minimizer(self, minimizer: MinimizerBase) -> None:
    #     self._minimizer = minimizer

    # # required dunder methods
    # def __str__(self):
    #     return f"Analysis: {self.name}"