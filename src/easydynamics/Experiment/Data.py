from easyscience.job.experiment import ExperimentBase

class Data(ExperimentBase):
    """
    Data class for storing experimental data.
    
    Attributes:
        data (sc.DataArray): The experimental data.
    """
    
    def __init__(self, data=None):
        super().__init__()
        self.data = data


    def append(self, new_data):
        """
        Append new data to the existing data.
        
        Args:
            new_data (sc.DataArray): New data to append.
        """
        if self.data is None:
            self.data = new_data
        else:
            NotImplementedError("Appending data is not implemented yet.")


    def get_data(self):
        """
        Get the stored data.
        
        Returns:
            sc.DataArray: The experimental data.
        """
        return self.data
    
    def remove(self):
        """
        Remove the stored data.
        """
        self.data = None