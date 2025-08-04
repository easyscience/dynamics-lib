from easyscience.job.experiment import ExperimentBase

class Data1D(ExperimentBase):
    """
    Data class for storing experimental data.
    
    Attributes:
        data : The experimental data.
    """
    
    def __init__(self, name):
        super().__init__(name)
        self.data = None


    def append(self, new_data):
        """
        Append new data to the existing data.
        
        Args:
            new_data (sc.DataArray): New data to append.
        """
        if self.data is None:
            self.data = new_data
        else:
            raise NotImplementedError("Appending data is not implemented yet.")

    def get_data(self):
        """
        Get the stored data.
        
        Returns:
            : The experimental data.
        """
        return self.data
    
    def remove(self):
        """
        Remove the stored data.
        """
        self.data = None

    def remove_outliers(self):
        """
        Remove outliers from the data.
        
        This method is a placeholder and should be implemented based on specific criteria for outlier removal.
        """
        raise NotImplementedError("Outlier removal is not implemented yet.")
    
    def __repr__(self):
        """
        String representation of the Data object.
        
        Returns:
            str: Representation of the Data object.
        """
        return f"Data(data={self.name})"
    
    def plot(self):
        raise NotImplementedError("Plotting is not implemented yet.")
    
