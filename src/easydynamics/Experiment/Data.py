from easyscience.job.experiment import ExperimentBase
import numpy as np
import scipp as sc
class Data(ExperimentBase):
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
        raise NotImplementedError("Plotting is not implemented yet. Use a plotting library to visualize the data.")
    

    
    @staticmethod
    def load_example_vanadium_data():
        """
        Load example vanadium data from files.
        
        Returns:
            sc.DataArray: DataArray containing the vanadium data with energy and Q as coordinates.
        """
        NUMBER_OF_Q_POINTS=16
        NUMBER_OF_E_POINTS=1024
        # TODO Add the correct Q values
        # [  0.5708,    0.7002,    0.8262 ,   0.9485 ,   1.0664  ,  1.1793   , 1.2868 ,   1.3883 ,   1.4833 ,   1.5716  ,  1.6525  ,  1.7258  ,  1.7910 ,   1.8480  ,  1.8965 ,   1.9361],unit='1/angstrom'



        intensity_values=np.zeros((NUMBER_OF_Q_POINTS,NUMBER_OF_E_POINTS))
        error_values=np.zeros((NUMBER_OF_Q_POINTS,NUMBER_OF_E_POINTS))

        # Load data into a matrix
        for Q in range(NUMBER_OF_Q_POINTS):
            filename = '../examples/QENS_example/IN16b_GGG_data/vanadium_Q' +str(Q+1) +'.dat'

            data_array = np.loadtxt(filename)
            energy_values=data_array[:, 0] #should be the same for all Q
            # EnergyValues[Q,:]=data_array[:, 0]
            intensity_values[Q,:]=data_array[:,1]
            error_values[Q,:]=data_array[:,2]

        # Define energy, q and intensity as scipp variables with units, and make a DataArray
        Q=sc.array(dims=['Q'],values=range(NUMBER_OF_Q_POINTS))
        energy=sc.array(dims=['energy'],values=energy_values/1000,unit='meV')
        intensity=sc.array(dims=['Q','energy'],values=intensity_values,variances=error_values*error_values) #The variance is the square of the uncertainty!

        vanadium_data = sc.DataArray(data=intensity, coords={'Q':Q,'energy': energy})
        

        return vanadium_data
    
    @staticmethod
    def load_example_vanadium_data_1d():
            """
            Load example vanadium data from files.
            
            Returns:
                sc.DataArray: DataArray containing the vanadium data with energy and Q as coordinates.
            """
            NUMBER_OF_Q_POINTS=16
            # TODO Add the correct Q values
            # [  0.5708,    0.7002,    0.8262 ,   0.9485 ,   1.0664  ,  1.1793   , 1.2868 ,   1.3883 ,   1.4833 ,   1.5716  ,  1.6525  ,  1.7258  ,  1.7910 ,   1.8480  ,  1.8965 ,   1.9361],unit='1/angstrom'



            # Load data into a matrix
            for Q in [5]:
                filename = '../examples/QENS_example/IN16b_GGG_data/vanadium_Q' +str(Q+1) +'.dat'

                data_array = np.loadtxt(filename)
                energy_values=data_array[:, 0] #should be the same for all Q
                # EnergyValues[Q,:]=data_array[:, 0]
                intensity_values=data_array[:,1]
                error_values=data_array[:,2]

            # Define energy, q and intensity as scipp variables with units, and make a DataArray
            Q=sc.array(dims=['Q'],values=range(NUMBER_OF_Q_POINTS))
            energy=sc.array(dims=['energy'],values=energy_values/1000,unit='meV')
            intensity=sc.array(dims=['energy'],values=intensity_values,variances=error_values*error_values) #The variance is the square of the uncertainty!

            vanadium_data = sc.DataArray(data=intensity, coords={'energy': energy})
            

            return vanadium_data    

    @staticmethod
    def load_example_data_1d():
        #Preallocate
        # Load data into a matrix

        NUMBER_OF_Q_POINTS=16

        for Q in [5]:
            filename = '../examples/QENS_example/IN16b_GGG_data/data_450mK_Q' +str(Q+1) +'.dat'

            data_array = np.loadtxt(filename)
            energy_values=data_array[:, 0] #should be the same for all Q
            # EnergyValues[Q,:]=data_array[:, 0]
            intensity_values=data_array[:,1]
            error_values=data_array[:,2]

        # Define energy, Q and intensity as scipp variables with units, and make a DataArray
        energy=sc.array(dims=['energy'],values=energy_values/1000,unit='meV')
        Q=sc.array(dims=['Q'],values=range(NUMBER_OF_Q_POINTS))
        intensity=sc.array(dims=['energy'],values=intensity_values,variances=error_values*error_values) #The variance is the square of the uncertainty!

        GGG_data_450mK = sc.DataArray(data=intensity, coords={'energy': energy})

        return GGG_data_450mK


    @staticmethod
    def load_example_anesthetics_data_lowT():
        data = np.loadtxt('../examples/Anesthetics/data/BVC2K_q4.inx', skiprows=4)

        # Extract columns
        E = data[:, 0]    # Energy
        I = data[:, 1]    # Intensity
        dI = data[:, 2]   # Error


        indices=np.where(E>-2.0)

        # Create Scipp DataArray
        da = sc.DataArray(
            data=sc.array(dims=['energy'], values=I[indices], variances=dI[indices]**2),
            coords={'energy': sc.array(dims=['energy'], values=E[indices], unit='meV')}
        )
        return da
    
    @staticmethod
    def load_example_anesthetics_data_midT():
        data = np.loadtxt('../examples/Anesthetics/data/BVC50K_q4.inx', skiprows=4)

        # Extract columns
        E = data[:, 0]    # Energy
        I = data[:, 1]    # Intensity
        dI = data[:, 2]   # Error

        indices=np.where(E>-2.0)

        # Create Scipp DataArray
        da = sc.DataArray(
            data=sc.array(dims=['energy'], values=I[indices], variances=dI[indices]**2),
            coords={'energy': sc.array(dims=['energy'], values=E[indices], unit='meV')}
        )
        return da
    
    @staticmethod
    def load_example_anesthetics_data_highT():
        data = np.loadtxt('../examples/Anesthetics/data/BVC250K_q4.inx', skiprows=4)

        # Extract columns
        E = data[:, 0]    # Energy
        I = data[:, 1]    # Intensity
        dI = data[:, 2]   # Error

        indices=np.where(E>-2.0)

        # Create Scipp DataArray
        da = sc.DataArray(
            data=sc.array(dims=['energy'], values=I[indices], variances=dI[indices]**2),
            coords={'energy': sc.array(dims=['energy'], values=E[indices], unit='meV')}
        )
        return da