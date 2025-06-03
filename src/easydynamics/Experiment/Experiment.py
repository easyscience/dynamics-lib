import scipp as sc
import numpy as np



class Experiment:

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
    