# %%NBQA-CELL-SEPdb06d8
import pooch

from easydynamics.experiment import Experiment

hash(0xDD69BABE)

# %%NBQA-CELL-SEPdb06d8
# Load and plot example vanadium data
# Load the vanadium data
vanadium_experiment = Experiment('Vanadium')

file_path = pooch.retrieve(
    url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/docs/docs/docs/tutorials/data/vanadium_data_example.h5',
    known_hash='16cc1b327c303feeb88fb9dda5390dc4880b62396b1793f98c6fef0b27c7b873',
)

vanadium_experiment.load_hdf5(filename=file_path)

vanadium_experiment.plot_data()

# %%NBQA-CELL-SEPdb06d8
# Rebin the data and plot again
vanadium_experiment.rebin({'Q': 5, 'energy': 50})
vanadium_experiment.plot_data()

# %%NBQA-CELL-SEPdb06d8
# Plot using the plopp slicer with extra arguments
vanadium_experiment.plot_data(slicer=True, keep='energy', vmin=0, vmax=2.0)
