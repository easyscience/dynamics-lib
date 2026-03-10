# %%NBQA-CELL-SEPdb06d8
import pooch

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.sample_model.resolution_model import ResolutionModel
from easydynamics.sample_model.sample_model import SampleModel

hash(0xD652F76B)

# %%NBQA-CELL-SEPdb06d8
vanadium_experiment = Experiment('Vanadium')

file_path = pooch.retrieve(
    url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/docs/docs/docs/tutorials/data/vanadium_data_example.h5',
    known_hash='16cc1b327c303feeb88fb9dda5390dc4880b62396b1793f98c6fef0b27c7b873',
)


vanadium_experiment.load_hdf5(filename=file_path)

# %%NBQA-CELL-SEPdb06d8
# Example of Analysis1d with a simple sample model and instrument model
delta_function = DeltaFunction(display_name='DeltaFunction', area=1)
sample_model = SampleModel(
    components=delta_function,
)

res_gauss = Gaussian(width=0.1)
resolution_model = ResolutionModel(components=res_gauss)


background_model = BackgroundModel(components=Polynomial(coefficients=[0.001]))

instrument_model = InstrumentModel(
    resolution_model=resolution_model,
    background_model=background_model,
)

my_analysis = Analysis1d(
    display_name='Vanadium Analysis',
    experiment=vanadium_experiment,
    sample_model=sample_model,
    instrument_model=instrument_model,
    Q_index=5,
)

fit_result = my_analysis.fit()
fig = my_analysis.plot_data_and_model()
fig
