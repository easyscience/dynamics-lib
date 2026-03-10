# %%NBQA-CELL-SEPdb06d8
import numpy as np

from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.sample_model.resolution_model import ResolutionModel

hash(0xA46DB22)

# %%NBQA-CELL-SEPdb06d8
# Create a BackgroundModel and a ResolutionModel and add them to an
# InstrumentModel

Q = np.linspace(0.1, 2.0, 5)

background_model = BackgroundModel()
background_model.components = Polynomial(coefficients=[1, 0.1, 0.01])

resolution_model = ResolutionModel()
resolution_model.append_component(Gaussian(width=0.05))

instrument_model = InstrumentModel(
    Q=Q,
    resolution_model=resolution_model,
    background_model=background_model,
)

# %%NBQA-CELL-SEPdb06d8
instrument_model.get_all_variables(Q_index=1)

# %%NBQA-CELL-SEPdb06d8
instrument_model.fix_resolution_parameters()
instrument_model.get_all_variables(Q_index=1)
