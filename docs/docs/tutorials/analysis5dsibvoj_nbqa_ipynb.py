# %%NBQA-CELL-SEP5932b7
# Imports
import pooch

from easydynamics.analysis.analysis import Analysis
from easydynamics.experiment import Experiment
from easydynamics.sample_model import BrownianTranslationalDiffusion
from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.sample_model.resolution_model import ResolutionModel
from easydynamics.sample_model.sample_model import SampleModel

hash(0x5D2FE732)

# %%NBQA-CELL-SEP5932b7
# Load the vanadium data
vanadium_experiment = Experiment('Vanadium')
file_path = pooch.retrieve(
    url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/master/docs/docs/tutorials/data/vanadium_data_example.h5',
    known_hash='16cc1b327c303feeb88fb9dda5390dc4880b62396b1793f98c6fef0b27c7b873',
)

vanadium_experiment.load_hdf5(filename=file_path)

# %%NBQA-CELL-SEP5932b7
# Example of Analysis with a simple sample model and instrument model
# The scattering from vanadium is purely elastic, so we model it with a
# delta function
delta_function = DeltaFunction(display_name='DeltaFunction', area=1)
sample_model = SampleModel(
    components=delta_function,
)

# The resolution is in this case modeled as a Gaussian. However, we can
# add as many components as we like to the resolution model
res_gauss = Gaussian(width=0.1)
res_gauss.area.fixed = True
resolution_components = ComponentCollection()
resolution_components.append_component(res_gauss)
resolution_model = ResolutionModel(components=resolution_components)

# The background model is created in the same way. In this case, we use
# a flat background
background_model = BackgroundModel(components=Polynomial(coefficients=[0.001]))

# We combine the resolution abd background model into an instrument
# model. This model also contains a small energy offset to account for
# instrument misalignment.

instrument_model = InstrumentModel(
    resolution_model=resolution_model,
    background_model=background_model,
)

# Collect everything into an analysis object.
vanadium_analysis = Analysis(
    display_name='Vanadium Full Analysis',
    experiment=vanadium_experiment,
    sample_model=sample_model,
    instrument_model=instrument_model,
)

# Let us first fit a single Q index and plot the data and model to see
# how it looks
fit_result_independent_single_Q = vanadium_analysis.fit(fit_method='independent', Q_index=5)
vanadium_analysis.plot_data_and_model(Q_index=5)

# %%NBQA-CELL-SEP5932b7
# It looks good, so let us fit all Q indices independently and plot the
# results
fit_result_independent_all_Q = vanadium_analysis.fit(fit_method='independent')
vanadium_analysis.plot_data_and_model()

# %%NBQA-CELL-SEP5932b7
# Inspect the Parameters as a scipp Dataset
vanadium_analysis.parameters_to_dataset()

# %%NBQA-CELL-SEP5932b7
# Plot some of fitted parameters as a function of Q
vanadium_analysis.plot_parameters(names=['DeltaFunction area'])

# %%NBQA-CELL-SEP5932b7
vanadium_analysis.plot_parameters(names=['Gaussian width'])

# %%NBQA-CELL-SEP5932b7
vanadium_analysis.plot_parameters(names=['energy_offset'])

# %%NBQA-CELL-SEP5932b7
# Now it's time to look at the data we want to fit. We first load the
# data
diffusion_experiment = Experiment('Diffusion')

file_path = pooch.retrieve(
    url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/master/docs/docs/tutorials/data/diffusion_data_example.h5',
    known_hash='5fe846b19aacbda8b8b936eb2e5310d025dc56c25b0b353521e7d6b921f229ab',
)

diffusion_experiment.load_hdf5(filename=file_path)

# %%NBQA-CELL-SEP5932b7
# Now we set up the model, similarly to how we set up the model for the
# vanadium data.

delta_function = DeltaFunction(display_name='DeltaFunction', area=0.2)
lorentzian = Lorentzian(display_name='Lorentzian', area=0.5, width=0.3)
component_collection = ComponentCollection(
    components=[delta_function, lorentzian],
)

sample_model = SampleModel(
    components=component_collection,
)

background_model = BackgroundModel(components=Polynomial(coefficients=[0.001]))

instrument_model = InstrumentModel(
    background_model=background_model,
)

diffusion_analysis = Analysis(
    display_name='Diffusion Full Analysis',
    experiment=diffusion_experiment,
    sample_model=sample_model,
    instrument_model=instrument_model,
)

# We need to hack in the resolution model from the vanadium analysis,
# since the setters and getters overwrite the model. This will be fixed
# asap.
diffusion_analysis.instrument_model._resolution_model = (
    vanadium_analysis.instrument_model.resolution_model
)

# We fix all parameters of the resolution model.
diffusion_analysis.instrument_model.resolution_model.fix_all_parameters()

# %%NBQA-CELL-SEP5932b7
# Let us see how good the starting parameters are
diffusion_analysis.plot_data_and_model()

# %%NBQA-CELL-SEP5932b7
# Now we fit the data and plot the result. Looks good!
diffusion_analysis.fit(fit_method='independent')
diffusion_analysis.plot_data_and_model()

# %%NBQA-CELL-SEP5932b7
# Let us look at the most interesting fit parameters
diffusion_analysis.plot_parameters(names=['Lorentzian width', 'Lorentzian area'])

# %%NBQA-CELL-SEP5932b7
# It will be possible to fit this to a DiffusionModel, but that will
# come later.

# %%NBQA-CELL-SEP5932b7
# Let us now fit directly to a diffusion model. We replace the
# Lorentzian with a Brownian translational diffusion model and keep the
# other parameters the same.
delta_function = DeltaFunction(display_name='DeltaFunction', area=0.2)
component_collection = ComponentCollection(
    components=[delta_function],
)
diffusion_model = BrownianTranslationalDiffusion(
    display_name='Brownian Translational Diffusion', diffusion_coefficient=2.4e-9, scale=0.5
)

sample_model = SampleModel(
    components=component_collection,
    diffusion_models=diffusion_model,
)

background_model = BackgroundModel(components=Polynomial(coefficients=[0.001]))

instrument_model = InstrumentModel(
    background_model=background_model,
)

diffusion_model_analysis = Analysis(
    display_name='Diffusion Full Analysis',
    experiment=diffusion_experiment,
    sample_model=sample_model,
    instrument_model=instrument_model,
)

# We again need to hack in the resolution model from the vanadium
# analysis, since the setters and getters overwrite the model. This will
# be fixed asap.
diffusion_model_analysis.instrument_model._resolution_model = (
    vanadium_analysis.instrument_model.resolution_model
)
diffusion_model_analysis.instrument_model.resolution_model.fix_all_parameters()

# Let us see how good the starting parameters are
diffusion_model_analysis.plot_data_and_model()

# %%NBQA-CELL-SEP5932b7
# We now fit all the data simultaneously to the diffusion model, then
# plot the result. Looks good.
diffusion_model_analysis.fit(fit_method='simultaneous')
diffusion_model_analysis.plot_data_and_model()

# %%NBQA-CELL-SEP5932b7
# Let us look at the fitted diffusion coefficient
diffusion_model.get_all_parameters()
