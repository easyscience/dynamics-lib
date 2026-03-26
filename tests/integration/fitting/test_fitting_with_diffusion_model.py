# Imports
import numpy as np
import pooch

from easydynamics.analysis.analysis import Analysis
from easydynamics.experiment import Experiment
from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.diffusion_model.brownian_translational_diffusion import (
    BrownianTranslationalDiffusion,
)
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.sample_model.resolution_model import ResolutionModel
from easydynamics.sample_model.sample_model import SampleModel


class TestFittingWithDiffusionModel:
    def test_fitting_with_diffusion_model(self):
        # Load the vanadium data
        vanadium_experiment = Experiment('Vanadium')

        file_path = pooch.retrieve(
            url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/master/docs/docs/tutorials/data/vanadium_data_example.h5',
            known_hash='16cc1b327c303feeb88fb9dda5390dc4880b62396b1793f98c6fef0b27c7b873',
        )

        vanadium_experiment.load_hdf5(filename=file_path)

        delta_function = DeltaFunction(display_name='DeltaFunction', area=1)
        sample_model = SampleModel(components=delta_function)

        resolution_components = ComponentCollection()
        res_gauss = Gaussian(width=0.1, area=1, display_name='Res. Gauss')
        res_gauss.area.fixed = True
        resolution_components.append_component(res_gauss)
        resolution_model = ResolutionModel(components=resolution_components)

        background_model = BackgroundModel(components=Polynomial(coefficients=[0.001]))

        instrument_model = InstrumentModel(
            resolution_model=resolution_model,
            background_model=background_model,
        )

        vanadium_analysis = Analysis(
            display_name='Vanadium Full Analysis',
            experiment=vanadium_experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )

        fit_result_independent_single_Q = vanadium_analysis.fit(
            fit_method='independent', Q_index=5
        )

        assert fit_result_independent_single_Q.success
        assert fit_result_independent_single_Q.chi2 < 75.0
        assert fit_result_independent_single_Q.reduced_chi < 0.4

        diffusion_experiment = Experiment('Diffusion')

        file_path = pooch.retrieve(
            url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/master/docs/docs/tutorials/data/diffusion_data_example.h5',
            known_hash='5fe846b19aacbda8b8b936eb2e5310d025dc56c25b0b353521e7d6b921f229ab',
        )

        diffusion_experiment.load_hdf5(filename=file_path)

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
            resolution_model=vanadium_analysis.instrument_model.resolution_model,
        )
        instrument_model.resolution_model.fix_all_parameters()

        diffusion_analysis = Analysis(
            display_name='Diffusion Analysis',
            experiment=diffusion_experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )

        fit_result = diffusion_analysis.fit(fit_method='independent')

        assert fit_result[0].success
        assert fit_result[0].chi2 < 43.0
        assert fit_result[0].reduced_chi < 0.3

        ###############
        # Diffusion model
        ###############

        delta_function = DeltaFunction(display_name='DeltaFunction', area=0.2)
        component_collection = ComponentCollection(
            components=[delta_function],
        )
        diffusion_model = BrownianTranslationalDiffusion(
            display_name='Brownian Translational Diffusion',
            diffusion_coefficient=2.4e-9,
            scale=0.5,
        )

        sample_model = SampleModel(
            components=component_collection,
            diffusion_models=diffusion_model,
        )

        background_model = BackgroundModel(components=Polynomial(coefficients=[0.001]))

        instrument_model = InstrumentModel(
            background_model=background_model,
            resolution_model=vanadium_analysis.instrument_model.resolution_model,
        )

        diffusion_model_analysis = Analysis(
            display_name='Diffusion Full Analysis',
            experiment=diffusion_experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )

        diffusion_model_analysis.instrument_model.resolution_model.fix_all_parameters()

        fit_result = diffusion_model_analysis.fit(fit_method='simultaneous')

        assert fit_result[0].success
        assert fit_result[0].chi2 < 56.0
        assert fit_result[0].reduced_chi < 0.4

        pars = diffusion_model.get_all_parameters()

        tol = 10 * pars[0].error
        assert np.isclose(pars[0].value, 1.1258025622851164e-08, atol=tol)
        tol = 10 * pars[1].error
        assert np.isclose(pars[1].value, 0.6937774083152299, atol=tol)
