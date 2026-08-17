# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest

from easydynamics.analysis.fit_binding import FitBinding
from easydynamics.sample_model.components.gaussian import Gaussian
from easydynamics.sample_model.components.polynomial import Polynomial
from easydynamics.sample_model.diffusion_model.brownian_translational_diffusion import (
    BrownianTranslationalDiffusion,
)
from easydynamics.sample_model.diffusion_model.delta_lorentz import DeltaLorentz
from easydynamics.utils.fit_target import FitTarget


class TestFitBinding:
    @pytest.fixture
    def component_binding(self):
        model = Gaussian(display_name='GaussianModel')
        return FitBinding(model=model, targets='parameter1')

    @pytest.fixture
    def diffusion_binding(self):
        model = BrownianTranslationalDiffusion(lorentzian_name='Lorentzian')
        return FitBinding(model=model)

    #############
    # Initialization and validation
    #############

    def test_initialization(self, component_binding):
        # WHEN THEN EXPECT
        assert isinstance(component_binding, FitBinding)
        assert component_binding.targets == 'parameter1'
        assert isinstance(component_binding.model, Gaussian)

    def test_initialization_invalid_model_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='model must be a ModelComponent'):
            FitBinding(model='not_a_model', targets='parameter1')

    @pytest.mark.parametrize(
        'targets',
        [None, ['parameter1'], {'value': 'parameter1'}, 123],
        ids=['none', 'list', 'dict', 'int'],
    )
    def test_component_model_requires_string_targets(self, targets):
        # WHEN THEN EXPECT: component models have a single prediction, so targets must be
        # the dataset key
        with pytest.raises(TypeError, match='targets must be the dataset key'):
            FitBinding(model=Gaussian(), targets=targets)

    def test_diffusion_unknown_prediction_raises(self):
        # WHEN THEN EXPECT
        model = BrownianTranslationalDiffusion()
        with pytest.raises(ValueError, match=r'Unknown prediction.*Available predictions'):
            FitBinding(model=model, targets=['nonsense'])

    def test_diffusion_unknown_prediction_in_dict_raises(self):
        # WHEN THEN EXPECT
        model = BrownianTranslationalDiffusion()
        with pytest.raises(ValueError, match='Unknown prediction'):
            FitBinding(model=model, targets={'delta_area': 'Elastic area'})

    def test_diffusion_invalid_targets_type_raises(self):
        # WHEN THEN EXPECT
        model = BrownianTranslationalDiffusion()
        with pytest.raises(TypeError, match='targets must be None'):
            FitBinding(model=model, targets=123)

    def test_diffusion_non_string_dataset_key_raises(self):
        # WHEN THEN EXPECT
        model = BrownianTranslationalDiffusion()
        with pytest.raises(TypeError, match='dataset keys'):
            FitBinding(model=model, targets={'width': 123})

    #############
    # Properties
    #############

    def test_model_setter_revalidates_targets(self):
        # WHEN: a binding using DeltaLorentz's delta_area prediction
        binding = FitBinding(model=DeltaLorentz(), targets=['delta_area'])

        # THEN EXPECT: switching to a model without that prediction fails
        with pytest.raises(ValueError, match='Unknown prediction'):
            binding.model = BrownianTranslationalDiffusion()

    def test_model_setter_invalid_type_raises(self, component_binding):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='model must be a ModelComponent'):
            component_binding.model = 'not_a_model'

    def test_targets_setter(self, diffusion_binding):
        # WHEN THEN
        diffusion_binding.targets = ['width']

        # EXPECT
        assert [t.name for t in diffusion_binding.get_targets()] == ['width']

    def test_targets_setter_invalid_raises(self, diffusion_binding):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='Unknown prediction'):
            diffusion_binding.targets = ['nonsense']

    #############
    # get_targets
    #############

    def test_component_target(self, component_binding):
        # WHEN
        targets = component_binding.get_targets()

        # THEN EXPECT: a single target evaluating the component, with its units
        assert len(targets) == 1
        target = targets[0]
        assert isinstance(target, FitTarget)
        assert target.name == 'value'
        assert target.dataset_key == 'parameter1'
        assert target.label == 'GaussianModel'
        assert target.x_unit == component_binding.model.x_unit
        assert target.y_unit == component_binding.model.y_unit

    def test_component_target_function_evaluates_model(self, component_binding):
        # WHEN
        target = component_binding.get_targets()[0]
        x = np.linspace(-1, 1, 5)

        # THEN EXPECT
        np.testing.assert_allclose(target.function(x), component_binding.model.evaluate(x))

    def test_diffusion_default_targets(self, diffusion_binding):
        # WHEN
        targets = diffusion_binding.get_targets()

        # THEN EXPECT: all predictions with default dataset keys from the Lorentzian name
        assert [t.name for t in targets] == ['area', 'width']
        assert [t.dataset_key for t in targets] == ['Lorentzian area', 'Lorentzian width']

    def test_diffusion_string_target(self):
        # WHEN
        model = BrownianTranslationalDiffusion(lorentzian_name='QENS')
        binding = FitBinding(model=model, targets='width')

        # THEN
        targets = binding.get_targets()

        # EXPECT
        assert len(targets) == 1
        assert targets[0].name == 'width'
        assert targets[0].dataset_key == 'QENS width'

    def test_diffusion_list_targets(self):
        # WHEN
        model = BrownianTranslationalDiffusion(lorentzian_name='QENS')
        binding = FitBinding(model=model, targets=['width', 'area'])

        # THEN EXPECT: order follows the user's list
        assert [t.name for t in binding.get_targets()] == ['width', 'area']

    def test_diffusion_dict_targets_map_dataset_keys(self):
        # WHEN: mapping DeltaLorentz predictions to custom dataset keys
        model = DeltaLorentz()
        binding = FitBinding(
            model=model,
            targets={
                'width': 'L width',
                'area': 'L area',
                'delta_area': 'Elastic area',
            },
        )

        # THEN
        targets = {t.name: t for t in binding.get_targets()}

        # EXPECT
        assert targets['width'].dataset_key == 'L width'
        assert targets['area'].dataset_key == 'L area'
        assert targets['delta_area'].dataset_key == 'Elastic area'

    def test_delta_lorentz_default_targets(self):
        # WHEN: default keys derive from the component names
        model = DeltaLorentz(lorentzian_name='Lorentzian', delta_name='Delta function')
        binding = FitBinding(model=model)

        # THEN
        targets = {t.name: t for t in binding.get_targets()}

        # EXPECT
        assert set(targets) == {'area', 'width', 'delta_area'}
        assert targets['area'].dataset_key == 'Lorentzian area'
        assert targets['width'].dataset_key == 'Lorentzian width'
        assert targets['delta_area'].dataset_key == 'Delta function area'

    def test_diffusion_target_units(self, diffusion_binding):
        # WHEN
        targets = {t.name: t for t in diffusion_binding.get_targets()}

        # THEN EXPECT: predictions take Q in 1/angstrom; widths are in the model's x_unit and
        # areas in the scale unit
        assert targets['width'].x_unit == '1/angstrom'
        assert targets['width'].y_unit == diffusion_binding.model.x_unit
        assert targets['area'].y_unit == str(diffusion_binding.model.scale.unit)

    def test_diffusion_target_units_track_conversions(self, diffusion_binding):
        # WHEN: targets are snapshots of the live model, so unit conversions are reflected
        diffusion_binding.model.convert_x_unit('ueV')

        # THEN
        targets = {t.name: t for t in diffusion_binding.get_targets()}

        # EXPECT
        assert targets['width'].y_unit == 'ueV'

    def test_diffusion_target_functions(self):
        # WHEN
        model = BrownianTranslationalDiffusion(diffusion_coefficient=2.4e-9, scale=0.5)
        binding = FitBinding(model=model)
        Q = np.array([0.5, 1.0])

        # THEN
        targets = {t.name: t for t in binding.get_targets()}

        # EXPECT
        np.testing.assert_allclose(targets['width'].function(Q), model.calculate_width(Q))
        np.testing.assert_allclose(
            targets['area'].function(Q), model.calculate_QISF(Q) * model.scale.value
        )

    def test_delta_lorentz_delta_area_function(self):
        # WHEN
        model = DeltaLorentz(A_0=0.5, mean_u_squared=0.3, scale=2.0, lorentzian_width=0.1)
        binding = FitBinding(model=model, targets=['delta_area'])
        Q = np.array([0.5, 1.0])

        # THEN
        target = binding.get_targets()[0]

        # EXPECT
        np.testing.assert_allclose(target.function(Q), model.calculate_EISF(Q) * model.scale.value)

    #############
    # dunder methods
    #############

    def test_repr(self, diffusion_binding):
        # WHEN THEN
        repr_str = repr(diffusion_binding)

        # EXPECT
        assert 'FitBinding' in repr_str
        assert 'model=' in repr_str
        assert 'targets=' in repr_str

    #############
    # Workflows: end-to-end regression tests for the standard ParameterAnalysis workflows
    #############

    def test_polynomial_targets_gaussian_area(self):
        # WHEN: fitting a Polynomial to a 'Gaussian area' dataset key
        polynomial = Polynomial(
            coefficients=[1.0, 0.5], x_unit='1/angstrom', y_unit='meV', display_name='Line'
        )
        binding = FitBinding(model=polynomial, targets='Gaussian area')

        # THEN
        target = binding.get_targets()[0]

        # EXPECT
        assert target.dataset_key == 'Gaussian area'
        assert target.label == 'Line'
