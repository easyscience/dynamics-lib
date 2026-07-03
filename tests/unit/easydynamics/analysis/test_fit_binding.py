# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from unittest.mock import Mock

import pytest

from easydynamics.analysis.fit_binding import FitBinding
from easydynamics.sample_model.components.gaussian import Gaussian
from easydynamics.sample_model.diffusion_model.brownian_translational_diffusion import (
    BrownianTranslationalDiffusion,
)


class TestFitBinding:
    @pytest.fixture
    def fit_binding(self):
        model = Gaussian()
        return FitBinding(parameter_name='parameter1', model=model)

    @pytest.fixture
    def diffusion_fit_binding(self):
        model = BrownianTranslationalDiffusion()
        return FitBinding(parameter_name='parameter3', model=model)

    def test_initialization(self, fit_binding):
        # WHEN THEN EXPECT
        assert isinstance(fit_binding, FitBinding)
        assert fit_binding.parameter_name == 'parameter1'
        assert isinstance(fit_binding.model, Gaussian)
        assert fit_binding.modes is None

    def test_init_with_string_modes(self):
        model = Gaussian()
        fb = FitBinding(parameter_name='param', model=model, modes='area')
        assert fb.modes == ['area']

    @pytest.mark.parametrize(
        'parameter_name, model, modes, error_msg',
        [
            # parameter_name errors
            (123, Gaussian(), None, 'parameter_name must be a string'),
            (None, Gaussian(), None, 'parameter_name must be a string'),
            # model errors
            (
                'param',
                123,
                None,
                'model must be a ModelComponent, ComponentCollection, or DiffusionModelBase',
            ),
            (
                'param',
                'not_a_model',
                None,
                'model must be a ModelComponent, ComponentCollection, or DiffusionModelBase',
            ),
            # modes type errors
            (
                'param',
                Gaussian(),
                123,
                'modes must be a string, list of strings, or None',
            ),
            (
                'param',
                Gaussian(),
                {'mode': 'area'},
                'modes must be a string, list of strings, or None',
            ),
            # modes list content errors
            (
                'param',
                Gaussian(),
                ['area', 123],
                'All modes in the list must be strings',
            ),
            ('param', Gaussian(), [None], 'All modes in the list must be strings'),
        ],
    )
    def test_fitbinding_init_errors(self, parameter_name, model, modes, error_msg):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=error_msg):
            FitBinding(
                parameter_name=parameter_name,
                model=model,
                modes=modes,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def test_parameter_name_setter(self, fit_binding):
        # WHEN
        fit_binding.parameter_name = 'new_parameter'

        # THEN EXPECT
        assert fit_binding.parameter_name == 'new_parameter'

    def test_parameter_name_setter_errors(self, fit_binding):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='parameter_name must be a string'):
            fit_binding.parameter_name = 123

    def test_model_setter(self, fit_binding):
        # WHEN
        model = BrownianTranslationalDiffusion()

        # THEN
        fit_binding.model = model

        # EXPECT
        assert fit_binding.model is model

    def test_model_setter_errors(self, fit_binding):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='model must be a ModelComponent, ComponentCollection, or DiffusionModelBase',
        ):
            fit_binding.model = 'not_a_model'

    def test_modes_setter(self, fit_binding):
        # WHEN

        # THEN
        fit_binding.modes = 'area'

        # EXPECT
        assert fit_binding.modes == ['area']

        # THEN
        fit_binding.modes = ['area', 'width']

        # EXPECT
        assert fit_binding.modes == ['area', 'width']

    def test_modes_setter_errors(self, fit_binding):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='modes must be a string, list of strings, or None'):
            fit_binding.modes = 123

        with pytest.raises(TypeError, match='modes must be a string, list of strings, or None'):
            fit_binding.modes = {'mode': 'area'}

        with pytest.raises(TypeError, match='All modes in the list must be strings'):
            fit_binding.modes = ['area', 123]

        with pytest.raises(TypeError, match='All modes in the list must be strings'):
            fit_binding.modes = [None]

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def test_build_callables_component(self, fit_binding):
        # WHEN
        mock_model = Mock()
        mock_model.evaluate.return_value = 1.0
        fit_binding._model = mock_model

        # THEN
        callables = fit_binding.build_callables()

        # EXPECT
        assert len(callables) == 1
        assert callable(callables[0])
        assert callables[0](0) == pytest.approx(1.0)
        mock_model.evaluate.assert_called_once_with(0)

    def test_build_callables_diffusion(self, diffusion_fit_binding):
        # WHEN
        mock_model = Mock(spec=BrownianTranslationalDiffusion)
        mock_model.calculate_QISF.return_value = 2.0
        mock_model.scale.value = 3.0
        mock_model.calculate_width.return_value = 0.5
        diffusion_fit_binding._model = mock_model

        # THEN
        callables = diffusion_fit_binding.build_callables()

        # EXPECT
        assert len(callables) == 2
        assert callable(callables[0])
        assert callable(callables[1])
        assert callables[0](0) == pytest.approx(6.0)  # 2.0 * 3.0
        assert callables[1](0) == pytest.approx(0.5)
        mock_model.calculate_QISF.assert_called_once_with(0)
        mock_model.calculate_width.assert_called_once_with(0)

    def test_build_callables_diffusion_with_modes(self, diffusion_fit_binding):
        # WHEN
        diffusion_fit_binding.modes = 'area'
        mock_model = Mock(spec=BrownianTranslationalDiffusion)
        mock_model.calculate_QISF.return_value = 2.0
        mock_model.scale.value = 3.0
        diffusion_fit_binding._model = mock_model

        # THEN
        callables = diffusion_fit_binding.build_callables()

        # EXPECT
        assert len(callables) == 1
        assert callable(callables[0])
        assert callables[0](0) == pytest.approx(6.0)  # 2.0 * 3.0
        mock_model.calculate_QISF.assert_called_once_with(0)

    def test_get_model_names(self, fit_binding):
        # WHEN THEN
        model_names = fit_binding.get_model_names()

        # EXPECT
        assert model_names == ['Gaussian']

    def test_get_model_names_diffusion(self, diffusion_fit_binding):
        # WHEN THEN
        model_names = diffusion_fit_binding.get_model_names()

        # EXPECT
        assert model_names == [
            'BrownianTranslationalDiffusion area',
            'BrownianTranslationalDiffusion width',
        ]

    def test_get_parameter_names(self, fit_binding):
        # WHEN THEN
        parameter_names = fit_binding.get_parameter_names()

        # EXPECT
        assert parameter_names == ['parameter1']

    def test_get_parameter_names_diffusion(self, diffusion_fit_binding):
        # WHEN THEN
        parameter_names = diffusion_fit_binding.get_parameter_names()

        # EXPECT
        assert parameter_names == ['parameter3 area', 'parameter3 width']

    def test_unit_contract_for_component_model(self, fit_binding):
        # WHEN THEN EXPECT: component bindings expose the model's units, so ParameterAnalysis
        # converts data into them before fitting
        assert fit_binding.x_unit == fit_binding.model.x_unit
        assert fit_binding.y_unit == fit_binding.model.y_unit

    def test_unit_contract_for_diffusion_model(self, diffusion_fit_binding):
        # WHEN THEN EXPECT: diffusion-model callables take raw Q values, so the binding
        # declares no units and no conversion is applied (regression: an isinstance check in
        # ParameterAnalysis used to encode this knowledge)
        assert diffusion_fit_binding.x_unit is None
        assert diffusion_fit_binding.y_unit is None

    def test_get_parameter_names_diffusion_delta_mode_maps_to_area(self, diffusion_fit_binding):
        # WHEN: the delta contribution is stored in the parameters Dataset as the
        # DeltaFunction's area parameter ('<name> area'), so mode 'delta' must map to
        # '<parameter_name> area' while other modes keep their own suffix.
        diffusion_fit_binding.modes = ['delta', 'width']

        # THEN
        parameter_names = diffusion_fit_binding.get_parameter_names()

        # EXPECT
        assert parameter_names == ['parameter3 area', 'parameter3 width']

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def test_build_diffusion_callable(self, diffusion_fit_binding):

        # WHEN
        mock_model = Mock()
        mock_model.calculate_QISF.return_value = 2.0
        mock_model.scale.value = 3.0
        mock_model.calculate_width.return_value = 0.5
        diffusion_fit_binding._model = mock_model

        # THEN
        area_callable = diffusion_fit_binding._build_diffusion_callable(mode='area')
        width_callable = diffusion_fit_binding._build_diffusion_callable(mode='width')

        # EXPECT
        assert area_callable(0) == pytest.approx(6.0)  # 2.0 * 3.0
        mock_model.calculate_QISF.assert_called_once_with(0)

        assert width_callable(0) == pytest.approx(0.5)
        mock_model.calculate_width.assert_called_once_with(0)

        # THEN
        result_area = area_callable(0, unused_arg=123)
        result_width = width_callable(0, unused_arg=123)

        # EXPECT
        assert result_area == pytest.approx(6.0)  # Should ignore unused_arg
        assert result_width == pytest.approx(0.5)  # Should ignore unused_arg

    def test_build_diffusion_callable_delta_mode(self, diffusion_fit_binding):
        # WHEN
        mock_model = Mock()
        mock_model.calculate_EISF.return_value = 0.8
        mock_model.scale.value = 2.0
        diffusion_fit_binding._model = mock_model

        # THEN
        delta_callable = diffusion_fit_binding._build_diffusion_callable(mode='delta')

        # EXPECT
        assert delta_callable(0) == pytest.approx(1.6)
        mock_model.calculate_EISF.assert_called_once_with(0)

    def test_build_diffusion_callable_errors(self, diffusion_fit_binding):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='Unknown diffusion mode: invalid_mode'):
            diffusion_fit_binding._build_diffusion_callable(mode='invalid_mode')

    def test_get_modes(self, diffusion_fit_binding):
        # WHEN
        modes = diffusion_fit_binding._get_modes()

        # EXPECT
        assert modes == ['area', 'width']

        # THEN
        diffusion_fit_binding.modes = 'area'
        modes = diffusion_fit_binding._get_modes()
        # EXPECT
        assert modes == ['area']

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------
    def test_repr(self, fit_binding):
        # WHEN
        repr_str = repr(fit_binding)

        # THEN EXPECT
        assert 'FitBinding' in repr_str
        assert "parameter_name='parameter1'" in repr_str
        assert "model='Gaussian'" in repr_str
        assert 'modes=None' in repr_str
