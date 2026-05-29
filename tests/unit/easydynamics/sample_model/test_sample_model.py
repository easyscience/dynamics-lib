# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
from scipp import UnitError

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.diffusion_model.brownian_translational_diffusion import (
    BrownianTranslationalDiffusion,
)
from easydynamics.sample_model.diffusion_model.delta_lorentz import DeltaLorentz
from easydynamics.sample_model.sample_model import SampleModel
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings


class TestSampleModel:
    @pytest.fixture
    def sample_model(self):
        component1 = Gaussian(
            name='TestGaussian1Name',
            display_name='TestGaussian1Display',
            area=1.0,
            center=0.0,
            width=1.0,
            unit='meV',
        )
        component2 = Lorentzian(
            name='TestLorentzian1Name',
            display_name='TestLorentzian1Display',
            area=2.0,
            center=1.0,
            width=0.5,
            unit='meV',
        )
        component_collection = ComponentCollection()
        component_collection.append_component(component1)
        component_collection.append_component(component2)

        diffusion_model = BrownianTranslationalDiffusion(
            display_name='DiffusionModelDisplay', name='DiffusionModelName'
        )

        return SampleModel(
            display_name='InitModel',
            components=component_collection,
            diffusion_models=diffusion_model,
            unit='meV',
            Q=np.array([1.0, 2.0, 3.0]),
            temperature=10.0,
        )

    def test_init(self, sample_model):

        # WHEN THEN
        model = sample_model

        # EXPECT
        assert model.display_name == 'InitModel'
        assert model.unit == 'meV'
        assert len(model.components) == 2
        assert isinstance(model.diffusion_models, list)
        assert len(model.diffusion_models) == 1
        assert isinstance(model.diffusion_models[0], BrownianTranslationalDiffusion)
        assert model.temperature.value == pytest.approx(10.0)
        assert model.normalize_detailed_balance is True
        assert model.use_detailed_balance is True
        assert isinstance(model.detailed_balance_settings, DetailedBalanceSettings)
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))

    def test_init_custom_input(self):
        # WHEN THEN
        diffusion_model1 = BrownianTranslationalDiffusion()
        diffusion_model2 = BrownianTranslationalDiffusion()

        detailed_balance_settings = DetailedBalanceSettings(
            use_detailed_balance=False,
            normalize_detailed_balance=False,
        )
        sample_model = SampleModel(
            diffusion_models=[diffusion_model1, diffusion_model2],
            detailed_balance_settings=detailed_balance_settings,
        )

        # EXPECT
        assert len(sample_model.diffusion_models) == 2
        assert sample_model.diffusion_models[0] is diffusion_model1
        assert sample_model.diffusion_models[1] is diffusion_model2
        assert sample_model.use_detailed_balance is False
        assert sample_model.normalize_detailed_balance is False
        assert sample_model.detailed_balance_settings is detailed_balance_settings

    @pytest.mark.parametrize(
        'invalid_input, expected_exception, match',
        [
            # diffusion_models
            (
                {'diffusion_models': 'invalid_diffusion_model'},
                TypeError,
                'diffusion_models must be a DiffusionModelBase',
            ),
            # temperature
            (
                {'temperature': 'invalid_temperature'},
                TypeError,
                'temperature must be a number or None',
            ),
            (
                {'temperature': -5.0},
                ValueError,
                'temperature must be non-negative',
            ),
            # detailed_balance_settings
            (
                {'detailed_balance_settings': 'invalid_settings'},
                TypeError,
                'detailed_balance_settings must be a DetailedBalanceSettings or None',
            ),
        ],
        ids=[
            'diffusion_models_invalid_type',
            'temperature_not_numeric',
            'temperature_negative',
            'detailed_balance_settings_invalid_type',
        ],
    )
    def test_init_raises_for_invalid_input(self, invalid_input, expected_exception, match):
        """
        Test that initialization raises appropriate exceptions for
        invalid input parameters.
        """
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=match):
            SampleModel(**invalid_input)

    def test_append_and_remove_and_clear_diffusion_model(self, sample_model):
        # WHEN
        model = sample_model
        new_diffusion_model = BrownianTranslationalDiffusion(
            name='new_diffusion_model',
        )

        # THEN
        model.append_diffusion_model(new_diffusion_model)

        # EXPECT
        assert len(model.diffusion_models) == 2
        assert model.diffusion_models[1] is new_diffusion_model

        # THEN
        model.remove_diffusion_model('new_diffusion_model')

        # EXPECT
        assert len(model.diffusion_models) == 1

        # THEN
        model.clear_diffusion_models()
        # EXPECT
        assert len(model.diffusion_models) == 0

    def test_append_diffusion_model_raises_with_invalid_type(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='diffusion_model must be a DiffusionModelBase',
        ):
            sample_model.append_diffusion_model('invalid_diffusion_model')

    def test_remove_diffusion_model_raises_with_invalid_name(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            ValueError,
            match='No DiffusionModel',
        ):
            sample_model.remove_diffusion_model('non_existent_model')

    def test_diffusion_model_setter(self, sample_model):
        # WHEN
        model = sample_model
        new_diffusion_model1 = BrownianTranslationalDiffusion(name='new_diffusion_model1')
        new_diffusion_model2 = BrownianTranslationalDiffusion(name='new_diffusion_model2')

        # THEN
        model.diffusion_models = [new_diffusion_model1, new_diffusion_model2]

        # EXPECT
        assert len(model.diffusion_models) == 2
        assert model.diffusion_models[0] is new_diffusion_model1
        assert model.diffusion_models[1] is new_diffusion_model2

        # THEN
        model.diffusion_models = None

        # EXPECT
        assert len(model.diffusion_models) == 0

        # THEN
        model.diffusion_models = new_diffusion_model1

        # EXPECT
        assert len(model.diffusion_models) == 1
        assert model.diffusion_models[0] is new_diffusion_model1

    @pytest.mark.parametrize(
        'invalid_value',
        [
            'invalid_diffusion_model',
            123,
            [
                BrownianTranslationalDiffusion(name='valid_diffusion_model'),
                'invalid_diffusion_model',
            ],
        ],
        ids=['string', 'integer', 'list with invalid type'],
    )
    def test_diffusion_model_setter_raises_with_invalid_type(self, invalid_value, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='diffusion_models must be ',
        ):
            sample_model.diffusion_models = invalid_value

    def test_temperature_setter(self, sample_model):
        # WHEN
        model = sample_model

        # THEN
        model.temperature = 20.0

        # EXPECT
        assert model.temperature.value == pytest.approx(20.0)

        # THEN
        model.temperature = None

        # EXPECT
        assert model.temperature is None

        # THEN
        model.temperature = 0.0

        # EXPECT
        assert model.temperature.value == pytest.approx(0.0)

    @pytest.mark.parametrize(
        'invalid_value',
        [
            'invalid_temperature',
            [1, 2, 3],
            {'temp': 10},
            -5.0,
        ],
        ids=['string', 'list', 'dict', 'negative'],
    )
    def test_temperature_setter_raises_with_invalid_type(self, invalid_value, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            (TypeError, ValueError),
            match=r'temperature must be a number or None|temperature must be non-negative',
        ):
            sample_model.temperature = invalid_value

    def test_temperature_unit_setter_raises(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            AttributeError,
            match='Temperature_unit is read-only',
        ):
            sample_model.temperature_unit = 123

    def test_convert_temperature_unit(self, sample_model):
        # WHEN
        model = sample_model

        # THEN
        model.convert_temperature_unit('mK')

        # EXPECT
        assert model.temperature_unit == 'mK'
        assert model.temperature.value == 10 * 1000

    def test_convert_temperature_unit_raises_with_no_temperature(self, sample_model):
        # WHEN
        model = sample_model
        model.temperature = None

        # THEN / EXPECT
        with pytest.raises(
            ValueError,
            match='Temperature is not set, cannot convert unit',
        ):
            model.convert_temperature_unit('mK')

    def test_convert_temperature_unit_raises_with_invalid_unit(self, sample_model):
        # WHEN
        model = sample_model

        # THEN / EXPECT
        with pytest.raises(
            UnitError,
            match='Failed to',
        ):
            model.convert_temperature_unit('invalid_unit')

    def test_normalize_detailed_balance_setter(self, sample_model):
        # WHEN
        model = sample_model

        # THEN
        model.normalize_detailed_balance = False

        # EXPECT
        assert model.normalize_detailed_balance is False

        # THEN
        model.normalize_detailed_balance = True

        # EXPECT
        assert model.normalize_detailed_balance is True

    def test_normalize_detailed_balance_setter_raises_with_invalid_type(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='normalize_detailed_balance must be True or False',
        ):
            sample_model.normalize_detailed_balance = 'invalid_value'

    def test_use_detailed_balance_setter(self, sample_model):
        # WHEN
        model = sample_model

        # THEN
        model.use_detailed_balance = False

        # EXPECT
        assert model.use_detailed_balance is False

        # THEN
        model.use_detailed_balance = True

        # EXPECT
        assert model.use_detailed_balance is True

    def test_use_detailed_balance_setter_raises_with_invalid_type(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='use_detailed_balance must be True or False',
        ):
            sample_model.use_detailed_balance = 'invalid_value'

    def test_detailed_balance_settings_property(self, sample_model):
        # WHEN
        new_settings = DetailedBalanceSettings(
            use_detailed_balance=False, normalize_detailed_balance=False
        )

        # THEN
        sample_model.detailed_balance_settings = new_settings

        # EXPECT
        assert sample_model.detailed_balance_settings is new_settings
        assert sample_model.use_detailed_balance is False
        assert sample_model.normalize_detailed_balance is False

    def test_detailed_balance_settings_setter_invalid(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='detailed_balance_settings must be a DetailedBalanceSettings',
        ):
            sample_model.detailed_balance_settings = 'invalid_settings'

    def test_evaluate_calls_dbf(self, sample_model):
        # WHEN
        x = np.array([0.0, 1.0, 2.0])

        collection1 = Mock()
        collection2 = Mock()

        collection1.evaluate.return_value = np.array([1.0, 2.0, 3.0])
        collection2.evaluate.return_value = np.array([4.0, 5.0, 6.0])

        sample_model._component_collections = [collection1, collection2]

        with patch('easydynamics.sample_model.sample_model.detailed_balance_factor') as mock_dbf:
            mock_dbf.return_value = np.array([10.0, 10.0, 10.0])  # simplified DBF
            # THEN
            result = sample_model.evaluate(x)

            # EXPECT
            # Check that DBF was called with correct arguments
            mock_dbf.assert_called_once_with(
                energy=x,
                temperature=sample_model.temperature,
                divide_by_temperature=sample_model.normalize_detailed_balance,
                energy_unit=sample_model.unit,
            )

            # Check that evaluate was called on each component
            collection1.evaluate.assert_called_once_with(x)
            collection2.evaluate.assert_called_once_with(x)

            # Check that DBF was applied elementwise
            np.testing.assert_allclose(result[0], np.array([1.0, 2.0, 3.0]) * 10.0)
            np.testing.assert_allclose(result[1], np.array([4.0, 5.0, 6.0]) * 10.0)

    @pytest.mark.parametrize(
        'temperature, use_detailed_balance',
        [
            (None, True),  # DB disabled because temperature is None
            (300.0, False),  # DB disabled explicitly
        ],
        ids=[
            'temperature_none',
            'use_detailed_balance_false',
        ],
    )
    def test_evaluate_doesnt_call_dbf_when_disabled(
        self, sample_model, temperature, use_detailed_balance
    ):
        # WHEN
        x = np.array([0.0, 1.0, 2.0])

        collection1 = Mock()
        collection2 = Mock()

        collection1.evaluate.return_value = np.array([1.0, 2.0, 3.0])
        collection2.evaluate.return_value = np.array([4.0, 5.0, 6.0])

        sample_model._component_collections = [collection1, collection2]

        sample_model.temperature = temperature
        sample_model.use_detailed_balance = use_detailed_balance

        with patch('easydynamics.sample_model.sample_model.detailed_balance_factor') as mock_dbf:
            mock_dbf.return_value = np.array([10.0, 10.0, 10.0])  # simplified DBF
            # THEN
            result = sample_model.evaluate(x)

            # EXPECT
            # Check that DBF was not called since detailed balance is disabled
            mock_dbf.assert_not_called()

            # Check that evaluate was called on each component
            collection1.evaluate.assert_called_once_with(x)
            collection2.evaluate.assert_called_once_with(x)

            # Check that results were not modified by DBF
            np.testing.assert_allclose(result[0], np.array([1.0, 2.0, 3.0]))
            np.testing.assert_allclose(result[1], np.array([4.0, 5.0, 6.0]))

    def test_generate_component_collections(self, sample_model):
        # WHEN THEN
        sample_model._generate_component_collections()

        # EXPECT
        assert len(sample_model._component_collections) == 3  # 3 Q values
        for collection in sample_model._component_collections:
            assert isinstance(collection, ComponentCollection)
            assert len(list(collection)) == 3  # 3 components
            assert collection[0].name == 'TestGaussian1Name'
            assert collection[0].area.value == pytest.approx(1.0)
            assert collection[1].name == 'TestLorentzian1Name'
            assert collection[1].area.value == pytest.approx(2.0)
            assert isinstance(collection[2], Lorentzian)

    def test_get_all_variables(self, sample_model):
        # WHEN

        # THEN
        all_vars = sample_model.get_all_variables()

        # EXPECT
        # Should include temperature and variables from diffusion model
        expected_num_vars = 3 * 3 * 3  # 3 components, each with 3 parameters, across 3 Q values
        expected_num_vars += 2  # diffusion model has 2 parameters
        expected_num_vars += 1  # temperature variable

        assert len(all_vars) == expected_num_vars
        assert sample_model.temperature in all_vars
        for var in sample_model.diffusion_models[0].get_all_variables():
            assert var in all_vars

        # Template component variables should NOT be included
        template_vars = []
        for component in sample_model.components:
            template_vars.extend(component.get_all_variables())

        for var in template_vars:
            assert var not in all_vars

        assert len(set(all_vars)) == len(all_vars)  # all variables should be unique

    def test_get_all_variables_delta_lorentz(self):
        # WHEN
        sample_model = SampleModel(
            Q=np.array([1.0, 2.0, 3.0]),
        )
        delta_lorentz = DeltaLorentz(allow_Q_variation={'A_0': True, 'lorentzian_width': True})
        sample_model.diffusion_models = delta_lorentz

        # THEN
        variables = sample_model.get_all_variables()

        # EXPECT
        delta_lorentz_vars = delta_lorentz.get_all_variables()

        assert set(variables) == set(delta_lorentz_vars)

        # THEN
        variables = sample_model.get_all_variables(Q_index=0)

        # EXPECT
        delta_lorentz_vars = delta_lorentz.get_all_variables(Q_index=0)

        assert set(variables) == set(delta_lorentz_vars)

    def test_repr(self, sample_model):
        # WHEN
        repr_str = repr(sample_model)

        # THEN / EXPECT
        assert 'SampleModel' in repr_str
        assert 'unit=' in repr_str
        assert 'Q = ' in repr_str
        assert 'components' in repr_str
        assert 'diffusion_models' in repr_str
        assert 'temperature' in repr_str
        assert 'normalize_detailed_balance' in repr_str
