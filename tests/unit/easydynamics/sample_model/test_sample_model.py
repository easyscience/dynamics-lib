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
from easydynamics.sample_model.sample_model import SampleModel


class TestSampleModel:
    @pytest.fixture
    def sample_model(self):
        component1 = Gaussian(
            display_name='TestGaussian1',
            unique_name='TestGaussian1',
            area=1.0,
            center=0.0,
            width=1.0,
            unit='meV',
        )
        component2 = Lorentzian(
            display_name='TestLorentzian1',
            unique_name='TestLorentzian1',
            area=2.0,
            center=1.0,
            width=0.5,
            unit='meV',
        )
        component_collection = ComponentCollection()
        component_collection.append_component(component1)
        component_collection.append_component(component2)

        diffusion_model = BrownianTranslationalDiffusion(
            display_name='DiffusionModel', unique_name='DiffusionModel'
        )

        sample_model = SampleModel(
            display_name='InitModel',
            components=component_collection,
            diffusion_models=diffusion_model,
            unit='meV',
            Q=np.array([1.0, 2.0, 3.0]),
            temperature=10.0,
        )

        return sample_model

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
        assert model.temperature.value == 10.0
        assert model.divide_by_temperature is True
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))

    def test_init_list_of_diffusion_model(self):
        # WHEN THEN
        diffusion_model1 = BrownianTranslationalDiffusion()
        diffusion_model2 = BrownianTranslationalDiffusion()
        sample_model = SampleModel(diffusion_models=[diffusion_model1, diffusion_model2])

        # EXPECT
        assert len(sample_model.diffusion_models) == 2
        assert sample_model.diffusion_models[0] is diffusion_model1
        assert sample_model.diffusion_models[1] is diffusion_model2

    def test_init_raises_with_invalid_diffusion_model(self):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='diffusion_models must be ',
        ):
            SampleModel(diffusion_models='invalid_diffusion_model')

    def test_init_raises_with_invalid_temperature(self):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='temperature must be a number or None',
        ):
            SampleModel(temperature='invalid_temperature')

    def test_init_raises_with_negative_temperature(self):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            ValueError,
            match='temperature must be non-negative',
        ):
            SampleModel(temperature=-5.0)

    def test_init_raises_with_invalid_divide_by_temperature(self):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='divide_by_temperature must be True or False',
        ):
            SampleModel(divide_by_temperature='invalid_value')

    def test_append_and_remove_and_clear_diffusion_model(self, sample_model):
        # WHEN
        model = sample_model
        new_diffusion_model = BrownianTranslationalDiffusion(
            unique_name='new_diffusion_model',
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
        new_diffusion_model1 = BrownianTranslationalDiffusion()
        new_diffusion_model2 = BrownianTranslationalDiffusion()

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
            [BrownianTranslationalDiffusion(), 'invalid_diffusion_model'],
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
        assert model.temperature.value == 20.0

        # THEN
        model.temperature = None

        # EXPECT
        assert model.temperature is None

        # THEN
        model.temperature = 0.0

        # EXPECT
        assert model.temperature.value == 0.0

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
            match='temperature must be a number or None|temperature must be non-negative',
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

    def test_divide_by_temperature_setter(self, sample_model):
        # WHEN
        model = sample_model

        # THEN
        model.divide_by_temperature = False

        # EXPECT
        assert model.divide_by_temperature is False

        # THEN
        model.divide_by_temperature = True

        # EXPECT
        assert model.divide_by_temperature is True

    def test_divide_by_temperature_setter_raises_with_invalid_type(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='divide_by_temperature must be True or False',
        ):
            sample_model.divide_by_temperature = 'invalid_value'

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
                divide_by_temperature=sample_model.divide_by_temperature,
                energy_unit=sample_model.unit,
            )

            # Check that evaluate was called on each component
            collection1.evaluate.assert_called_once_with(x)
            collection2.evaluate.assert_called_once_with(x)

            # Check that DBF was applied elementwise
            np.testing.assert_allclose(result[0], np.array([1.0, 2.0, 3.0]) * 10.0)
            np.testing.assert_allclose(result[1], np.array([4.0, 5.0, 6.0]) * 10.0)

    def test_generate_component_collections(self, sample_model):
        # WHEN THEN
        sample_model._generate_component_collections()

        # EXPECT
        assert len(sample_model._component_collections) == 3  # 3 Q values
        for collection in sample_model._component_collections:
            assert isinstance(collection, ComponentCollection)
            assert len(collection.components) == 3  # 3 components
            assert collection.components[0].display_name == 'TestGaussian1'
            assert collection.components[0].area.value == 1.0
            assert collection.components[1].display_name == 'TestLorentzian1'
            assert collection.components[1].area.value == 2.0
            assert collection.components[2].display_name == 'Brownian diffusion'
            assert isinstance(collection.components[2], Lorentzian)

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
        assert 'divide_by_temperature' in repr_str
