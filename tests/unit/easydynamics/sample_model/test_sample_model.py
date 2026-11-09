# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
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
            x_unit='meV',
        )
        component2 = Lorentzian(
            name='TestLorentzian1Name',
            display_name='TestLorentzian1Display',
            area=2.0,
            center=1.0,
            width=0.5,
            x_unit='meV',
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
            x_unit='meV',
            Q=np.array([1.0, 2.0, 3.0]),
            temperature=10.0,
        )

    def test_init(self, sample_model):

        # WHEN THEN
        model = sample_model

        # EXPECT
        assert model.display_name == 'InitModel'
        assert model.x_unit == 'meV'
        assert model.y_unit == 'dimensionless'
        assert len(model.components) == 2
        assert isinstance(model.diffusion_models, list)
        assert len(model.diffusion_models) == 1
        assert isinstance(model.diffusion_models[0], BrownianTranslationalDiffusion)
        assert model.temperature.value == pytest.approx(10.0)
        assert model.normalize_detailed_balance is True
        assert model.use_detailed_balance is True
        assert isinstance(model.detailed_balance_settings, DetailedBalanceSettings)
        np.testing.assert_array_equal(model.Q.values, np.array([1.0, 2.0, 3.0]))

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

    def test_convert_x_unit_propagates_to_diffusion_models(self):
        # WHEN: a SampleModel with an attached diffusion model
        Q = np.array([1.0, 2.0])
        brownian = BrownianTranslationalDiffusion(diffusion_coefficient=2.4e-9, Q=Q)
        model = SampleModel(Q=Q, diffusion_models=brownian)
        width_mev = model.get_component_collection(0)[0].width.value

        # THEN
        model.convert_x_unit('ueV')

        # EXPECT: the diffusion model follows the SampleModel's new unit, and the merged
        # collections carry rescaled widths in the new unit
        assert sc.Unit(str(brownian.x_unit)) == sc.Unit('ueV')
        assert sc.Unit(str(brownian.scale.unit)) == sc.Unit('ueV')
        collection = model.get_component_collection(0)
        assert sc.Unit(str(collection[0].width.unit)) == sc.Unit('ueV')
        assert collection[0].width.value == pytest.approx(width_mev * 1000)

    def test_convert_x_unit_does_not_mark_collections_dirty(self):
        # WHEN: a SampleModel with an attached diffusion model and built collections
        Q = np.array([1.0, 2.0])
        brownian = BrownianTranslationalDiffusion(diffusion_coefficient=2.4e-9, Q=Q)
        model = SampleModel(Q=Q, diffusion_models=brownian)
        collection_before = model.get_component_collection(0)
        width = collection_before[0].width

        # THEN
        model.convert_x_unit('ueV')

        # EXPECT: conversion works in place — no dirty flag, no rebuild, and the converted
        # dependent width stays in the new unit through later dependency-graph updates
        # (regression: conversion used to mark the collections dirty, and the rebuild
        # discarded per-Q state and object references)
        assert model._component_collections_is_dirty is False
        assert model.get_component_collection(0) is collection_before
        width_uev = width.value
        brownian.diffusion_coefficient = 4.8e-9
        assert sc.Unit(str(width.unit)) == sc.Unit('ueV')
        assert width.value == pytest.approx(2 * width_uev)

    def test_convert_y_unit_propagates_to_diffusion_models(self):
        # WHEN: SampleModel and diffusion model sharing y_unit='1/meV'
        Q = np.array([1.0])
        brownian = BrownianTranslationalDiffusion(
            diffusion_coefficient=2.4e-9, Q=Q, y_unit='1/meV'
        )
        model = SampleModel(Q=Q, y_unit='1/meV', diffusion_models=brownian)

        # THEN
        model.convert_y_unit('1/eV')

        # EXPECT
        assert sc.Unit(str(brownian.y_unit)) == sc.Unit('1/eV')
        assert sc.Unit(str(brownian.scale.unit)) == sc.Unit('meV/eV')
        assert sc.Unit(str(model.y_unit)) == sc.Unit('1/eV')

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
        sample_model._component_collections_is_dirty = False
        sample_model._built_diffusion_versions = sample_model._diffusion_versions()

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
                energy_unit=sample_model.x_unit,
            )

            # Check that evaluate was called on each component
            collection1.evaluate.assert_called_once_with(x, output='numpy')
            collection2.evaluate.assert_called_once_with(x, output='numpy')

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
        sample_model._component_collections_is_dirty = False
        sample_model._built_diffusion_versions = sample_model._diffusion_versions()

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
            collection1.evaluate.assert_called_once_with(x, output='numpy')
            collection2.evaluate.assert_called_once_with(x, output='numpy')

            # Check that results were not modified by DBF
            np.testing.assert_allclose(result[0], np.array([1.0, 2.0, 3.0]))
            np.testing.assert_allclose(result[1], np.array([4.0, 5.0, 6.0]))

    def test_evaluate_scipp_output_with_detailed_balance(self, sample_model):
        # WHEN the fixture has temperature set, so detailed balance is applied
        x = np.linspace(-2.0, 2.0, 21)

        # THEN (regression: multiplying an sc.Variable with the numpy DBF used to raise)
        balanced = sample_model.evaluate(x, output='scipp')

        # EXPECT scipp output matches numpy output, with the model's y_unit kept
        reference = sample_model.evaluate(x, output='numpy')
        assert len(balanced) == 3
        for scipp_values, numpy_values in zip(balanced, reference, strict=True):
            assert isinstance(scipp_values, sc.Variable)
            assert scipp_values.unit == sc.Unit('dimensionless')
            np.testing.assert_allclose(scipp_values.values, numpy_values)

        # THEN disabling detailed balance
        sample_model.use_detailed_balance = False
        unbalanced = sample_model.evaluate(x, output='scipp')

        # EXPECT the detailed balance factor really was applied above
        assert not np.allclose(balanced[0].values, unbalanced[0].values)

    def test_evaluate_dataarray_input_with_and_without_detailed_balance(self, sample_model):
        # WHEN
        x = np.linspace(-2.0, 2.0, 21)
        data_array = sc.DataArray(
            data=sc.array(dims=['energy'], values=np.zeros_like(x)),
            coords={'energy': sc.array(dims=['energy'], values=x, unit='meV')},
        )

        # THEN (regression: detailed balance used to reject DataArray x, which the
        # component pipeline explicitly supports)
        with_temperature = sample_model.evaluate(data_array)
        reference_with = sample_model.evaluate(x)

        sample_model.temperature = None
        without_temperature = sample_model.evaluate(data_array)
        reference_without = sample_model.evaluate(x)

        # EXPECT DataArray input matches plain numpy input in both modes
        for result, reference in zip(with_temperature, reference_with, strict=True):
            np.testing.assert_allclose(result, reference)
        for result, reference in zip(without_temperature, reference_without, strict=True):
            np.testing.assert_allclose(result, reference)

    def test_init_invalid_temperature_does_not_mutate_diffusion_models(self):
        # WHEN a diffusion model without Q and an invalid temperature
        diffusion_model = BrownianTranslationalDiffusion()

        # THEN EXPECT construction fails on the temperature validation
        with pytest.raises(TypeError, match='temperature must be a number or None'):
            SampleModel(
                diffusion_models=diffusion_model,
                Q=np.array([1.0, 2.0]),
                temperature='cold',
            )

        # EXPECT the failed construction did not mutate the passed diffusion model
        assert diffusion_model.Q is None
        assert diffusion_model.get_component_collections() == []

    def test_temperature_unit_is_normalized_to_str(self, sample_model):
        # WHEN constructed with a scipp Unit instead of a string
        model = SampleModel(temperature=10.0, temperature_unit=sc.Unit('K'))

        # EXPECT the stored unit is normalized to a string
        assert isinstance(model.temperature_unit, str)
        assert model.temperature_unit == 'K'

        # THEN converting with a scipp Unit
        sample_model.convert_temperature_unit(sc.Unit('mK'))

        # EXPECT the stored unit is normalized to a string as well
        assert isinstance(sample_model.temperature_unit, str)
        assert sample_model.temperature_unit == 'mK'

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
        assert 'Q = ' in repr_str
        assert 'components' in repr_str
        assert 'diffusion_models' in repr_str
        assert 'temperature' in repr_str
        # Regression: a stray ')' and a missing separator used to mangle this into
        # 'x_unit=meV), y_unit=dimensionlessQ = ...', and the closing ')' was missing
        assert 'x_unit=meV, y_unit=dimensionless' in repr_str
        assert repr_str.rstrip().endswith(')')
        assert 'normalize_detailed_balance' in repr_str

    def test_y_unit_setter_raises(self, sample_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(AttributeError):
            sample_model.y_unit = '1/meV'

    def test_convert_y_unit(self):
        # WHEN
        g = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        model = SampleModel(components=g, x_unit='meV')
        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        model.convert_y_unit('1/eV')
        # EXPECT
        assert model.y_unit == '1/eV'
        assert model.components[0].y_unit == '1/eV'
        assert g.area.value == pytest.approx(1e3)

    def test_remove_diffusion_model_raises_with_duplicate_names(self):
        # WHEN a SampleModel with two DiffusionModels sharing a name
        Q = np.linspace(0.5, 2.0, 3)
        model = SampleModel(
            Q=Q,
            diffusion_models=[
                BrownianTranslationalDiffusion(name='dup'),
                BrownianTranslationalDiffusion(name='dup'),
            ],
        )
        # THEN EXPECT
        with pytest.raises(ValueError, match=r'Multiple DiffusionModels share the name'):
            model.remove_diffusion_model('dup')

    def test_convert_x_unit_rolls_back_when_diffusion_model_conversion_fails(self):
        # WHEN a SampleModel whose diffusion model raises during x-unit conversion
        Q = np.linspace(0.5, 2.0, 3)
        brownian = BrownianTranslationalDiffusion()
        model = SampleModel(Q=Q, diffusion_models=brownian)
        original_unit = model.x_unit
        # THEN EXPECT the conversion fails partway through
        with (
            patch.object(brownian, 'convert_x_unit', side_effect=RuntimeError('boom')),
            pytest.raises(RuntimeError, match='boom'),
        ):
            model.convert_x_unit('ueV')
        # EXPECT the model's own x_unit to be rolled back to the original
        assert model.x_unit == original_unit

    #############
    # Incremental per-Q updates (issue #220)
    #############

    def test_append_component_inserts_before_diffusion_components(self, sample_model):
        # WHEN per-Q collections are built
        _ = sample_model.get_component_collection(0)

        # THEN
        sample_model.append_component(Gaussian(name='NewGaussian'))

        # EXPECT the order matches a full rebuild: template components first
        patched = [
            sample_model.get_component_collection(i).list_component_names()
            for i in range(len(sample_model.Q))
        ]
        sample_model._generate_component_collections()
        rebuilt = [
            sample_model.get_component_collection(i).list_component_names()
            for i in range(len(sample_model.Q))
        ]
        assert patched == rebuilt

    def test_append_diffusion_model_keeps_per_Q_state(self, sample_model):
        # WHEN per-Q collections are built and one per-Q parameter is fitted
        width = sample_model.get_component_collection(1)['TestGaussian1Name'].width
        width.value = 0.123
        version = sample_model.state_version
        new_model = BrownianTranslationalDiffusion(name='NewDiffusion', lorentzian_name='NewL')

        # THEN
        sample_model.append_diffusion_model(new_model)

        # EXPECT the new model's components are added at every Q, and the fitted value is kept
        for i in range(len(sample_model.Q)):
            collection = sample_model.get_component_collection(i)
            for component in new_model.get_component_collections(i):
                assert any(c is component for c in collection)
        assert sample_model.get_component_collection(1)['TestGaussian1Name'].width is width
        assert width.value == pytest.approx(0.123)
        assert sample_model.state_version > version
        assert sample_model.component_collections_is_dirty is False

    def test_remove_diffusion_model_keeps_per_Q_state(self, sample_model):
        # WHEN per-Q collections are built and one per-Q parameter is fitted
        width = sample_model.get_component_collection(1)['TestGaussian1Name'].width
        width.value = 0.123
        removed = sample_model.diffusion_models[0]
        version = sample_model.state_version

        # THEN
        sample_model.remove_diffusion_model('DiffusionModelName')

        # EXPECT only the template components remain, with the fitted value kept
        template_names = sample_model.components.list_component_names()
        for i in range(len(sample_model.Q)):
            collection = sample_model.get_component_collection(i)
            assert collection.list_component_names() == template_names
            for component in removed.get_component_collections(i):
                assert not any(c is component for c in collection)
        assert sample_model.get_component_collection(1)['TestGaussian1Name'].width is width
        assert width.value == pytest.approx(0.123)
        assert sample_model.state_version > version
        assert sample_model.component_collections_is_dirty is False

    def test_remove_diffusion_model_rebuilds_when_components_not_found(self, sample_model):
        # WHEN per-Q collections are built, but the diffusion model has since regenerated its own
        _ = sample_model.get_component_collection(0)
        sample_model.diffusion_models[0].create_component_collections()

        # THEN
        sample_model.remove_diffusion_model('DiffusionModelName')

        # EXPECT a full rebuild is scheduled instead of a partial in-place removal
        assert sample_model.component_collections_is_dirty is True
        names = sample_model.get_component_collection(0).list_component_names()
        assert names == sample_model.components.list_component_names()

    def test_append_diffusion_model_without_Q_does_not_raise(self):
        "Regression: a diffusion model with its own Q on a Q-less SampleModel raised in zip"
        # WHEN a SampleModel without Q whose (empty) per-Q collections are current
        model = SampleModel(components=Gaussian(name='A'))
        model.get_all_variables()
        diffusion_model = BrownianTranslationalDiffusion(name='D', Q=[0.5, 1.0])

        # THEN
        model.append_diffusion_model(diffusion_model)

        # EXPECT the model is added and a rebuild is scheduled instead of a patch
        assert model.diffusion_models == [diffusion_model]
        assert model.component_collections_is_dirty is True

    def test_append_diffusion_model_rebuilds_when_collections_stale(self, sample_model):
        # WHEN per-Q collections have not been built yet
        new_model = BrownianTranslationalDiffusion(name='NewDiffusion', lorentzian_name='NewL')

        # THEN
        sample_model.append_diffusion_model(new_model)

        # EXPECT nothing is patched; the next access rebuilds with the new model's components
        assert sample_model.component_collections_is_dirty is True
        collection = sample_model.get_component_collection(0)
        for component in new_model.get_component_collections(0):
            assert any(c is component for c in collection)

    @pytest.mark.filterwarnings('ignore:Duplicate component names')
    def test_remove_component_rebuilds_when_edit_hidden_by_name_clash(self):
        "Regression: a diffusion component sharing a template name masked a per-Q edit"
        # WHEN a template component and a diffusion component share the name 'L', and the
        # template 'L' was popped from one per-Q collection directly
        diffusion_model = BrownianTranslationalDiffusion(name='D', lorentzian_name='L')
        model = SampleModel(
            components=ComponentCollection(components=[Gaussian(name='A'), Lorentzian(name='L')]),
            diffusion_models=diffusion_model,
            Q=[0.5, 1.0],
        )
        model.get_component_collection(0).pop(1)

        # THEN
        model.remove_component('L')

        # EXPECT a full rebuild, so the diffusion component survives at every Q
        assert model.component_collections_is_dirty is True
        for i in range(len(model.Q)):
            collection = model.get_component_collection(i)
            for component in diffusion_model.get_component_collections(i):
                assert any(c is component for c in collection)

    #############
    # Diffusion models shared between SampleModels (union Q grid)
    #############

    @staticmethod
    def diffusion_components(sample_model, Q_index):
        collection = sample_model.get_component_collection(Q_index)
        template_names = sample_model.components.list_component_names()
        return [c for c in collection if c.name not in template_names]

    def test_diffusion_model_grid_follows_its_sample_models(self):
        # WHEN a diffusion model created with a Q grid of its own
        diffusion_model = BrownianTranslationalDiffusion(name='D', Q=[0.5, 1.0, 1.5, 2.0])

        # THEN
        model = SampleModel(diffusion_models=diffusion_model, Q=[1.0, 2.0])

        # EXPECT the grid is the Q of the SampleModels using it, and each Q picks its components
        np.testing.assert_allclose(diffusion_model.Q.values, [1.0, 2.0])
        assert self.diffusion_components(model, 0) == list(
            diffusion_model.get_component_collections(0)
        )
        assert self.diffusion_components(model, 1) == list(
            diffusion_model.get_component_collections(1)
        )
        assert (
            model._generated_component_names(1)
            == diffusion_model.get_component_collections(1).list_component_names()
        )

    def test_two_sample_models_extend_a_shared_diffusion_model_to_the_union_Q(self):
        # WHEN two SampleModels with partly overlapping Q share one diffusion model
        diffusion_model = BrownianTranslationalDiffusion(name='D')
        first = SampleModel(diffusion_models=diffusion_model, Q=[0.5, 1.0, 1.5])

        # THEN
        second = SampleModel(diffusion_models=diffusion_model, Q=[1.0, 1.5, 2.0])

        # EXPECT the grid is the union, both pick their own Q, and the common Q share components
        np.testing.assert_allclose(diffusion_model.Q.values, [0.5, 1.0, 1.5, 2.0])
        assert self.diffusion_components(first, 1) == self.diffusion_components(second, 0)
        assert self.diffusion_components(first, 2) == self.diffusion_components(second, 1)
        assert self.diffusion_components(first, 0) == list(
            diffusion_model.get_component_collections(0)
        )
        assert self.diffusion_components(second, 2) == list(
            diffusion_model.get_component_collections(3)
        )

    def test_sample_model_refreshes_diffusion_components_when_shared_model_regenerates(self):
        # WHEN a SampleModel with a fitted per-Q template parameter, whose diffusion model is then
        # extended by a second SampleModel
        diffusion_model = BrownianTranslationalDiffusion(name='D')
        first = SampleModel(
            components=Gaussian(name='G'), diffusion_models=diffusion_model, Q=[0.5, 1.0]
        )
        width = first.get_component_collection(0)['G'].width
        width.value = 0.123
        width.fixed = True
        stale = self.diffusion_components(first, 0)
        version = first.state_version
        assert first.component_collections_is_dirty is False

        # THEN
        second = SampleModel(diffusion_models=diffusion_model, Q=[1.0, 1.5])

        # EXPECT the version moves at once, and on next use the diffusion components are swapped
        # in place while the template component keeps its fitted state
        assert first.state_version > version
        assert first.component_collections_is_dirty is True
        assert self.diffusion_components(first, 0) != stale
        assert self.diffusion_components(first, 0) == list(
            diffusion_model.get_component_collections(0)
        )
        assert self.diffusion_components(first, 1) == self.diffusion_components(second, 0)
        assert first.get_component_collection(0)['G'].width is width
        assert width.value == pytest.approx(0.123)
        assert width.fixed is True
        assert first.component_collections_is_dirty is False

    def test_get_all_variables_maps_Q_index_onto_the_diffusion_model_grid(self):
        # WHEN a DeltaLorentz with per-Q parameters shared by two SampleModels
        diffusion_model = DeltaLorentz(name='DL', allow_Q_variation={'A_0': True})
        first = SampleModel(diffusion_models=diffusion_model, Q=[0.5, 1.0])
        second = SampleModel(diffusion_models=diffusion_model, Q=[1.5])
        assert first.Q is not None

        # THEN
        variables = second.get_all_variables(Q_index=0)

        # EXPECT the per-Q variables are those at the grid's Q index 2, not index 0
        expected = diffusion_model.get_independent_variables(Q_index=2)
        assert expected
        assert all(any(v is e for v in variables) for e in expected)
        unexpected = diffusion_model.get_independent_variables(Q_index=0)
        assert not any(any(v is u for v in variables) for u in unexpected)

    def test_clear_Q_keeps_the_diffusion_model_Q(self, sample_model):
        # WHEN THEN clearing Q, as a SampleModel created before its data is loaded has none
        sample_model.clear_Q(confirm=True)

        # EXPECT the diffusion model keeps its grid, so other SampleModels using it are unaffected
        np.testing.assert_allclose(sample_model.diffusion_models[0].Q.values, [1.0, 2.0, 3.0])

    def test_append_diffusion_model_with_its_own_Q_grid_patches_in_place(self, sample_model):
        # WHEN built collections and a new diffusion model created with another grid
        width = sample_model.get_component_collection(1)['TestGaussian1Name'].width
        width.value = 0.123
        new_model = BrownianTranslationalDiffusion(
            name='New', lorentzian_name='NewL', Q=[1.0, 2.0, 3.0, 4.0]
        )

        # THEN
        sample_model.append_diffusion_model(new_model)

        # EXPECT the grid follows the SampleModel, its components are added, the value is kept
        assert sample_model.component_collections_is_dirty is False
        np.testing.assert_allclose(new_model.Q.values, [1.0, 2.0, 3.0])
        for Q_index in range(3):
            collection = sample_model.get_component_collection(Q_index)
            for component in new_model.get_component_collections(Q_index):
                assert any(c is component for c in collection)
        assert width.value == pytest.approx(0.123)

    def test_new_Q_after_clear_Q_replaces_the_diffusion_model_grid(self, sample_model):
        # WHEN the only SampleModel using the diffusion model gets new Q values, as in a rebin
        sample_model.clear_Q(confirm=True)

        # THEN
        sample_model.Q = [1.5, 2.5]

        # EXPECT the grid is the new Q only, without the old values
        np.testing.assert_allclose(sample_model.diffusion_models[0].Q.values, [1.5, 2.5])

    def test_grid_shrinks_when_a_sample_model_stops_using_the_diffusion_model(self):
        # WHEN two SampleModels share a diffusion model
        diffusion_model = BrownianTranslationalDiffusion(name='D')
        first = SampleModel(diffusion_models=diffusion_model, Q=[0.5, 1.0])
        second = SampleModel(diffusion_models=diffusion_model, Q=[1.0, 1.5])
        np.testing.assert_allclose(diffusion_model.Q.values, [0.5, 1.0, 1.5])

        # THEN the second drops the diffusion model
        second.remove_diffusion_model('D')

        # EXPECT the grid is the first's Q again, and the first follows the regenerated components
        np.testing.assert_allclose(diffusion_model.Q.values, [0.5, 1.0])
        assert self.diffusion_components(first, 1) == list(
            diffusion_model.get_component_collections(1)
        )
        assert second.get_component_collection(0).list_component_names() == []

    def test_duplicate_Q_values_share_one_grid_point(self):
        # WHEN a SampleModel with a repeated Q value
        diffusion_model = BrownianTranslationalDiffusion(name='D')

        # THEN
        model = SampleModel(diffusion_models=diffusion_model, Q=[1.0, 1.0])

        # EXPECT one grid point, used by both per-Q collections
        np.testing.assert_allclose(diffusion_model.Q.values, [1.0])
        assert self.diffusion_components(model, 0) == self.diffusion_components(model, 1)

    def test_remove_diffusion_model_with_larger_Q_grid_patches_in_place(self):
        # WHEN a SampleModel sharing a diffusion model on a larger grid, with fitted per-Q state
        diffusion_model = BrownianTranslationalDiffusion(name='D')
        first = SampleModel(
            components=Gaussian(name='G'), diffusion_models=diffusion_model, Q=[0.5, 1.0]
        )
        second = SampleModel(diffusion_models=diffusion_model, Q=[1.0, 1.5])
        width = first.get_component_collection(0)['G'].width
        width.value = 0.123
        version = first.state_version

        # THEN
        first.remove_diffusion_model('D')

        # EXPECT only the template component remains, with its value; the grid is the second's
        assert first.get_component_collection(0).list_component_names() == ['G']
        assert first.get_component_collection(0)['G'].width is width
        assert width.value == pytest.approx(0.123)
        assert first.state_version > version
        np.testing.assert_allclose(diffusion_model.Q.values, [1.0, 1.5])
        assert self.diffusion_components(second, 0) == list(
            diffusion_model.get_component_collections(0)
        )

    def test_refresh_falls_back_to_a_rebuild_when_the_grid_lacks_Q(self):
        # WHEN a diffusion model whose grid was set by hand without the SampleModel's Q values,
        # while another registered SampleModel keeps those values in the union
        diffusion_model = BrownianTranslationalDiffusion(name='D')
        first = SampleModel(
            components=Gaussian(name='G'), diffusion_models=diffusion_model, Q=[0.5, 1.0]
        )
        _ = first.get_component_collection(0)
        diffusion_model.clear_Q(confirm=True)
        diffusion_model.Q = [7.0]

        # THEN
        collection = first.get_component_collection(0)

        # EXPECT the grid is restored and the collection holds the live components
        np.testing.assert_allclose(diffusion_model.Q.values, [0.5, 1.0])
        assert self.diffusion_components(first, 0) == list(
            diffusion_model.get_component_collections(0)
        )
        assert collection.list_component_names()[0] == 'G'

    def test_refresh_does_not_bump_the_state_counter_twice(self):
        # WHEN a SampleModel whose shared diffusion model is extended by a second SampleModel
        diffusion_model = BrownianTranslationalDiffusion(name='D')
        first = SampleModel(diffusion_models=diffusion_model, Q=[0.5, 1.0])
        _ = first.get_component_collection(0)
        second = SampleModel(diffusion_models=diffusion_model, Q=[1.0, 1.5])
        assert second.Q is not None
        version = first.state_version

        # THEN the first refreshes its collections
        _ = first.get_component_collection(0)

        # EXPECT the version seen before the refresh is the version after it
        assert first.state_version == version

    def test_state_version_is_monotonic_across_diffusion_model_replacement(self, sample_model):
        # WHEN
        versions = [sample_model.state_version]
        removed = sample_model.diffusion_models[0]
        new_model = BrownianTranslationalDiffusion(name='New', lorentzian_name='NewL')

        # THEN replacing, clearing, and re-adding diffusion models
        sample_model.diffusion_models = [new_model]
        versions.append(sample_model.state_version)
        sample_model.diffusion_models = None
        versions.append(sample_model.state_version)
        sample_model.append_diffusion_model(removed)
        versions.append(sample_model.state_version)
        _ = sample_model.get_component_collection(0)
        versions.append(sample_model.state_version)

        # EXPECT
        assert versions == sorted(versions)
        assert len(set(versions)) >= 4
