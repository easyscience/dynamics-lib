# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.sample_model.resolution_model import ResolutionModel


class TestInstrumentModel:
    @pytest.fixture
    def instrument_model(self):
        Q = np.array([1.0, 2.0, 3.0])
        component1 = Polynomial(coefficients=[1.0, 2.0])
        background_model = BackgroundModel(components=component1, Q=Q)

        component2 = Gaussian()
        resolution_model = ResolutionModel(components=component2, Q=Q)

        instrument_model = InstrumentModel(
            display_name='TestInstrumentModel',
            background_model=background_model,
            resolution_model=resolution_model,
            Q=Q,
        )

        return instrument_model

    @pytest.fixture
    def resolution_model(self):
        component = Gaussian()
        resolution_model = ResolutionModel(components=component)
        return resolution_model

    @pytest.fixture
    def background_model(self):
        component = Polynomial(coefficients=[1.0, 2.0])
        background_model = BackgroundModel(components=component)
        return background_model

    def test_init(self, instrument_model):
        # WHEN THEN
        model = instrument_model

        # EXPECT
        assert model.display_name == 'TestInstrumentModel'
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))
        assert isinstance(model.background_model, BackgroundModel)
        assert isinstance(model.resolution_model, ResolutionModel)
        np.testing.assert_array_equal(model.background_model.Q, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(model.resolution_model.Q, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))

    def test_init_defaults(self):
        # WHEN THEN
        model = InstrumentModel()

        # EXPECT
        assert model.display_name == 'MyInstrumentModel'
        assert isinstance(model.background_model, BackgroundModel)
        assert isinstance(model.resolution_model, ResolutionModel)
        assert model.Q is None

    @pytest.mark.parametrize(
        'kwargs, expected_exception, expected_message',
        [
            (
                {'resolution_model': 123},
                TypeError,
                'resolution_model must be a ResolutionModel',
            ),
            (
                {'background_model': 'not a model'},
                TypeError,
                'background_model must be a BackgroundModel',
            ),
            (
                {'energy_offset': 'abc'},
                TypeError,
                'energy_offset must be a number',
            ),
            (
                {'unit': 123},
                TypeError,
                'unit must be',
            ),
        ],
        ids=[
            'invalid resolution_model',
            'invalid background_model',
            'invalid energy_offset',
            'invalid unit',
        ],
    )
    def test_instrument_model_init_invalid_inputs(
        self, kwargs, expected_exception, expected_message
    ):
        with pytest.raises(expected_exception, match=expected_message):
            InstrumentModel(**kwargs)

    def test_resolution_model_setter_calls_update(self, instrument_model, resolution_model):
        # WHEN
        instrument_model._on_resolution_model_change = MagicMock()

        # THEN
        instrument_model.resolution_model = resolution_model

        # EXPECT
        assert instrument_model._resolution_model is resolution_model
        instrument_model._on_resolution_model_change.assert_called_once()

    def test_resolution_model_setter_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='resolution_model must be a ResolutionModel',
        ):
            instrument_model.resolution_model = 'invalid_model'

    def test_background_model_setter_calls_update(self, instrument_model, background_model):
        # WHEN
        instrument_model._on_background_model_change = MagicMock()

        # THEN
        instrument_model.background_model = background_model

        # EXPECT
        assert instrument_model._background_model is background_model
        instrument_model._on_background_model_change.assert_called_once()

    def test_background_model_setter_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='background_model must be a BackgroundModel',
        ):
            instrument_model.background_model = 123

    def test_Q_setter(self, instrument_model):
        "Test that Q setter calls the appropriate methods."
        # WHEN
        new_Q = np.array([4.0, 5.0, 6.0])

        instrument_model._on_Q_change = MagicMock()

        # THEN EXPECT
        with patch(
            'easydynamics.sample_model.instrument_model._validate_and_convert_Q',
            return_value=new_Q,
        ) as mock_validate:
            instrument_model.Q = new_Q

            np.testing.assert_array_equal(instrument_model.Q, new_Q)
            mock_validate.assert_called_once_with(new_Q)
            instrument_model._on_Q_change.assert_called_once()

    def test_unit_setter_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            AttributeError,
            match='Unit is read-only. Use convert_unit to change the unit between allowed types ',
        ):
            instrument_model.unit = 'meV'

    def test_energy_offset_setter(self, instrument_model):
        # WHEN
        instrument_model._on_energy_offset_change = MagicMock()

        # THEN
        instrument_model.energy_offset = 1.0

        # EXPECT
        assert instrument_model.energy_offset.value == 1.0
        instrument_model._on_energy_offset_change.assert_called_once()

    def test_energy_offset_setter_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='energy_offset must be a number',
        ):
            instrument_model.energy_offset = 'invalid_offset'

    def test_convert_unit_calls_all_children(self, instrument_model):
        # WHEN
        new_unit = 'eV'

        # THEN
        # Mock downstream convert_unit calls
        instrument_model._background_model.convert_unit = MagicMock()
        instrument_model._resolution_model.convert_unit = MagicMock()
        instrument_model._energy_offset.convert_unit = MagicMock()
        for offset in instrument_model._energy_offsets:
            offset.convert_unit = MagicMock()

        with patch(
            'easydynamics.sample_model.instrument_model._validate_unit',
            return_value=new_unit,
        ) as mock_validate:
            instrument_model.convert_unit(new_unit)

            # EXPECT
            mock_validate.assert_called_once_with(new_unit)

            instrument_model._background_model.convert_unit.assert_called_once_with(new_unit)
            instrument_model._resolution_model.convert_unit.assert_called_once_with(new_unit)
            instrument_model._energy_offset.convert_unit.assert_called_once_with(new_unit)

            for offset in instrument_model._energy_offsets:
                offset.convert_unit.assert_called_once_with(new_unit)

            # final state
            assert instrument_model.unit == new_unit

    def test_convert_unit_None_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            ValueError,
            match=' must be a valid unit',
        ):
            instrument_model.convert_unit(None)

    def test_fix_resolution_parameters(self, instrument_model):
        # WHEN
        instrument_model.resolution_model.fix_all_parameters = MagicMock()

        # THEN
        instrument_model.fix_resolution_parameters()

        # EXPECT
        instrument_model.resolution_model.fix_all_parameters.assert_called_once()

    def test_free_all_resolution_parameters(self, instrument_model):
        # WHEN
        instrument_model.resolution_model.free_all_parameters = MagicMock()

        # THEN
        instrument_model.free_resolution_parameters()

        # EXPECT
        instrument_model.resolution_model.free_all_parameters.assert_called_once()

    def test_get_all_variables(self, instrument_model):
        # WHEN
        all_vars = instrument_model.get_all_variables()

        # THEN
        expected_var_names = {
            'energy_offset',
            'Polynomial_c0',
            'Polynomial_c1',
            'Gaussian area',
            'Gaussian center',
            'Gaussian width',
        }

        retrieved_var_names = {var.name for var in all_vars}

        assert expected_var_names == retrieved_var_names
        assert len(all_vars) == 18

    def test_get_all_variables_no_Q(self, instrument_model):
        # WHEN
        instrument_model.Q = None

        # THEN
        all_vars = instrument_model.get_all_variables()

        # EXPECT
        assert all_vars == []

    def test_get_all_variables_with_Q_index(self, instrument_model):
        # WHEN
        all_vars = instrument_model.get_all_variables(Q_index=1)

        # THEN
        expected_var_names = {
            'energy_offset',
            'Polynomial_c0',
            'Polynomial_c1',
            'Gaussian area',
            'Gaussian center',
            'Gaussian width',
        }

        retrieved_var_names = {var.name for var in all_vars}

        assert expected_var_names == retrieved_var_names
        assert len(all_vars) == 6

    def test_get_all_variables_with_invalid_Q_index_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            IndexError,
            match='Q_index 5 is out of bounds',
        ):
            instrument_model.get_all_variables(Q_index=5)

    def test_get_all_variables_with_nonint_Q_index_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='Q_index must be an int or None, got str',
        ):
            instrument_model.get_all_variables(Q_index='invalid_index')

    def test_generate_energy_offsets_Q_none(self, instrument_model):
        # WHEN
        instrument_model._Q = None

        # THEN
        instrument_model._generate_energy_offsets()

        # EXPECT
        assert instrument_model._energy_offsets == []

    def test_generate_energy_offsets(self, instrument_model):
        # WHEN
        instrument_model._Q = np.array([1.0, 2.0, 3.0, 4.0])

        # THEN
        instrument_model._generate_energy_offsets()

        # EXPECT
        assert len(instrument_model._energy_offsets) == 4
        for offset in instrument_model._energy_offsets:
            assert offset.name == 'energy_offset'
            assert offset.unit == instrument_model.unit
            assert offset.value == instrument_model.energy_offset.value

    def test_on_Q_change(self, instrument_model):
        # WHEN
        instrument_model._generate_energy_offsets = MagicMock()
        new_Q = np.array([1.0, 2.0, 3.0, 4.0])

        # THEN
        instrument_model._Q = new_Q
        instrument_model._on_Q_change()

        # EXPECT
        instrument_model._generate_energy_offsets.assert_called_once()
        instrument_model._background_model.Q = new_Q
        instrument_model._resolution_model.Q = new_Q

    def test_on_energy_offset_change(self, instrument_model):
        # WHEN
        new_offset = 2.0

        # THEN
        instrument_model._energy_offset.value = new_offset
        instrument_model._on_energy_offset_change()

        # EXPECT
        for offset in instrument_model._energy_offsets:
            assert offset.value == new_offset

    def test_on_resolution_model_change(self, instrument_model, resolution_model):
        # WHEN
        new_resolution_model = resolution_model

        # THEN
        instrument_model._resolution_model = new_resolution_model
        instrument_model._on_resolution_model_change()

        # EXPECT
        assert instrument_model._resolution_model is new_resolution_model

    def test_on_background_model_change(self, instrument_model, background_model):
        # WHEN
        new_background_model = background_model

        # THEN
        instrument_model._background_model = new_background_model
        instrument_model._on_background_model_change()

        # EXPECT
        assert instrument_model._background_model is new_background_model

    def test_repr_contains_expected_fields(self, instrument_model):
        # WHEN THEN
        repr_str = repr(instrument_model)

        # EXPECT
        assert repr_str.startswith('InstrumentModel(')
        assert f'unique_name={instrument_model.unique_name!r}' in repr_str
        assert f'unit={instrument_model.unit}' in repr_str
        assert 'Q_len=3' in repr_str
        assert f'resolution_model={instrument_model._resolution_model!r}' in repr_str
        assert f'background_model={instrument_model._background_model!r}' in repr_str
        assert repr_str.endswith(')')
