# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError

from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.sample_model.resolution_model import ResolutionModel
from easydynamics.sample_model.sample_model import SampleModel


class TestInstrumentModel:
    @pytest.fixture
    def instrument_model(self):
        Q = np.array([1.0, 2.0, 3.0])
        component1 = Polynomial(coefficients=[1.0, 2.0])
        background_model = BackgroundModel(components=component1, Q=Q)

        component2 = Gaussian()
        resolution_model = ResolutionModel(components=component2, Q=Q)

        return InstrumentModel(
            display_name='TestInstrumentModel',
            background_model=background_model,
            resolution_model=resolution_model,
            Q=Q,
        )

    @pytest.fixture
    def instrument_model_without_Q(self):
        component1 = Polynomial(coefficients=[1.0, 2.0])
        background_model = BackgroundModel(components=component1)

        component2 = Gaussian()
        resolution_model = ResolutionModel(components=component2)

        return InstrumentModel(
            display_name='TestInstrumentModel',
            background_model=background_model,
            resolution_model=resolution_model,
        )

    @pytest.fixture
    def resolution_model(self):
        component = Gaussian()
        return ResolutionModel(components=component)

    @pytest.fixture
    def background_model(self):
        component = Polynomial(coefficients=[1.0, 2.0])
        return BackgroundModel(components=component)

    @pytest.fixture
    def sample_model(self):
        component = Gaussian()
        return SampleModel(components=component)

    def test_init(self, instrument_model):
        # WHEN THEN
        model = instrument_model

        # EXPECT
        assert model.display_name == 'TestInstrumentModel'
        assert isinstance(model.Q, sc.Variable)
        assert model.Q.unit == sc.Unit('1/angstrom')
        np.testing.assert_array_equal(model.Q.values, np.array([1.0, 2.0, 3.0]))
        assert isinstance(model.background_model, BackgroundModel)
        assert isinstance(model.resolution_model, ResolutionModel)
        np.testing.assert_array_equal(model.background_model.Q.values, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(model.resolution_model.Q.values, np.array([1.0, 2.0, 3.0]))

    def test_init_defaults(self):
        # WHEN THEN
        model = InstrumentModel()

        # EXPECT
        assert model.display_name == 'MyInstrumentModel'
        assert isinstance(model.background_model, BackgroundModel)
        assert isinstance(model.resolution_model, ResolutionModel)
        assert model.Q is None

    def test_init_sample_model_as_resolution_model(self, sample_model):
        # WHEN THEN
        model = InstrumentModel(resolution_model=sample_model)

        # EXPECT
        assert model.display_name == 'MyInstrumentModel'
        assert isinstance(model.background_model, BackgroundModel)
        assert isinstance(model.resolution_model, ResolutionModel)

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
                {'x_unit': 123},
                UnitError,
                'unit must be',
            ),
        ],
        ids=[
            'invalid resolution_model',
            'invalid background_model',
            'invalid energy_offset',
            'invalid x unit',
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

    def test_resolution_model_setter_converts_sample_model(
        self,
        instrument_model,
    ):
        """
        Test that setting a SampleModel as the resolution_model calls the method to convert it to
        a ResolutionModel, and updates the resolution_model and calls the change callback.
        """
        # WHEN
        sample_model = SampleModel()
        converted_model = ResolutionModel()
        instrument_model._on_resolution_model_change = MagicMock()

        with patch.object(
            ResolutionModel,
            'from_sample_model',
            return_value=converted_model,
        ) as mock_from_sample_model:
            # THEN
            instrument_model.resolution_model = sample_model

        # EXPECT
        mock_from_sample_model.assert_called_once_with(sample_model)
        assert instrument_model._resolution_model is converted_model
        instrument_model._on_resolution_model_change.assert_called_once_with()

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

    def test_clear_Q(self, instrument_model):
        # WHEN
        instrument_model.clear_Q(confirm=True)

        # THEN / EXPECT
        assert instrument_model.Q is None
        assert instrument_model.background_model.Q is None
        assert instrument_model.resolution_model.Q is None

    def test_clear_Q_raises_without_confirm(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(ValueError, match='Clearing Q values requires confirmation'):
            instrument_model.clear_Q()

    def test_x_unit_setter_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(AttributeError):
            instrument_model.x_unit = 'meV'

    def test_energy_offset_setter(self, instrument_model):
        # WHEN
        instrument_model._on_energy_offset_change = MagicMock()

        # THEN
        instrument_model.energy_offset = 1.0

        # EXPECT
        assert instrument_model.energy_offset.value == pytest.approx(1.0)
        instrument_model._on_energy_offset_change.assert_called_once()

    def test_energy_offset_setter_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='energy_offset must be a number',
        ):
            instrument_model.energy_offset = 'invalid_offset'

    def test_get_energy_offset(self, instrument_model):
        # WHEN

        # THEN
        offset_at_Q0 = instrument_model.get_energy_offset(0)

        # EXPECT
        assert offset_at_Q0.value == instrument_model.energy_offset.value

    def test_get_energy_offset_invalid_index_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            IndexError,
            match='Q_index 5 is out of bounds',
        ):
            instrument_model.get_energy_offset(5)

    def test_get_energy_offset_nonint_index_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='Q_index must be an int or None, got str',
        ):
            instrument_model.get_energy_offset('invalid_index')

    def test_get_energy_offset_no_Q_raises(self, instrument_model):
        # WHEN
        instrument_model.clear_Q(confirm=True)

        # THEN / EXPECT
        with pytest.raises(
            ValueError,
            match='No Q values are set',
        ):
            instrument_model.get_energy_offset(0)

    def test_convert_x_unit_calls_all_children(self, instrument_model):
        # WHEN
        new_unit = 'eV'

        # THEN
        # Ensure energy offsets are built before mocking
        instrument_model._ensure_energy_offsets_current()
        # Mock downstream convert_x_unit calls
        instrument_model._background_model.convert_x_unit = MagicMock()
        instrument_model._resolution_model.convert_x_unit = MagicMock()
        instrument_model._energy_offset.convert_unit = MagicMock()
        for offset in instrument_model._energy_offsets:
            offset.convert_unit = MagicMock()

        with patch(
            'easydynamics.sample_model.instrument_model._validate_unit',
            return_value=new_unit,
        ) as mock_validate:
            instrument_model.convert_x_unit(new_unit)

            # EXPECT
            mock_validate.assert_called_once_with(new_unit)

            instrument_model._background_model.convert_x_unit.assert_called_once_with(new_unit)
            instrument_model._resolution_model.convert_x_unit.assert_called_once_with(new_unit)
            instrument_model._energy_offset.convert_unit.assert_called_once_with(new_unit)

            for offset in instrument_model._energy_offsets:
                offset.convert_unit.assert_called_once_with(new_unit)

            # final state
            assert instrument_model.x_unit == new_unit

    def test_convert_x_unit_None_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            ValueError,
            match=' must be a valid unit',
        ):
            instrument_model.convert_x_unit(None)

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

    #############
    # Scale
    #############

    def test_scale_is_fixed_at_one_by_default(self, instrument_model):
        # WHEN THEN EXPECT
        assert instrument_model.scale.value == pytest.approx(1.0)
        assert instrument_model.scale.fixed is True
        assert instrument_model.scale.name == 'instrument_scale'

    def test_init_with_scale(self):
        # WHEN THEN
        model = InstrumentModel(scale=2.5)

        # EXPECT
        assert model.scale.value == pytest.approx(2.5)
        assert model.scale.fixed is True

    def test_scale_setter_sets_the_value(self, instrument_model):
        # WHEN
        scale = instrument_model.scale

        # THEN
        instrument_model.scale = 0.5

        # EXPECT the same Parameter, with the new value
        assert instrument_model.scale is scale
        assert scale.value == pytest.approx(0.5)

    @pytest.mark.parametrize(
        'value, expected_exception',
        [('2', TypeError), (None, TypeError), (-1.0, ValueError)],
        ids=['string', 'None', 'negative'],
    )
    def test_scale_setter_invalid_raises(self, instrument_model, value, expected_exception):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match='scale must be'):
            instrument_model.scale = value

    def test_scale_is_one_variable_for_all_Q(self, instrument_model):
        # WHEN THEN
        per_Q = [instrument_model.get_all_variables(Q_index=Q_index) for Q_index in range(3)]

        # EXPECT the same Parameter at every Q, listed once for all Q together
        scale = instrument_model.scale
        assert all(any(variable is scale for variable in variables) for variables in per_Q)
        assert sum(variable is scale for variable in instrument_model.get_all_variables()) == 1

    def test_get_all_variables(self, instrument_model):
        # WHEN
        all_vars = instrument_model.get_all_variables()

        # THEN
        expected_var_names = {
            'energy_offset',
            'instrument_scale',
            'Polynomial_c0',
            'Polynomial_c1',
            'Gaussian area',
            'Gaussian center',
            'Gaussian width',
        }

        retrieved_var_names = {var.name for var in all_vars}

        assert expected_var_names == retrieved_var_names
        assert len(all_vars) == 19
        assert len(all_vars) == len(set(all_vars))

    def test_get_all_variables_no_Q(self, instrument_model):
        # WHEN
        instrument_model.clear_Q(confirm=True)

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
            'instrument_scale',
            'Polynomial_c0',
            'Polynomial_c1',
            'Gaussian area',
            'Gaussian center',
            'Gaussian width',
        }

        retrieved_var_names = {var.name for var in all_vars}

        assert expected_var_names == retrieved_var_names
        assert len(all_vars) == 7

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

    def test_normalize_resolution(self, instrument_model):
        # WHEN
        instrument_model.resolution_model.normalize_area = MagicMock()

        # THEN
        instrument_model.normalize_resolution()

        # EXPECT
        instrument_model.resolution_model.normalize_area.assert_called_once()

    def test_generate_energy_offsets_Q_none(self, instrument_model):
        # WHEN
        instrument_model._Q = None

        # THEN
        instrument_model._generate_energy_offsets()

        # EXPECT
        assert instrument_model._energy_offsets == []

    def test_generate_energy_offsets(self, instrument_model):
        # WHEN
        instrument_model._Q = sc.Variable(
            dims=['Q'], values=[1.0, 2.0, 3.0, 4.0], unit='1/angstrom'
        )

        # THEN
        instrument_model._generate_energy_offsets()

        # EXPECT
        assert len(instrument_model._energy_offsets) == 4
        for offset in instrument_model._energy_offsets:
            assert offset.name == 'energy_offset'
            assert offset.unit == instrument_model.x_unit
            assert offset.value == instrument_model.energy_offset.value

    def test_Q_setter_keeps_Q_within_the_tolerance(self, instrument_model):
        # WHEN
        old_Q = instrument_model.Q

        # THEN Q values that differ in rounding only
        instrument_model.Q = old_Q.values + 5e-5

        # EXPECT the model keeps its own Q values
        assert instrument_model.Q is old_Q

    def test_Q_setter_raises_beyond_the_tolerance(self, instrument_model):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='increase Q_tolerance'):
            instrument_model.Q = instrument_model.Q.values + 2e-4

    def test_Q_tolerance(self, instrument_model):
        # WHEN
        old_Q = instrument_model.Q
        assert instrument_model.Q_tolerance == pytest.approx(1e-4)

        # THEN
        instrument_model.Q_tolerance = 0.01
        instrument_model.Q = old_Q.values + 0.005

        # EXPECT the larger difference is accepted, and the model keeps its own Q values
        assert instrument_model.Q is old_Q

        # EXPECT the resolution and background models have the tolerance too
        assert instrument_model.Q_tolerance == pytest.approx(0.01)
        assert instrument_model.resolution_model.Q_tolerance == pytest.approx(0.01)
        assert instrument_model.background_model.Q_tolerance == pytest.approx(0.01)

    def test_Q_tolerance_invalid_raises(self, instrument_model):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q_tolerance must be a number'):
            instrument_model.Q_tolerance = 'large'

    def test_Q_setter(self, instrument_model_without_Q):
        # WHEN
        first_new_Q = np.array([1.0, 2.0, 3.0])

        # THEN
        instrument_model_without_Q.Q = first_new_Q

        # EXPECT
        assert instrument_model_without_Q._energy_offsets_is_dirty is True
        np.testing.assert_array_equal(
            instrument_model_without_Q.background_model.Q.values, first_new_Q
        )
        np.testing.assert_array_equal(
            instrument_model_without_Q.resolution_model.Q.values, first_new_Q
        )

        # THEN
        new_Q = np.array([4.0, 5.0, 6.0])

        # EXPECT
        with pytest.raises(ValueError, match='New Q values are not similar to the old ones'):
            instrument_model_without_Q.Q = new_Q

        # THEN
        new_Q = None
        instrument_model_without_Q.Q = new_Q

        # EXPECT
        # Q values remain unchanged
        np.testing.assert_array_equal(
            instrument_model_without_Q.background_model.Q.values, first_new_Q
        )
        np.testing.assert_array_equal(
            instrument_model_without_Q.resolution_model.Q.values, first_new_Q
        )

        # THEN - set Q to an equivalent scipp Variable; values match so should be accepted
        new_Q = sc.Variable(dims=['Q'], values=[1.0, 2.0, 3.0], unit='1/angstrom')
        instrument_model_without_Q.Q = new_Q

        # EXPECT - Q propagated to child models, offsets marked dirty again
        assert instrument_model_without_Q._energy_offsets_is_dirty is True
        np.testing.assert_array_equal(
            instrument_model_without_Q.background_model.Q.values, first_new_Q
        )
        np.testing.assert_array_equal(
            instrument_model_without_Q.resolution_model.Q.values, first_new_Q
        )

    def test_fix_and_free_offset(self, instrument_model):
        # WHEN
        # EXPECT
        for offset in instrument_model._energy_offsets:
            assert offset.fixed is False

        # THEN
        instrument_model.fix_energy_offset()

        # EXPECT
        for offset in instrument_model._energy_offsets:
            assert offset.fixed is True
        # THEN
        instrument_model.free_energy_offset()

        # EXPECT
        for offset in instrument_model._energy_offsets:
            assert offset.fixed is False

        # THEN
        instrument_model.fix_energy_offset(Q_index=1)

        # EXPECT
        for i, offset in enumerate(instrument_model._energy_offsets):
            if i == 1:
                assert offset.fixed is True
            else:
                assert offset.fixed is False

    def test_fix_or_free_energy_offset_invalid_Q_index_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            IndexError,
            match='Q_index 5 is out of bounds',
        ):
            instrument_model.fix_energy_offset(Q_index=5)

        with pytest.raises(
            IndexError,
            match='Q_index 5 is out of bounds',
        ):
            instrument_model.free_energy_offset(Q_index=5)

    def test_fix_or_free_energy_offset_nonint_Q_index_raises(self, instrument_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match='Q_index must be an int',
        ):
            instrument_model.fix_energy_offset(Q_index='invalid_index')

        with pytest.raises(
            TypeError,
            match='Q_index must be an int',
        ):
            instrument_model.free_energy_offset(Q_index='invalid_index')

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
        # WHEN a resolution model that does not know the instrument's Q yet
        new_resolution_model = resolution_model
        assert new_resolution_model.Q is None

        # THEN
        instrument_model.resolution_model = new_resolution_model

        # EXPECT the change callback propagated the instrument's Q to the new model
        assert instrument_model.resolution_model is new_resolution_model
        np.testing.assert_array_equal(new_resolution_model.Q.values, np.array([1.0, 2.0, 3.0]))

    def test_on_background_model_change(self, instrument_model, background_model):
        # WHEN a background model that does not know the instrument's Q yet
        new_background_model = background_model
        assert new_background_model.Q is None

        # THEN
        instrument_model.background_model = new_background_model

        # EXPECT the change callback propagated the instrument's Q to the new model
        assert instrument_model.background_model is new_background_model
        np.testing.assert_array_equal(new_background_model.Q.values, np.array([1.0, 2.0, 3.0]))

    def test_repr_contains_expected_fields(self, instrument_model):
        # WHEN THEN
        repr_str = repr(instrument_model)

        # EXPECT values pinned from the fixture's known construction inputs, so a wrong
        # attribute value cannot satisfy its own interpolation
        assert repr_str.startswith('InstrumentModel(')
        assert repr_str.endswith(')')
        assert "unique_name='" in repr_str
        assert 'x_unit=meV' in repr_str
        assert 'Q_len=3' in repr_str
        assert 'resolution_model=ResolutionModel(' in repr_str
        assert 'background_model=BackgroundModel(' in repr_str
