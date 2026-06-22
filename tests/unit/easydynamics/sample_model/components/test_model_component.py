# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.model_component import ModelComponent


class DummyComponent(ModelComponent):
    def __init__(self):
        super().__init__(display_name='Dummy')
        self.area = Parameter(name='area', value=1.0, unit='meV', fixed=False)
        self.center = Parameter(name='center', value=2.0, unit='meV', fixed=True)
        self.width = Parameter(name='width', value=3.0, unit='meV', fixed=True)
        self._x_unit = 'meV'

    def get_all_parameters(self):
        return [self.area, self.center, self.width]

    def evaluate(self, x):
        return np.zeros_like(x)


class TestModelComponent:
    @pytest.fixture
    def dummy(self):
        return DummyComponent()

    def test_unit_cannot_be_set_directly(self, dummy: ModelComponent):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match='read-only'):
            dummy.x_unit = 'K'

    def test_convert_unit(self, dummy: DummyComponent):
        # WHEN THEN
        dummy.convert_x_unit('microeV')

        # EXPECT
        assert dummy.x_unit == 'microeV'
        assert dummy.area.value == pytest.approx(1 * 1e3)
        assert dummy.center.value == pytest.approx(2 * 1e3)
        assert dummy.width.value == pytest.approx(3 * 1e3)

    def test_convert_unit_incorrect_unit_raises(self, dummy: DummyComponent):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'unit must be a string or sc.Unit'):
            dummy.convert_x_unit(123)

    def test_free_and_fix_all_parameters(self, dummy):
        # WHEN THEN EXPECT
        dummy.free_all_parameters()
        assert all(not p.fixed for p in dummy.get_all_parameters())

        # THEN EXPECT
        dummy.fix_all_parameters()
        assert all(p.fixed for p in dummy.get_all_parameters())

    def test_repr(self, dummy):
        # WHEN THEN EXPECT
        repr_str = repr(dummy)
        assert 'DummyComponent' in repr_str

    @pytest.mark.parametrize(
        'x_input, expected_array',
        [
            (5.0, np.array([5.0])),
            ([1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
            (sc.scalar(5.0, unit='meV'), np.array([5.0])),
            (
                sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='meV'),
                np.array([1.0, 2.0, 3.0]),
            ),
            (
                sc.DataArray(
                    data=sc.array(dims=['x'], values=[10.0, 20.0, 30.0]),
                    coords={'x': sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='meV')},
                ),
                np.array([1.0, 2.0, 3.0]),
            ),
        ],
        ids=[
            'python_scalar',
            'python_list',
            'numpy_array',
            'scipp_scalar',
            'scipp_array',
            'scipp_dataarray',
        ],
    )
    def test_prepare_x_for_evaluate_various_inputs(self, dummy, x_input, expected_array):
        result = dummy._prepare_x_for_evaluate(x_input)
        x_prepared, _detected_unit, _dim = result

        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == expected_array.shape
        np.testing.assert_array_equal(x_prepared, expected_array)

    def test_prepare_x_for_evaluate_with_scipp_data_array_multiple_coords_raises(self, dummy):
        # WHEN
        x = sc.array(dims=['x'], values=[0.0, 0.5, 1.0], unit='meV')
        y = sc.array(dims=['y'], values=[0.0, 1.0, 2.0], unit='meV')
        var = sc.array(
            dims=['x', 'y'],
            values=[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
        )
        array = sc.DataArray(data=var, coords={'x': x, 'y': y})

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='must have exactly one coordinate',
        ):
            dummy._prepare_x_for_evaluate(array)

    @pytest.mark.parametrize(
        'x, expected_message',
        [
            (np.array([0.0, np.nan, 1.0]), 'Input x contains NaN values.'),
            (np.array([0.0, np.inf, 1.0]), 'Input x contains infinite values.'),
        ],
        ids=['nan', 'infinite'],
    )
    def test_prepare_x_for_evaluate_with_invalid_input_raises(
        self, dummy: DummyComponent, x, expected_message
    ):
        # THEN EXPECT
        with pytest.raises(ValueError, match=expected_message):
            dummy._prepare_x_for_evaluate(x)

    def test_prepare_x_for_evaluate_with_incompatible_unit_raises(self, dummy):
        # WHEN
        x = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='nm')

        # THEN EXPECT
        with pytest.raises(
            Exception,
            match='Input x has unit nm',
        ):
            dummy._prepare_x_for_evaluate(x)

    def test_prepare_x_for_evaluate_with_different_unit_no_warn(self, dummy):
        # WHEN
        x = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], unit='microeV')

        # THEN EXPECT: compatible units are accepted without warning;
        # the component's x_unit is NOT mutated and x values are returned as-is.
        x_prepared, _detected_unit, _dim = dummy._prepare_x_for_evaluate(x)

        # EXPECT
        assert isinstance(x_prepared, np.ndarray)
        assert x_prepared.shape == (3,)
        np.testing.assert_array_equal(x_prepared, [1.0, 2.0, 3.0])
        assert dummy.x_unit == 'meV'  # component unit unchanged
        assert dummy.area.value == pytest.approx(1.0)  # parameter values unchanged
        assert dummy.center.value == pytest.approx(2.0)
        assert dummy.width.value == pytest.approx(3.0)
