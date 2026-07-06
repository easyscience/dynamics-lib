# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.delta_function import DeltaFunction
from easydynamics.sample_model.components.lorentzian import Lorentzian
from easydynamics.sample_model.diffusion_model.delta_lorentz import DeltaLorentz


class TestDeltaLorentz:
    @pytest.fixture
    def delta_lorentz_model(self):
        return DeltaLorentz()

    def test_mean_u_squared_unit_conversion_leaves_physics_unchanged(self):
        # WHEN: a model with a non-zero Debye-Waller factor
        model = DeltaLorentz(A_0=0.5, mean_u_squared=1.0, lorentzian_width=0.1, Q=np.array([1.0]))
        eisf_before = model.calculate_EISF()
        qisf_before = model.calculate_QISF()
        delta_area_before = model.get_component_collections(0)[1].area.value

        # THEN: convert mean_u_squared to nm**2 (same physical value, different number)
        model.mean_u_squared.convert_unit('nm**2')

        # EXPECT: EISF, QISF, and the dependent delta area are unchanged
        np.testing.assert_allclose(model.calculate_EISF(), eisf_before)
        np.testing.assert_allclose(model.calculate_QISF(), qisf_before)
        assert model.get_component_collections(0)[1].area.value == pytest.approx(delta_area_before)

    def test_convert_x_unit_converts_lorentzian_width(self):
        # WHEN
        model = DeltaLorentz(lorentzian_width=0.1, Q=np.array([1.0]))

        # THEN
        model.convert_x_unit('ueV')

        # EXPECT: the width template is rescaled and the regenerated component follows
        assert sc.Unit(str(model.lorentzian_width.unit)) == sc.Unit('ueV')
        assert model.lorentzian_width.value == pytest.approx(100.0)
        collection = model.get_component_collections(0)
        assert sc.Unit(str(collection[0].width.unit)) == sc.Unit('ueV')
        assert collection[0].width.value == pytest.approx(100.0)

    def test_convert_x_unit_with_q_varying_width_preserves_state(self):
        # WHEN: a model with per-Q widths, one of which has been changed from the template
        model = DeltaLorentz(
            lorentzian_width=0.1,
            Q=np.array([1.0, 2.0]),
            allow_Q_variation={'lorentzian_width': True},
        )
        model._lorentzian_width_list[0].value = 0.2
        collection_before = model.get_component_collections(0)

        # THEN
        model.convert_x_unit('ueV')

        # EXPECT: conversion happens in place — per-Q values are converted, not reset to the
        # template, and the collections are not regenerated (regression: conversion used to
        # rebuild the collections, discarding per-Q state)
        assert model.get_component_collections(0) is collection_before
        assert model._lorentzian_width_list[0].value == pytest.approx(200.0)
        assert model._lorentzian_width_list[1].value == pytest.approx(100.0)
        for width in model._lorentzian_width_list:
            assert sc.Unit(str(width.unit)) == sc.Unit('ueV')

    @pytest.fixture
    def delta_lorentz_model_with_Q(self):
        Q = np.linspace(0.5, 2, 7)
        return DeltaLorentz(
            Q=Q,
            A_0=0.5,
            lorentzian_width=0.0015,
            allow_Q_variation={'A_0': True, 'lorentzian_width': True},
        )

    @pytest.fixture
    def delta_lorentz_model_with_Q_no_variation(self):
        Q = np.linspace(0.5, 2, 7)
        return DeltaLorentz(
            Q=Q,
            A_0=0.5,
            lorentzian_width=0.0015,
            allow_Q_variation={'A_0': False, 'lorentzian_width': False},
        )

    def test_init_default(self, delta_lorentz_model):
        # WHEN THEN EXPECT
        assert delta_lorentz_model.display_name == 'DeltaLorentz'
        assert delta_lorentz_model.x_unit == 'meV'
        assert delta_lorentz_model.y_unit == 'dimensionless'
        assert delta_lorentz_model.scale.value == pytest.approx(1.0)
        assert delta_lorentz_model.mean_u_squared.value == pytest.approx(0.0)
        assert delta_lorentz_model.A_0.value == pytest.approx(1.0)
        assert delta_lorentz_model.lorentzian_width.value == pytest.approx(1.0)

    def test_y_unit_setter_raises(self, delta_lorentz_model):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=r'read-only'):
            delta_lorentz_model.y_unit = '1/meV'

    def test_init_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN THEN EXPECT
        assert delta_lorentz_model_with_Q.display_name == 'DeltaLorentz'
        assert delta_lorentz_model_with_Q.x_unit == 'meV'
        assert delta_lorentz_model_with_Q.scale.value == pytest.approx(1.0)
        assert delta_lorentz_model_with_Q.mean_u_squared.value == pytest.approx(0.0)
        assert delta_lorentz_model_with_Q.A_0.value == pytest.approx(0.5)
        assert delta_lorentz_model_with_Q.lorentzian_width.value == pytest.approx(0.0015)
        assert delta_lorentz_model_with_Q._allow_Q_variation == {
            'A_0': True,
            'lorentzian_width': True,
        }
        assert len(delta_lorentz_model_with_Q._A_0_list) == len(delta_lorentz_model_with_Q.Q)
        assert len(delta_lorentz_model_with_Q._lorentzian_width_list) == len(
            delta_lorentz_model_with_Q.Q
        )
        assert all(pytest.approx(a.value) == 0.5 for a in delta_lorentz_model_with_Q._A_0_list)
        assert all(
            pytest.approx(lw.value) == 0.0015
            for lw in delta_lorentz_model_with_Q._lorentzian_width_list
        )

    @pytest.mark.parametrize(
        'kwargs,expected_exception, expected_message',
        [
            (
                {
                    'mean_u_squared': -1.0,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                ValueError,
                'mean_u_squared must be non-negative',
            ),
            (
                {
                    'mean_u_squared': 'not a number',
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'mean_u_squared must be a number',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': -1.0,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                ValueError,
                'A_0 must be between 0 and 1',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 'not a number',
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'A_0 must be a number',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': -1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                ValueError,
                'lorentzian_width must be ',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 'not a number',
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'lorentzian_width must be a number',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': 'Not a dict',
                    'delta_name': 'Delta',
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'allow_Q_variation must be a dict',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 123,
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'delta_name must be a string',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': None,
                    'delta_display_name': 'DeltaDisplay',
                },
                TypeError,
                'delta_name must be a string',
            ),
            (
                {
                    'mean_u_squared': 0.1,
                    'A_0': 0.5,
                    'lorentzian_width': 1.0,
                    'allow_Q_variation': {'A_0': True, 'lorentzian_width': True},
                    'delta_name': 'Delta',
                    'delta_display_name': 123,
                },
                TypeError,
                'delta_display_name must be a string',
            ),
        ],
        ids=[
            'mean_u_squared negative',
            'mean_u_squared not a number',
            'A_0 negative',
            'A_0 not a number',
            'lorentzian_width negative',
            'lorentzian_width not a number',
            'allow_Q_variation not a dict',
            'delta_name not a string',
            'delta_name not a string (None)',
            'delta_display_name not a string',
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_exception, expected_message):
        with pytest.raises(expected_exception, match=expected_message):
            DeltaLorentz(**kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        ('attribute', 'value', 'expected'),
        [
            ('mean_u_squared', 2.0, 2.0),
            ('mean_u_squared', 0.0, 0.0),
            ('mean_u_squared', 5, 5.0),
            ('A_0', 0.0, 0.0),
            ('A_0', 1.0, 1.0),
            ('A_0', 0.5, 0.5),
            ('lorentzian_width', 1.5, 1.5),
            ('delta_name', 'delta', 'delta'),
            ('delta_display_name', 'display', 'display'),
            ('delta_display_name', None, None),
        ],
        ids=[
            'mean_u_squared set to 2.0',
            'mean_u_squared set to 0.0',
            'mean_u_squared set to 5 (int)',
            'A_0 set to 0.0',
            'A_0 set to 1.0',
            'A_0 set to 0.5',
            'lorentzian_width set to 1.5',
            "delta_name set to 'delta'",
            "delta_display_name set to 'display'",
            'delta_display_name set to None',
        ],
    )
    def test_setters_valid(
        self,
        delta_lorentz_model,
        attribute,
        value,
        expected,
    ):
        # WHEN
        setattr(delta_lorentz_model, attribute, value)

        # THEN
        result = getattr(delta_lorentz_model, attribute)

        # Handle Parameters
        if isinstance(result, Parameter):
            result = result.value

        # EXPECT
        assert result == expected

    @pytest.mark.parametrize(
        ('attribute', 'value', 'exception', 'message'),
        [
            (
                'mean_u_squared',
                -1.0,
                ValueError,
                r'mean_u_squared must be non-negative.',
            ),
            (
                'mean_u_squared',
                'invalid',
                TypeError,
                r'mean_u_squared must be a number.',
            ),
            (
                'A_0',
                -0.1,
                ValueError,
                r'A_0 must be between 0 and 1.',
            ),
            (
                'A_0',
                1.1,
                ValueError,
                r'A_0 must be between 0 and 1.',
            ),
            (
                'A_0',
                'invalid',
                TypeError,
                r'A_0 must be a number.',
            ),
            (
                'A_1',
                0.5,
                AttributeError,
                r'A_1 is a dependent parameter and cannot be set directly.',
            ),
            (
                'lorentzian_width',
                -0.1,
                ValueError,
                r'lorentzian_width must be.',
            ),
            (
                'lorentzian_width',
                'invalid',
                TypeError,
                r'lorentzian_width must be a number.',
            ),
            (
                'delta_name',
                1,
                TypeError,
                r'delta_name must be a string.',
            ),
            (
                'delta_name',
                None,
                TypeError,
                r'delta_name must be a string.',
            ),
            (
                'delta_display_name',
                1,
                TypeError,
                r'delta_display_name must be a string or None.',
            ),
            (
                'delta_display_name',
                [],
                TypeError,
                r'delta_display_name must be a string or None.',
            ),
        ],
        ids=[
            'mean_u_squared negative',
            'mean_u_squared not a number',
            'A_0 less than 0',
            'A_0 greater than 1',
            'A_0 not a number',
            'A_1 set directly',
            'lorentzian_width negative',
            'lorentzian_width not a number',
            'delta_name not a string',
            'delta_name not a string (None)',
            'delta_display_name not a string',
            'delta_display_name not a string (list)',
        ],
    )
    def test_setters_invalid(
        self,
        delta_lorentz_model,
        attribute,
        value,
        exception,
        message,
    ):
        # WHEN THEN EXPECT
        with pytest.raises(exception, match=message):
            setattr(delta_lorentz_model, attribute, value)

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def test_calculate_width_without_Q(self, delta_lorentz_model):
        # WHEN THEN
        width = delta_lorentz_model.calculate_width(Q=0.5)

        # EXPECT
        assert len(width) == 1
        assert width[0] == pytest.approx(1.0)

    def test_calculate_width_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN THEN
        width = delta_lorentz_model_with_Q.calculate_width()

        # EXPECT
        assert len(width) == len(delta_lorentz_model_with_Q.Q)
        assert all(width_i == pytest.approx(0.0015) for width_i in width)

    def test_calculate_EISF(self, delta_lorentz_model):
        # WHEN

        # THEN
        eisf = delta_lorentz_model.calculate_EISF(Q=0.5)

        # EXPECT
        assert len(eisf) == 1
        expected = delta_lorentz_model.A_0.value * np.exp(
            -delta_lorentz_model.mean_u_squared.value * 0.5**2
        )
        assert eisf[0] == pytest.approx(expected)

    def test_calculate_EISF_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN

        # THEN
        eisf = delta_lorentz_model_with_Q.calculate_EISF()

        # EXPECT
        assert len(eisf) == len(delta_lorentz_model_with_Q.Q)
        for i in range(len(eisf)):
            expected = delta_lorentz_model_with_Q._A_0_list[i].value * np.exp(
                -delta_lorentz_model_with_Q.mean_u_squared.value
                * delta_lorentz_model_with_Q.Q.values[i] ** 2
            )
            assert eisf[i] == pytest.approx(expected)

    def test_calculate_QISF(self, delta_lorentz_model):
        # WHEN THEN
        qisf = delta_lorentz_model.calculate_QISF(Q=0.5)

        # EXPECT
        assert len(qisf) == 1
        expected = delta_lorentz_model.A_1.value * np.exp(
            -delta_lorentz_model.mean_u_squared.value * 0.5**2
        )
        assert qisf[0] == pytest.approx(expected)

    def test_calculate_QISF_with_Q(self, delta_lorentz_model_with_Q):
        # WHEN THEN
        qisf = delta_lorentz_model_with_Q.calculate_QISF()

        # EXPECT
        assert len(qisf) == len(delta_lorentz_model_with_Q.Q)
        for i in range(len(qisf)):
            expected = delta_lorentz_model_with_Q._A_1_list[i].value * np.exp(
                -delta_lorentz_model_with_Q.mean_u_squared.value
                * delta_lorentz_model_with_Q.Q.values[i] ** 2
            )

            assert qisf[i] == pytest.approx(expected)

    def test_create_component_collections_no_Q(self, delta_lorentz_model):
        # WHEN

        # THEN
        collections = delta_lorentz_model.create_component_collections()

        # EXPECT
        assert collections == []

    def test_create_component_collections_with_Q_variation(self, delta_lorentz_model_with_Q):
        # WHEN

        # THEN
        collections = delta_lorentz_model_with_Q.create_component_collections()

        # EXPECT
        assert len(collections) == 7

        for collection in collections:
            assert len(collection) == 2
            # Check Lorentzian
            assert isinstance(collection[0], Lorentzian)
            assert collection[0].width.value == pytest.approx(0.0015)
            assert collection[0].width.independent is True
            assert collection[0].area.independent is False
            assert (
                'scale * exp(-mean_u_squared_ratio.value *'
                in collection[0].area.dependency_expression
            )
            assert 'A_1' in collection[0].area.dependency_expression

            # Check DeltaFunction
            assert isinstance(collection[1], DeltaFunction)
            assert collection[1].area.independent is False
            assert (
                'scale * exp(-mean_u_squared_ratio.value *'
                in collection[1].area.dependency_expression
            )
            assert 'A_0' in collection[1].area.dependency_expression

    def test_create_component_collections_with_no_Q_variation(
        self, delta_lorentz_model_with_Q_no_variation
    ):
        # WHEN

        # THEN
        collections = delta_lorentz_model_with_Q_no_variation.create_component_collections()

        # EXPECT
        assert len(collections) == 7

        for collection in collections:
            assert len(collection) == 2
            # Check Lorentzian
            assert isinstance(collection[0], Lorentzian)
            assert collection[0].width.value == pytest.approx(0.0015)
            assert collection[0].width.independent is False
            assert collection[0].width.dependency_expression == 'lorentzian_width'
            assert collection[0].area.independent is False
            assert (
                'scale * exp(-mean_u_squared_ratio.value *'
                in collection[0].area.dependency_expression
            )
            assert 'A_1' in collection[0].area.dependency_expression

            # Check DeltaFunction
            assert isinstance(collection[1], DeltaFunction)
            assert collection[1].area.independent is False
            assert (
                'scale * exp(-mean_u_squared_ratio.value *'
                in collection[1].area.dependency_expression
            )
            assert 'A_0' in collection[1].area.dependency_expression

    @pytest.mark.parametrize(
        ('Q_index', 'expected_exception', 'expected_message'),
        [
            (-1, IndexError, r'Q_index must be non-negative'),
            (100, IndexError, r'Q_index 100 is out of bounds'),
            ('string', TypeError, r'Q_index must be an int or None, got str'),
        ],
        ids=[
            'negative index',
            'index too large',
            'non-integer index',
        ],
    )
    def test_get_independent_variables_raises(
        self, delta_lorentz_model_with_Q, Q_index, expected_exception, expected_message
    ):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            delta_lorentz_model_with_Q.get_independent_variables(Q_index=Q_index)

    def test_get_all_variables_no_Q_index(self, delta_lorentz_model_with_Q):
        # WHEN

        # THEN
        variables = delta_lorentz_model_with_Q.get_all_variables()

        # EXPECT
        # scale, mean_u_squared, + Q-dependent A_0, A_1
        # + 3 Lorentzian and 2 DeltaFunction parameters
        # for each Q
        assert len(variables) == 2 + (2 + 3 + 2) * len(delta_lorentz_model_with_Q.Q)
        assert len(variables) == len(set(variables))

    def test_get_all_variables_with_Q_index(self, delta_lorentz_model_with_Q):
        # WHEN

        # THEN
        variables = delta_lorentz_model_with_Q.get_all_variables(Q_index=0)

        # EXPECT
        # scale, mean_u_squared, + A_0, A_1
        # + 3 Lorentzian and 2 DeltaFunction parameters

        assert len(variables) == 2 + (2 + 3 + 2)
        assert len(variables) == len(set(variables))

    def test_get_all_variables_no_Q_index_no_Q_variation(
        self, delta_lorentz_model_with_Q_no_variation
    ):
        # WHEN

        # THEN
        variables = delta_lorentz_model_with_Q_no_variation.get_all_variables()

        # EXPECT
        # scale, mean_u_squared, + Q-independent A_0, A_1 and lorentzian_width
        # + 3 Lorentzian and 2 DeltaFunction parameters for each Q
        assert len(variables) == 2 + 3 + (3 + 2) * len(delta_lorentz_model_with_Q_no_variation.Q)
        assert len(variables) == len(set(variables))

    def test_get_all_variables_with_Q_index_no_Q_variation(
        self, delta_lorentz_model_with_Q_no_variation
    ):
        # WHEN

        # THEN
        variables = delta_lorentz_model_with_Q_no_variation.get_all_variables(Q_index=0)

        # EXPECT
        # scale, mean_u_squared, + A_0, A_1 and lorentzian_width
        # + 3 Lorentzian and 2 DeltaFunction parameters

        assert len(variables) == 2 + (3 + 3 + 2)
        assert len(variables) == len(set(variables))

    def test_get_all_variables_invalid_Q_index(self, delta_lorentz_model_with_Q):
        # WHEN THEN EXPECT
        with pytest.raises(IndexError, match='Q_index must be non-negative'):
            delta_lorentz_model_with_Q.get_all_variables(Q_index=-1)

        with pytest.raises(IndexError, match=r'Q_index \d+ is out of bounds'):
            delta_lorentz_model_with_Q.get_all_variables(Q_index=len(delta_lorentz_model_with_Q.Q))

    @pytest.mark.parametrize(
        ('allow_Q_variation', 'expected'),
        [
            (
                None,
                {
                    'A_0': False,
                    'lorentzian_width': False,
                },
            ),
            (
                {},
                {
                    'A_0': False,
                    'lorentzian_width': False,
                },
            ),
            (
                {'A_0': True},
                {
                    'A_0': True,
                    'lorentzian_width': False,
                },
            ),
            (
                {'lorentzian_width': True},
                {
                    'A_0': False,
                    'lorentzian_width': True,
                },
            ),
            (
                {
                    'A_0': True,
                    'lorentzian_width': True,
                },
                {
                    'A_0': True,
                    'lorentzian_width': True,
                },
            ),
        ],
    )
    def test_create_Q_variation_dict_valid(
        self,
        delta_lorentz_model,
        allow_Q_variation,
        expected,
    ):
        result = delta_lorentz_model._create_Q_variation_dict(allow_Q_variation)

        assert result == expected

    @pytest.mark.parametrize(
        'allow_Q_variation',
        [
            1,
            'invalid',
            [1, 2, 3],
            1.5,
        ],
    )
    def test_create_Q_variation_dict_raises_type_error(
        self,
        delta_lorentz_model,
        allow_Q_variation,
    ):
        with pytest.raises(
            TypeError,
            match=r'allow_Q_variation must be a dict or None.',
        ):
            delta_lorentz_model._create_Q_variation_dict(allow_Q_variation)

    @pytest.mark.parametrize(
        'allow_Q_variation',
        [
            {'invalid_key': True},
            {'A_0': True, 'bad_key': False},
        ],
    )
    def test_create_Q_variation_dict_raises_value_error(
        self,
        delta_lorentz_model,
        allow_Q_variation,
    ):
        with pytest.raises(
            ValueError,
            match=r'Unknown keys in allow_Q_variation',
        ):
            delta_lorentz_model._create_Q_variation_dict(allow_Q_variation)

    def test_create_mean_u_squared_parameter(self, delta_lorentz_model):
        # WHEN

        # THEN
        mean_u_squared_param = delta_lorentz_model._create_mean_u_squared_parameter(2.0)

        # THEN EXPECT
        assert isinstance(mean_u_squared_param, Parameter)
        assert mean_u_squared_param.value == pytest.approx(2.0)
        assert mean_u_squared_param.independent is True
        assert mean_u_squared_param.fixed is False
        assert mean_u_squared_param.min == pytest.approx(0.0)
        assert mean_u_squared_param.unit == 'Å^2'

    def test_create_mean_u_squared_parameter_invalid(self, delta_lorentz_model):
        with pytest.raises(ValueError, match=r'mean_u_squared must be non-negative.'):
            delta_lorentz_model._create_mean_u_squared_parameter(-1.0)

        with pytest.raises(TypeError, match=r'mean_u_squared must be a number.'):
            delta_lorentz_model._create_mean_u_squared_parameter('invalid')

    def test_create_A0_A1_parameters(self, delta_lorentz_model):
        # WHEN
        # THEN
        A_0_param, A_1_param = delta_lorentz_model._create_A0_A1_parameters(0.75)

        # THEN EXPECT
        assert isinstance(A_0_param, Parameter)
        assert A_0_param.value == pytest.approx(0.75)
        assert A_0_param.independent is True
        assert A_0_param.fixed is False
        assert A_0_param.min == pytest.approx(0.0)
        assert A_0_param.max == pytest.approx(1.0)

        assert isinstance(A_1_param, Parameter)
        assert A_1_param.value == pytest.approx(0.25)
        assert A_1_param.independent is False
        assert A_1_param.fixed is False
        assert A_1_param.dependency_expression == '1 - A_0'

    def test_create_A0_A1_parameters_invalid(self, delta_lorentz_model):
        with pytest.raises(ValueError, match=r'A_0 must be between 0 and 1.'):
            delta_lorentz_model._create_A0_A1_parameters(-0.1)

        with pytest.raises(ValueError, match=r'A_0 must be between 0 and 1.'):
            delta_lorentz_model._create_A0_A1_parameters(1.1)

        with pytest.raises(TypeError, match=r'A_0 must be a number.'):
            delta_lorentz_model._create_A0_A1_parameters('invalid')

    def test_create_lorentzian_width_parameter(self, delta_lorentz_model):
        # WHEN

        # THEN
        width_param = delta_lorentz_model._create_lorentzian_width_parameter(0.5)

        # THEN EXPECT
        assert isinstance(width_param, Parameter)
        assert width_param.value == pytest.approx(0.5)
        assert width_param.independent is True
        assert width_param.fixed is False
        assert width_param.unit == 'meV'

    def test_create_lorentzian_width_parameter_invalid(self, delta_lorentz_model):
        with pytest.raises(ValueError, match=r'lorentzian_width must be '):
            delta_lorentz_model._create_lorentzian_width_parameter(-0.1)

        with pytest.raises(TypeError, match=r'lorentzian_width must be a number.'):
            delta_lorentz_model._create_lorentzian_width_parameter('invalid')

    def test_create_A0_A1_parameter_lists(self, delta_lorentz_model_with_Q):
        # WHEN
        A_0 = delta_lorentz_model_with_Q.A_0
        A_0.value = 0.75

        # THEN
        A_0_list, A_1_list = delta_lorentz_model_with_Q._create_A0_A1_parameter_lists(A_0)

        # THEN EXPECT
        assert isinstance(A_0_list, list)
        assert isinstance(A_1_list, list)
        assert len(A_0_list) == len(delta_lorentz_model_with_Q.Q)
        assert len(A_1_list) == len(delta_lorentz_model_with_Q.Q)

        for A_0_param in A_0_list:
            assert isinstance(A_0_param, Parameter)
            assert A_0_param.value == pytest.approx(0.75)
            assert A_0_param.independent is True
            assert A_0_param.fixed is False
            assert A_0_param.min == pytest.approx(0.0)
            assert A_0_param.max == pytest.approx(1.0)

        for A_1_param in A_1_list:
            assert isinstance(A_1_param, Parameter)
            assert A_1_param.value == pytest.approx(0.25)
            assert A_1_param.independent is False
            assert A_1_param.fixed is False
            assert A_1_param.dependency_expression == '1 - A_0'

    def test_create_lorentzian_width_parameter_list(self, delta_lorentz_model_with_Q):
        # WHEN
        width = delta_lorentz_model_with_Q.lorentzian_width
        width.value = 0.5

        # THEN
        width_list = delta_lorentz_model_with_Q._create_lorentzian_width_parameter_list(width)

        # EXPECT
        assert isinstance(width_list, list)
        assert len(width_list) == len(delta_lorentz_model_with_Q.Q)

        for width_param in width_list:
            assert isinstance(width_param, Parameter)
            assert width_param.value == pytest.approx(0.5)
            assert width_param.independent is True
            assert width_param.fixed is False
            assert width_param.unit == 'meV'

    def test_on_Q_change_Q_set(self, delta_lorentz_model_with_Q):
        # WHEN

        # THEN

        # This calls on_Q_change
        delta_lorentz_model_with_Q.clear_Q(confirm=True)

        # EXPECT
        assert delta_lorentz_model_with_Q.Q is None
        assert delta_lorentz_model_with_Q._A_0_list == []
        assert delta_lorentz_model_with_Q._A_1_list == []
        assert delta_lorentz_model_with_Q._lorentzian_width_list == []

        # THEN
        new_Q = np.linspace(0.5, 2, 7)

        delta_lorentz_model_with_Q.Q = new_Q

        # EXPECT
        assert len(delta_lorentz_model_with_Q._A_0_list) == len(new_Q)
        assert len(delta_lorentz_model_with_Q._lorentzian_width_list) == len(new_Q)
        assert all(pytest.approx(a.value) == 0.5 for a in delta_lorentz_model_with_Q._A_0_list)
        assert all(
            pytest.approx(lw.value) == 0.0015
            for lw in delta_lorentz_model_with_Q._lorentzian_width_list
        )

    def test_on_Q_change_Q_set_no_variation(self, delta_lorentz_model_with_Q_no_variation):
        # WHEN

        # THEN

        # This calls on_Q_change
        delta_lorentz_model_with_Q_no_variation.clear_Q(confirm=True)

        # EXPECT
        assert delta_lorentz_model_with_Q_no_variation.Q is None
        assert delta_lorentz_model_with_Q_no_variation._A_0_list == []
        assert delta_lorentz_model_with_Q_no_variation._A_1_list == []
        assert delta_lorentz_model_with_Q_no_variation._lorentzian_width_list == []

        # THEN
        # This calls on_Q_change
        new_Q = np.linspace(0.5, 2, 7)

        delta_lorentz_model_with_Q_no_variation.Q = new_Q

        # EXPECT
        assert delta_lorentz_model_with_Q_no_variation._A_0_list == []
        assert delta_lorentz_model_with_Q_no_variation._A_1_list == []
        assert delta_lorentz_model_with_Q_no_variation._lorentzian_width_list == []

    def test_repr(self, delta_lorentz_model):
        # WHEN THEN
        repr_str = repr(delta_lorentz_model)

        # EXPECT
        assert 'DeltaLorentz' in repr_str
        assert 'scale' in repr_str
        assert 'mean_u_squared' in repr_str
        assert 'A_0' in repr_str
        assert 'lorentzian_width' in repr_str
        # Regression: a stray ')' used to mangle this into 'x_unit=meV), y_unit=...'
        assert 'x_unit=meV, y_unit=dimensionless' in repr_str

    # ───── Regression tests ─────

    def test_calculate_width_with_Q_subset(self, delta_lorentz_model_with_Q):
        # WHEN: Q-varying widths with distinguishable per-Q values
        model = delta_lorentz_model_with_Q
        for i, width in enumerate(model._lorentzian_width_list):
            width.value = 1.0 + i

        # THEN: request a subset of the stored Q values (as ParameterAnalysis does when it
        # drops rows with missing data)
        subset = model.Q.values[[1, 3, 5]]
        widths = model.calculate_width(subset)

        # EXPECT: one width per requested Q, matching the stored per-Q parameters
        np.testing.assert_allclose(widths, [2.0, 4.0, 6.0])

    def test_calculate_EISF_with_Q_subset(self, delta_lorentz_model_with_Q):
        # WHEN: Q-varying A_0 with distinguishable per-Q values
        model = delta_lorentz_model_with_Q
        for i, A_0 in enumerate(model._A_0_list):
            A_0.value = (i + 1) / 10

        # THEN
        subset = model.Q.values[[0, 6]]
        eisf = model.calculate_EISF(subset)

        # EXPECT: mean_u_squared is 0, so EISF equals the per-Q A_0 values
        np.testing.assert_allclose(eisf, [0.1, 0.7])

    def test_calculate_width_with_unknown_Q_raises(self, delta_lorentz_model_with_Q):
        # WHEN THEN EXPECT: a Q value not stored in the model has no per-Q width
        with pytest.raises(ValueError, match='do not match the Q values stored'):
            delta_lorentz_model_with_Q.calculate_width(np.array([10.0]))

    def test_calculate_width_raises_after_clear_Q_when_allow_Q_variation(
        self, delta_lorentz_model_with_Q
    ):
        # WHEN: model with Q-variation enabled for lorentzian_width
        assert delta_lorentz_model_with_Q._allow_Q_variation['lorentzian_width'] is True
        assert len(delta_lorentz_model_with_Q._lorentzian_width_list) > 0

        # THEN: clear Q (empties _lorentzian_width_list)
        delta_lorentz_model_with_Q.clear_Q(confirm=True)
        assert len(delta_lorentz_model_with_Q._lorentzian_width_list) == 0

        # THEN: before the fix, calculate_width() silently returned [] instead of raising.
        with pytest.raises(ValueError, match='Q must be provided'):
            delta_lorentz_model_with_Q.calculate_width()
