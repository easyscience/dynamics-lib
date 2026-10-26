# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.diffusion_model.delta_lorentzian_diffusion_model_base import (
    DeltaLorentzianDiffusionModelBase,
)
from easydynamics.utils.utils import Q_type

# Weights of orders 0 (delta function) to 3; they sum to 1.
WEIGHTS = [0.5, 0.3, 0.15, 0.05]


class DummyDiffusionModel(DeltaLorentzianDiffusionModelBase):
    """Test-only model with constant weights and widths order * w."""

    _order_symbol = 'k'

    def __init__(self, number_of_lorentzians: int = 3, **kwargs) -> None:
        kwargs.setdefault('name', 'Dummy')
        kwargs.setdefault('display_name', 'Dummy')
        super().__init__(number_of_lorentzians, **kwargs)
        self.w = Parameter(name='w', value=0.1, unit='meV')
        self.create_component_collections()

    def calculate_width(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        self._verify_order(order)
        return order * self.w.value * np.ones_like(self._ensure_Q(Q))

    def calculate_EISF(self, Q: Q_type | None = None) -> np.ndarray:
        return WEIGHTS[0] * np.ones_like(self._ensure_Q(Q))

    def calculate_QISF_order(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        self._verify_order(order)
        return WEIGHTS[order] * np.ones_like(self._ensure_Q(Q))

    def _create_weight(self, Q: float, order: int) -> Parameter:
        return Parameter(name=f'weight_k{order}_Q{Q:.2f}', value=WEIGHTS[order], fixed=True)

    def _write_width_dependency_expression(self, order: int) -> str:
        return f'{order} * w'

    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        return {'w': self.w}


@pytest.fixture
def dummy():
    return DummyDiffusionModel(scale=2.0, Q=np.array([0.5, 1.0]))


# ----------------------------------------------------------------------
# Components
# ----------------------------------------------------------------------


def test_component_names_use_symbol(dummy):
    # WHEN
    components = dummy.get_component_collections()[0][:]

    # THEN EXPECT
    assert dummy.lorentzian_component_name(2) == 'Dummy_k2'
    assert [c.name for c in components] == ['Delta function', 'Dummy_k1', 'Dummy_k2', 'Dummy_k3']
    assert components[1].display_name == 'Dummy_k1'


def test_components_have_weights_and_widths(dummy):
    # WHEN THEN EXPECT
    for collection in dummy.get_component_collections():
        delta, *lorentzians = collection[:]
        assert delta.area.value == pytest.approx(1.0)
        assert [c.area.value for c in lorentzians] == pytest.approx([0.6, 0.3, 0.1])
        assert [c.width.value for c in lorentzians] == pytest.approx([0.1, 0.2, 0.3])


def test_areas_follow_scale(dummy):
    # WHEN
    dummy.scale.value = 4.0

    # THEN EXPECT
    assert dummy.get_component_collections()[0][1].area.value == pytest.approx(1.2)


def test_setting_Q_builds_collections():
    # WHEN
    dummy = DummyDiffusionModel()

    # THEN EXPECT
    assert dummy.get_component_collections() == []
    dummy.Q = [0.5]
    assert len(dummy.get_component_collections()[0][:]) == 4


# ----------------------------------------------------------------------
# Orders and validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize('order', [0, 4, 1.5, True])
def test_order_out_of_range_raises(dummy, order):
    # WHEN THEN EXPECT
    with pytest.raises((TypeError, ValueError), match='order must be'):
        dummy.lorentzian_component_name(order)


@pytest.mark.parametrize('n, error', [(0, ValueError), (1.0, TypeError), (True, TypeError)])
def test_number_of_lorentzians_validated(n, error):
    # WHEN THEN EXPECT
    with pytest.raises(error, match='number_of_lorentzians must be'):
        DummyDiffusionModel(number_of_lorentzians=n)


def test_number_of_lorentzians_read_only(dummy):
    # WHEN THEN EXPECT
    assert dummy.number_of_lorentzians == 3
    with pytest.raises(AttributeError):
        dummy.number_of_lorentzians = 2


@pytest.mark.parametrize('kwargs', [{'delta_name': 1}, {'delta_display_name': 1}])
def test_delta_names_validated(kwargs):
    # WHEN THEN EXPECT
    with pytest.raises(TypeError):
        DummyDiffusionModel(**kwargs)


@pytest.mark.parametrize('display_name', ['Elastic line', None])
def test_delta_name_setters(dummy, display_name):
    # WHEN
    dummy.delta_name = 'Elastic'
    dummy.delta_display_name = display_name

    # THEN EXPECT
    assert dummy.delta_name == 'Elastic'
    assert dummy.delta_display_name == display_name


@pytest.mark.parametrize(
    'attribute, message',
    [
        ('delta_name', 'delta_name must be a string'),
        ('delta_display_name', 'delta_display_name must be a string or None'),
    ],
)
def test_delta_name_setters_reject_non_strings(dummy, attribute, message):
    # WHEN THEN EXPECT
    with pytest.raises(TypeError, match=message):
        setattr(dummy, attribute, 1)


def test_incomplete_subclass_cannot_be_created():
    # WHEN
    class NoEISF(DummyDiffusionModel):
        calculate_EISF = DeltaLorentzianDiffusionModelBase.calculate_EISF

    # THEN EXPECT
    with pytest.raises(TypeError):
        NoEISF()


# ----------------------------------------------------------------------
# Fit targets
# ----------------------------------------------------------------------


def test_fit_target_names(dummy):
    # WHEN THEN EXPECT
    assert [t.name for t in dummy.get_fit_targets()] == [
        'elastic_area',
        'area_1',
        'width_1',
        'area_2',
        'width_2',
        'area_3',
        'width_3',
    ]


def test_fit_target_values(dummy):
    # WHEN
    targets = {t.name: t for t in dummy.get_fit_targets()}
    Q = np.array([0.5])

    # THEN EXPECT
    np.testing.assert_allclose(targets['elastic_area'].function(Q), [1.0])
    np.testing.assert_allclose(targets['area_2'].function(Q), [0.3])
    np.testing.assert_allclose(targets['width_3'].function(Q), [0.3])


def test_fit_target_keys_follow_names(dummy):
    # WHEN
    dummy.delta_name = 'Elastic'
    targets = dummy.get_fit_targets()

    # THEN EXPECT
    assert targets[0].dataset_key == 'Elastic area'
    assert targets[1].dataset_key == 'Dummy_k1 area'
