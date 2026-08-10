# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import FrozenInstanceError

import pytest

from easydynamics.utils.fit_target import FitTarget


def test_fit_target_holds_prediction_metadata():
    # WHEN a FitTarget is created
    target = FitTarget(
        name='width',
        dataset_key='Lorentzian width',
        function=lambda x: x * 2,
        label='DeltaLorentz width',
        x_unit='1/angstrom',
        y_unit='meV',
    )
    # EXPECT its attributes to be preserved and the function callable
    assert target.name == 'width'
    assert target.dataset_key == 'Lorentzian width'
    assert target.function(3) == 6
    assert target.label == 'DeltaLorentz width'
    assert target.x_unit == '1/angstrom'
    assert target.y_unit == 'meV'


def test_fit_target_allows_none_key_and_units():
    # WHEN a component-style FitTarget without a default key/units is created
    target = FitTarget(
        name='value',
        dataset_key=None,
        function=lambda x: x,
        label='value',
        x_unit=None,
        y_unit=None,
    )
    # EXPECT the optional fields to be None
    assert target.dataset_key is None
    assert target.x_unit is None
    assert target.y_unit is None


def test_fit_target_is_frozen():
    # GIVEN a FitTarget
    target = FitTarget(
        name='value',
        dataset_key=None,
        function=lambda x: x,
        label='value',
        x_unit=None,
        y_unit=None,
    )
    # WHEN THEN EXPECT: it is immutable
    with pytest.raises(FrozenInstanceError):
        target.name = 'other'
