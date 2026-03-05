# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model.resolution_model import ResolutionModel


class TestResolutionModel:
    @pytest.fixture
    def resolution_model(self):
        component1 = Gaussian(
            display_name='TestGaussian1',
            area=1.0,
            center=0.0,
            width=1.0,
            unit='meV',
        )
        component2 = Lorentzian(
            display_name='TestLorentzian1',
            area=2.0,
            center=1.0,
            width=0.5,
            unit='meV',
        )
        component_collection = ComponentCollection()
        component_collection.append_component(component1)
        component_collection.append_component(component2)
        resolution_model = ResolutionModel(
            display_name='InitModel',
            components=component_collection,
            unit='meV',
            Q=np.array([1.0, 2.0, 3.0]),
        )

        return resolution_model

    def test_init(self, resolution_model):
        # WHEN THEN
        model = resolution_model

        # EXPECT
        assert model.display_name == 'InitModel'
        assert model.unit == 'meV'
        assert len(model.components) == 2
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize(
        'invalid_component, expected_error_msg',
        [
            ('invalid_component', 'must be '),
            (123, 'must be '),
            (45.6, 'must be '),
            (DeltaFunction(), 'cannot be a DeltaFunction'),
            (Polynomial(), 'cannot be a Polynomial'),
            (
                [Gaussian(), 'invalid_in_list'],
                'must be ',
            ),
        ],
        ids=[
            'string',
            'int',
            'float',
            'DeltaFunction',
            'Polynomial',
            'list_with_invalid',
        ],
    )
    def test_init_raises_with_invalid_components(self, invalid_component, expected_error_msg):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match=expected_error_msg,
        ):
            ResolutionModel(components=invalid_component)

        with pytest.raises(
            TypeError,
            match=expected_error_msg,
        ):
            collection = ComponentCollection()
            collection.append_component(invalid_component)
            ResolutionModel(components=collection)

    def test_append_and_remove_and_clear_component(self, resolution_model):
        # WHEN
        new_component = Gaussian(unique_name='NewGaussian')

        # THEN
        resolution_model.append_component(new_component)

        # EXPECT
        assert len(resolution_model.components) == 3
        assert resolution_model.components[-1] is new_component

        # THEN
        resolution_model.remove_component('NewGaussian')

        # EXPECT
        assert len(resolution_model.components) == 2

        # THEN
        resolution_model.clear_components()

        # EXPECT
        assert len(resolution_model.components) == 0

    def test_append_component_collection(self, resolution_model):
        # WHEN
        new_collection = ComponentCollection()
        new_component1 = Lorentzian()
        new_component2 = Gaussian()
        new_collection.append_component(new_component1)
        new_collection.append_component(new_component2)

        # THEN
        resolution_model.append_component(new_collection)

        # EXPECT
        assert len(resolution_model.components) == 4
        assert resolution_model.components[-2] is new_component1
        assert resolution_model.components[-1] is new_component2

    @pytest.mark.parametrize(
        'invalid_component',
        [
            DeltaFunction(),
            Polynomial(),
        ],
        ids=['DeltaFunction', 'Polynomial'],
    )
    def test_append_invalid_component_type_raises(self, resolution_model, invalid_component):
        # WHEN / THEN / EXPECT
        # appending a single component
        with pytest.raises(
            TypeError,
            match='cannot be ',
        ):
            resolution_model.append_component(invalid_component)

        # appending a collection with invalid component
        with pytest.raises(
            TypeError,
            match='cannot be ',
        ):
            collection = ComponentCollection()
            collection.append_component(invalid_component)
            resolution_model.append_component(collection)
