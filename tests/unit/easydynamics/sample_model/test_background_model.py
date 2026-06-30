# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.background_model import BackgroundModel


class TestBackgroundModel:
    @pytest.fixture
    def background_model(self):
        component1 = Gaussian(
            name='TestGaussian1',
            area=1.0,
            center=0.0,
            width=1.0,
            x_unit='meV',
        )
        component2 = Lorentzian(
            display_name='TestLorentzian1',
            area=2.0,
            center=1.0,
            width=0.5,
            x_unit='meV',
        )
        component_collection = ComponentCollection()
        component_collection.append_component(component1)
        component_collection.append_component(component2)
        return BackgroundModel(
            display_name='InitModel',
            components=component_collection,
            x_unit='meV',
            Q=np.array([1.0, 2.0, 3.0]),
        )

    def test_init(self, background_model):
        # WHEN THEN
        model = background_model

        # EXPECT
        assert model.display_name == 'InitModel'
        assert model.x_unit == 'meV'
        assert len(model.components) == 2
        np.testing.assert_array_equal(model.Q, np.array([1.0, 2.0, 3.0]))

    @pytest.mark.parametrize(
        'invalid_component, expected_error_msg',
        [
            ('invalid_component', 'must be '),
            (123, 'must be '),
            (45.6, 'must be '),
            (
                [Gaussian(), 'invalid_in_list'],
                'must be ',
            ),
        ],
        ids=[
            'string',
            'int',
            'float',
            'list_with_invalid',
        ],
    )
    def test_init_raises_with_invalid_components(self, invalid_component, expected_error_msg):
        # WHEN / THEN / EXPECT
        with pytest.raises(
            TypeError,
            match=expected_error_msg,
        ):
            BackgroundModel(components=invalid_component)

        with pytest.raises(
            TypeError,
            match=expected_error_msg,
        ):
            collection = ComponentCollection()
            collection.append_component(invalid_component)
            BackgroundModel(components=collection)

    def test_y_unit_default(self, background_model):
        # WHEN THEN EXPECT
        assert background_model.y_unit == 'dimensionless'

    def test_y_unit_setter_raises(self, background_model):
        # WHEN / THEN / EXPECT
        with pytest.raises(AttributeError):
            background_model.y_unit = '1/meV'

    def test_convert_y_unit(self):
        # WHEN
        g = Gaussian(area=1.0, x_unit='meV', y_unit='1/meV')
        model = BackgroundModel(components=g, x_unit='meV')
        # THEN: convert y_unit to '1/eV' (same dimension, different scale)
        model.convert_y_unit('1/eV')
        # EXPECT
        assert model.y_unit == '1/eV'
        assert model.components[0].y_unit == '1/eV'
        assert g.area.value == pytest.approx(1e3)
