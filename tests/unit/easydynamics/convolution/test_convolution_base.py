# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipp import UnitError

from easydynamics.convolution.convolution_base import ConvolutionBase
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model.component_collection import ComponentCollection


class TestConvolutionBase:
    @pytest.fixture
    def convolution_base(self):
        energy = np.linspace(-10, 10, 100)
        sample_components = ComponentCollection(display_name='ComponentCollection')
        resolution_components = ComponentCollection(display_name='ResolutionModel')

        return ConvolutionBase(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

    def test_init(self, convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(convolution_base, ConvolutionBase)
        assert isinstance(convolution_base.energy, sc.Variable)
        assert np.allclose(convolution_base.energy.values, np.linspace(-10, 10, 100))
        assert isinstance(convolution_base._sample_components, ComponentCollection)
        assert isinstance(convolution_base._resolution_components, ComponentCollection)

    def test_init_with_model_component(self):
        # WHEN
        energy = np.linspace(-10, 10, 100)
        sample_component = Gaussian()
        resolution_component = Gaussian()

        convolution_base = ConvolutionBase(
            energy=energy,
            sample_components=sample_component,
            resolution_components=resolution_component,
        )

        # THEN EXPECT
        assert isinstance(convolution_base, ConvolutionBase)
        assert isinstance(convolution_base.energy, sc.Variable)
        assert np.allclose(convolution_base.energy.values, np.linspace(-10, 10, 100))
        assert isinstance(convolution_base.sample_components, ComponentCollection)
        assert isinstance(convolution_base.resolution_components, ComponentCollection)
        assert convolution_base.sample_components.components[0] == sample_component
        assert convolution_base.resolution_components.components[0] == resolution_component

    def test_init_energy_numerical_none_offset(self):
        # WHEN
        energy = 1

        convolution_base = ConvolutionBase(
            energy=energy,
        )

        # THEN EXPECT
        assert isinstance(convolution_base, ConvolutionBase)
        assert isinstance(convolution_base.energy, sc.Variable)
        assert convolution_base.energy.values == np.array([1.0])
        assert convolution_base.energy.unit == 'meV'
        assert convolution_base._sample_components is None
        assert convolution_base._resolution_components is None

    @pytest.mark.parametrize(
        'kwargs, expected_message',
        [
            (
                {
                    'energy': 'invalid',
                    'sample_components': ComponentCollection(),
                    'resolution_components': ComponentCollection(),
                    'unit': 'meV',
                    'energy_offset': 0,
                },
                'Energy must be',
            ),
            (
                {
                    'energy': np.linspace(-10, 10, 100),
                    'sample_components': 'invalid',
                    'resolution_components': ComponentCollection(),
                    'unit': 'meV',
                    'energy_offset': 0,
                },
                (
                    '`sample_components` is an instance of str, '
                    'but must be a ComponentCollection or ModelComponent.'
                ),
            ),
            (
                {
                    'energy': np.linspace(-10, 10, 100),
                    'sample_components': ComponentCollection(),
                    'resolution_components': 'invalid',
                    'unit': 'meV',
                    'energy_offset': 0,
                },
                (
                    '`resolution_components` is an instance of str, '
                    'but must be a ComponentCollection or ModelComponent.'
                ),
            ),
            (
                {
                    'energy': np.linspace(-10, 10, 100),
                    'sample_components': ComponentCollection(),
                    'resolution_components': ComponentCollection(),
                    'unit': 123,
                    'energy_offset': 0,
                },
                'unit must be ',
            ),
            (
                {
                    'energy': np.linspace(-10, 10, 100),
                    'sample_components': ComponentCollection(),
                    'resolution_components': ComponentCollection(),
                    'unit': 'meV',
                    'energy_offset': 'invalid',
                },
                'Energy_offset must be ',
            ),
        ],
    )
    def test_input_type_validation_raises(self, kwargs, expected_message):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=expected_message):
            ConvolutionBase(**kwargs)

    @pytest.mark.parametrize(
        'energy, expected_energy',
        [
            (
                1,
                sc.array(dims=['energy'], values=[1.0], unit='meV'),
            ),
            (
                1.0,
                sc.array(dims=['energy'], values=[1.0], unit='meV'),
            ),
            (
                np.linspace(-5, 5, 50),
                sc.array(dims=['energy'], values=np.linspace(-5, 5, 50), unit='meV'),
            ),
            (
                sc.array(dims=['energy'], values=np.linspace(-5, 5, 50), unit='meV'),
                sc.array(dims=['energy'], values=np.linspace(-5, 5, 50), unit='meV'),
            ),
        ],
        ids=['int', 'float', 'np.ndarray', 'scipp.Variable'],
    )
    def test_energy_setter(self, convolution_base, energy, expected_energy):
        # WHEN
        convolution_base.energy = energy

        # THEN
        assert sc.identical(convolution_base.energy, expected_energy)

    def test_energy_setter_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match=r'Energy must be a Number, a numpy ndarray or a scipp Variable.',
        ):
            convolution_base.energy = 'invalid'

    def test_unit_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert convolution_base.energy.unit == 'meV'

    def test_unit_setter_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match=r'Use convert_unit to change the unit between allowed types ',
        ):
            convolution_base.unit = 'K'

    def test_convert_unit(self, convolution_base):
        # WHEN THEN
        convolution_base.convert_unit('eV')

        # EXPECT
        assert convolution_base.energy.unit == 'eV'
        assert convolution_base.unit == 'eV'
        assert np.allclose(convolution_base.energy.values, np.linspace(-0.01, 0.01, 100))

    def test_convert_unit_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match=r'Energy unit must be a string or scipp unit.',
        ):
            convolution_base.convert_unit(123)

    def test_convert_unit_invalid_unit_rollback(self, convolution_base):
        # WHEN THEN
        with pytest.raises(
            UnitError,
            match=r'Conversion from `meV` to `s` is not valid.',
        ):
            convolution_base.convert_unit('s')

        # EXPECT
        assert convolution_base.unit == 'meV'
        assert np.allclose(convolution_base.energy.values, np.linspace(-10, 10, 100))

    def test_convert_unit_invalid_offset_unit_rollback(self, convolution_base):
        # WHEN
        convolution_base.energy_offset = Parameter(name='energy_offset', value=5, unit='s')

        # THEN
        with pytest.raises(
            UnitError,
            match=r'Conversion from `s` to `meV` is not valid.',
        ):
            convolution_base.convert_unit('meV')

        # EXPECT
        assert convolution_base.unit == 'meV'
        assert convolution_base.energy_offset.unit == 's'

    def test_energy_offset_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert convolution_base.energy_offset.value == 0

        # THEN
        convolution_base.energy_offset = 5
        assert convolution_base.energy_offset.value == 5

        # THEN
        convolution_base.energy_offset = Parameter(name='energy_offset', value=10, unit='meV')
        assert convolution_base.energy_offset.value == 10
        assert convolution_base.energy_offset.unit == 'meV'

    def test_energy_offset_setter_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match=r'Energy_offset must be a number or a Parameter.',
        ):
            convolution_base.energy_offset = 'invalid'

    def test_energy_with_offset_setter_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match='is a read-only property',
        ):
            convolution_base.energy_with_offset = 5

    def test_sample_components_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(convolution_base.sample_components, ComponentCollection)

    def test_sample_components_setter(self, convolution_base):
        # WHEN
        new_sample_components = ComponentCollection(display_name='NewComponentCollection')

        # THEN
        convolution_base.sample_components = new_sample_components

        # EXPECT
        assert convolution_base.sample_components == new_sample_components

    def test_sample_components_setter_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match=(
                r'`sample_components` is an instance of str, '
                r'but must be a ComponentCollection or ModelComponent.'
            ),
        ):
            convolution_base.sample_components = 'invalid'

    def test_resolution_components_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert isinstance(convolution_base.resolution_components, ComponentCollection)

    def test_resolution_components_setter(self, convolution_base):
        # WHEN
        new_resolution_components = ComponentCollection(display_name='NewResolutionModel')
        # THEN
        convolution_base.resolution_components = new_resolution_components

        # EXPECT
        assert convolution_base.resolution_components == new_resolution_components

    def test_resolution_components_setter_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match=(
                r'`resolution_components` is an instance of str, '
                r'but must be a ComponentCollection or ModelComponent.'
            ),
        ):
            convolution_base.resolution_components = 'invalid'
