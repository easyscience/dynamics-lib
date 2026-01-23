import numpy as np
import pytest
import scipp as sc

from easydynamics.convolution.convolution_base import ConvolutionBase
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
                    'energy_unit': 'meV',
                },
                'Energy must be',
            ),
            (
                {
                    'energy': np.linspace(-10, 10, 100),
                    'sample_components': 'invalid',
                    'resolution_components': ComponentCollection(),
                    'energy_unit': 'meV',
                },
                '`sample_components` is an instance of str, but must be a ComponentCollection or ModelComponent.',  # noqa: E501
            ),
            (
                {
                    'energy': np.linspace(-10, 10, 100),
                    'sample_components': ComponentCollection(),
                    'resolution_components': 'invalid',
                    'energy_unit': 'meV',
                },
                '`resolution_components` is an instance of str, but must be a ComponentCollection or ModelComponent.',  # noqa: E501
            ),
            (
                {
                    'energy': np.linspace(-10, 10, 100),
                    'sample_components': ComponentCollection(),
                    'resolution_components': ComponentCollection(),
                    'energy_unit': 123,
                },
                'Energy_unit must be ',
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
            match='Energy must be a Number, a numpy ndarray or a scipp Variable.',
        ):
            convolution_base.energy = 'invalid'

    def test_energy_unit_property(self, convolution_base):
        # WHEN THEN EXPECT
        assert convolution_base.energy.unit == 'meV'

    def test_energy_unit_setter_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            AttributeError,
            match='Use convert_unit to change the unit between allowed types ',
        ):
            convolution_base.energy_unit = 'K'

    def test_convert_energy_unit(self, convolution_base):
        # WHEN THEN
        convolution_base.convert_energy_unit('eV')

        # EXPECT
        assert convolution_base.energy.unit == 'eV'
        assert convolution_base.energy_unit == 'eV'
        assert np.allclose(convolution_base.energy.values, np.linspace(-0.01, 0.01, 100))

    def test_convert_energy_unit_invalid_type_raises(self, convolution_base):
        # WHEN THEN EXPECT
        with pytest.raises(
            TypeError,
            match='Energy unit must be a string or scipp unit.',
        ):
            convolution_base.convert_energy_unit(123)

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
            match='`sample_components` is an instance of str, but must be a ComponentCollection or ModelComponent.',  # noqa: E501
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
            match='`resolution_components` is an instance of str, but must be a ComponentCollection or ModelComponent.',  # noqa: E501
        ):
            convolution_base.resolution_components = 'invalid'
