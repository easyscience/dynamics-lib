# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.convolution_settings import ConvolutionSettings
from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.convolution.numerical_convolution_base import NumericalConvolutionBase
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model.component_collection import ComponentCollection


class TestNumericalConvolutionBase:
    @pytest.fixture
    def default_numerical_convolution_base(self):
        energy = np.linspace(-10, 10, 101)
        sample_components = ComponentCollection(display_name='ComponentCollection')
        resolution_components = ComponentCollection(display_name='ResolutionModel')

        return NumericalConvolutionBase(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

    def test_init(self, default_numerical_convolution_base):
        """
        Test initialization of NumericalConvolutionBase with default
        parameters.
        """
        # WHEN THEN EXPECT
        assert isinstance(default_numerical_convolution_base, NumericalConvolutionBase)
        assert isinstance(default_numerical_convolution_base.energy, sc.Variable)
        assert np.allclose(
            default_numerical_convolution_base.energy.values, np.linspace(-10, 10, 101)
        )
        assert isinstance(
            default_numerical_convolution_base._sample_components, ComponentCollection
        )
        assert isinstance(
            default_numerical_convolution_base._resolution_components,
            ComponentCollection,
        )
        assert default_numerical_convolution_base.upsample_factor == 5
        assert default_numerical_convolution_base.extension_factor == pytest.approx(0.2)
        assert default_numerical_convolution_base.temperature is None
        assert default_numerical_convolution_base.unit == 'meV'
        assert default_numerical_convolution_base.normalize_detailed_balance is True
        assert isinstance(default_numerical_convolution_base._energy_grid, EnergyGrid)

    def test_init_with_custom_parameters(self):
        """
        Test initialization of NumericalConvolutionBase with custom
        parameters.
        """
        # WHEN
        energy = np.linspace(-5, 5, 50)
        sample_components = ComponentCollection(display_name='ComponentCollection')
        resolution_components = ComponentCollection(display_name='ResolutionModel')
        resolution_settings = ConvolutionSettings(
            upsample_factor=10, extension_factor=0.5, normalize_detailed_balance=False
        )
        temperature = 300.0
        temperature_unit = 'K'
        unit = 'meV'

        # THEN
        numerical_convolution_base = NumericalConvolutionBase(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            convolution_settings=resolution_settings,
            temperature=temperature,
            temperature_unit=temperature_unit,
            unit=unit,
        )

        # EXPECT
        assert numerical_convolution_base.upsample_factor == 10
        assert numerical_convolution_base.extension_factor == pytest.approx(0.5)
        assert numerical_convolution_base.temperature.value == temperature
        assert numerical_convolution_base.temperature.unit == temperature_unit
        assert numerical_convolution_base.unit == unit
        assert numerical_convolution_base.normalize_detailed_balance is False
        assert isinstance(numerical_convolution_base._energy_grid, EnergyGrid)

    def test_init_raises_type_error_for_invalid_temperature(self):
        """
        Test that initialization raises TypeError for invalid
        temperature.
        """
        # WHEN
        energy = np.linspace(-5, 5, 50)
        sample_components = ComponentCollection(display_name='ComponentCollection')
        resolution_components = ComponentCollection(display_name='ResolutionModel')
        invalid_temperature = 'invalid_temperature'

        # THEN EXPECT
        with pytest.raises(TypeError, match=r'Temperature must be None, a number or a Parameter.'):
            NumericalConvolutionBase(
                energy=energy,
                sample_components=sample_components,
                resolution_components=resolution_components,
                temperature=invalid_temperature,
            )

    def test_init_raises_type_error_for_invalid_temperature_unit(self):
        """
        Test that initialization raises TypeError for invalid
        temperature_unit.
        """
        # WHEN
        energy = np.linspace(-5, 5, 50)
        sample_components = ComponentCollection(display_name='ComponentCollection')
        resolution_components = ComponentCollection(display_name='ResolutionModel')
        invalid_temperature_unit = 123  # Not a string or sc.Unit

        # THEN EXPECT
        with pytest.raises(TypeError, match=r'Temperature_unit must be a string or sc.Unit.'):
            NumericalConvolutionBase(
                energy=energy,
                sample_components=sample_components,
                resolution_components=resolution_components,
                temperature_unit=invalid_temperature_unit,
            )

    def test_energy_setter(self, default_numerical_convolution_base):
        """
        Test setting a new energy array updates the energy grid
        accordingly.
        """
        # WHEN
        new_energy = np.linspace(-20, 20, 201)

        # THEN
        default_numerical_convolution_base.energy = new_energy

        # EXPECT
        assert isinstance(default_numerical_convolution_base.energy, sc.Variable)
        assert np.allclose(default_numerical_convolution_base.energy.values, new_energy)

        # EXPECT: plan invalidated
        assert (
            default_numerical_convolution_base.convolution_settings.convolution_plan_is_valid
            is False
        )

        # THEN
        # Force regeneration of energy grid
        default_numerical_convolution_base._create_energy_grid()

        # EXPECT
        assert default_numerical_convolution_base._energy_grid.energy_dense.shape[0] == round(
            201 * default_numerical_convolution_base.upsample_factor
        )

    @pytest.mark.parametrize(
        'new_upsample_factor, expected_size',
        [
            (10, (101 * 10)),
            (None, 101),
        ],
        ids=['upsample_10', 'no_upsampling'],
    )
    def test_upsample_factor_setter(
        self,
        default_numerical_convolution_base,
        new_upsample_factor,
        expected_size,
    ):
        """
        Test setting upsample factor updates the energy grid correctly,
        including disabling upsampling when set to None.
        """
        # WHEN
        default_numerical_convolution_base.upsample_factor = new_upsample_factor

        # EXPECT: plan invalidated
        assert (
            default_numerical_convolution_base.convolution_settings.convolution_plan_is_valid
            is False
        )

        # Force regeneration of energy grid
        default_numerical_convolution_base._create_energy_grid()

        # EXPECT: correct factor + grid size
        assert default_numerical_convolution_base.upsample_factor == new_upsample_factor
        assert (
            default_numerical_convolution_base._energy_grid.energy_dense.shape[0] == expected_size
        )

    @pytest.mark.parametrize(
        'invalid_upsample_factor, expected_exception',
        [
            (-1, ValueError),  # numeric < 1 → ValueError
            (0, ValueError),  # numeric < 1 → ValueError
            (1.0, ValueError),  # numeric = 1 → ValueError
            ('invalid', TypeError),  # non-numeric → TypeError
        ],
        ids=['negative', 'zero', 'one', 'string'],
    )
    def test_upsample_setter_raises(
        self,
        default_numerical_convolution_base,
        invalid_upsample_factor,
        expected_exception,
    ):
        """
        Test that setting invalid upsample factors raises appropriate
        exceptions.
        """
        # WHEN THEN EXPECT
        with pytest.raises(
            expected_exception,
        ):
            default_numerical_convolution_base.upsample_factor = invalid_upsample_factor

    def test_extension_factor_setter(self, default_numerical_convolution_base):
        """
        Test setting a new extension factor updates the energy grid
        accordingly.
        """
        # WHEN
        new_extension_factor = 0.5

        # THEN
        default_numerical_convolution_base.extension_factor = new_extension_factor

        # EXPECT: plan invalidated
        assert (
            default_numerical_convolution_base.convolution_settings.convolution_plan_is_valid
            is False
        )

        # THEN
        # Force regeneration of energy grid
        default_numerical_convolution_base._create_energy_grid()

        # EXPECT
        assert default_numerical_convolution_base.extension_factor == new_extension_factor
        expected_span = 20 + (0.5 * 20)  # original span + extension
        assert np.isclose(
            default_numerical_convolution_base._energy_grid.energy_span_dense,
            expected_span,
        )

    @pytest.mark.parametrize(
        'invalid_extension_factor, expected_exception',
        [
            (-0.1, ValueError),  # negative → ValueError
            ('invalid', TypeError),  # non-numeric → TypeError
        ],
        ids=['negative', 'string'],
    )
    def test_extension_factor_setter_raises(
        self,
        default_numerical_convolution_base,
        invalid_extension_factor,
        expected_exception,
    ):
        """
        Test that setting invalid extension factors raises appropriate
        exceptions.
        """

        # WHEN THEN EXPECT
        with pytest.raises(
            expected_exception,
        ):
            default_numerical_convolution_base.extension_factor = invalid_extension_factor

    @pytest.mark.parametrize(
        'temperature_input, expected_value',
        [
            (1, 1.0),
            (100.0, 100.0),
            (Parameter(name='TempParam', value=250.0, unit='K'), 250.0),
        ],
        ids=['int', 'float', 'parameter'],
    )
    def test_temperature_setter(
        self, default_numerical_convolution_base, temperature_input, expected_value
    ):
        """
        Test setting various valid temperature inputs.
        """
        # WHEN
        default_numerical_convolution_base.temperature = temperature_input

        # THEN EXPECT
        assert default_numerical_convolution_base.temperature.value == expected_value
        assert default_numerical_convolution_base.temperature.unit == 'K'

    def test_temperature_setter_none(self, default_numerical_convolution_base):
        """
        Test setting temperature to None.
        """
        # WHEN
        default_numerical_convolution_base.temperature = None

        # THEN EXPECT
        assert default_numerical_convolution_base.temperature is None

    def test_temperature_setter_does_not_replace_parameter(
        self, default_numerical_convolution_base
    ):
        """
        Test that if setting the temperature to a value when it already
        exists does not create a new Parameter.
        """
        # WHEN
        temp_param = Parameter(name='TempParam', value=300.0, unit='K')
        default_numerical_convolution_base.temperature = temp_param

        # THEN
        default_numerical_convolution_base.temperature = 350.0

        # EXPECT
        assert default_numerical_convolution_base.temperature is temp_param
        assert default_numerical_convolution_base.temperature.value == pytest.approx(350.0)

    def test_temperature_setter_raises(self, default_numerical_convolution_base):
        """
        Test that setting an invalid temperature raises TypeError.
        """
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Temperature must be'):
            default_numerical_convolution_base.temperature = 'invalid_temperature'

    def test_normalize_detailed_balance_setter(self, default_numerical_convolution_base):
        """
        Test setting normalize_detailed_balance to False.
        """
        # WHEN
        default_numerical_convolution_base.normalize_detailed_balance = False

        # THEN EXPECT
        assert default_numerical_convolution_base.normalize_detailed_balance is False

    def test_normalize_detailed_balance_setter_raises(self, default_numerical_convolution_base):
        """
        Test that setting an invalid normalize_detailed_balance raises
        TypeError.
        """
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='normalize_detailed_balance must be'):
            default_numerical_convolution_base.normalize_detailed_balance = 'invalid'

    def test_create_energy_grid_upsample_none(self, default_numerical_convolution_base):
        """
        Test creating energy grid with upsample_factor set to None (no
        upsampling). In this case, the energy grid is equal to the input
        energy.
        """
        # WHEN
        default_numerical_convolution_base.upsample_factor = None
        energy_grid = default_numerical_convolution_base._create_energy_grid()

        # THEN EXPECT
        assert isinstance(energy_grid, EnergyGrid)
        assert np.allclose(
            energy_grid.energy_dense,
            np.linspace(-10, 10, 101),  # no extension
        )
        assert np.allclose(energy_grid.energy_dense_centered, np.linspace(-10, 10, 101))
        assert np.isclose(energy_grid.energy_dense_step, 0.2)
        assert np.isclose(energy_grid.energy_span_dense, 20.0)
        assert np.isclose(energy_grid.energy_even_length_offset, 0.0)

    def test_create_energy_grid_upsample_none_non_uniform_raises(
        self, default_numerical_convolution_base
    ):
        """
        Test that creating energy grid with upsample_factor set to None
        and non-uniform energy raises ValueError (the energy grid must
        always be uniform).
        """
        # WHEN
        default_numerical_convolution_base.energy = np.array([0, 1, 3, 6, 10])
        default_numerical_convolution_base.upsample_factor = None
        with pytest.raises(
            ValueError,
            match='Input array `energy` must be uniformly spaced if upsample_factor is not given',
        ):
            default_numerical_convolution_base._create_energy_grid()

    @pytest.mark.parametrize('num_points', [100, 101], ids=['even', 'odd'])
    def test_create_energy_grid_upsample_and_extension(
        self, default_numerical_convolution_base, num_points
    ):
        """
        Test creating energy grid with upsampling and extension. The
        even_length_offset is tested for both even and odd number of
        input energy points.
        """
        # WHEN
        default_numerical_convolution_base.energy = np.linspace(-10, 10, num_points)

        # THEN
        energy_grid = default_numerical_convolution_base._create_energy_grid()

        # EXPECT
        assert isinstance(energy_grid, EnergyGrid)

        expected_extended_energy = np.linspace(
            -12,
            12,
            round(num_points * default_numerical_convolution_base.upsample_factor),
        )
        assert np.allclose(energy_grid.energy_dense, expected_extended_energy)

        expected_centered_energy = expected_extended_energy
        assert np.allclose(energy_grid.energy_dense_centered, expected_centered_energy)

        expected_step = expected_extended_energy[1] - expected_extended_energy[0]
        assert np.isclose(energy_grid.energy_dense_step, expected_step)

        expected_span = 24.0  # original span + extension
        assert np.isclose(energy_grid.energy_span_dense, expected_span)

        if num_points % 2 == 0:
            expected_offset = -expected_step / 2
            assert np.isclose(energy_grid.energy_even_length_offset, expected_offset)
        else:
            assert np.isclose(energy_grid.energy_even_length_offset, 0.0)

    def test_create_energy_grid_non_centered_energy(self, default_numerical_convolution_base):
        """
        Test creating energy grid when input energy is not centered
        around zero. The centered energy grid should be shifted
        accordingly.
        """
        # WHEN
        default_numerical_convolution_base.energy = np.linspace(5, 25, 101)
        energy_grid = default_numerical_convolution_base._create_energy_grid()

        # THEN EXPECT
        assert isinstance(energy_grid, EnergyGrid)

        expected_extended_energy = np.linspace(
            3,
            27,
            round(101 * default_numerical_convolution_base.upsample_factor),
        )
        assert np.allclose(energy_grid.energy_dense, expected_extended_energy)

        expected_centered_energy = expected_extended_energy - 15.0
        assert np.allclose(energy_grid.energy_dense_centered, expected_centered_energy)

        expected_step = expected_extended_energy[1] - expected_extended_energy[0]
        assert np.isclose(energy_grid.energy_dense_step, expected_step)

        expected_span = 24.0  # original span + extension
        assert np.isclose(energy_grid.energy_span_dense, expected_span)

        assert np.isclose(energy_grid.energy_even_length_offset, 0.0)

    def test_check_width_large_threshold(self, default_numerical_convolution_base):
        """
        Test that _check_width_thresholds warns when model widths are
        too large compared to energy grid span.
        """
        # WHEN
        wide_gaussian = Gaussian(
            display_name='ComponentCollection', area=1.0, center=0.0, width=15.0
        )

        # THEN EXPECT
        with pytest.warns(
            UserWarning,
            match='Increase extension_factor to improve',
        ):
            default_numerical_convolution_base._check_width_thresholds(
                model=wide_gaussian,
                model_name='ComponentCollection',
            )

    def test_check_width_small_threshold(self, default_numerical_convolution_base):
        """
        Test that _check_width_thresholds warns when model widths are
        too small compared to energy grid step.
        """
        # WHEN
        narrow_gaussian = Gaussian(
            display_name='ComponentCollection', area=1.0, center=0.0, width=0.000001
        )

        # THEN EXPECT
        with pytest.warns(
            UserWarning,
            match='Increase upsample_factor to improve',
        ):
            default_numerical_convolution_base._check_width_thresholds(
                model=narrow_gaussian,
                model_name='ComponentCollection',
            )

    def test_check_width_no_warnings(self, default_numerical_convolution_base):
        """
        Test that _check_width_thresholds does not warn when model
        widths are within acceptable range. Also tests that
        ComponentCollection components are checked correctly.
        """
        # WHEN
        good_gaussian = Gaussian(
            display_name='ComponentCollection', area=1.0, center=0.0, width=1.0
        )
        sample_components = ComponentCollection(display_name='ComponentCollection')
        sample_components.append_component(good_gaussian)

        # THEN EXPECT
        default_numerical_convolution_base._check_width_thresholds(
            model=sample_components,
            model_name='ComponentCollection',
        )

    def test_repr(self, default_numerical_convolution_base):
        """
        Test the __repr__ method of NumericalConvolutionBase.
        """
        # WHEN
        repr_str = repr(default_numerical_convolution_base)

        # THEN EXPECT
        assert 'NumericalConvolutionBase(' in repr_str
        assert 'energy=array of shape' in repr_str
        assert '(101,' in repr_str  # correct shape

        # Sample and resolution models
        assert 'ComponentCollection' in repr_str
        assert 'Components: No components' in repr_str
        assert 'sample_components=' in repr_str
        assert 'resolution_components=' in repr_str

        # Important parameters
        assert 'unit=meV' in repr_str
        assert 'upsample_factor=5' in repr_str
        assert 'extension_factor=0.2' in repr_str
        assert 'temperature=None' in repr_str
        assert 'normalize_detailed_balance=True' in repr_str
