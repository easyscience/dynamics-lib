# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.convolution.analytical_convolution import AnalyticalConvolution
from easydynamics.convolution.convolution import Convolution
from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.convolution.numerical_convolution import NumericalConvolution
from easydynamics.sample_model import DampedHarmonicOscillator
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model import Voigt
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings


class TestConvolution:
    @pytest.fixture
    def default_convolution(self):
        energy = np.linspace(-10, 10, 5001)
        sample_components = ComponentCollection(display_name='ComponentCollection')

        sample_components.append_component(
            Gaussian(name='Gaussian1', area=2.0, center=0.1, width=0.4)
        )

        sample_components.append_component(
            DampedHarmonicOscillator(name='DHO1', area=2.0, center=1.0, width=0.1)
        )

        sample_components.append_component(DeltaFunction(name='Delta1', area=2.0, center=0.3))

        resolution_components = ComponentCollection(name='ResolutionModel')
        resolution_components.append_component(
            Gaussian(name='GaussianRes', area=3.0, center=0.2, width=0.5)
        )

        return Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

    @pytest.fixture
    def convolution_with_components(self):
        energy = np.linspace(-10, 10, 5001)
        sample_components = Gaussian(name='Gaussian1', area=2.0, center=0.1, width=0.4)

        resolution_components = Gaussian(name='GaussianRes', area=3.0, center=0.2, width=0.5)

        return Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

    def test_init(self, default_convolution):
        "Test initialization of Convolution with default parameters."
        # WHEN THEN EXPECT
        assert isinstance(default_convolution, Convolution)
        assert isinstance(default_convolution.energy, sc.Variable)
        assert np.allclose(default_convolution.energy.values, np.linspace(-10, 10, 5001))
        assert isinstance(default_convolution._sample_components, ComponentCollection)
        assert isinstance(default_convolution._resolution_components, ComponentCollection)
        assert default_convolution.upsample_factor == 5
        assert default_convolution.extension_factor == pytest.approx(0.2)
        assert default_convolution.temperature is None
        assert default_convolution.x_unit == 'meV'
        assert default_convolution.y_unit == 'dimensionless'
        assert default_convolution.detailed_balance_settings.normalize_detailed_balance is True
        assert isinstance(default_convolution._energy_grid, EnergyGrid)

        assert isinstance(default_convolution._analytical_sample_components, ComponentCollection)
        assert (
            default_convolution._analytical_sample_components[0]
            is default_convolution.sample_components[0]
        )
        assert isinstance(default_convolution._numerical_sample_components, ComponentCollection)
        assert (
            default_convolution._numerical_sample_components[0]
            is default_convolution.sample_components[1]
        )

        assert isinstance(default_convolution._delta_sample_components, ComponentCollection)
        assert (
            default_convolution._delta_sample_components[0]
            is default_convolution.sample_components[2]
        )
        assert default_convolution._convolution_plan_is_current() is True
        assert default_convolution._reactions_enabled is True

    def test_init_components(self, convolution_with_components):
        "Test initialization of Convolution with default parameters."
        # WHEN THEN EXPECT
        assert isinstance(convolution_with_components, Convolution)
        assert isinstance(convolution_with_components.energy, sc.Variable)
        assert np.allclose(convolution_with_components.energy.values, np.linspace(-10, 10, 5001))
        assert isinstance(convolution_with_components._sample_components, ComponentCollection)
        assert isinstance(convolution_with_components._resolution_components, ComponentCollection)
        assert convolution_with_components.upsample_factor == 5
        assert convolution_with_components.extension_factor == pytest.approx(0.2)
        assert convolution_with_components.temperature is None
        assert convolution_with_components.x_unit == 'meV'
        assert convolution_with_components.y_unit == 'dimensionless'
        assert (
            convolution_with_components.detailed_balance_settings.normalize_detailed_balance
            is True
        )
        assert isinstance(convolution_with_components._energy_grid, EnergyGrid)

        assert isinstance(
            convolution_with_components._analytical_sample_components,
            ComponentCollection,
        )
        assert (
            convolution_with_components._analytical_sample_components[0]
            is convolution_with_components.sample_components[0]
        )
        assert isinstance(
            convolution_with_components._numerical_sample_components,
            ComponentCollection,
        )
        assert convolution_with_components._numerical_sample_components.is_empty

        assert isinstance(
            convolution_with_components._delta_sample_components, ComponentCollection
        )
        assert convolution_with_components._delta_sample_components.is_empty
        assert convolution_with_components._convolution_plan_is_current() is True
        assert convolution_with_components._reactions_enabled is True

    def test_convolution_plan_collections_inherit_units(self):
        "Regression: internal plan collections must carry the Convolution's units"
        # WHEN
        convolution = Convolution(
            energy=np.linspace(-10, 10, 101),
            sample_components=Gaussian(
                name='Gaussian1', area=2.0, center=0.1, width=0.4, y_unit='counts'
            ),
            resolution_components=Gaussian(name='GaussianRes', area=3.0, center=0.2, width=0.5),
            y_unit='counts',
        )

        # THEN EXPECT
        for collection in (
            convolution._analytical_sample_components,
            convolution._delta_sample_components,
            convolution._numerical_sample_components,
        ):
            assert collection.x_unit == 'meV'
            assert collection.y_unit == 'counts'

        # Wrapping a bare ModelComponent must mirror the component's own units
        assert convolution._sample_components.y_unit == 'counts'
        assert convolution._resolution_components.y_unit == 'dimensionless'

    def test_convolution_plan_is_built_when_invalid(self, default_convolution):
        """
        Test that convolution plan is built when invalid.
        """
        # WHEN
        conv = default_convolution
        conv._plan_seen_version = None

        # THEN EXPECT
        with patch.object(conv, '_build_convolution_plan') as build_plan:
            conv.convolution()
            build_plan.assert_called_once()

    def test_convolution_calls_analytical_convolver(self, default_convolution):
        """
        Test that convolution calls analytical convolver when analytical
        components are present.
        """
        # WHEN
        conv = default_convolution

        # THEN EXPECT
        with patch.object(
            conv._analytical_convolver, 'convolution', return_value=np.array([1.0])
        ) as analytical_conv:
            conv.convolution()
            analytical_conv.assert_called_once()

    def test_convolution_calls_numerical_convolver(self, default_convolution):
        """
        Test that convolution calls numerical convolver when numerical
        components are present.
        """
        # WHEN
        conv = default_convolution

        # THEN EXPECT
        with patch.object(
            conv._numerical_convolver, 'convolution', return_value=np.array([1.0])
        ) as numerical_conv:
            conv.convolution()
            numerical_conv.assert_called_once()

    def test_convolution_calls_convolve_delta_functions(self, default_convolution):
        """
        Test that convolution calls _convolve_delta_functions when delta
        components are present.
        """
        # WHEN
        conv = default_convolution

        # THEN EXPECT
        with patch.object(
            conv,
            '_convolve_delta_functions',
            return_value=np.array([1.0]),
        ) as delta_eval:
            conv.convolution()
            delta_eval.assert_called_once()

    @pytest.mark.parametrize(
        'analytical_component',
        [True, False],
        ids=['with_analytical', 'without_analytical'],
    )
    @pytest.mark.parametrize(
        'numerical_component',
        [True, False],
        ids=['with_numerical', 'without_numerical'],
    )
    @pytest.mark.parametrize('delta_component', [True, False], ids=['with_delta', 'without_delta'])
    def test_convolution_calls_correct_methods(
        self,
        default_convolution,
        analytical_component,
        numerical_component,
        delta_component,
    ):
        """
        Tests that convolution calls the correct methods depending on
        which component types are present.
        """

        # WHEN
        conv = default_convolution
        sample_components = ComponentCollection()

        if analytical_component:
            sample_components.append_component(
                Gaussian(name='Gaussian', area=1.0, center=0.0, width=0.1)
            )

        if numerical_component:
            sample_components.append_component(
                DampedHarmonicOscillator(
                    display_name='DampedHarmonicOscillator',
                    area=1.0,
                    center=1.0,
                    width=0.1,
                )
            )

        if delta_component:
            sample_components.append_component(
                DeltaFunction(display_name='DeltaFunction', area=1.0, center=0.0)
            )

        conv.sample_components = sample_components  # This updates the internal sample models
        conv._build_convolution_plan()  # Ensure the plan is built with the new components

        # THEN
        # Mock the methods to be tested. Use nullcontext if the
        # component type is not present.
        if analytical_component:
            patch_analytical = patch.object(
                conv._analytical_convolver, 'convolution', return_value=np.array([1.0])
            )
        else:
            patch_analytical = nullcontext()

        if numerical_component:
            patch_numerical = patch.object(
                conv._numerical_convolver, 'convolution', return_value=np.array([1.0])
            )
        else:
            patch_numerical = nullcontext()

        patch_delta = patch.object(
            conv,
            '_convolve_delta_functions',
            return_value=np.array([1.0]),
        )

        # EXPECT
        # Each method is called only if the corresponding component
        # type is present.
        with (
            patch_analytical as mock_analytical_method,
            patch_numerical as mock_numerical_method,
            patch_delta as mock_delta_method,
        ):
            # The plan was built above; keep it current so convolution() does not rebuild
            # the convolvers and discard the mocks.
            conv._mark_convolution_plan_current()
            conv.convolution()

            if analytical_component:
                mock_analytical_method.assert_called_once()
            else:
                assert conv._analytical_convolver is None

            if numerical_component:
                mock_numerical_method.assert_called_once()
            else:
                assert conv._numerical_convolver is None

            if delta_component:
                mock_delta_method.assert_called_once()
            else:
                mock_delta_method.assert_not_called()

    def test_convolve_delta_functions(self, default_convolution):
        """
        Test that _convolve_delta_functions returns expected values.
        """
        # WHEN
        conv = default_convolution

        # THEN
        result = conv._convolve_delta_functions()

        # EXPECT
        expected_area = 2.0 * 3.0  # Delta area * Resolution area
        expected_center = 0.3 + 0.2  # Delta center + Resolution center
        expected_width = 0.5  # Resolution width
        expected_values = Gaussian(
            name='ExpectedGaussian',
            area=expected_area,
            center=expected_center,
            width=expected_width,
        ).evaluate(conv.energy.values)

        assert np.allclose(result, expected_values)

    # List of analytic functions
    analytic_functions: ClassVar[list[object]] = [
        Gaussian(name='G', area=1.0, center=0.0, width=0.1),
        Lorentzian(display_name='L', area=1.0, center=0.0, width=0.1),
        Voigt(
            display_name='V',
            area=1.0,
            center=0.0,
            gaussian_width=0.1,
            lorentzian_width=0.1,
        ),
    ]

    # List of non-analytic functions
    non_analytic_functions: ClassVar[list[object]] = [
        DampedHarmonicOscillator(display_name='DHO', area=1.0, center=1.0, width=0.1),
        Polynomial(display_name='P', coefficients=[1.0, 0.0, 0.0]),
    ]

    all_functions_except_delta: ClassVar[list[object]] = [
        *analytic_functions,
        *non_analytic_functions,
    ]

    all_functions: ClassVar[list[object]] = [
        *all_functions_except_delta,
        DeltaFunction(display_name='Delta', area=1.0, center=0.0),
    ]

    @pytest.mark.parametrize('function1', all_functions, ids=lambda f: f.__class__.__name__)
    @pytest.mark.parametrize(
        'function2', all_functions_except_delta, ids=lambda f: f.__class__.__name__
    )
    def test_check_if_pair_is_analytic(self, default_convolution, function1, function2):
        """
        Test _check_if_pair_is_analytic for all combinations.
        Analytic functions combined → True, otherwise False.
        """
        # WHEN
        conv = default_convolution

        # THEN
        # skip delta function in resolution
        result = conv._check_if_pair_is_analytic(
            sample_component=function1, resolution_component=function2
        )

        # EXPECT
        # Determine expected result
        is_analytic1 = isinstance(function1, (Gaussian, Lorentzian, Voigt))
        is_analytic2 = isinstance(function2, (Gaussian, Lorentzian, Voigt))
        expected = is_analytic1 and is_analytic2
        assert result == expected

    def test_check_if_pair_is_analytic_raises_with_delta_in_resolution(self, default_convolution):
        """
        Test that _check_if_pair_is_analytic raises ValueError when
        resolution component is DeltaFunction.
        """
        # WHEN
        conv = default_convolution
        sample_component = Gaussian(name='G', area=1.0, center=0.0, width=0.1)
        resolution_component = DeltaFunction(display_name='Delta', area=1.0, center=0.0)

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='This is not supported',
        ):
            conv._check_if_pair_is_analytic(
                sample_component=sample_component,
                resolution_component=resolution_component,
            )

    @pytest.mark.parametrize(
        'analytical_component',
        [True, False],
        ids=['with_analytical', 'without_analytical'],
    )
    @pytest.mark.parametrize(
        'numerical_component',
        [True, False],
        ids=['with_numerical', 'without_numerical'],
    )
    @pytest.mark.parametrize('delta_component', [True, False], ids=['with_delta', 'without_delta'])
    @pytest.mark.parametrize(
        'temperature', [None, 100], ids=['without_temperature', 'with_temperature']
    )
    def test_build_convolution_plan(
        self,
        default_convolution,
        analytical_component,
        numerical_component,
        delta_component,
        temperature,
    ):
        """
        Tests that convolution calls the correct methods depending on
        which component types are present.
        """

        # WHEN
        conv = default_convolution
        sample_components = ComponentCollection()

        if analytical_component:
            sample_components.append_component(
                Gaussian(name='Gaussian', area=1.0, center=0.0, width=0.1)
            )

        if numerical_component:
            sample_components.append_component(
                DampedHarmonicOscillator(
                    name='DampedHarmonicOscillator',
                    area=1.0,
                    center=1.0,
                    width=0.1,
                )
            )

        if delta_component:
            sample_components.append_component(
                DeltaFunction(name='DeltaFunction', area=1.0, center=0.0)
            )

        # THEN
        conv.sample_components = sample_components  # This updates the internal sample models
        if temperature is not None:
            conv.temperature = temperature
        conv._build_convolution_plan()

        # EXPECT
        assert isinstance(conv._analytical_sample_components, ComponentCollection)
        if analytical_component and not temperature:
            assert len(conv._analytical_sample_components) == 1
            assert conv._analytical_sample_components[0].name == 'Gaussian'
        else:
            assert len(conv._analytical_sample_components) == 0

        assert isinstance(conv._delta_sample_components, ComponentCollection)
        if delta_component:
            assert len(conv._delta_sample_components) == 1
            assert conv._delta_sample_components[0].name == 'DeltaFunction'
        else:
            assert len(conv._delta_sample_components) == 0

        assert isinstance(conv._numerical_sample_components, ComponentCollection)

        if not temperature:
            if numerical_component:
                assert len(conv._numerical_sample_components) == 1
                assert conv._numerical_sample_components[0].name == 'DampedHarmonicOscillator'
            else:
                assert len(conv._numerical_sample_components) == 0
        else:
            # analytical and numerical components go to numerical when
            # temperature is set
            expected_numerical_count = 0
            if numerical_component:
                expected_numerical_count += 1
            if analytical_component:
                expected_numerical_count += 1
            assert len(conv._numerical_sample_components) == expected_numerical_count

        assert conv._convolution_plan_is_current() is True

    @pytest.mark.parametrize(
        'analytical_component',
        [True, False],
        ids=['with_analytical', 'without_analytical'],
    )
    @pytest.mark.parametrize(
        'numerical_component',
        [True, False],
        ids=['with_numerical', 'without_numerical'],
    )
    def test_set_convolvers(
        self,
        default_convolution,
        analytical_component,
        numerical_component,
    ):
        """
        Tests that convolution sets the correct methods depending on
        which component types are present.
        """

        # WHEN
        conv = default_convolution
        sample_components = ComponentCollection()

        if analytical_component:
            sample_components.append_component(
                Gaussian(name='Gaussian', area=1.0, center=0.0, width=0.1)
            )

        if numerical_component:
            sample_components.append_component(
                DampedHarmonicOscillator(
                    display_name='DampedHarmonicOscillator',
                    area=1.0,
                    center=1.0,
                    width=0.1,
                )
            )

        # THEN
        conv.sample_components = sample_components
        # Should already have been called by _build_convolution_plan,
        # but we now call it explicitly
        conv._build_convolution_plan()  # Ensure the plan is built with the new components
        conv._set_convolvers()

        # EXPECT
        if analytical_component:
            assert isinstance(conv._analytical_convolver, AnalyticalConvolution)
        else:
            assert conv._analytical_convolver is None

        if numerical_component:
            assert isinstance(conv._numerical_convolver, NumericalConvolution)
        else:
            assert conv._numerical_convolver is None

    def test_setattr_does_not_invalidate_plan_for_non_tracked_attribute(
        self,
        default_convolution,
    ):
        # WHEN
        conv = default_convolution
        assert conv._convolution_plan_is_current() is True

        # Capture current identity of internal state to ensure no rebuild
        old_plan_id = id(conv._analytical_sample_components)
        old_numerical_id = id(conv._numerical_sample_components)
        old_delta_id = id(conv._delta_sample_components)

        # THEN (NOT in _invalidate_plan_on_change)
        conv.display_name = 'new_name'

        # EXPECT
        assert conv._convolution_plan_is_current() is True
        assert id(conv._analytical_sample_components) == old_plan_id
        assert id(conv._numerical_sample_components) == old_numerical_id
        assert id(conv._delta_sample_components) == old_delta_id
        assert conv.display_name == 'new_name'

    def test_setattr_invalidates_plan_for_tracked_attribute(
        self,
        default_convolution,
    ):
        # WHEN
        conv = default_convolution
        assert conv._convolution_plan_is_current() is True

        # THEN
        conv.upsample_factor = 10

        # EXPECT
        assert conv._convolution_plan_is_current() is False

    def test_convert_y_unit_propagates_to_sub_convolvers(self):
        # WHEN: Convolution with Gaussian sample and resolution components,
        #       both with y_unit='1/meV'
        energy = np.linspace(-10, 10, 5001)
        sample_components = ComponentCollection()
        sample_components.append_component(
            Gaussian(name='G', area=1.0, center=0.0, width=0.5, x_unit='meV', y_unit='1/meV')
        )
        resolution_components = ComponentCollection()
        resolution_components.append_component(
            Gaussian(name='R', area=1.0, center=0.0, width=0.3, x_unit='meV', y_unit='1/meV')
        )
        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

        # THEN: convert y_unit to '1/eV'
        conv.convert_y_unit('1/eV')

        # EXPECT: y_unit updated and propagated to sub-convolvers via Convolution.convert_y_unit
        assert conv.y_unit == '1/eV'
        assert conv._analytical_convolver._y_unit == '1/eV'

    #############
    # Plan invalidation regressions
    #############

    def test_invalidate_plan_on_change_names_are_real_attributes(self, default_convolution):
        "Regression: the tracked-attribute set used to contain names that never exist"
        # WHEN THEN EXPECT every tracked name is an actual attribute of a built convolver
        for name in Convolution._invalidate_plan_on_change:
            assert hasattr(default_convolution, name), name

    def test_in_place_sample_append_contributes_to_convolution(self):
        "Regression: appending to the live sample collection used to leave output unchanged"
        # WHEN a convolver that has already produced output
        energy = np.linspace(-10, 10, 1001)
        sample_components = ComponentCollection(
            components=[Gaussian(name='G', area=2.0, center=0.1, width=0.4)]
        )
        resolution_components = ComponentCollection(
            components=[Gaussian(name='R', area=1.0, center=0.0, width=0.5)]
        )
        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )
        result_before = conv.convolution()

        # THEN mutating the live sample collection in place
        conv.sample_components.append_component(
            Lorentzian(name='L', area=1.0, center=0.0, width=0.3)
        )
        result_after = conv.convolution()

        # EXPECT the new component contributes, matching a freshly built convolver
        fresh = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )
        assert not np.allclose(result_after, result_before)
        np.testing.assert_allclose(result_after, fresh.convolution(), rtol=1e-10)

    def test_in_place_resolution_append_contributes_to_convolution(self):
        "Regression: appending to the live resolution collection used to leave output unchanged"
        # WHEN a convolver that has already produced output
        energy = np.linspace(-10, 10, 1001)
        sample_components = ComponentCollection(
            components=[Gaussian(name='G', area=2.0, center=0.1, width=0.4)]
        )
        resolution_components = ComponentCollection(
            components=[Gaussian(name='R', area=1.0, center=0.0, width=0.5)]
        )
        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )
        result_before = conv.convolution()

        # THEN mutating the live resolution collection in place
        conv.resolution_components.append_component(
            Gaussian(name='R2', area=0.5, center=0.0, width=0.2)
        )
        result_after = conv.convolution()

        # EXPECT the new component contributes, matching a freshly built convolver
        fresh = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )
        assert not np.allclose(result_after, result_before)
        np.testing.assert_allclose(result_after, fresh.convolution(), rtol=1e-10)

    def test_detailed_balance_toggle_changes_output(self):
        "Regression: toggling use_detailed_balance after construction used to be ignored"
        # WHEN a convolver built with detailed balance off
        energy = np.linspace(-10, 10, 1001)
        sample_components = ComponentCollection(
            components=[Lorentzian(name='L', area=2.0, center=0.0, width=0.4)]
        )
        resolution_components = ComponentCollection(
            components=[Gaussian(name='R', area=1.0, center=0.0, width=0.5)]
        )
        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            temperature=300.0,
            detailed_balance_settings=DetailedBalanceSettings(use_detailed_balance=False),
        )
        result_off = conv.convolution()

        # THEN toggling detailed balance on after construction
        conv.detailed_balance_settings.use_detailed_balance = True
        result_on = conv.convolution()

        # EXPECT the output changes and matches a convolver built with detailed balance on
        fresh = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            temperature=300.0,
            detailed_balance_settings=DetailedBalanceSettings(use_detailed_balance=True),
        )
        assert not np.allclose(result_on, result_off)
        np.testing.assert_allclose(result_on, fresh.convolution(), rtol=1e-10)

    def test_energy_offset_rebind_reaches_sub_convolvers(self):
        "Regression: rebinding energy_offset used to leave sub-convolvers on the old Parameter"
        # WHEN a convolver with analytical, numerical and delta components and offset 0
        energy = np.linspace(-10, 10, 1001)
        sample_components = ComponentCollection(
            components=[
                Gaussian(name='G', area=2.0, center=0.1, width=0.4),
                DampedHarmonicOscillator(name='DHO', area=2.0, center=1.0, width=0.1),
                DeltaFunction(name='D', area=1.0, center=0.3),
            ]
        )
        resolution_components = ComponentCollection(
            components=[Gaussian(name='R', area=1.0, center=0.0, width=0.5)]
        )
        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )
        result_before = conv.convolution()

        # THEN rebinding the offset to a brand-new Parameter
        conv.energy_offset = Parameter(name='energy_offset', value=1.0, unit='meV')
        result_after = conv.convolution()

        # EXPECT every path (analytical, numerical, delta) sees the new offset, matching a
        # freshly built convolver
        fresh = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            energy_offset=1.0,
        )
        assert not np.allclose(result_after, result_before)
        np.testing.assert_allclose(result_after, fresh.convolution(), rtol=1e-10)

    #############
    # Dispatch and validation regressions
    #############

    def test_subclass_of_analytical_component_convolves_like_base(self):
        "Regression: a Lorentzian subclass was routed analytically but rejected by dispatch"

        # WHEN a subclass of Lorentzian in the sample model
        class MyLorentzian(Lorentzian):
            pass

        energy = np.linspace(-10, 10, 1001)
        conv = Convolution(
            energy=energy,
            sample_components=MyLorentzian(name='MyL', area=2.0, center=0.1, width=0.4),
            resolution_components=Gaussian(name='R', area=1.0, center=0.0, width=0.5),
        )

        # THEN it is routed to the analytical convolver and convolved with the base rules
        result = conv.convolution()

        # EXPECT
        assert len(conv._analytical_sample_components) == 1
        reference = Convolution(
            energy=energy,
            sample_components=Lorentzian(name='L', area=2.0, center=0.1, width=0.4),
            resolution_components=Gaussian(name='R2', area=1.0, center=0.0, width=0.5),
        )
        np.testing.assert_allclose(result, reference.convolution(), rtol=1e-10)

    def test_empty_resolution_raises(self):
        "Regression: an empty resolution used to silently produce zeros"
        # WHEN THEN EXPECT at construction
        with pytest.raises(ValueError, match=r'resolution_components is empty'):
            Convolution(
                energy=np.linspace(-10, 10, 101),
                sample_components=Gaussian(name='G', area=1.0, center=0.0, width=0.4),
                resolution_components=ComponentCollection(),
            )

    def test_emptying_resolution_in_place_raises_on_next_convolution(self, default_convolution):
        # WHEN the live resolution collection is emptied after construction
        conv = default_convolution
        conv.resolution_components.pop('GaussianRes')

        # THEN EXPECT the next convolution rebuilds the plan and refuses to silently
        # return zeros
        with pytest.raises(ValueError, match=r'resolution_components is empty'):
            conv.convolution()

    #############
    # Registry and label housekeeping
    #############

    def test_plan_rebuilds_do_not_leak_registry_entries(self, default_convolution):
        "Regression: every plan rebuild used to register new objects in the global map forever"
        # WHEN a convolver that has built its plan at least once
        conv = default_convolution
        conv.convolution()
        vertices_before = len(conv._global_object.map.vertices())

        # THEN forcing several full plan rebuilds
        for _ in range(3):
            conv._plan_seen_version = None
            conv.convolution()

        # EXPECT the global map did not grow
        assert len(conv._global_object.map.vertices()) == vertices_before

    def test_convert_y_unit_updates_plan_collection_labels(self):
        "Regression: plan-collection y_unit labels used to stay stale until the next rebuild"
        # WHEN a convolver with analytical, numerical and delta components in 1/meV
        energy = np.linspace(-10, 10, 1001)
        sample_components = ComponentCollection(
            components=[
                Gaussian(name='G', area=1.0, center=0.0, width=0.4, y_unit='1/meV'),
                DampedHarmonicOscillator(
                    name='DHO', area=1.0, center=1.0, width=0.1, y_unit='1/meV'
                ),
                DeltaFunction(name='D', area=1.0, center=0.0, y_unit='1/meV'),
            ],
            y_unit='1/meV',
        )
        resolution_components = ComponentCollection(
            components=[Gaussian(name='R', area=1.0, center=0.0, width=0.5)]
        )
        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
            y_unit='1/meV',
        )

        # THEN
        conv.convert_y_unit('1/eV')

        # EXPECT the plan collections' labels follow without waiting for a rebuild
        assert conv._analytical_sample_components.y_unit == '1/eV'
        assert conv._numerical_sample_components.y_unit == '1/eV'
        assert conv._delta_sample_components.y_unit == '1/eV'

    def test_convert_y_unit_propagates_to_numerical_convolver(self):
        # WHEN: a DHO sample component forces a numerical convolver
        energy = np.linspace(-10, 10, 5001)
        sample_components = ComponentCollection()
        sample_components.append_component(
            DampedHarmonicOscillator(
                name='DHO', area=1.0, center=1.0, width=0.5, x_unit='meV', y_unit='1/meV'
            )
        )
        resolution_components = ComponentCollection()
        resolution_components.append_component(
            Gaussian(name='R', area=1.0, center=0.0, width=0.3, x_unit='meV', y_unit='1/meV')
        )
        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

        # THEN
        conv.convert_y_unit('1/eV')

        # EXPECT
        assert conv.y_unit == '1/eV'
        assert conv._numerical_convolver._y_unit == '1/eV'
