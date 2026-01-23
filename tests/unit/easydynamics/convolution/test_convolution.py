from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc

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


class TestConvolution:
    @pytest.fixture
    def default_convolution(self):
        energy = np.linspace(-10, 10, 5001)
        sample_components = ComponentCollection(display_name='ComponentCollection')

        sample_components.append_component(
            Gaussian(display_name='Gaussian1', area=2.0, center=0.1, width=0.4)
        )

        sample_components.append_component(
            DampedHarmonicOscillator(display_name='DHO1', area=2.0, center=1.0, width=0.1)
        )

        sample_components.append_component(
            DeltaFunction(display_name='Delta1', area=2.0, center=0.3)
        )

        resolution_components = ComponentCollection(display_name='ResolutionModel')
        resolution_components.append_component(
            Gaussian(display_name='GaussianRes', area=3.0, center=0.2, width=0.5)
        )

        conv = Convolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )
        return conv

    def test_init(self, default_convolution):
        "Test initialization of Convolution with default parameters."
        # WHEN THEN EXPECT
        assert isinstance(default_convolution, Convolution)
        assert isinstance(default_convolution.energy, sc.Variable)
        assert np.allclose(default_convolution.energy.values, np.linspace(-10, 10, 5001))
        assert isinstance(default_convolution._sample_components, ComponentCollection)
        assert isinstance(default_convolution._resolution_components, ComponentCollection)
        assert default_convolution.upsample_factor == 5
        assert default_convolution.extension_factor == 0.2
        assert default_convolution.temperature is None
        assert default_convolution.energy_unit == 'meV'
        assert default_convolution.normalize_detailed_balance is True
        assert isinstance(default_convolution._energy_grid, EnergyGrid)

        assert isinstance(default_convolution._analytical_sample_components, ComponentCollection)
        assert (
            default_convolution._analytical_sample_components.components[0]
            is default_convolution.sample_components.components[0]
        )
        assert isinstance(default_convolution._numerical_sample_components, ComponentCollection)
        assert (
            default_convolution._numerical_sample_components.components[0]
            is default_convolution.sample_components.components[1]
        )

        assert isinstance(default_convolution._delta_sample_components, ComponentCollection)
        assert (
            default_convolution._delta_sample_components.components[0]
            is default_convolution.sample_components.components[2]
        )
        assert default_convolution._convolution_plan_is_valid is True
        assert default_convolution._reactions_enabled is True

    def test_convolution_plan_is_built_when_invalid(self, default_convolution):
        """
        Test that convolution plan is built when invalid.
        """
        # WHEN
        conv = default_convolution
        conv._convolution_plan_is_valid = False

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
                Gaussian(display_name='Gaussian', area=1.0, center=0.0, width=0.1)
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
            conv._convolution_plan_is_valid = True
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
            display_name='ExpectedGaussian',
            area=expected_area,
            center=expected_center,
            width=expected_width,
        ).evaluate(conv.energy.values)

        assert np.allclose(result, expected_values)

    # List of analytic functions
    analytic_functions = [
        Gaussian(display_name='G', area=1.0, center=0.0, width=0.1),
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
    non_analytic_functions = [
        DampedHarmonicOscillator(display_name='DHO', area=1.0, center=1.0, width=0.1),
        Polynomial(display_name='P', coefficients=[1.0, 0.0, 0.0]),
    ]

    all_functions_except_delta = analytic_functions + non_analytic_functions
    all_functions = all_functions_except_delta + [
        DeltaFunction(display_name='Delta', area=1.0, center=0.0)
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
        Test that _check_if_pair_is_analytic raises TypeError when
        resolution component is DeltaFunction.
        """
        # WHEN
        conv = default_convolution
        sample_component = Gaussian(display_name='G', area=1.0, center=0.0, width=0.1)
        resolution_component = DeltaFunction(display_name='Delta', area=1.0, center=0.0)

        # THEN EXPECT
        with pytest.raises(
            TypeError,
            match='This is not supported',
        ):
            conv._check_if_pair_is_analytic(
                sample_component=sample_component,
                resolution_component=resolution_component,
            )

    @pytest.mark.parametrize(
        'sample_component,resolution_component',
        [
            (
                'NotAModelComponent',
                Gaussian(display_name='G', area=1.0, center=0.0, width=0.1),
            ),
            (
                Gaussian(display_name='G', area=1.0, center=0.0, width=0.1),
                'NotAModelComponent',
            ),
        ],
        ids=['invalid_sample_component', 'invalid_resolution_component'],
    )
    def test_check_if_pair_is_analytic_raises_with_invalid_types(
        self, default_convolution, sample_component, resolution_component
    ):
        """
        Test that _check_if_pair_is_analytic raises TypeError when given
        invalid component types.
        """
        # WHEN
        conv = default_convolution

        # THEN EXPECT
        with pytest.raises(
            TypeError,
            match='must be a ModelComponent',
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
        'temperature', [None, 100], ids=['with_temperature', 'without_temperature']
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
                Gaussian(display_name='Gaussian', area=1.0, center=0.0, width=0.1)
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

        # THEN
        conv.sample_components = sample_components  # This updates the internal sample models
        if temperature is not None:
            conv.temperature = temperature
        # It is already called by sample_components setter, but we now
        # call it explicitly
        conv._build_convolution_plan()

        # EXPECT
        assert isinstance(conv._analytical_sample_components, ComponentCollection)
        if analytical_component and not temperature:
            assert len(conv._analytical_sample_components.components) == 1
            assert conv._analytical_sample_components.components[0].display_name == 'Gaussian'
        else:
            assert len(conv._analytical_sample_components.components) == 0

        assert isinstance(conv._delta_sample_components, ComponentCollection)
        if delta_component:
            assert len(conv._delta_sample_components.components) == 1
            assert conv._delta_sample_components.components[0].display_name == 'DeltaFunction'
        else:
            assert len(conv._delta_sample_components.components) == 0

        assert isinstance(conv._numerical_sample_components, ComponentCollection)

        if not temperature:
            if numerical_component:
                assert len(conv._numerical_sample_components.components) == 1
                assert (
                    conv._numerical_sample_components.components[0].display_name
                    == 'DampedHarmonicOscillator'
                )
            else:
                assert len(conv._numerical_sample_components.components) == 0
        else:
            # analytical and numerical components go to numerical when
            # temperature is set
            expected_numerical_count = 0
            if numerical_component:
                expected_numerical_count += 1
            if analytical_component:
                expected_numerical_count += 1
            assert len(conv._numerical_sample_components.components) == expected_numerical_count

        assert conv._convolution_plan_is_valid is True

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
                Gaussian(display_name='Gaussian', area=1.0, center=0.0, width=0.1)
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
        conv.sample_components = sample_components  # This updates the internal sample models
        # Should already have been called by sample_components setter,
        # but we now call it explicitly
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
