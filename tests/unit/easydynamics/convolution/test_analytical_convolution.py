# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from scipy.signal import fftconvolve

# from easyscience.variable import Parameter
from easydynamics.convolution.analytical_convolution import AnalyticalConvolution
from easydynamics.sample_model import DampedHarmonicOscillator
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Voigt
from easydynamics.sample_model.component_collection import ComponentCollection

NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE = 1e-5
NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE = 1e-6


@pytest.fixture
def function1(request):
    # request.param will be e.g. "gaussian1"
    return request.getfixturevalue(request.param)


@pytest.fixture
def function2(request):
    return request.getfixturevalue(request.param)


class TestAnalyticalConvolution:
    @pytest.fixture
    def default_analytical_convolution(self):
        # Energy needs to be odd to avoid issues with shifts in
        # fftconvolve.
        energy = np.linspace(-100, 100, 2**15 + 1)
        sample_components = ComponentCollection(display_name='ComponentCollection')
        resolution_components = ComponentCollection(display_name='ResolutionModel')

        return AnalyticalConvolution(
            energy=energy,
            sample_components=sample_components,
            resolution_components=resolution_components,
        )

    @pytest.fixture
    def gaussian1(self):
        return Gaussian(area=2.0, center=1.0, width=0.5)

    @pytest.fixture
    def gaussian2(self):
        return Gaussian(area=3.0, center=-1.0, width=0.4)

    @pytest.fixture
    def lorentzian1(self):
        return Lorentzian(area=2.0, center=1.0, width=0.55)

    @pytest.fixture
    def lorentzian2(self):
        return Lorentzian(area=3.0, center=-1.0, width=0.45)

    @pytest.fixture
    def voigt1(self):
        return Voigt(area=2.0, center=1.0, gaussian_width=0.33, lorentzian_width=0.28)

    @pytest.fixture
    def voigt2(self):
        return Voigt(area=3.0, center=-1.0, gaussian_width=0.36, lorentzian_width=0.42)

    @pytest.fixture
    def dho1(self):
        return DampedHarmonicOscillator(area=2.0, center=2.0, width=0.2)

    def test_init(self, default_analytical_convolution):
        # WHEN THEN EXPECT
        assert isinstance(default_analytical_convolution, AnalyticalConvolution)
        assert isinstance(default_analytical_convolution.energy, sc.Variable)
        assert np.allclose(
            default_analytical_convolution.energy.values,
            np.linspace(-100, 100, 2**15 + 1),
        )
        assert isinstance(default_analytical_convolution._sample_components, ComponentCollection)
        assert isinstance(
            default_analytical_convolution._resolution_components, ComponentCollection
        )

    def test_convolution(
        self,
        default_analytical_convolution,
        gaussian1,
        gaussian2,
        lorentzian1,
        lorentzian2,
    ):
        """
        Test that the convolute method calls _convolute_analytic_pair
        for all component pairs.
        """

        # WHEN
        sample_components = ComponentCollection(display_name='ComponentCollection')
        sample_components.append_component(gaussian1)
        sample_components.append_component(lorentzian1)

        resolution_components = ComponentCollection(display_name='ResolutionModel')
        resolution_components.append_component(gaussian2)
        resolution_components.append_component(lorentzian2)
        default_analytical_convolution.sample_components = sample_components
        default_analytical_convolution.resolution_components = resolution_components

        # THEN
        # Mock _convolute_analytic_pair to return 1.0 for any input pair
        def mock_convolute_analytic_pair(sample_component, resolution_component):  # noqa: ARG001
            return np.full_like(default_analytical_convolution.energy.values, fill_value=1.0)

        with patch.object(
            default_analytical_convolution,
            '_convolute_analytic_pair',
            side_effect=mock_convolute_analytic_pair,
        ) as mocked_pair:
            result = default_analytical_convolution.convolution()

            # EXPECT
            # 2 sample components x 2 resolution components
            assert np.all(result == pytest.approx(4.0))
            assert mocked_pair.call_count == 4

            # Gather the actual calls to verify correct pairs
            calls = []
            for c in mocked_pair.call_args_list:
                kwargs = c.kwargs

                # Ensure no accidental extra arguments
                assert set(kwargs.keys()) == {
                    'sample_component',
                    'resolution_component',
                }

                calls.append((kwargs['sample_component'], kwargs['resolution_component']))

            expected_calls = [
                (gaussian1, gaussian2),
                (gaussian1, lorentzian2),
                (lorentzian1, gaussian2),
                (lorentzian1, lorentzian2),
            ]
            assert calls == expected_calls

    def test_convolution_components(
        self,
        default_analytical_convolution,
        gaussian1,
        lorentzian1,
    ):
        """
        Test that the convolute method also works for components.
        """

        # WHEN
        default_analytical_convolution.sample_components = gaussian1
        default_analytical_convolution.resolution_components = lorentzian1

        # THEN
        # Mock _convolute_analytic_pair to return 1.0 for any input pair
        def mock_convolute_analytic_pair(sample_component, resolution_component):  # noqa: ARG001
            return np.full_like(default_analytical_convolution.energy.values, fill_value=1.0)

        with patch.object(
            default_analytical_convolution,
            '_convolute_analytic_pair',
            side_effect=mock_convolute_analytic_pair,
        ) as mocked_pair:
            result = default_analytical_convolution.convolution()

            # EXPECT
            # 1 sample component x 1 resolution component
            assert np.all(result == pytest.approx(1.0))
            assert mocked_pair.call_count == 1

            # Gather the actual calls to verify correct pairs
            calls = []
            for c in mocked_pair.call_args_list:
                kwargs = c.kwargs

                # Ensure no accidental extra arguments
                assert set(kwargs.keys()) == {
                    'sample_component',
                    'resolution_component',
                }

                calls.append((kwargs['sample_component'], kwargs['resolution_component']))

            expected_calls = [
                (gaussian1, lorentzian1),
            ]
            assert calls == expected_calls

    @pytest.mark.parametrize(
        'function1, function2, expected_method, swapped',
        [
            # Normal cases
            (
                'gaussian1',
                'gaussian2',
                '_convolute_gaussian_gaussian',
                False,
            ),
            (
                'gaussian1',
                'lorentzian1',
                '_convolute_gaussian_lorentzian',
                False,
            ),
            (
                'gaussian1',
                'voigt1',
                '_convolute_gaussian_voigt',
                False,
            ),
            (
                'lorentzian1',
                'lorentzian2',
                '_convolute_lorentzian_lorentzian',
                False,
            ),
            (
                'lorentzian1',
                'voigt1',
                '_convolute_lorentzian_voigt',
                False,
            ),
            ('voigt1', 'voigt2', '_convolute_voigt_voigt', False),
            # Swapped cases
            ('lorentzian1', 'gaussian1', '_convolute_gaussian_lorentzian', True),
            ('voigt1', 'gaussian1', '_convolute_gaussian_voigt', True),
            ('voigt1', 'lorentzian1', '_convolute_lorentzian_voigt', True),
        ],
        indirect=['function1', 'function2'],
        ids=[
            'gauss-gauss',
            'gauss-lorentz',
            'gauss-voigt',
            'lorentz-lorentz',
            'lorentz-voigt',
            'voigt-voigt',
            'lorentz-gauss',
            'voigt-gauss',
            'voigt-lorentz',
        ],
    )
    def test_convolute_analytic_pair_calls_correct_helper(
        self,
        default_analytical_convolution,
        function1,
        function2,
        expected_method,
        swapped,
    ):
        """
        Test that _convolute_analytic_pair calls the correct internal
        helper with correct order.
        """

        with patch.object(
            default_analytical_convolution,
            expected_method,
            return_value='mocked_result',
        ) as mocked_func:
            result = default_analytical_convolution._convolute_analytic_pair(function1, function2)
            expected_args = (function2, function1) if swapped else (function1, function2)

            mocked_func.assert_called_once_with(*expected_args)
            assert result == 'mocked_result'

    @pytest.mark.parametrize(
        'function1',
        ['gaussian1', 'lorentzian1', 'voigt1', 'dho1'],
        indirect=True,
        ids=['gaussian', 'lorentzian', 'voigt', 'dho'],
    )
    def test_convolute_analytic_pair_delta(self, default_analytical_convolution, function1):
        """
        Test that convolution with delta function returns the other
        function.
        """
        # WHEN THEN
        delta_function = DeltaFunction(area=2.0, center=0.5)
        function1 = Gaussian(area=3.0, center=-1.0, width=1.0)

        convoluted = default_analytical_convolution._convolute_analytic_pair(
            delta_function, function1
        )

        # EXPECT
        expected = 2.0 * function1.evaluate(default_analytical_convolution.energy.values - 0.5)

        assert np.allclose(
            convoluted,
            expected,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
        )

    def test_convolute_analytic_pair_resolution_delta_raises(self, default_analytical_convolution):
        """
        Test that an error is raised if the resolution function is a
        delta function.
        """
        # WHEN
        sample_function = Gaussian(area=2.0, center=0.0, width=1.0)
        resolution_function = DeltaFunction(area=1.0, center=0.0)

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='not supported',
        ):
            default_analytical_convolution._convolute_analytic_pair(
                sample_function, resolution_function
            )

    def test_convolute_analytic_pair_non_analytical_pair_raises(
        self, default_analytical_convolution, gaussian1, dho1
    ):
        """
        Test that an error is raised if the function pair is not
        supported for analytical convolution.
        """
        # WHEN
        sample_function = dho1
        resolution_function = gaussian1

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match='not supported',
        ):
            default_analytical_convolution._convolute_analytic_pair(
                sample_function, resolution_function
            )

    @pytest.mark.parametrize(
        'method_name, function1, function2',
        [
            ('_convolute_gaussian_gaussian', 'gaussian1', 'gaussian2'),
            ('_convolute_gaussian_lorentzian', 'gaussian1', 'lorentzian1'),
            ('_convolute_gaussian_voigt', 'gaussian1', 'voigt1'),
            ('_convolute_lorentzian_lorentzian', 'lorentzian1', 'lorentzian2'),
            ('_convolute_lorentzian_voigt', 'lorentzian1', 'voigt1'),
            ('_convolute_voigt_voigt', 'voigt1', 'voigt2'),
        ],
        indirect=['function1', 'function2'],
        ids=[
            'gauss-gauss',
            'gauss-lorentz',
            'gauss-voigt',
            'lorentz-lorentz',
            'lorentz-voigt',
            'voigt-voigt',
        ],
    )
    def test_convolute_function1_function2(
        self, default_analytical_convolution, function1, function2, method_name
    ):
        # This test is perhaps superfluous since the methods get tested
        # indirectly above.
        """
        Test that the analytical convolution methods are correct by
        comparing to numerical convolution.
        """
        # WHEN THEN
        convoluted = getattr(default_analytical_convolution, method_name)(function1, function2)

        # EXPECT
        expected = fftconvolve(
            function1.evaluate(default_analytical_convolution.energy.values),
            function2.evaluate(default_analytical_convolution.energy.values),
            mode='same',
        ) * (
            default_analytical_convolution.energy.values[1]
            - default_analytical_convolution.energy.values[0]
        )

        # Numerical convolution can be inaccurate at the edges, so only
        # compare the central part.
        N = len(convoluted)
        start = N // 4
        end = 3 * N // 4

        assert np.allclose(
            convoluted[start:end],
            expected[start:end],
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
        )

    @pytest.mark.parametrize(
        'function1',
        ['gaussian1', 'lorentzian1', 'voigt1', 'dho1'],
        indirect=True,
        ids=['gaussian', 'lorentzian', 'voigt', 'dho'],
    )
    def test_convolute_delta_any(self, default_analytical_convolution, function1):
        """
        Test that convolution with delta function returns the other
        function.
        """
        # WHEN THEN
        delta_function = DeltaFunction(area=2.0, center=0.5)
        convoluted = default_analytical_convolution._convolute_delta_any(delta_function, function1)

        # EXPECT
        expected = 2.0 * function1.evaluate(default_analytical_convolution.energy.values - 0.5)

        assert np.allclose(
            convoluted,
            expected,
            rtol=NUMERICAL_CONVOLUTION_RELATIVE_TOLERANCE,
            atol=NUMERICAL_CONVOLUTION_ABSOLUTE_TOLERANCE,
        )

    def test_gaussian_eval(self, default_analytical_convolution):
        # WHEN
        area = 2.0
        center = 0.0
        width = 1.0

        # THEN
        gaussian_eval = default_analytical_convolution._gaussian_eval(area, center, width)

        # EXPECT
        expected = Gaussian(area=area, center=center, width=width).evaluate(
            default_analytical_convolution.energy.values
        )
        assert np.allclose(gaussian_eval, expected)

    def test_lorentzian_eval(self, default_analytical_convolution):
        # WHEN
        area = 2.0
        center = 0.0
        width = 1.0

        # THEN
        lorentzian_eval = default_analytical_convolution._lorentzian_eval(area, center, width)

        # EXPECT
        expected = Lorentzian(area=area, center=center, width=width).evaluate(
            default_analytical_convolution.energy.values
        )
        assert np.allclose(lorentzian_eval, expected)

    def test_voigt_eval(self, default_analytical_convolution):
        # WHEN
        area = 2.0
        center = 0.0
        gaussian_width = 1.0
        lorentzian_width = 1.0

        # THEN
        voigt_eval = default_analytical_convolution._voigt_eval(
            area, center, gaussian_width, lorentzian_width
        )

        # EXPECT
        expected = Voigt(
            area=area,
            center=center,
            gaussian_width=gaussian_width,
            lorentzian_width=lorentzian_width,
        ).evaluate(default_analytical_convolution.energy.values)

        assert np.allclose(voigt_eval, expected)
