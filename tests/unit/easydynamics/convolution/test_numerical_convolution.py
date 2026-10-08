# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter
from scipy.signal import fftconvolve

from easydynamics.convolution.energy_grid import EnergyGrid
from easydynamics.convolution.numerical_convolution import NumericalConvolution
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.settings.convolution_settings import ConvolutionSettings
from easydynamics.utils.detailed_balance import detailed_balance_factor


def _make_numerical_convolution(convolution_settings=None):
    energy = np.linspace(-10, 10, 5001)
    sample_components = ComponentCollection(display_name='ComponentCollection')
    sample_components.append_component(Gaussian(name='Gaussian1', area=2.0, center=0.1, width=0.4))
    resolution_components = ComponentCollection(display_name='ResolutionModel')
    resolution_components.append_component(
        Gaussian(name='GaussianRes', area=3.0, center=0.2, width=0.5)
    )

    return NumericalConvolution(
        energy=energy,
        sample_components=sample_components,
        resolution_components=resolution_components,
        convolution_settings=convolution_settings,
    )


class TestNumericalConvolution:
    @pytest.fixture
    def default_numerical_convolution(self):
        return _make_numerical_convolution()

    def test_settings_change_invalidates_all_sharing_convolvers(self):
        # WHEN: two convolvers sharing one ConvolutionSettings, and an accuracy knob changes
        settings = ConvolutionSettings()
        convolver_a = _make_numerical_convolution(settings)
        convolver_b = _make_numerical_convolution(settings)
        settings.upsample_factor = 10

        # THEN: A consumes the change by rebuilding its plan
        convolver_a.convolution()

        # EXPECT: A is current, but B still sees the stale plan (regression: A's rebuild
        # setting the shared flag used to mask the settings change from B)
        assert convolver_a._convolution_plan_is_current() is True
        assert convolver_b._convolution_plan_is_current() is False

        # and B becomes current after its own rebuild
        convolver_b.convolution()
        assert convolver_b._convolution_plan_is_current() is True

    def test_invalidate_plan_invalidates_all_sharing_convolvers(self):
        # WHEN: two current convolvers sharing one ConvolutionSettings
        settings = ConvolutionSettings()
        convolver_a = _make_numerical_convolution(settings)
        convolver_b = _make_numerical_convolution(settings)

        # THEN: an invalidation affects every convolver, and one convolver rebuilding does
        # not mask the invalidation from the other
        settings._invalidate_plan()
        convolver_a.convolution()

        # EXPECT
        assert convolver_a._convolution_plan_is_current() is True
        assert convolver_b._convolution_plan_is_current() is False

    def test_init(self, default_numerical_convolution):
        """
        Test initialization of NumericalConvolution with
        default parameters.
        """
        # WHEN THEN EXPECT
        assert isinstance(default_numerical_convolution, NumericalConvolution)
        assert isinstance(default_numerical_convolution.energy, sc.Variable)
        assert np.allclose(default_numerical_convolution.energy.values, np.linspace(-10, 10, 5001))
        assert isinstance(default_numerical_convolution._sample_components, ComponentCollection)
        assert isinstance(
            default_numerical_convolution._resolution_components, ComponentCollection
        )
        assert default_numerical_convolution.upsample_factor == 5
        assert default_numerical_convolution.extension_factor == pytest.approx(0.2)
        assert default_numerical_convolution.temperature is None
        assert default_numerical_convolution.x_unit == 'meV'
        assert default_numerical_convolution.y_unit == 'dimensionless'
        assert (
            default_numerical_convolution.detailed_balance_settings.normalize_detailed_balance
            is True
        )
        assert isinstance(default_numerical_convolution._energy_grid, EnergyGrid)

    @pytest.mark.parametrize('upsample_factor', [None, 5])
    def test_convolution(self, default_numerical_convolution, upsample_factor):
        """
        Test that convolution of two Gaussians produces the
        expected result.
        """
        # WHEN THEN
        default_numerical_convolution.upsample_factor = upsample_factor
        default_numerical_convolution.energy_offset = 0.4
        result = default_numerical_convolution.convolution()

        # EXPECT
        expected_area = 2.0 * 3.0  # area of sample_components * area of resolution_components
        expected_center = (
            0.1 + 0.2 + 0.4
        )  # center of sample_components + center of resolution_components
        expected_width = np.sqrt(0.4**2 + 0.5**2)  # sqrt(width_sample^2 + width_res^2)
        expected_result = Gaussian(
            name='ExpectedConvolution',
            area=expected_area,
            center=expected_center,
            width=expected_width,
        ).evaluate(default_numerical_convolution.energy)
        assert np.allclose(result, expected_result, rtol=1e-4)

    def test_convolution_with_temperature(
        self,
        default_numerical_convolution,
    ):
        """
        Test that convolution includes detailed balance correction
        when temperature is provided.
        """

        # WHEN
        default_numerical_convolution.temperature = 5.0  # Kelvin

        # THEN
        result = default_numerical_convolution.convolution()

        # EXPECT
        sample_valds = default_numerical_convolution._sample_components.evaluate(
            default_numerical_convolution.energy.values
        )
        resolution_vals = default_numerical_convolution._resolution_components.evaluate(
            default_numerical_convolution.energy.values
        )
        DBF = detailed_balance_factor(energy=default_numerical_convolution.energy, temperature=5.0)
        expected_result = fftconvolve(sample_valds * DBF, resolution_vals, mode='same') * (
            default_numerical_convolution.energy.values[1]
            - default_numerical_convolution.energy.values[0]
        )

        assert np.allclose(result, expected_result, rtol=1e-4)

    @pytest.mark.parametrize(
        'plan_valid, suppress_warnings, use_db, upsample',
        [
            (True, True, False, None),
            (False, True, False, None),
            (True, False, False, None),
            (True, False, True, None),
            (True, False, True, 10),
            (False, False, True, 10),
        ],
        ids=[
            'plan_valid=True, suppress_warnings=True, use_db=False, upsample=None',
            'plan_valid=False, suppress_warnings=True, use_db=False, upsample=None',
            'plan_valid=True, suppress_warnings=False, use_db=False, upsample=None',
            'plan_valid=True, suppress_warnings=False, use_db=True, upsample=None',
            'plan_valid=True, suppress_warnings=False, use_db=True, upsample=10',
            'plan_valid=False, suppress_warnings=False, use_db=True, upsample=10',
        ],
    )
    def test_convolution_branches(
        self,
        default_numerical_convolution,
        monkeypatch,
        plan_valid,
        suppress_warnings,
        use_db,
        upsample,
    ):
        "Test that convolution branches are executed as expected based on settings."
        # WHEN
        conv = default_numerical_convolution

        # --- Configure branches ---
        conv.convolution_settings.suppress_warnings = suppress_warnings

        conv.detailed_balance_settings.use_detailed_balance = use_db
        conv.temperature = 10.0 if use_db else None

        conv.upsample_factor = upsample

        if plan_valid:
            conv._mark_convolution_plan_current()
        else:
            conv._plan_seen_version = None

        # --- Track calls ---
        create_grid_called = False
        check_width_calls = []

        # Mock the methods that would be called in the branches to track
        # whether they were called or not.
        def fake_create_energy_grid():
            nonlocal create_grid_called
            create_grid_called = True
            return conv._energy_grid

        def fake_check_width_thresholds(*args, **kwargs):
            check_width_calls.append((args, kwargs))

        monkeypatch.setattr(conv, '_create_energy_grid', fake_create_energy_grid)
        monkeypatch.setattr(conv, '_check_width_thresholds', fake_check_width_thresholds)

        # --- Simplify numerics ---
        dense = conv._energy_grid.energy_dense

        monkeypatch.setattr(
            conv.sample_components,
            'evaluate',
            lambda x: np.ones_like(dense),  # ruff: ignore[unused-lambda-argument]
        )
        monkeypatch.setattr(
            conv.resolution_components,
            'evaluate',
            lambda x: np.ones_like(dense),  # ruff: ignore[unused-lambda-argument]
        )

        db_called = False

        def fake_db(*args, **kwargs):  # ruff: ignore[unused-function-argument]
            nonlocal db_called
            db_called = True
            return np.ones_like(dense)

        monkeypatch.setattr(
            'easydynamics.convolution.numerical_convolution.detailed_balance_factor',
            fake_db,
        )

        monkeypatch.setattr(
            'easydynamics.convolution.numerical_convolution.fftconvolve',
            lambda a, b, mode: np.ones_like(dense),  # ruff: ignore[unused-lambda-argument]
        )

        interp_called = False

        def fake_interp(*args, **kwargs):  # ruff: ignore[unused-function-argument]
            nonlocal interp_called
            interp_called = True
            return np.ones_like(conv.energy.values)

        monkeypatch.setattr(np, 'interp', fake_interp)

        # THEN
        result = conv.convolution()

        # EXPECT
        # Branch 1: energy grid recreation
        assert create_grid_called is (not plan_valid)

        # Branch 2: warnings
        if suppress_warnings:
            assert len(check_width_calls) == 0
        else:
            assert len(check_width_calls) == 2

        # Branch 3: detailed balance
        assert db_called is (use_db and conv.temperature is not None)

        # Branch 4: interpolation
        assert interp_called is (upsample is not None)

        # Sanity: result shape
        if upsample is not None:
            assert result.shape == conv.energy.values.shape
        else:
            assert result.shape == dense.shape

    def test_convolution_with_energy_offset_in_different_unit(self, default_numerical_convolution):
        # WHEN: energy grid is in meV, offset given as a Parameter in eV (0.001 eV = 1.0 meV)
        conv = default_numerical_convolution
        conv.energy_offset = Parameter(name='energy_offset', value=0.001, unit='eV')
        result_ev = conv.convolution()

        # THEN: run again with the numerically equivalent offset as an explicit meV Parameter
        conv.energy_offset = Parameter(name='energy_offset', value=1.0, unit='meV')
        result_mev = conv.convolution()

        # EXPECT: unit conversion is transparent — results are numerically identical
        np.testing.assert_allclose(result_ev, result_mev, rtol=1e-6)

    def test_repr(self, default_numerical_convolution):
        r = repr(default_numerical_convolution)
        assert 'NumericalConvolution' in r
        assert 'energy_len=' in r

    # ───── Regression tests ─────

    def test_detailed_balance_energy_includes_even_length_offset(
        self, default_numerical_convolution, monkeypatch
    ):
        # WHEN: even-length energy grid with temperature
        nc = default_numerical_convolution
        nc.energy = np.linspace(-10, 10, 100)  # even-length → non-zero energy_even_length_offset
        nc.temperature = 300.0

        captured = {}

        # spy_dbf wraps detailed_balance_factor: captures the energy argument passed to it,
        # then delegates to the real function so the convolution still produces a valid result.
        def spy_dbf(**kwargs):
            captured['energy'] = kwargs['energy']
            return detailed_balance_factor(**kwargs)

        monkeypatch.setattr(
            'easydynamics.convolution.numerical_convolution.detailed_balance_factor', spy_dbf
        )

        # THEN: run convolution
        nc.convolution()

        # EXPECT: DBF receives energy_dense - energy_even_length_offset
        # Before the fix, energy_even_length_offset was omitted, causing a half-bin error.
        grid = nc._energy_grid
        np.testing.assert_allclose(
            captured['energy'], grid.energy_dense - grid.energy_even_length_offset, atol=1e-12
        )
