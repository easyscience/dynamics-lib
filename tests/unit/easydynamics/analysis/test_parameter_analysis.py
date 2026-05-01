# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc

from easydynamics.analysis.fit_bindings import FitBinding
from easydynamics.analysis.parameter_analysis import ParameterAnalysis
from easydynamics.sample_model.components.polynomial import Polynomial
from easydynamics.sample_model.diffusion_model.brownian_translational_diffusion import (
    BrownianTranslationalDiffusion,
)


class TestParameterAnalysis:
    @pytest.fixture
    def dataset(self):
        Q = sc.array(dims=['Q'], values=[0.1, 0.2])
        return sc.Dataset(
            data={
                'parameter1': sc.DataArray(
                    data=sc.array(dims=['Q'], values=[1.0, 2.0], variances=[0.1, 0.2], unit='meV'),
                    coords={'Q': Q},
                ),
                'parameter2': sc.DataArray(
                    data=sc.array(
                        dims=['Q'],
                        values=[1.5, 2.5],
                        variances=[0.15, 0.25],
                        unit='1/meV',
                    ),
                    coords={'Q': Q},
                ),
                'parameter3 area': sc.DataArray(
                    data=sc.array(dims=['Q'], values=[4.0, 5.0], variances=[0.3, 0.5], unit='meV'),
                    coords={'Q': Q},
                ),
                'parameter3 width': sc.DataArray(
                    data=sc.array(dims=['Q'], values=[6.0, 7.0], variances=[0.6, 0.7], unit='meV'),
                    coords={'Q': Q},
                ),
            }
        )

    @pytest.fixture
    def parameter_analysis(self, dataset):
        model = Polynomial(coefficients=[1.0, 0.5])
        diffusion_model = BrownianTranslationalDiffusion()

        fit_binding1 = FitBinding(parameter_name='parameter1', model=model)
        fit_binding2 = FitBinding(parameter_name='parameter3', model=diffusion_model)

        return ParameterAnalysis(parameters=dataset, bindings=[fit_binding1, fit_binding2])

    def test_initialization(self, parameter_analysis):
        # WHEN THEN EXPECT
        assert isinstance(parameter_analysis, ParameterAnalysis)
        assert len(parameter_analysis.bindings) == 2
        assert parameter_analysis.bindings[0].parameter_name == 'parameter1'
        assert parameter_analysis.bindings[1].parameter_name == 'parameter3'

    def test_parameter_property(self, parameter_analysis):
        # WHEN
        parameters = parameter_analysis.parameters

        # THEN EXPECT
        assert isinstance(parameters, sc.Dataset)
        assert set(parameters.keys()) == {
            'parameter1',
            'parameter2',
            'parameter3 area',
            'parameter3 width',
        }

        # WHEN
        Q = sc.array(dims=['Q'], values=[0.1, 0.2])
        new_data = sc.Dataset(
            data={
                'parameter4': sc.DataArray(
                    data=sc.array(
                        dims=['Q'],
                        values=[71.0, 12.0],
                        variances=[1.1, 2.2],
                        unit='meV',
                    ),
                    coords={'Q': Q},
                ),
                'parameter5': sc.DataArray(
                    data=sc.array(
                        dims=['Q'],
                        values=[8.5, 0.5],
                        variances=[2.15, 1.25],
                        unit='1/meV',
                    ),
                    coords={'Q': Q},
                ),
            }
        )

        # THEN
        parameter_analysis.parameters = new_data

        # EXPECT
        assert parameter_analysis.parameters is new_data

    def test_bindings_property(self, parameter_analysis):
        # WHEN
        bindings = parameter_analysis.bindings

        # THEN EXPECT
        assert isinstance(bindings, list)
        assert len(bindings) == 2
        assert all(isinstance(b, FitBinding) for b in bindings)

        # WHEN
        model = Polynomial(coefficients=[2.0, 1.0])
        new_binding = FitBinding(parameter_name='parameter2', model=model)
        parameter_analysis.bindings = new_binding

        # THEN EXPECT
        assert parameter_analysis.bindings == [new_binding]

    def test_fit_no_bindings_raises(self, parameter_analysis):
        # WHEN

        # THEN
        parameter_analysis.bindings = None

        # EXPECT
        with pytest.raises(ValueError, match='No fit bindings provided'):
            parameter_analysis.fit()

    def test_fit_no_parameters_raises(self, parameter_analysis):
        # WHEN

        # THEN
        parameter_analysis.parameters = None

        # EXPECT
        with pytest.raises(ValueError, match='No parameters Dataset provided'):
            parameter_analysis.fit()

    def test_fit_wrong_parameter_name_raises(self, parameter_analysis):
        # WHEN
        model = Polynomial(coefficients=[2.0, 1.0])
        incorrect_binding = FitBinding(parameter_name='nonexistent_parameter', model=model)
        parameter_analysis.bindings = incorrect_binding

        # THEN EXPECT
        with pytest.raises(
            ValueError,
            match="Parameter 'nonexistent_parameter' from binding",
        ):
            parameter_analysis.fit()

    def test_fit_success(self, parameter_analysis):
        # WHEN
        mock_result = MagicMock()

        with patch('easydynamics.analysis.parameter_analysis.MultiFitter') as MockMultiFitter:
            instance = MockMultiFitter.return_value
            instance.fit.return_value = mock_result

            # THEN
            result = parameter_analysis.fit()

            # EXPECT
            assert MockMultiFitter.called

            kwargs = MockMultiFitter.call_args.kwargs
            assert 'fit_objects' in kwargs
            assert 'fit_functions' in kwargs

            # Expect 3 fits:
            # - parameter1 → 1 callable
            # - parameter3 → 2 callables (area + width)
            assert len(kwargs['fit_objects']) == 3
            assert len(kwargs['fit_functions']) == 3

            # --- Fit called correctly ---
            instance.fit.assert_called_once()

            call_kwargs = instance.fit.call_args.kwargs

            x = call_kwargs['x']
            y = call_kwargs['y']
            w = call_kwargs['weights']

            assert len(x) == 3
            assert len(y) == 3
            assert len(w) == 3

            # Check one concrete value
            np.testing.assert_allclose(x[0], [0.1, 0.2])
            np.testing.assert_allclose(y[0], [1.0, 2.0])

            expected_w = 1 / np.sqrt([0.1, 0.2])
            np.testing.assert_allclose(w[0], expected_w)

            assert result is mock_result


# TEST PLOT
