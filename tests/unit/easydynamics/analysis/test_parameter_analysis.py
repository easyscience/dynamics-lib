# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc

from easydynamics.analysis.fit_bindings import FitBinding
from easydynamics.analysis.parameter_analysis import ParameterAnalysis
from easydynamics.sample_model.components.gaussian import Gaussian
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
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT
    # TEST PLOT

    @pytest.mark.parametrize(
        'set_pars_none, bindings, expected_exception, match',
        [
            # No parameters
            (
                True,
                [MagicMock(spec=FitBinding)],
                ValueError,
                'No parameters Dataset provided',
            ),
            # No bindings
            (
                False,
                [],
                ValueError,
                'No fit bindings provided',
            ),
            # Wrong bindings type
            (
                False,
                'not_a_list',
                TypeError,
                'bindings must be a list of FitBinding objects',
            ),
        ],
        ids=['no_parameters', 'no_bindings', 'wrong_bindings_type'],
    )
    def test_calculate_model_dataset_invalid_inputs(
        self, parameter_analysis, set_pars_none, bindings, expected_exception, match
    ):
        # WHEN
        if set_pars_none:
            parameter_analysis.parameters = None
        else:
            # keep existing valid dataset
            pass

        # THEN EXPECT
        with pytest.raises(expected_exception, match=match):
            parameter_analysis.calculate_model_dataset(bindings)

    def test_calculate_model_dataset_missing_parameter_raises(self, parameter_analysis):
        binding = parameter_analysis.bindings[0]

        with (
            patch.object(binding, 'get_parameter_names', return_value=['missing_name']),
            patch.object(binding, 'get_model_names', return_value=['model']),
            patch.object(binding, 'build_callables', return_value=[lambda x: x]),
            pytest.raises(ValueError, match='not found in parameters Dataset'),
        ):
            parameter_analysis.calculate_model_dataset([binding])

    def test_calculate_model_dataset_single_binding(self, parameter_analysis):
        # WHEN
        binding = parameter_analysis.bindings[0]

        # Mock callable to return predictable output
        mock_callable = MagicMock(return_value=np.array([10.0, 20.0]))

        with (
            patch.object(binding, 'build_callables', return_value=[mock_callable]),
            patch.object(binding, 'get_model_names', return_value=['model1']),
            patch.object(binding, 'get_parameter_names', return_value=['parameter1']),
        ):
            # THEN
            result = parameter_analysis.calculate_model_dataset([binding])

        # EXPECT
        assert isinstance(result, sc.Dataset)
        assert 'model1' in result

        da = result['model1']

        np.testing.assert_allclose(da.values, [10.0, 20.0])
        np.testing.assert_allclose(da.coords['Q'].values, [0.1, 0.2])
        assert da.unit == parameter_analysis.parameters['parameter1'].unit
        mock_callable.assert_called_once()

        args, kwargs = mock_callable.call_args
        np.testing.assert_allclose(args[0], [0.1, 0.2])
        assert kwargs == {}

    def test_calculate_model_dataset_multiple_bindings(self, parameter_analysis):
        # WHEN
        binding1 = parameter_analysis.bindings[0]
        binding2 = parameter_analysis.bindings[1]

        # Mock callable to return predictable output
        mock_callable1 = MagicMock(return_value=np.array([10.0, 20.0]))
        mock_callable2 = MagicMock(return_value=np.array([30.0, 40.0]))

        with (
            patch.object(binding1, 'build_callables', return_value=[mock_callable1]),
            patch.object(binding1, 'get_model_names', return_value=['model1']),
            patch.object(binding1, 'get_parameter_names', return_value=['parameter1']),
            patch.object(binding2, 'build_callables', return_value=[mock_callable2]),
            patch.object(binding2, 'get_model_names', return_value=['model2']),
            patch.object(binding2, 'get_parameter_names', return_value=['parameter2']),
        ):
            # THEN
            result = parameter_analysis.calculate_model_dataset([binding1, binding2])

        # EXPECT
        assert isinstance(result, sc.Dataset)
        assert 'model1' in result
        assert 'model2' in result

        da1 = result['model1']
        da2 = result['model2']

        np.testing.assert_allclose(da1.values, [10.0, 20.0])
        np.testing.assert_allclose(da1.coords['Q'].values, [0.1, 0.2])
        assert da1.unit == parameter_analysis.parameters['parameter1'].unit

        np.testing.assert_allclose(da2.values, [30.0, 40.0])
        np.testing.assert_allclose(da2.coords['Q'].values, [0.1, 0.2])
        assert da2.unit == parameter_analysis.parameters['parameter2'].unit
        mock_callable1.assert_called_once()
        args, kwargs = mock_callable1.call_args
        np.testing.assert_allclose(args[0], [0.1, 0.2])
        assert kwargs == {}

        mock_callable2.assert_called_once()
        args, kwargs = mock_callable2.call_args
        np.testing.assert_allclose(args[0], [0.1, 0.2])
        assert kwargs == {}

    def test_calculate_model_dataset_multiple_bindings_diffusion(self, parameter_analysis):
        # WHEN
        binding1 = parameter_analysis.bindings[0]
        binding2 = parameter_analysis.bindings[1]

        # Mock callable to return predictable output
        mock_callable1 = MagicMock(return_value=np.array([10.0, 20.0]))
        mock_callable2a = MagicMock(return_value=np.array([30.0, 40.0]))
        mock_callable2w = MagicMock(return_value=np.array([50.0, 60.0]))

        with (
            patch.object(binding1, 'build_callables', return_value=[mock_callable1]),
            patch.object(binding1, 'get_model_names', return_value=['model1']),
            patch.object(binding1, 'get_parameter_names', return_value=['parameter1']),
            patch.object(
                binding2,
                'build_callables',
                return_value=[mock_callable2a, mock_callable2w],
            ),
            patch.object(binding2, 'get_model_names', return_value=['model2a', 'model2w']),
            patch.object(
                binding2,
                'get_parameter_names',
                return_value=['parameter3 area', 'parameter3 width'],
            ),
        ):
            # THEN
            result = parameter_analysis.calculate_model_dataset([binding1, binding2])

        # EXPECT
        assert isinstance(result, sc.Dataset)
        assert 'model1' in result
        assert 'model2a' in result
        assert 'model2w' in result

        da1 = result['model1']
        da2a = result['model2a']
        da2w = result['model2w']

        np.testing.assert_allclose(da1.values, [10.0, 20.0])
        np.testing.assert_allclose(da1.coords['Q'].values, [0.1, 0.2])
        assert da1.unit == parameter_analysis.parameters['parameter1'].unit

        np.testing.assert_allclose(da2a.values, [30.0, 40.0])
        np.testing.assert_allclose(da2a.coords['Q'].values, [0.1, 0.2])
        assert da2a.unit == parameter_analysis.parameters['parameter3 area'].unit

        np.testing.assert_allclose(da2w.values, [50.0, 60.0])
        np.testing.assert_allclose(da2w.coords['Q'].values, [0.1, 0.2])
        assert da2w.unit == parameter_analysis.parameters['parameter3 width'].unit

        args, kwargs = mock_callable1.call_args
        np.testing.assert_allclose(args[0], [0.1, 0.2])
        assert kwargs == {}

        args, kwargs = mock_callable2a.call_args
        np.testing.assert_allclose(args[0], [0.1, 0.2])
        assert kwargs == {}

        args, kwargs = mock_callable2w.call_args
        np.testing.assert_allclose(args[0], [0.1, 0.2])
        assert kwargs == {}

    def test_append_binding(self, parameter_analysis):
        # WHEN
        model = Polynomial(coefficients=[2.0, 1.0])
        new_binding = FitBinding(parameter_name='parameter2', model=model)

        # THEN
        parameter_analysis.append_binding(new_binding)

        # EXPECT
        assert len(parameter_analysis.bindings) == 3
        assert parameter_analysis.bindings[-1] == new_binding

    def test_append_binding_wrong_type_raises(self, parameter_analysis):
        # WHEN
        new_binding = 'not_a_fit_binding'

        # THEN EXPECT
        with pytest.raises(TypeError, match='binding must be a FitBinding object'):
            parameter_analysis.append_binding(new_binding)

    def test_clear_bindings(self, parameter_analysis):
        # WHEN
        parameter_analysis.clear_bindings()

        # THEN EXPECT
        assert len(parameter_analysis.bindings) == 0

    def test_get_all_variables(self, parameter_analysis):
        # WHEN

        # THEN
        variables = parameter_analysis.get_all_variables()

        # EXPECT
        expected_variables = []
        expected_variables.extend(parameter_analysis.bindings[0].model.get_all_variables())
        expected_variables.extend(parameter_analysis.bindings[1].model.get_all_variables())
        assert set(variables) == set(expected_variables)

    def test_get_all_variables_no_bindings(self, parameter_analysis):
        # WHEN
        parameter_analysis.clear_bindings()

        # THEN EXPECT
        variables = parameter_analysis.get_all_variables()
        assert variables == []

    def test_get_all_variables_overlapping_variables(self, parameter_analysis):
        # WHEN
        # Create two bindings with overlapping variables
        model1 = Gaussian(display_name='model1')

        binding1 = FitBinding(parameter_name='parameter1', model=model1)
        binding2 = FitBinding(parameter_name='parameter2', model=model1)

        parameter_analysis.bindings = [binding1, binding2]

        # THEN
        variables = parameter_analysis.get_all_variables()

        # EXPECT
        assert isinstance(variables, list)
        assert set(variables) == set(model1.get_all_variables())
        assert len(variables) == len(model1.get_all_variables())

    @pytest.mark.parametrize(
        'input_bindings, expected_length',
        [
            (None, 0),
            ('single', 1),
            ('list', 2),
        ],
    )
    def test_verify_bindings_valid(self, parameter_analysis, input_bindings, expected_length):
        # WHEN
        b1 = parameter_analysis.bindings[0]
        b2 = parameter_analysis.bindings[1]

        if input_bindings == 'single':
            input_bindings = b1
        elif input_bindings == 'list':
            input_bindings = [b1, b2]

        # THEN
        result = parameter_analysis._verify_bindings(input_bindings)

        # EXPECT
        assert isinstance(result, list)
        assert len(result) == expected_length

        if expected_length == 1:
            assert result[0] is b1
        if expected_length == 2:
            assert result == [b1, b2]

    @pytest.mark.parametrize(
        'invalid_bindings',
        [
            'not_a_list_or_fit_binding',
            [MagicMock(spec=FitBinding), 'not_a_fit_binding'],
        ],
        ids=['wrong_type', 'list_with_wrong_type'],
    )
    def test_verify_bindings_invalid(self, parameter_analysis, invalid_bindings):
        # WHEN THEN EXPECT
        with pytest.raises((ValueError, TypeError)):
            parameter_analysis._verify_bindings(invalid_bindings)
