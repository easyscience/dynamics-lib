# from unittest.mock import Mock


from collections import Counter
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.experiment import Experiment
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.components.gaussian import Gaussian


class TestAnalysis1d:
    @pytest.fixture
    def analysis1d(self):
        Q = sc.array(dims=["Q"], values=[1, 2, 3], unit="1/Angstrom")
        energy = sc.array(dims=["energy"], values=[10, 20, 30], unit="meV")
        data = sc.array(dims=["Q", "energy"], values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data_array = sc.DataArray(data=data, coords={"Q": Q, "energy": energy})

        experiment = Experiment(data=data_array)
        sample_model = SampleModel(components=Gaussian())
        instrument_model = InstrumentModel()
        analysis1d = Analysis1d(
            display_name="TestAnalysis",
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            Q_index=0,
            extra_parameters=None,
        )

        return analysis1d

    def test_init(self, analysis1d):
        # WHEN THEN

        # EXPECT
        assert analysis1d.display_name == "TestAnalysis"
        assert isinstance(analysis1d._experiment, Experiment)
        assert isinstance(analysis1d._sample_model, SampleModel)
        assert isinstance(analysis1d._instrument_model, InstrumentModel)
        assert analysis1d._extra_parameters == []
        assert np.array_equal(analysis1d.Q.values, [1, 2, 3])
        assert analysis1d.Q_index == 0

    def test_Q_index_setter(self, analysis1d):
        # WHEN
        analysis1d.Q_index = 1

        # THEN / EXPECT
        assert analysis1d.Q_index == 1

    @pytest.mark.parametrize(
        "invalid_Q_index, expected_exception, expected_message",
        [
            (-1, IndexError, "Q_index must be"),
            (10, IndexError, "Q_index must be"),
            ("invalid", IndexError, "Q_index must be "),
            (np.nan, IndexError, "Q_index must be "),
            ([1, 2], IndexError, "Q_index must be "),
        ],
        ids=[
            "Negative index",
            "Index out of range",
            "Non-integer string",
            "NaN value",
            "List instead of integer",
        ],
    )
    def test_Q_index_setter_incorrect_Q(
        self, analysis1d, invalid_Q_index, expected_exception, expected_message
    ):
        # WHEN / THEN / EXPECT
        with pytest.raises(expected_exception, match=expected_message):
            analysis1d.Q_index = invalid_Q_index

    def test_calculate_updates_convolver_and_calls_calculate(self, analysis1d):
        # WHEN

        # mock the _create_convolver and _calculate methods to verify
        # they are called
        fake_convolver = object()
        expected_result = np.array([42.0])

        analysis1d._create_convolver = MagicMock(return_value=fake_convolver)
        analysis1d._calculate = MagicMock(return_value=expected_result)

        # THEN
        result = analysis1d.calculate()

        # EXPECT

        analysis1d._create_convolver.assert_called_once()
        assert analysis1d._convolver is fake_convolver
        analysis1d._calculate.assert_called_once()
        np.testing.assert_array_equal(result, expected_result)

    def test__calculate_adds_sample_and_background(self, analysis1d):
        sample = np.array([1.0, 2.0, 3.0])
        background = np.array([0.5, 0.5, 0.5])

        analysis1d._evaluate_sample = MagicMock(return_value=sample)
        analysis1d._evaluate_background = MagicMock(return_value=background)

        result = analysis1d._calculate()

        np.testing.assert_array_equal(result, sample + background)

        analysis1d._evaluate_sample.assert_called_once()
        analysis1d._evaluate_background.assert_called_once()

    def test_fit_raises_if_no_experiment(self, analysis1d):
        # WHEN THEN
        analysis1d._experiment = None

        # EXPECT
        with pytest.raises(ValueError, match="No experiment"):
            analysis1d.fit()

    def test_fit_calls_fitter_with_correct_arguments(self, analysis1d):

        # WHEN

        # Mock all the methods that are called during fit to verify they
        # are called with the correct arguments
        fake_x = np.array([1, 2, 3])
        fake_y = np.array([10, 20, 30])
        fake_weights = np.array([0.1, 0.2, 0.3])

        analysis1d._extract_x_y_weights_from_experiment = MagicMock(
            return_value=(fake_x, fake_y, fake_weights)
        )

        analysis1d._create_convolver = MagicMock(return_value="fake_convolver")

        fake_fit_result = object()
        fake_fitter_instance = MagicMock()
        fake_fitter_instance.fit.return_value = fake_fit_result

        with patch(
            "easydynamics.analysis.analysis1d.EasyScienceFitter",
            return_value=fake_fitter_instance,
        ) as mock_fitter:
            analysis1d.as_fit_function = MagicMock(return_value="fit_func")

            # THEN
            result = analysis1d.fit()

        # EXPECT

        # Check that all the mocked methods were called with the correct
        # arguments
        analysis1d._create_convolver.assert_called_once()

        mock_fitter.assert_called_once_with(
            fit_object=analysis1d,
            fit_function="fit_func",
        )

        analysis1d._extract_x_y_weights_from_experiment.assert_called_once()

        fake_fitter_instance.fit.assert_called_once_with(
            x=fake_x,
            y=fake_y,
            weights=fake_weights,
        )

        # And that the result is returned
        assert analysis1d._fit_result is fake_fit_result
        assert result is fake_fit_result

    def test_as_fit_function_calls_calculate(self, analysis1d):
        # WHEN
        expected = np.array([1.0, 2.0, 3.0])
        analysis1d._calculate = MagicMock(return_value=expected)

        # THEN
        fit_func = analysis1d.as_fit_function()

        # EXPECT
        assert callable(fit_func)

        # THEN
        # call the fit function with some x values
        result = fit_func(x=[1, 2, 3])  # should be ignored

        # EXPECT
        analysis1d._calculate.assert_called_once()

        assert result is expected

    def test_get_all_variables(self, analysis1d):
        # WHEN
        extra_par1 = Parameter(name="extra_par1", value=1.0)
        extra_par2 = Parameter(name="extra_par2", value=2.0)
        analysis1d._extra_parameters = [extra_par1, extra_par2]

        # THEN
        variables = analysis1d.get_all_variables()

        # EXPECT
        assert isinstance(variables, list)
        sample_vars = analysis1d.sample_model.get_all_variables(
            Q_index=analysis1d.Q_index
        )
        instrument_vars = analysis1d.instrument_model.get_all_variables(
            Q_index=analysis1d.Q_index
        )
        extra_vars = [extra_par1, extra_par2]
        expected_vars = sample_vars + instrument_vars + extra_vars
        assert Counter(variables) == Counter(expected_vars)

    def test_plot_raises_if_no_data(self, analysis1d):
        analysis1d.experiment._data = None

        with pytest.raises(ValueError, match="No data"):
            analysis1d.plot_data_and_model()

    def test_plot_calls_plopp_with_correct_arguments(self, analysis1d):
        # WHEN

        # Mock the data and model components to be plotted
        fake_model = sc.DataArray(data=sc.array(dims=["energy"], values=[1, 2, 3]))
        analysis1d._create_sample_scipp_array = MagicMock(return_value=fake_model)

        fake_components = sc.Dataset(
            {
                "Component1": sc.DataArray(
                    data=sc.array(dims=["energy"], values=[0.1, 0.2, 0.3])
                )
            }
        )
        analysis1d._create_components_dataset_single_Q = MagicMock(
            return_value=fake_components
        )

        fake_fig = object()

        with patch("plopp.plot", return_value=fake_fig) as mock_plot:
            # THEN
            result = analysis1d.plot_data_and_model()

        # EXPECT

        # Ensure component dataset created
        analysis1d._create_components_dataset_single_Q.assert_called_once()

        # Ensure plot called
        mock_plot.assert_called_once()

        # Inspect arguments
        args, kwargs = mock_plot.call_args

        dataset_passed = args[0]

        assert "Data" in dataset_passed
        assert "Model" in dataset_passed
        assert "Component1" in dataset_passed

        assert result is fake_fig
