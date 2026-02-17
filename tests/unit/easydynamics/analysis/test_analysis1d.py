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
        energy = sc.array(dims=["energy"], values=[10.0, 20.0, 30.0], unit="meV")
        data = sc.array(
            dims=["Q", "energy"],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            variances=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        )

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

    #############
    # Private methods: small utilities
    #############

    def test_require_Q_index(self, analysis1d):
        # WHEN THEN
        Q_index = analysis1d._require_Q_index()

        # EXPECT
        assert Q_index == analysis1d.Q_index

    def test_require_Q_index_raises_if_no_Q_index(self, analysis1d):
        # WHEN THEN
        analysis1d._Q_index = None

        # EXPECT
        with pytest.raises(ValueError, match="Q_index must be set"):
            analysis1d._require_Q_index()

    def test_on_Q_index_changed(self, analysis1d):
        # WHEN
        analysis1d._create_convolver = MagicMock()

        # THEN
        analysis1d._on_Q_index_changed()

        # EXPECT
        analysis1d._create_convolver.assert_called_once()

    def test_extract_x_y_weights_from_experiment(self, analysis1d):
        # WHEN THEN
        x, y, weights = analysis1d._extract_x_y_weights_from_experiment()

        # EXPECT
        assert np.array_equal(x, analysis1d.experiment.energy.values)
        assert np.array_equal(y, analysis1d.experiment.data.values[analysis1d.Q_index])
        assert np.array_equal(
            weights, 1 / analysis1d.experiment.data.variances[analysis1d.Q_index] ** 0.5
        )

    #############
    # Private methods: evaluation
    #############

    #############
    # Private methods: create scipp arrays for plotting
    #############

    @pytest.mark.parametrize(
        "background",
        [
            None,
            np.array([0.5, 0.5, 0.5]),
        ],
        ids=[
            "No background",
            "With background",
        ],
    )
    def test_create_component_scipp_array(self, analysis1d, background):
        """Test that _create_component_scipp_array correctly evaluates
        the component, adds the background and calls _to_scipp_array
        with the correct values."""
        ""
        # WHEN

        # Mock the functions that will be called.
        analysis1d._evaluate_sample_component = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )

        analysis1d._to_scipp_array = MagicMock()

        component = object()

        # THEN
        analysis1d._create_component_scipp_array(
            component=component, background=background
        )

        # EXPECT
        analysis1d._evaluate_sample_component.assert_called_once_with(
            component=component
        )

        expected_values = np.array([1.0, 2.0, 3.0])
        if background is not None:
            expected_values += background

        analysis1d._to_scipp_array.assert_called_once()

        # Extract the actual call
        _, kwargs = analysis1d._to_scipp_array.call_args

        np.testing.assert_array_equal(
            kwargs["values"],
            expected_values,
        )

    def test_create_background_component_scipp_array(self, analysis1d):
        """Test that _create_background_component_scipp_array correctly
        evaluates the component, adds the background and calls
        _to_scipp_array with the correct values."""

        # WHEN

        # Mock the functions that will be called.
        analysis1d._evaluate_background_component = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )
        analysis1d._to_scipp_array = MagicMock()

        component = object()

        # THEN
        analysis1d._create_background_component_scipp_array(component=component)

        # EXPECT
        analysis1d._evaluate_background_component.assert_called_once_with(
            component=component
        )

        analysis1d._to_scipp_array.assert_called_once()

        # Extract the actual call
        _, kwargs = analysis1d._to_scipp_array.call_args

        np.testing.assert_array_equal(
            kwargs["values"],
            np.array([1.0, 2.0, 3.0]),
        )

    def test_create_sample_scipp_array(self, analysis1d):
        """Test that _create_sample_scipp_array correctly
        evaluates the full model and calls _to_scipp_array with the
        correct values."""

        # WHEN

        # Mock the functions that will be called.
        analysis1d._calculate = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        analysis1d._to_scipp_array = MagicMock()

        # THEN
        analysis1d._create_sample_scipp_array()

        # EXPECT
        analysis1d._calculate.assert_called_once()

        analysis1d._to_scipp_array.assert_called_once()

        # Extract the actual call
        _, kwargs = analysis1d._to_scipp_array.call_args

        np.testing.assert_array_equal(
            kwargs["values"],
            np.array([1.0, 2.0, 3.0]),
        )

    @pytest.mark.parametrize(
        "add_background",
        [True, False],
        ids=["With background", "Without background"],
    )
    def test_create_components_dataset_single_Q(
        self,
        analysis1d,
        add_background,
    ):
        """Test orchestration of _create_components_dataset_single_Q."""

        # WHEN

        # Choose a particular Q_index, but without using the setter to
        # avoid validation logic
        analysis1d._Q_index = 5

        # Mock all the things

        # ---- Sample component ----
        sample_component = MagicMock()
        sample_component.display_name = "sample_comp"

        sample_collection = MagicMock()
        sample_collection.components = [sample_component]

        analysis1d.sample_model.get_component_collection = MagicMock(
            return_value=sample_collection
        )

        # ---- Background component ----
        background_component = MagicMock()
        background_component.display_name = "background_comp"

        background_collection = MagicMock()
        background_collection.components = [background_component]

        analysis1d.instrument_model.background_model.get_component_collection = (
            MagicMock(return_value=background_collection)
        )

        # ---- Background evaluation ----
        background_value = np.array([10.0, 20.0, 30.0])
        analysis1d._evaluate_background = MagicMock(return_value=background_value)

        # ---- Return scipp DataArrays ----
        fake_sample_da = sc.DataArray(
            data=sc.array(dims=["energy"], values=[1.0, 2.0, 3.0])
        )

        analysis1d._create_component_scipp_array = MagicMock(
            return_value=fake_sample_da
        )

        fake_background_da = sc.DataArray(
            data=sc.array(dims=["energy"], values=[4.0, 5.0, 6.0])
        )

        analysis1d._create_background_component_scipp_array = MagicMock(
            return_value=fake_background_da
        )

        # THEN
        dataset = analysis1d._create_components_dataset_single_Q(
            add_background=add_background
        )

        # EXPECT

        # The correct component collections are requested with the
        # correct Q_index
        analysis1d.sample_model.get_component_collection.assert_called_once_with(
            Q_index=analysis1d.Q_index
        )

        analysis1d.instrument_model.background_model.get_component_collection.assert_called_once_with(
            Q_index=analysis1d.Q_index
        )

        # Background is evaluated if add_background=True, and not
        # evaluated if False
        if add_background:
            analysis1d._evaluate_background.assert_called_once()
            expected_background = background_value
        else:
            analysis1d._evaluate_background.assert_not_called()
            expected_background = None

        # The sample component scipp array is created with the correct
        # component and background
        analysis1d._create_component_scipp_array.assert_called_once()
        _, kwargs = analysis1d._create_component_scipp_array.call_args

        assert kwargs["component"] is sample_component

        if expected_background is None:
            assert kwargs["background"] is None
        else:
            np.testing.assert_array_equal(
                kwargs["background"],
                expected_background,
            )

        # Background component creation
        analysis1d._create_background_component_scipp_array.assert_called_once_with(
            component=background_component
        )

        # Dataset content
        assert isinstance(dataset, sc.Dataset)
        assert "sample_comp" in dataset
        assert "background_comp" in dataset

    def test_to_scipp_array(self, analysis1d):
        # WHEN
        numpy_array = np.array([1.0, 2.0, 3.0])

        # THEN
        scipp_array = analysis1d._to_scipp_array(numpy_array)

        # EXPECT
        assert isinstance(scipp_array, sc.DataArray)
        np.testing.assert_array_equal(scipp_array.values, numpy_array)

        np.testing.assert_array_equal(
            scipp_array.coords["energy"].values, analysis1d.experiment.energy.values
        )

        np.testing.assert_array_equal(
            scipp_array.coords["Q"].values,
            analysis1d.experiment.Q[analysis1d.Q_index].values,
        )
