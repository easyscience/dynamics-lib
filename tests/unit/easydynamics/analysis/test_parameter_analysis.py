# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import scipp as sc

from easydynamics.analysis.parameter_analysis import ParameterAnalysis
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import Gaussian
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model import BrownianTranslationalDiffusion


class TestParameterAnalysis:
    @pytest.fixture
    def dataset(self):
        Q = sc.array(dims=["Q"], values=[0.1, 0.2])
        return sc.Dataset(
            data={
                "parameter1": sc.DataArray(
                    data=sc.array(
                        dims=["Q"], values=[1.0, 2.0], variances=[0.1, 0.2], unit="meV"
                    ),
                    coords={"Q": Q},
                ),
                "parameter2": sc.DataArray(
                    data=sc.array(
                        dims=["Q"],
                        values=[1.5, 2.5],
                        variances=[0.15, 0.25],
                        unit="1/meV",
                    ),
                    coords={"Q": Q},
                ),
                "parameter3 area": sc.DataArray(
                    data=sc.array(
                        dims=["Q"], values=[4.0, 5.0], variances=[0.3, 0.5], unit="meV"
                    ),
                    coords={"Q": Q},
                ),
                "parameter3 width": sc.DataArray(
                    data=sc.array(
                        dims=["Q"], values=[6.0, 7.0], variances=[0.6, 0.7], unit="meV"
                    ),
                    coords={"Q": Q},
                ),
            }
        )

    @pytest.fixture
    def parameter_analysis(self, dataset):
        func1 = Gaussian()
        func2 = Lorentzian()
        return ParameterAnalysis(
            parameters=dataset,
            fit_functions={"parameter1": func1, "parameter3 area": func2},
        )

    @pytest.fixture
    def parameter_analysis_diffusion(self, dataset):
        func = BrownianTranslationalDiffusion()
        return ParameterAnalysis(
            parameters=dataset,
            fit_functions={"parameter3": func},
            fit_settings={"parameter3": ["area", "width"]},
        )

    @pytest.fixture
    def parameter_analysis_diffusion_and_componentcollection(self, dataset):
        func1 = Gaussian()
        func2 = Lorentzian()
        funcs = ComponentCollection()
        funcs.append_component(func1)
        funcs.append_component(func2)

        func3 = BrownianTranslationalDiffusion()
        return ParameterAnalysis(
            parameters=dataset,
            fit_functions={"parameter1": funcs, "parameter3": func3},
            fit_settings={"parameter3": ["area", "width"]},
        )

    def test_parameter_analysis_initialization(self, parameter_analysis):
        # WHEN THEN EXPECT
        assert isinstance(parameter_analysis, ParameterAnalysis)

        # Parameters
        assert isinstance(parameter_analysis.parameters, sc.Dataset)
        assert set(parameter_analysis.parameters.keys()) == {
            "parameter1",
            "parameter2",
            "parameter3 area",
            "parameter3 width",
        }

        # Fit functions
        assert isinstance(parameter_analysis.fit_functions, dict)
        assert set(parameter_analysis.fit_functions.keys()) == {
            "parameter1",
            "parameter3 area",
        }

        # Fit settings default
        assert isinstance(parameter_analysis.fit_settings, dict)
        assert parameter_analysis.fit_settings == {}

        # Prepared fit data
        prepared = parameter_analysis._prepared_fit_data
        assert isinstance(prepared.fit_function_callables, list)
        assert isinstance(prepared.fit_objects, list)
        assert isinstance(prepared.fit_function_display_names, list)
        assert isinstance(prepared.parameter_names, list)
        assert isinstance(prepared.expanded_parameter_names, list)

        # Consistency checks
        n_funcs = len(prepared.fit_function_callables)
        assert len(prepared.fit_objects) == n_funcs
        assert len(prepared.fit_function_display_names) == n_funcs

        # Parameter names should match input keys
        assert prepared.parameter_names == ["parameter1", "parameter3 area"]

        # Expanded names should match what exists in dataset
        for name in prepared.expanded_parameter_names:
            assert name in parameter_analysis.parameters

    def test_parameter_property(self, parameter_analysis):
        # WHEN
        parameters = parameter_analysis.parameters

        # THEN EXPECT
        assert isinstance(parameters, sc.Dataset)
        assert set(parameters.keys()) == {
            "parameter1",
            "parameter2",
            "parameter3 area",
            "parameter3 width",
        }

        # WHEN
        Q = sc.array(dims=["Q"], values=[0.1, 0.2])
        new_data = sc.Dataset(
            data={
                "parameter4": sc.DataArray(
                    data=sc.array(
                        dims=["Q"],
                        values=[71.0, 12.0],
                        variances=[1.1, 2.2],
                        unit="meV",
                    ),
                    coords={"Q": Q},
                ),
                "parameter5": sc.DataArray(
                    data=sc.array(
                        dims=["Q"],
                        values=[8.5, 0.5],
                        variances=[2.15, 1.25],
                        unit="1/meV",
                    ),
                    coords={"Q": Q},
                ),
            }
        )

        # THEN
        parameter_analysis.parameters = new_data

        # EXPECT
        assert parameter_analysis.parameters is new_data
