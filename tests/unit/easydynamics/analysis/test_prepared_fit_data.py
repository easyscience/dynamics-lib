# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import pytest

from easydynamics.analysis.prepared_fit_data import _PreparedFitData


class TestPreparedFitData:
    @pytest.fixture
    def dummy_callable(self):
        def f(x):
            return x

        return f

    @pytest.fixture
    def prepared_data(self, dummy_callable):
        return _PreparedFitData(
            fit_function_callables=[dummy_callable],
            fit_objects=[object()],
            fit_function_display_names=['f'],
            parameter_names=['p'],
            expanded_parameter_names=['p'],
        )

    def test_initialization(self, prepared_data, dummy_callable):
        assert isinstance(prepared_data.fit_function_callables, list)
        assert isinstance(prepared_data.fit_objects, list)
        assert isinstance(prepared_data.fit_function_display_names, list)
        assert isinstance(prepared_data.parameter_names, list)
        assert isinstance(prepared_data.expanded_parameter_names, list)

        assert prepared_data.fit_function_callables[0] is dummy_callable
        assert callable(prepared_data.fit_function_callables[0])
