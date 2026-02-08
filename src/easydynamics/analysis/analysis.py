# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.variable import Parameter

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.experiment import Experiment
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.instrument_model import InstrumentModel


class Analysis(AnalysisBase):
    """For analysing data."""

    def __init__(
        self,
        display_name: str = "MyAnalysis",
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        extra_parameters: (
            Parameter | list[Parameter] | list[list[Parameter]] | None
        ) = None,
    ):

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
        )

        if experiment is not None and not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an instance of Experiment or None.")

        self._analysis_list = []
        if self.Q is not None:
            for Q_index in range(len(self.Q)):
                analysis = Analysis1d(
                    display_name=f"{self.display_name}_Q{Q_index}",
                    unique_name=(
                        f"{self.unique_name}_Q{Q_index}" if self.unique_name else None
                    ),
                    experiment=self.experiment,
                    sample_model=self.sample_model,
                    instrument_model=self.instrument_model,
                    extra_parameters=extra_parameters,
                    Q_index=Q_index,
                )
                self._analysis_list.append(analysis)

    #############
    # Properties
    #############

    #############
    # Other methods
    #############
    def calculate(self, Q_index: int | None = None) -> list[np.ndarray]:
        """Calculate model data for a specific Q index."""

        if Q_index is None:
            result = []
            for analysis in self._analysis_list:
                result.append(analysis.calculate())
            return result

        if Q_index < 0 or Q_index >= len(self._analysis_list):
            raise IndexError("Q_index out of range.")

        return self._analysis_list[Q_index].calculate()

    #############
    # Private methods
    #############

    def _fit_single_Q(self, Q_index: int) -> None:
        """Fit data for a single Q index."""

        if Q_index < 0 or Q_index >= len(self._analysis_list):
            raise IndexError("Q_index out of range.")

        self._analysis_list[Q_index].fit()

    def _fit_all_Q_independently(self) -> None:
        """Fit data for all Q indices independently."""

        for analysis in self._analysis_list:
            analysis.fit()

    def _fit_all_Q_simultaneously(self) -> None:
        """Fit data for all Q indices simultaneously."""

        xs = []
        ys = []
        ws = []

        for analysis in self._analysis_list:
            data = analysis.experiment.data["Q", analysis.Q_index]

            x = data.coords["energy"].values
            y = data.values
            e = np.sqrt(data.variances)

            analysis._convolver = analysis._create_convolver(
                Q_index=analysis.Q_index,
                energy=x,
            )

            xs.append(x)
            ys.append(y)
            ws.append(1.0 / e)

        fit_functions = []

        for analysis in self._analysis_list:

            def make_fit_func(a):
                def fit_func(_):
                    return a._calculate()

                return fit_func

            fit_functions.append(make_fit_func(analysis))

        mf = MultiFitter(
            fit_objects=self._analysis_list,
            fit_functions=fit_functions,
        )

        results = mf.fit(
            x=xs,
            y=ys,
            weights=ws,
        )
        return results

    #############
    # Dunder methods
    #############
