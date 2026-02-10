# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import plopp as pp
import scipp as sc
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.variable import Parameter

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.experiment import Experiment
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.utils.utils import _in_notebook


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
                    unique_name=(f"{self.unique_name}_Q{Q_index}"),
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

    @property
    def analysis_list(self) -> list[Analysis1d]:
        """List of Analysis1d objects, one for each Q index."""
        return self._analysis_list

    @analysis_list.setter
    def analysis_list(self, value: list[Analysis1d]) -> None:
        """analysis_list is read-only. To change the analysis list,
        modify the experiment, sample model, or instrument model."""

        raise AttributeError(
            "analysis_list is read-only. "
            "To change the analysis list, modify the experiment, sample model, "
            "or instrument model."
        )

    #############
    # Other methods
    #############
    def calculate(self, Q_index: int | None = None) -> list[np.ndarray] | np.ndarray:
        """Calculate model data for a specific Q index.
        If Q_index is None, calculate for all Q indices and return a
        list of arrays.

        Parameters: Q_index: Index of the Q value to calculate for. If
        None, calculate for all Q values.

        Returns: If Q_index is None, returns a list of numpy arrays, one
        for each Q index. If Q_index is an integer, returns a single
        numpy array for that Q index.
        """

        if Q_index is None:
            return [analysis.calculate() for analysis in self.analysis_list]

        self._verify_Q_index(Q_index)
        return self.analysis_list[Q_index].calculate()

    def fit(self, fit_method: str = "independent", Q_index: int | None = None):
        """Fit the model to the experimental data.

        Parameters: fit_method: Method to use for fitting. Options are
        "independent" (fit each Q index independently, one after the
        other) or "simultaneous" (fit all Q indices simultaneously).
        Q_index: If fit_method is "sequential", specify which Q index to
        fit. If None, fit all Q indices independently.

        Returns: Fit results, which may be a list of FitResults if
        fitting independently, or a single FitResults object if fitting
        simultaneously.
        """

        if fit_method == "independent":
            if Q_index is not None:
                return self._fit_single_Q(Q_index)
            else:
                return self._fit_all_Q_independently()
        elif fit_method == "simultaneous":
            return self._fit_all_Q_simultaneously()
        else:
            raise ValueError(
                "Invalid fit method. Choose 'independent' or 'simultaneous'."
            )

    def plot_data_and_model(
        self,
        plot_components: bool = True,
        Q_index: int | None = None,
        **kwargs,
    ) -> None:
        """Plot the dataset using plopp."""

        if self.experiment.binned_data is None:
            raise ValueError("No data to plot. Please load data first.")

        if not _in_notebook():
            raise RuntimeError(
                "plot_data() can only be used in a Jupyter notebook environment."
            )
        from IPython.display import display

        plot_kwargs_defaults = {
            "title": self.display_name,
            "linestyle": {"Data": "none", "Model": "-"},
            "marker": {"Data": "o", "Model": None},
            "color": {"Data": "black", "Model": "red"},
        }
        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)
        data_and_model = {
            "Data": self.experiment.binned_data,
            "Model": self._create_model_scipp_array(),
        }

        if plot_components:
            components_da, background_da = (
                self._create_components_and_background_scipp_arrays()
            )

            data_and_model["Background"] = background_da
            plot_kwargs_defaults["linestyle"]["Background"] = "--"
            plot_kwargs_defaults["marker"]["Background"] = None

            for icomp in range(components_da.sizes["component"]):
                Q_index = 0
                comp_name = (
                    self.sample_model.get_component_collection(Q_index)
                    .components[icomp]
                    .display_name
                )
                data_and_model[comp_name] = components_da["component", icomp]
                plot_kwargs_defaults["linestyle"][comp_name] = "--"
                plot_kwargs_defaults["marker"][comp_name] = None

        fig = pp.slicer(
            data_and_model,
            **plot_kwargs_defaults,
        )
        display(fig)

    #############
    # Private methods
    #############

    def _fit_single_Q(self, Q_index: int):
        """Fit data for a single Q index."""

        self._verify_Q_index(Q_index)

        return self.analysis_list[Q_index].fit()

    def _fit_all_Q_independently(self):
        """Fit data for all Q indices independently."""
        return [analysis.fit() for analysis in self.analysis_list]

    def _fit_all_Q_simultaneously(self):
        """Fit data for all Q indices simultaneously."""

        xs = []
        ys = []
        ws = []

        for analysis in self.analysis_list:
            data = analysis.experiment.data["Q", analysis.Q_index]

            x = data.coords["energy"].values
            y = data.values
            e = np.sqrt(data.variances)

            # Make sure the convolver is up to date for this Q index
            analysis._convolver = analysis._create_convolver(
                Q_index=analysis.Q_index,
                energy=x,
            )

            xs.append(x)
            ys.append(y)
            ws.append(1.0 / e)

        fit_functions = []
        for analysis in self.analysis_list:
            # Use the private method to avoid excessive checks
            def make_fit_func(a):
                def fit_func(_):
                    return a._calculate()

                return fit_func

            fit_functions.append(make_fit_func(analysis))

        mf = MultiFitter(
            fit_objects=self.analysis_list,
            fit_functions=fit_functions,
        )

        results = mf.fit(
            x=xs,
            y=ys,
            weights=ws,
        )
        return results

    def _verify_Q_index(self, Q_index: int) -> None:
        """Verify that the provided Q_index is valid."""
        if not isinstance(Q_index, int):
            raise TypeError("Q_index must be an integer.")
        if Q_index < 0 or Q_index >= len(self.analysis_list):
            raise IndexError("Q_index out of range.")

    def _create_model_scipp_array(self) -> sc.DataArray:
        """Create a scipp array for the model"""

        model = sc.array(dims=["Q", "energy"], values=self.calculate())
        model_data_array = sc.DataArray(
            data=model,
            coords={"Q": self.Q, "energy": self.experiment.energy},
        )
        return model_data_array

    def _create_components_and_background_scipp_arrays(
        self,
    ) -> tuple[sc.DataArray, sc.DataArray]:
        """
        Create:
        1) A DataArray with sample components + background
            dims = (component, Q, energy)
        2) A DataArray with summed background
            dims = (Q, energy)
        """

        component_values = None  # List[List[np.ndarray]]
        background_values = []  # List[np.ndarray]

        for analysis in self.analysis_list:
            sample_comps_q, background_comps_q = (
                analysis.calculate_individual_components()
            )

            # (energy,)
            background_sum_q = sum(background_comps_q)
            background_values.append(background_sum_q)

            if component_values is None:
                component_values = [[] for _ in range(len(sample_comps_q))]

            for icomp, sample_comp_q in enumerate(sample_comps_q):
                component_values[icomp].append(sample_comp_q + background_sum_q)

        # Sample components DataArray
        components_array = sc.array(
            dims=["component", "Q", "energy"],
            values=component_values,
        )

        components_da = sc.DataArray(
            data=components_array,
            coords={
                "component": sc.arange("component", len(component_values)),
                "Q": self.Q,
                "energy": self.experiment.energy,
            },
        )

        # Background-only DataArray
        background_array = sc.array(
            dims=["Q", "energy"],
            values=background_values,
        )

        background_da = sc.DataArray(
            data=background_array,
            coords={
                "Q": self.Q,
                "energy": self.experiment.energy,
            },
        )

        return components_da, background_da

    #############
    # Dunder methods
    #############
