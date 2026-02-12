# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import plopp as pp
import scipp as sc
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.variable import Parameter
from scipp import UnitError

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
        display_name: str = 'MyAnalysis',
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        extra_parameters: Parameter | list[Parameter] | None = None,
    ):

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            extra_parameters=extra_parameters,
        )

        if experiment is not None and not isinstance(experiment, Experiment):
            raise TypeError('experiment must be an instance of Experiment or None.')

        self._analysis_list = []
        if self.Q is not None:
            for Q_index in range(len(self.Q)):
                analysis = Analysis1d(
                    display_name=f'{self.display_name}_Q{Q_index}',
                    unique_name=(f'{self.unique_name}_Q{Q_index}'),
                    experiment=self.experiment,
                    sample_model=self.sample_model,
                    instrument_model=self.instrument_model,
                    extra_parameters=self._extra_parameters,
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
        """analysis_list is read-only.

        To change the analysis list, modify the experiment, sample
        model, or instrument model.
        """

        raise AttributeError(
            'analysis_list is read-only. '
            'To change the analysis list, modify the experiment, sample model, '
            'or instrument model.'
        )

    #############
    # Other methods
    #############
    def calculate(
        self,
        Q_index: int | None = None,
    ) -> list[np.ndarray] | np.ndarray:
        """Calculate model data for a specific Q index. If Q_index is
        None, calculate for all Q indices and return a list of arrays.

        Parameters: Q_index: Index of the Q value to calculate for. If
        None, calculate for all Q values.

        Returns: If Q_index is None, returns a list of numpy arrays, one
        for each Q index. If Q_index is an integer, returns a single
        numpy array for that Q index.
        """

        if Q_index is None:
            return [analysis.calculate() for analysis in self.analysis_list]

        Q_index = self._verify_Q_index(Q_index)
        return self.analysis_list[Q_index].calculate()

    def fit(
        self,
        fit_method: str = 'independent',
        Q_index: int | None = None,
    ) -> FitResults | list[FitResults]:
        """Fit the model to the experimental data.

        Parameters:
        ---------------
        fit_method: string, optional
            Method to use for fitting. Options are "independent" (fit
            each Q index independently, one after the other) or
            "simultaneous" (fit all Q indices simultaneously).
        Q_index: int or None, optional
            If fit_method is "independent", specify which Q index to
            fit. If None, fit all Q indices independently.

        Returns: Fit results, which may be a list of FitResults if
            fitting independently, or a single FitResults object if
            fitting simultaneously.
        """

        if self.Q is None:
            raise ValueError(
                'No Q values available for fitting. Please check the experiment data.'
            )

        Q_index = self._verify_Q_index(Q_index)

        if fit_method == 'independent':
            if Q_index is not None:
                return self._fit_single_Q(Q_index)
            else:
                return self._fit_all_Q_independently()
        elif fit_method == 'simultaneous':
            return self._fit_all_Q_simultaneously()
        else:
            raise ValueError("Invalid fit method. Choose 'independent' or 'simultaneous'.")

    def plot_data_and_model(
        self,
        Q_index: int | None = None,
        plot_components: bool = True,
        add_background: bool = True,
        **kwargs,
    ) -> None:
        """Plot the data and model using plopp."""

        if Q_index is not None:
            Q_index = self._verify_Q_index(Q_index)
            return self.analysis_list[Q_index].plot_data_and_model(
                plot_components=plot_components,
                add_background=add_background,
                **kwargs,
            )

        if self.experiment.binned_data is None:
            raise ValueError('No data to plot. Please load data first.')

        if not _in_notebook():
            raise RuntimeError('plot_data() can only be used in a Jupyter notebook environment.')

        if self.Q is None:
            raise ValueError(
                'No Q values available for plotting. Please check the experiment data.'
            )

        if not isinstance(plot_components, bool):
            raise TypeError('plot_components must be True or False.')

        if not isinstance(add_background, bool):
            raise TypeError('add_background must be True or False.')

        from IPython.display import display

        plot_kwargs_defaults = {
            'title': self.display_name,
            'linestyle': {'Data': 'none', 'Model': '-'},
            'marker': {'Data': 'o', 'Model': None},
            'color': {'Data': 'black', 'Model': 'red'},
            'markerfacecolor': {'Data': 'none', 'Model': 'none'},
        }
        data_and_model = {
            'Data': self.experiment.binned_data,
            'Model': self._create_model_array(),
        }

        if plot_components:
            components = self._create_components_dataset(add_background=add_background)
            for key in components.keys():
                data_and_model[key] = components[key]
                plot_kwargs_defaults['linestyle'][key] = '--'
                plot_kwargs_defaults['marker'][key] = None

        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)

        fig = pp.slicer(
            data_and_model,
            **plot_kwargs_defaults,
        )
        display(fig)

    def parameters_to_dataset(self) -> sc.Dataset:
        """Creates a scipp dataset with copies of the Parameters in the
        model.

        Ensures unit consistency across Q.
        """

        ds = sc.Dataset(coords={'Q': self.Q})

        # Collect all parameter names
        all_names = {
            param.name
            for analysis in self.analysis_list
            for param in analysis.get_all_parameters()
        }

        # Storage
        values = {name: [] for name in all_names}
        variances = {name: [] for name in all_names}
        units = {}

        for analysis in self.analysis_list:
            pars = {p.name: p for p in analysis.get_all_parameters()}

            for name in all_names:
                if name in pars:
                    p = pars[name]

                    # Unit consistency check
                    if name not in units:
                        units[name] = p.unit
                    elif units[name] != p.unit:
                        try:
                            p.unit.convert(units[name])
                        except Exception as e:
                            raise UnitError(
                                f"Inconsistent units for parameter '{name}': "
                                f'{units[name]} vs {p.unit}'
                            ) from e

                    values[name].append(p.value)
                    variances[name].append(p.variance)
                else:
                    values[name].append(np.nan)
                    variances[name].append(np.nan)

        # Build dataset variables
        for name in all_names:
            ds[name] = sc.Variable(
                dims=['Q'],
                values=np.asarray(values[name], dtype=float),
                variances=np.asarray(variances[name], dtype=float),
                unit=units.get(name, None),
            )

        return ds

    def plot_parameters(
        self,
        names: str | list[str] | None = None,
        **kwargs,
    ) -> None:
        """Plot fitted parameters as a function of Q.

        Parameters:
        ---------------
        names: str or list of str
            Name(s) of the parameter(s) to plot. If None, plots all
            parameters.
        kwargs: Additional keyword arguments passed to plopp.slicer for
            customizing the plot (e.g., title, linestyle, marker,
            color).

        Returns: A plopp figure.
        """

        ds = self.parameters_to_dataset()

        if not names:
            names = list(ds.keys())

        if isinstance(names, str):
            names = [names]

        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise TypeError('names must be a string or a list of strings.')

        for name in names:
            if name not in ds:
                raise ValueError(f"Parameter '{name}' not found in dataset.")

        data_to_plot = {name: ds[name] for name in names}
        plot_kwargs_defaults = {
            'linestyle': {name: 'none' for name in names},
            'marker': {name: 'o' for name in names},
            'markerfacecolor': {name: 'none' for name in names},
        }

        plot_kwargs_defaults.update(kwargs)
        fig = pp.plot(
            data_to_plot,
            **plot_kwargs_defaults,
        )
        return fig

    #############
    # Private methods
    #############

    def _fit_single_Q(self, Q_index: int) -> FitResults:
        """Fit data for a single Q index."""

        Q_index = self._verify_Q_index(Q_index)

        return self.analysis_list[Q_index].fit()

    def _fit_all_Q_independently(self) -> list[FitResults]:
        """Fit data for all Q indices independently."""
        return [analysis.fit() for analysis in self.analysis_list]

    def _fit_all_Q_simultaneously(self) -> FitResults:
        """Fit data for all Q indices simultaneously."""

        xs = []
        ys = []
        ws = []

        for analysis in self.analysis_list:
            data = analysis.experiment.data['Q', analysis.Q_index]

            x = data.coords['energy'].values
            y = data.values
            e = np.sqrt(data.variances)

            # Make sure the convolver is up to date for this Q index
            analysis._convolver = analysis._create_convolver()

            xs.append(x)
            ys.append(y)
            ws.append(1.0 / e)

        mf = MultiFitter(
            fit_objects=self.analysis_list,
            fit_functions=self.get_fit_functions(),
        )

        results = mf.fit(
            x=xs,
            y=ys,
            weights=ws,
        )
        return results

    def get_fit_functions(self) -> list[callable]:
        """Get fit functions for all Q indices, which can be used for
        simultaneous fitting.
        """
        return [analysis.as_fit_function() for analysis in self.analysis_list]

    def _create_model_array(self) -> sc.DataArray:
        """Create a scipp array for the model."""

        model = sc.array(dims=['Q', 'energy'], values=self.calculate())
        model_data_array = sc.DataArray(
            data=model,
            coords={'Q': self.Q, 'energy': self.experiment.energy},
        )
        return model_data_array

    def _create_components_dataset(self, add_background: bool = True) -> sc.Dataset:
        """Create a scipp dataset containing the individual components
        of the model for plotting.

        Parameters:
        ---------------
        add_background: bool, optional
            Whether to add background components to the sample model
            components. Default is True.

        Returns: A scipp Dataset where each variable is a component of
            the model, with dimensions "Q" and "energy".
        """
        if not isinstance(add_background, bool):
            raise TypeError('add_background must be True or False.')

        datasets = [
            analysis._create_components_dataset_single_Q(add_background=add_background)
            for analysis in self.analysis_list
        ]

        return sc.concat(datasets, dim='Q')

    #############
    # Dunder methods
    #############
