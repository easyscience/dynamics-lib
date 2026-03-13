# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipp as sc
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.variable import Parameter
from plopp.backends.matplotlib.figure import InteractiveFigure
from scipp import UnitError

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.experiment import Experiment
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.utils.utils import _in_notebook


class Analysis(AnalysisBase):
    """For analysing two-dimensional data, i.e. intensity as function of
    energy and Q. Supports independent fits of each Q value and
    simultaneous fits of all Q.

    Args:
        display_name (str): Display name of the analysis.
        unique_name (str or None): Unique name of the analysis. If None,
            a unique name is automatically generated.
        experiment (Experiment | None): The Experiment associated with
            this Analysis. If None, a default Experiment is created.
        sample_model (SampleModel | None): The SampleModel associated
            with this Analysis. If None, a default SampleModel is
            created.
        instrument_model (InstrumentModel | None): The InstrumentModel
            associated with this Analysis. If None, a default
            InstrumentModel is created.
        extra_parameters (Parameter | list[Parameter] | None): Extra
            parameters to be included in the analysis for advanced
            users. If None, no extra parameters are added.

    Attributes:
        experiment (Experiment): The Experiment associated with this
            Analysis.
        sample_model (SampleModel): The SampleModel associated with this
            Analysis.
        instrument_model (InstrumentModel): The InstrumentModel
            associated with this Analysis.
        Q (sc.Variable | None): The Q values from the associated
            Experiment, if available.
        energy (sc.Variable | None): The energy values from the
            associated Experiment, if available.
        temperature (Parameter | None): The temperature from the
            associated SampleModel, if available.
        extra_parameters (list[Parameter]): The extra parameters
            included in this Analysis.
    """

    def __init__(
        self,
        display_name: str = 'MyAnalysis',
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        extra_parameters: Parameter | list[Parameter] | None = None,
    ) -> None:
        """Initialize an Analysis object.

        Args:
            display_name (str): Display name of the analysis.
            unique_name (str or None): Unique name of the analysis. If
                None, a unique name is automatically generated.
            experiment (Experiment | None): The Experiment associated
                with this Analysis. If None, a default Experiment is
                created.
            sample_model (SampleModel | None): The SampleModel
                associated with this Analysis. If None, a default
                SampleModel is created.
            instrument_model (InstrumentModel | None): The
                InstrumentModel associated with this Analysis. If None,
                a default InstrumentModel is created.
            extra_parameters (Parameter | list[Parameter] | None): Extra
                parameters to be included in the analysis for advanced
                users. If None, no extra parameters are added.
        """

        # Avoid triggering updates before the object is fully
        # initialized
        self._call_updaters = False
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            extra_parameters=extra_parameters,
        )

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
        # Now we can allow updates to trigger recalculations
        self._call_updaters = True

    #############
    # Properties
    #############

    @property
    def analysis_list(self) -> list[Analysis1d]:
        """Get the Analysis1d objects associated with this Analysis.

        Returns:
            list[Analysis1d]: A list of Analysis1d objects, one for
                each Q index.
        """
        return self._analysis_list

    @analysis_list.setter
    def analysis_list(self, value: list[Analysis1d]) -> None:
        """analysis_list is read-only.

        To change the analysis list, modify the experiment, sample
        model, or instrument model.

        Raises:
            AttributeError: Always raised, since analysis_list is
                read-only.
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
        energy: sc.Variable | None = None,
    ) -> list[np.ndarray] | np.ndarray:
        """Calculate model data for a specific Q index. If Q_index is
        None, calculate for all Q indices and return a list of arrays.

        Args:
            Q_index (int or None): Index of the Q value to calculate
                for. If None, calculate for all Q values.

        Returns:
            list[np.ndarray] | np.ndarray: If Q_index is None, returns
                a list of numpy arrays, one for each Q index.
                If Q_index is an integer, returns a single numpy array
                for that Q index.

        Raises:
            IndexError: If Q_index is not None and is out of bounds.
        """
        if energy is None:
            energy = self.energy

        if Q_index is None:
            return [analysis.calculate(energy=energy) for analysis in self.analysis_list]

        Q_index = self._verify_Q_index(Q_index)
        return self.analysis_list[Q_index].calculate(energy=energy)

    def fit(
        self,
        fit_method: str = 'independent',
        Q_index: int | None = None,
    ) -> FitResults | list[FitResults]:
        """Fit the model to the experimental data.

        Args:
            fit_method (str): Method to use for fitting. Options are
                "independent" (fit each Q index independently, one after
                the other) or "simultaneous" (fit all Q indices
                simultaneously). Default is "independent".
            Q_index (int or None): If fit_method is "independent",
                specify which Q index to fit. If None, fit all Q indices
                independently. Ignored if fit_method is "simultaneous".
                Default is None.

        Returns:
            FitResults: a list of FitResults if fitting independently,
                or a single FitResults object if fitting simultaneously.

        Raises:
            ValueError: If fit_method is not "independent" or
                "simultaneous"
            IndexError: If fit_method is "independent" and Q_index is
                out of bounds.
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
        energy: sc.Variable | None = None,
        **kwargs,
    ) -> InteractiveFigure:
        """Plot the experimental data and the model prediction.
        Optionally also plot the individual components of the model.

        Uses Plopp for plotting: https://scipp.github.io/plopp/

        Args:
            Q_index (int or None): Index of the Q value to plot. If
                None, plot all Q values. Default is None.
            plot_components (bool): Whether to plot the individual
                components. Default is True.
            add_background (bool): Whether to add background components
                to the sample model components when plotting. Default is
                True.
            **kwargs (Any): Additional keyword arguments passed to plopp
                for customizing the plot.

        Raises:
            ValueError: If Q_index is out of bounds, or if there is no
                data to plot, or if there are no Q values available for
                plotting.
            RuntimeError: If not in a Jupyter notebook environment.
            TypeError: If plot_components or add_background is not True
                or False.

        Returns:
            InteractiveFigure: A Plopp InteractiveFigure containing the
                plot of the data and model.
        """

        if Q_index is not None:
            Q_index = self._verify_Q_index(Q_index)
            return self.analysis_list[Q_index].plot_data_and_model(
                plot_components=plot_components,
                add_background=add_background,
                energy=energy,
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

        if energy is None:
            energy = self.energy

        import plopp as pp

        plot_kwargs_defaults = {
            'title': self.display_name,
            'linestyle': {'Data': 'none', 'Model': '-'},
            'marker': {'Data': 'o', 'Model': None},
            'color': {'Data': 'black', 'Model': 'red'},
            'markerfacecolor': {'Data': 'none', 'Model': 'none'},
            'keep': 'energy',
        }
        data_and_model = {
            'Data': self.experiment.binned_data,
            'Model': self._create_model_array(energy=energy),
        }

        if plot_components:
            components = self._create_components_dataset(
                add_background=add_background, energy=energy
            )
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
        return fig

    def parameters_to_dataset(self) -> sc.Dataset:
        """Creates a scipp dataset with copies of the Parameters in the
        model.

        Ensures unit consistency across Q.

        Returns:
            sc.Dataset: A dataset where each entry is a parameter, with
            dimensions "Q" and values corresponding to the parameter
            values.

        Raises:
            UnitError: If there are inconsistent units for the same
                parameter across different Q values.
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
                            p.convert_unit(units[name])
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
    ) -> InteractiveFigure:
        """Plot fitted parameters as a function of Q.

        Args:
            names (str | list[str] | None): Name(s) of the parameter(s)
                to plot. If None, plots all parameters.
            kwargs (Any): Additional keyword arguments passed to
                plopp.slicer for customizing the plot (e.g., title,
                linestyle, marker, color).

        Returns:
            InteractiveFigure: A Plopp InteractiveFigure containing the
                plot of the parameters.
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

        import plopp as pp

        fig = pp.plot(
            data_to_plot,
            **plot_kwargs_defaults,
        )
        return fig

    #############
    # Private methods - updating models when things change
    #############

    def _on_experiment_changed(self) -> None:
        """Update the Q values in the sample and instrument models when
        the experiment changes.

        Also update all the Analysi1d objects with the new experiment.
        """
        if self._call_updaters:
            super()._on_experiment_changed()
            for analysis in self.analysis_list:
                analysis.experiment = self.experiment

    def _on_sample_model_changed(self) -> None:
        """Update the Q values in the sample model when the sample model
        changes.

        Also update all the Analysi1d objects with the new sample model.
        """
        if self._call_updaters:
            super()._on_sample_model_changed()
            for analysis in self.analysis_list:
                analysis.sample_model = self.sample_model

    def _on_instrument_model_changed(self) -> None:
        """Update the Q values in the instrument model when the
        instrument model changes.

        Also update all the Analysi1d objects with the new instrument
        model.
        """
        if self._call_updaters:
            super()._on_instrument_model_changed()
            for analysis in self.analysis_list:
                analysis.instrument_model = self.instrument_model

    #############
    # Private methods
    #############

    def _fit_single_Q(self, Q_index: int) -> FitResults:
        """Fit data for a single Q index.

        Args:
            Q_index (int): Index of the Q value to fit.

        Returns:
            FitResults: The results of the fit for the specified
                Q index.
        """

        Q_index = self._verify_Q_index(Q_index)

        return self.analysis_list[Q_index].fit()

    def _fit_all_Q_independently(self) -> list[FitResults]:
        """Fit data for all Q indices independently.

        Returns:
            list[FitResults]: A list of FitResults, one for each Q
                index.
        """
        return [analysis.fit() for analysis in self.analysis_list]

    def _fit_all_Q_simultaneously(self) -> FitResults:
        """Fit data for all Q indices simultaneously.

        Returns:
            FitResults: The results of the simultaneous fit across all
                Q indices.
        """

        xs = []
        ys = []
        ws = []

        for analysis in self.analysis_list:
            x, y, weight, _ = self.experiment._extract_x_y_weights_only_finite(analysis.Q_index)
            xs.append(x)
            ys.append(y)
            ws.append(weight)

            # Make sure the convolver is up to date for this Q index
            analysis._convolver = analysis._create_convolver(energy=x)

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

        Returns:
            list[callable]: A list of fit functions, one for each
                Q index.
        """
        return [analysis.as_fit_function() for analysis in self.analysis_list]

    def _create_model_array(self, energy: sc.Variable | None = None) -> sc.DataArray:
        """Create a scipp array for the model.

        Returns:
            sc.DataArray: A DataArray containing the model values, with
                dimensions "Q" and "energy".
        """
        if energy is None:
            energy = self.energy
        model = sc.array(dims=['Q', 'energy'], values=self.calculate(energy=energy))
        model_data_array = sc.DataArray(
            data=model,
            coords={'Q': self.Q, 'energy': energy},
        )
        return model_data_array

    def _create_components_dataset(
        self,
        add_background: bool = True,
        energy: sc.Variable | None = None,
    ) -> sc.Dataset:
        """Create a scipp dataset containing the individual components
        of the model for plotting.

        Args:
            add_background (bool): Whether to add background components
                to the sample model components when creating the
                dataset. Default is True.

        Raises:
            TypeError: If add_background is not True or False.

        Returns:
            sc.Dataset: A scipp Dataset where each entry is a component
                of the model, with dimensions "Q".
        """
        if not isinstance(add_background, bool):
            raise TypeError('add_background must be True or False.')

        if energy is None:
            energy = self.energy

        datasets = [
            analysis._create_components_dataset_single_Q(
                add_background=add_background, energy=energy
            )
            for analysis in self.analysis_list
        ]

        return sc.concat(datasets, dim='Q')

    #############
    # Dunder methods
    #############
