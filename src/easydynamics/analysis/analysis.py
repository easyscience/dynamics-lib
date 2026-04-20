# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import scipp as sc
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.variable import Parameter
from plopp.backends.matplotlib.figure import InteractiveFigure
from scipp import UnitError

from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.analysis.analysis_base import AnalysisBase
from easydynamics.convolution.convolution_settings import ConvolutionSettings
from easydynamics.experiment import Experiment
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.utils.utils import _in_notebook


class Analysis(AnalysisBase):
    """
    For analysing two-dimensional data, i.e. intensity as function of energy and Q.

    Supports independent fits of each Q value and simultaneous fits of all Q.
    """

    def __init__(
        self,
        display_name: str | None = 'MyAnalysis',
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        convolution_settings: ConvolutionSettings | None = None,
        extra_parameters: Parameter | list[Parameter] | None = None,
    ) -> None:
        """
        Initialize an Analysis object.

        Parameters
        ----------
        display_name : str | None, default='MyAnalysis'
            Display name of the analysis.
        unique_name : str | None, default=None
            Unique name of the analysis. If None, a unique name is automatically generated.
        experiment : Experiment | None, default=None
            The Experiment associated with this Analysis. If None, a default Experiment is created.
        sample_model : SampleModel | None, default=None
            The SampleModel associated with this Analysis. If None, a default SampleModel is
            created.
        instrument_model : InstrumentModel | None, default=None
            The InstrumentModel associated with this Analysis. If None, a default InstrumentModel
            is created.
        convolution_settings : ConvolutionSettings | None, default=None
             The settings for the convolution. If None, default settings will be used.
        extra_parameters : Parameter | list[Parameter] | None, default=None
            Extra parameters to be included in the analysis for advanced users. If None, no extra
            parameters are added.
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
            convolution_settings=convolution_settings,
            extra_parameters=extra_parameters,
        )

        self._analysis_list = []
        if self.Q is not None:
            self._create_analysis_list()

        # Now we can allow updates to trigger recalculations
        self._call_updaters = True

    #############
    # Properties
    #############

    @property
    def analysis_list(self) -> list[Analysis1d]:
        """
        Get the Analysis1d objects associated with this Analysis.

        Returns
        -------
        list[Analysis1d]
            A list of Analysis1d objects, one for each Q index.
        """
        return self._analysis_list

    @analysis_list.setter
    def analysis_list(self, _value: list[Analysis1d]) -> None:
        """
        Analysis_list is read-only.

        To change the analysis list, modify the experiment, sample model, or instrument model.

        Parameters
        ----------
        _value : list[Analysis1d]
            The new list of Analysis1d objects. This argument is ignored, as analysis_list is
            read-only.

        Raises
        ------
        AttributeError
            Always raised, since analysis_list is read-only.
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
        """
        Calculate model data for a specific Q index.

        If Q_index is None, calculate for all Q indices and return a list of arrays.

        Parameters
        ----------
        Q_index : int | None, default=None
            Index of the Q value to calculate for. If None, calculate for all Q values.
        energy : sc.Variable | None, default=None
            The energy values to use for calculating the model. If None, uses the energy from the
            experiment.

        Returns
        -------
        list[np.ndarray] | np.ndarray
            If Q_index is None, returns a list of numpy arrays, one for each Q index. If Q_index is
            an integer, returns a single numpy array for that Q index.
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
        """
        Fit the model to the experimental data.

        Parameters
        ----------
        fit_method : str, default='independent'
            Method to use for fitting. Options are "independent" (fit each Q index independently,
            one after the other) or "simultaneous" (fit all Q indices simultaneously).
        Q_index : int | None, default=None
            If fit_method is "independent", specify which Q index to fit. If None, fit all Q
            indices independently. Ignored if fit_method is "simultaneous".

        Raises
        ------
        ValueError
            If fit_method is not "independent" or "simultaneous" or if there are no Q values
            available for fitting.

        Returns
        -------
        FitResults | list[FitResults]
            A list of FitResults if fitting independently, or a single FitResults object if fitting
            simultaneously.
        """

        if self.Q is None:
            raise ValueError(
                'No Q values available for fitting. Please check the experiment data.'
            )

        Q_index = self._verify_Q_index(Q_index)

        if fit_method == 'independent':
            if Q_index is not None:
                return self._fit_single_Q(Q_index)
            return self._fit_all_Q_independently()
        if fit_method == 'simultaneous':
            return self._fit_all_Q_simultaneously()
        raise ValueError("Invalid fit method. Choose 'independent' or 'simultaneous'.")

    def plot_data_and_model(
        self,
        Q_index: int | None = None,
        plot_components: bool = True,
        add_background: bool = True,
        energy: sc.Variable | None = None,
        **kwargs: dict[str, Any],
    ) -> InteractiveFigure:
        """
        Plot the experimental data and the model prediction.

        Optionally also plot the individual components of the model.

        Uses Plopp for plotting: https://scipp.github.io/plopp/

        Parameters
        ----------
        Q_index : int | None, default=None
            Index of the Q value to plot. If None, plot all Q values.
        plot_components : bool, default=True
            Whether to plot the individual components.
        add_background : bool, default=True
            Whether to add background components to the sample model components when plotting.
            Default is True.
        energy : sc.Variable | None, default=None
            The energy values to use for calculating the model. If None, uses the energy from the
            experiment.
        **kwargs : dict[str, Any]
            Additional keyword arguments passed to plopp for customizing the plot.

        Raises
        ------
        ValueError
            If Q_index is out of bounds, or if there is no data to plot, or if there are no Q
            values available for plotting.
        RuntimeError
            If not in a Jupyter notebook environment.
        TypeError
            If plot_components or add_background is not True or False.

        Returns
        -------
        InteractiveFigure
            A Plopp InteractiveFigure containing the plot of the data and model.
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
            for key in components:
                data_and_model[key] = components[key]
                plot_kwargs_defaults['linestyle'][key] = '--'
                plot_kwargs_defaults['marker'][key] = None

        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)

        fig = pp.slicer(
            data_and_model,
            **plot_kwargs_defaults,
        )
        for widget in fig.bottom_bar[0].controls.values():
            widget.slider_toggler.value = '-o-'

        return fig

    def parameters_to_dataset(self) -> sc.Dataset:
        """
        Creates a scipp dataset with copies of the Parameters in the model.

        Ensures unit consistency across Q.

        Raises
        ------
        UnitError
            If there are inconsistent units for the same parameter across different Q values.

        Returns
        -------
        sc.Dataset
            A dataset where each entry is a parameter, with dimensions "Q" and values corresponding
            to the parameter values.
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
                unit=units.get(name),
            )

        return ds

    def plot_parameters(
        self,
        names: str | list[str] | None = None,
        **kwargs: dict[str, Any],
    ) -> InteractiveFigure:
        """
        Plot fitted parameters as a function of Q.

        Parameters
        ----------
        names : str | list[str] | None, default=None
            Name(s) of the parameter(s) to plot. If None, plots all parameters.
        **kwargs : dict[str, Any]
            Additional keyword arguments passed to plopp.slicer for customizing the plot (e.g.,
            title, linestyle, marker, color).

        Raises
        ------
        TypeError
            If names is not a string, list of strings, or None.
        ValueError
            If any of the specified parameter names are not found in the dataset.

        Returns
        -------
        InteractiveFigure
            A Plopp InteractiveFigure containing the plot of the parameters.
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
            'linestyle': dict.fromkeys(names, 'none'),
            'marker': dict.fromkeys(names, 'o'),
            'markerfacecolor': dict.fromkeys(names, 'none'),
        }

        plot_kwargs_defaults.update(kwargs)

        import plopp as pp

        return pp.plot(
            data_to_plot,
            **plot_kwargs_defaults,
        )

    def fix_energy_offset(self, Q_index: int | None = None) -> None:
        """
        Fix the energy offset parameter(s) for a specific Q index, or for all Q indices if Q_index
        is None.

        Parameters
        ----------
        Q_index : int | None, default=None
            Index of the Q value to fix the energy offset for. If None, fixes the energy offset for
            all Q values.
        """
        if Q_index is not None:
            Q_index = self._verify_Q_index(Q_index)
            self.analysis_list[Q_index].fix_energy_offset()
        else:
            for analysis in self.analysis_list:
                analysis.fix_energy_offset()

    def free_energy_offset(self, Q_index: int | None = None) -> None:
        """
        Free the energy offset parameter(s) for a specific Q index, or for all Q indices if Q_index
        is None.

        Parameters
        ----------
        Q_index : int | None, default=None
            Index of the Q value to free the energy offset for. If None, frees the energy offset
            for all Q values.
        """
        if Q_index is not None:
            Q_index = self._verify_Q_index(Q_index)
            self.analysis_list[Q_index].free_energy_offset()
        else:
            for analysis in self.analysis_list:
                analysis.free_energy_offset()

    #############
    # Private methods - updating models when things change
    #############

    def _on_experiment_changed(self) -> None:
        """
        Update the Q values in the sample and instrument models when the experiment changes.

        Also update all the Analysis1d objects with the new experiment.
        """
        if self._call_updaters:
            super()._on_experiment_changed()
            for analysis in self.analysis_list:
                analysis.experiment = self.experiment

    def _on_sample_model_changed(self) -> None:
        """
        Update the Q values in the sample model when the sample model changes.

        Also update all the Analysis1d objects with the new sample model.
        """
        if self._call_updaters:
            super()._on_sample_model_changed()
            for analysis in self.analysis_list:
                analysis.sample_model = self.sample_model

    def _on_instrument_model_changed(self) -> None:
        """
        Update the Q values in the instrument model when the instrument model changes.

        Also update all the Analysis1d objects with the new instrument model.
        """
        if self._call_updaters:
            super()._on_instrument_model_changed()
            for analysis in self.analysis_list:
                analysis.instrument_model = self.instrument_model

    def _on_convolution_settings_changed(self) -> None:
        """
        Update the convolution settings in all Analysis1d objects when the convolution settings
        change.
        """
        if self._call_updaters:
            super()._on_convolution_settings_changed()
            for analysis1d in self.analysis_list:
                analysis1d.convolution_settings = self.convolution_settings

    def _create_analysis_list(self) -> None:
        """
        Create the list of Analysis1d objects, one for each Q index, based on the current
        experiment, sample model, and instrument model.
        """
        self._analysis_list = []
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
    # Private methods
    #############

    def _fit_single_Q(self, Q_index: int) -> FitResults:
        """
        Fit data for a single Q index.

        Parameters
        ----------
        Q_index : int
            Index of the Q value to fit.

        Returns
        -------
        FitResults
            The results of the fit for the specified Q index.
        """

        Q_index = self._verify_Q_index(Q_index)

        return self.analysis_list[Q_index].fit()

    def _fit_all_Q_independently(self) -> list[FitResults]:
        """
        Fit data for all Q indices independently.

        Returns
        -------
        list[FitResults]
            A list of FitResults, one for each Q index.
        """
        return [analysis.fit() for analysis in self.analysis_list]

    def _fit_all_Q_simultaneously(self) -> FitResults:
        """
        Fit data for all Q indices simultaneously.

        Returns
        -------
        FitResults
            The results of the simultaneous fit across all Q indices.
        """

        xs = []
        ys = []
        ws = []

        for analysis1d in self.analysis_list:
            x, y, weight, _ = self.experiment._extract_x_y_weights_only_finite(  # noqa: SLF001
                analysis1d.Q_index
            )
            xs.append(x)
            ys.append(y)
            ws.append(weight)

            # Make sure the convolver is up to date for this Q index
            analysis1d._convolver = analysis1d._create_convolver(  # noqa: SLF001
                energy=x
            )

        mf = MultiFitter(
            fit_objects=self.analysis_list,
            fit_functions=self.get_fit_functions(),
        )

        return mf.fit(
            x=xs,
            y=ys,
            weights=ws,
        )

    def get_fit_functions(self) -> list[callable]:
        """
        Get fit functions for all Q indices, which can be used for simultaneous fitting.

        Returns
        -------
        list[callable]
            A list of fit functions, one for each Q index.
        """
        return [analysis.as_fit_function() for analysis in self.analysis_list]

    def _create_model_array(self, energy: sc.Variable | None = None) -> sc.DataArray:
        """
        Create a scipp array for the model.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            The energy values to use for calculating the model. If None, uses the energy from the
            experiment.

        Returns
        -------
        sc.DataArray
            A DataArray containing the model values, with dimensions "Q" and "energy".
        """
        if energy is None:
            energy = self.energy
        model = sc.array(dims=['Q', 'energy'], values=self.calculate(energy=energy))
        return sc.DataArray(
            data=model,
            coords={'Q': self.Q, 'energy': energy},
        )

    def _create_components_dataset(
        self,
        add_background: bool = True,
        energy: sc.Variable | None = None,
    ) -> sc.Dataset:
        """
        Create a scipp dataset containing the individual components of the model for plotting.

        Parameters
        ----------
        add_background : bool, default=True
            Whether to add background components to the sample model components when creating the
            dataset.
        energy : sc.Variable | None, default=None
            The energy values to use for calculating the components. If None, uses the energy from
            the experiment.

        Raises
        ------
        TypeError
            If add_background is not True or False.

        Returns
        -------
        sc.Dataset
            A scipp Dataset where each entry is a component of the model, with dimensions "Q".
        """
        if not isinstance(add_background, bool):
            raise TypeError('add_background must be True or False.')

        if energy is None:
            energy = self.energy

        datasets = [
            analysis1d._create_components_dataset_single_Q(  # noqa: SLF001
                add_background=add_background, energy=energy
            )
            for analysis1d in self.analysis_list
        ]

        ds = sc.concat(datasets, dim='Q')
        return ds.assign_coords(Q=self.Q)

    #############
    # Dunder methods
    #############
