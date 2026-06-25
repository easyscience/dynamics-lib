# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy
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
from easydynamics.experiment import Experiment
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model.instrument_model import InstrumentModel
from easydynamics.settings.convolution_settings import ConvolutionSettings
from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings
from easydynamics.utils.plotting import slicerplot_with_residuals
from easydynamics.utils.utils import _in_notebook


class Analysis(AnalysisBase):
    """
    For analysing two-dimensional data, i.e. intensity as function of energy and Q.

    Supports independent fits of each Q value and simultaneous fits of all Q.

    Examples
    --------
    **Fitting vanadium data for instrument calibration**

    The standard workflow builds a sample model, resolution model, background model, and instrument
    model, then combines them into an Analysis before fitting:
    ```python
    import pooch
    import easydynamics as edyn
    import easydynamics.sample_model as sm

    file_path = pooch.retrieve(
        url='https://github.com/easyscience/dynamics-lib/raw/refs/heads/master/docs/docs/tutorials/data/vanadium_data_example.h5',
        known_hash='16cc1b327c303feeb88fb9dda5390dc4880b62396b1793f98c6fef0b27c7b873',
    )
    experiment = edyn.Experiment('Vanadium')
    experiment.load_hdf5(filename=file_path)

    sample_model = sm.SampleModel(components=sm.DeltaFunction(area=1))
    resolution_model = sm.ResolutionModel(components=sm.Gaussian(width=0.1))
    background_model = sm.BackgroundModel(components=sm.Polynomial(coefficients=[0.001]))
    instrument_model = sm.InstrumentModel(
        resolution_model=resolution_model,
        background_model=background_model,
    )

    analysis = edyn.Analysis(
        display_name='Vanadium Analysis',
        experiment=experiment,
        sample_model=sample_model,
        instrument_model=instrument_model,
    )
    analysis.fit(fit_method='independent')
    analysis.plot_data_and_model()
    ```

    **Inspecting fitted parameters and fitting a single Q first**

    Use ``Q_index`` to fit and plot a single Q slice before fitting all Q:
    ```python
    analysis.fit(fit_method='independent', Q_index=5)
    analysis.plot_data_and_model(Q_index=5)

    analysis.fit(fit_method='independent')
    analysis.plot_parameters(names=['Gaussian width'])
    ```
    """

    def __init__(
        self,
        display_name: str | None = 'MyAnalysis',
        unique_name: str | None = None,
        experiment: Experiment | None = None,
        sample_model: SampleModel | None = None,
        instrument_model: InstrumentModel | None = None,
        convolution_settings: ConvolutionSettings | None = None,
        detailed_balance_settings: DetailedBalanceSettings | None = None,
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
        detailed_balance_settings : DetailedBalanceSettings | None, default=None
            The settings for detailed balance. If None, default settings will be used.
        extra_parameters : Parameter | list[Parameter] | None, default=None
            Extra parameters to be included in the analysis for advanced users. If None, no extra
            parameters are added.
        """

        self._analysis_list: list[Analysis1d] = []
        self._analysis_list_is_dirty = True
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            experiment=experiment,
            sample_model=sample_model,
            instrument_model=instrument_model,
            convolution_settings=convolution_settings,
            detailed_balance_settings=detailed_balance_settings,
            extra_parameters=extra_parameters,
        )

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
        self._ensure_analysis_list_current()
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
    def rebin(
        self,
        dimensions: dict[str, int | sc.Variable],
        confirm: bool = False,
    ) -> None:
        """
        Rebin the experiment data along specified dimensions and update the analysis.

        If Q values change (in count or magnitude), ``confirm=True`` is required. This clears Q
        from ``sample_model`` and ``instrument_model`` (including resolution and background
        sub-models) so they can accept the new Q values when the analysis list is next rebuilt.

        Parameters
        ----------
        dimensions : dict[str, int | sc.Variable]
            A dictionary mapping dimension names to number of bins (int) or bin edges
            (sc.Variable).
        confirm : bool, default=False
            Must be ``True`` when rebinning changes the Q values (count or magnitude), since this
            clears Q from all models. Raises ``ValueError`` otherwise.

        Raises
        ------
        ValueError
            If rebinning changes Q and ``confirm`` is not ``True``.
        """
        old_Q = np.asarray(self.Q.values) if self.Q is not None else None
        old_binned_data = self.experiment._binned_data  # noqa: SLF001

        self.experiment.rebin(dimensions)
        new_Q = np.asarray(self.Q.values) if self.Q is not None else None

        q_changed = (
            old_Q is not None
            and new_Q is not None
            and (len(old_Q) != len(new_Q) or not np.allclose(old_Q, new_Q))
        )

        if q_changed and not confirm:
            self.experiment._binned_data = old_binned_data  # noqa: SLF001
            raise ValueError(
                'Rebinning changed Q values, which requires clearing Q from sample_model and '
                'instrument_model (including resolution and background sub-models). '
                'Pass confirm=True to proceed.'
            )

        if q_changed:
            self.sample_model.clear_Q(confirm=True)
            self.instrument_model.clear_Q(confirm=True)

        self._analysis_list_is_dirty = True

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
        plot_residuals: bool = False,
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
        plot_residuals : bool, default=False
            Whether to plot the residuals (data - model). Default is False.
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
                plot_residuals=plot_residuals,
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

        self._verify_bool(plot_components, 'plot_components')
        self._verify_bool(add_background, 'add_background')
        self._verify_bool(plot_residuals, 'plot_residuals')

        if energy is None:
            energy = self.energy

        import plopp as pp

        data_and_model = self.data_and_model_to_datagroup(
            energy=energy,
            add_background=add_background,
            include_components=plot_components,
            include_residuals=plot_residuals,
        )

        plot_kwargs_defaults = self._build_plot_style_defaults(data_and_model)
        plot_kwargs_defaults['keep'] = 'energy'
        plot_kwargs_defaults.update(kwargs)

        if plot_residuals:
            fig = slicerplot_with_residuals(
                data_and_model,
                residuals_key='Residuals',
                operation='sum',
                **plot_kwargs_defaults,
            )

        else:
            fig = pp.slicer(
                data_and_model,
                **plot_kwargs_defaults,
            )
            for widget in fig.bottom_bar[0].controls.values():
                widget.slider_toggler.value = '-o-'
        fig.autoscale()
        return fig

    def data_and_model_to_datagroup(
        self,
        energy: sc.Variable | None = None,
        add_background: bool = True,
        include_components: bool = True,
        include_residuals: bool = False,
    ) -> sc.DataGroup:
        """
        Create a scipp DataGroup containing the experimental data, model calculation and optionally
        the individual components of the model.

        Parameters
        ----------
        energy : sc.Variable | None, default=None
            The energy values to use for calculating the model. If None, uses the energy from the
            experiment.
        add_background : bool, default=True
            Whether to add background components to the sample model components when creating the
            DataGroup.
        include_components : bool, default=True
            Whether to include the individual components of the model in the DataGroup. If False,
            only the total model will be included.
        include_residuals : bool, default=False
            Whether to include the residuals (data - model) in the DataGroup.

        Raises
        ------
        ValueError
            If there is no data to include in the DataGroup, or if there are no Q values available
            for creating the DataGroup.

        Returns
        -------
        sc.DataGroup
            A DataGroup containing the experimental data, model calculation, and optionally the
            individual components of the model.
        """
        if self.experiment.binned_data is None:
            raise ValueError('No data to include in DataGroup. Please load data first.')

        if self.Q is None:
            raise ValueError(
                'No Q values available for creating DataGroup. Please check the experiment data.'
            )

        self._verify_bool(add_background, 'add_background')
        self._verify_bool(include_components, 'include_components')
        self._verify_bool(include_residuals, 'include_residuals')

        energy = self._verify_energy(energy) if energy is not None else self.energy

        data_and_model = {
            'Data': self.experiment.binned_data,
            'Model': self._create_model_array(energy=energy),
        }

        if include_components:
            components = self._create_components_dataset(
                add_background=add_background, energy=energy
            )
            for key in components:
                data_and_model[key] = components[key]

        if include_residuals:
            data_and_model['Residuals'] = self._create_residuals_array()

        return sc.DataGroup(data_and_model)

    def parameters_to_dataset(self) -> sc.Dataset:
        """
        Creates a scipp dataset with copies of the Parameters in the model.

        Ensures unit consistency across Q.

        Raises
        ------
        UnitError
            If there are inconsistent units for the same parameter across different Q values.
        ValueError
            If duplicate parameter names exist for the same Q index.

        Returns
        -------
        sc.Dataset
            A dataset where each entry is a parameter, with dimensions "Q" and values corresponding
            to the parameter values.
        """

        ds = sc.Dataset(coords={'Q': self.Q})

        # Collect all parameter names in first-seen order
        all_names = dict.fromkeys(
            param.name
            for analysis in self.analysis_list
            for param in analysis.get_all_parameters()
        )

        # Storage
        values = {name: [] for name in all_names}
        variances = {name: [] for name in all_names}
        units = {}

        for analysis in self.analysis_list:
            all_params = analysis.get_all_parameters()
            param_names = [p.name for p in all_params]
            if len(param_names) != len(set(param_names)):
                dups = sorted({n for n in param_names if param_names.count(n) > 1})
                raise ValueError(
                    f'Duplicate parameter names at Q_index {analysis.Q_index}: {dups}. '
                    'Rename components so all parameters have unique names.'
                )
            pars = {p.name: p for p in all_params}

            for name in all_names:
                if name in pars:
                    p = pars[name]

                    # Unit consistency check
                    if name not in units:
                        units[name] = p.unit
                    elif units[name] != p.unit:
                        try:
                            p = copy(p)
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

        if names is None:
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
        """
        super()._on_experiment_changed()
        self._analysis_list_is_dirty = True

    def _on_sample_model_changed(self) -> None:
        """
        Update the Q values in the sample model when the sample model changes.
        """
        super()._on_sample_model_changed()
        self._analysis_list_is_dirty = True

    def _on_instrument_model_changed(self) -> None:
        """
        Update the Q values in the instrument model when the instrument model changes.
        """
        super()._on_instrument_model_changed()
        self._analysis_list_is_dirty = True

    def _on_convolution_settings_changed(self) -> None:
        """
        Update the convolution settings when they change.
        """
        super()._on_convolution_settings_changed()
        self._analysis_list_is_dirty = True

    def _ensure_analysis_list_current(self) -> None:
        """Rebuild the analysis list if any dependency has changed since it was last built."""
        if self._analysis_list_is_dirty and self.Q is not None:
            self._create_analysis_list()
            self._analysis_list_is_dirty = False

    def _create_analysis_list(self) -> None:
        """
        Create the list of Analysis1d objects, one for each Q index, based on the current
        experiment, sample model, and instrument model.
        """
        self._analysis_list = []
        for Q_index in range(len(self.Q)):
            # Each Analysis1d gets its own ConvolutionSettings so that
            # convolution_plan_is_valid is tracked independently per Q index.
            per_q_settings = copy(self.convolution_settings)
            analysis = Analysis1d(
                display_name=f'{self.display_name}_Q{Q_index}',
                experiment=self.experiment,
                sample_model=self.sample_model,
                instrument_model=self.instrument_model,
                convolution_settings=per_q_settings,
                detailed_balance_settings=self.detailed_balance_settings,
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
            x, y, weight, mask = self.experiment._extract_x_y_weights_only_finite(  # noqa: SLF001
                analysis1d.Q_index
            )
            xs.append(x)
            ys.append(y)
            ws.append(weight)

            # Slice the scipp energy Variable to finite points only.
            mask_sc = sc.array(dims=['energy'], values=mask)
            analysis1d.refresh_convolver(energy=self.experiment.energy[mask_sc])

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

    def _create_residuals_array(self) -> sc.DataArray:
        """
        Create a scipp array for the residuals (data - model).

        Returns
        -------
        sc.DataArray
            A DataArray containing the residuals, with dimensions "Q" and "energy".
        """
        data = self.experiment.binned_data
        model = self._create_model_array()
        return data.copy(deep=True) - model

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

        Returns
        -------
        sc.Dataset
            A scipp Dataset where each entry is a component of the model, with dimensions "Q".
        """
        self._verify_bool(add_background, 'add_background')

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

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'display_name={self.display_name!r}, '
            f'unique_name={self.unique_name!r}, '
            f'n_analyses={len(self._analysis_list)})'
        )
