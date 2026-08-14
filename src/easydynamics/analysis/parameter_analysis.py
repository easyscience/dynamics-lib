# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Any

import numpy as np
import plopp as pp
import scipp as sc
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.fitting.multi_fitter import MultiFitter
from easyscience.variable import Parameter
from matplotlib import rcParams
from plopp.backends.matplotlib.figure import InteractiveFigure

from easydynamics.analysis.analysis import Analysis
from easydynamics.analysis.fit_binding import FitBinding
from easydynamics.analysis.posterior_labels import ParameterLabels
from easydynamics.analysis.posterior_sampling import PosteriorSampler
from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.utils.fit_target import FitTarget
from easydynamics.utils.utils import _in_notebook
from easydynamics.utils.utils import convert_value_unit


class ParameterAnalysis(EasyDynamicsModelBase):
    """
    For analysing fitted parameters.

    Can be used to fit parameters to ModelComponents, ComponentCollections, or DiffusionModelBase
    objects, and to plot the parameters and fit results. The parameters to be analyzed can be
    provided as a sc.Dataset or directly as an Analysis object. Multiple parameters can be fitted
    simultaneously, and each binding maps its model's predictions onto the dataset keys they are
    fitted against (for diffusion models e.g. 'area', 'width', or 'delta_area').

    Examples
    --------
    **Fitting Lorentzian widths to a diffusion model**

    After a full Analysis fit, pass the Analysis directly and bind the model's predictions to
    dataset keys using a ``FitBinding``:
    ```python
    import easydynamics as edyn

    # analysis is an edyn.Analysis object with previously fitted parameters
    diffusion_model = edyn.BrownianTranslationalDiffusion(diffusion_coefficient=2.4e-9, scale=0.5)
    binding = edyn.FitBinding(
        model=diffusion_model,
        targets={'width': 'Lorentzian width'},
    )

    param_analysis = edyn.ParameterAnalysis(
        parameters=analysis,
        bindings=binding,
    )
    param_analysis.fit()
    param_analysis.plot()
    ```

    **Fitting multiple parameters with separate bindings**

    Component models declare the units their evaluate expects: here the Polynomial's x is the
    dataset's Q coordinate and its y is the fitted parameter, so construct it with matching units
    (or pass ``x_unit=None`` / ``y_unit=None`` to fit raw values):
    ```python
    area_binding = edyn.FitBinding(
        model=edyn.Polynomial(coefficients=[0.5, 0.0], x_unit='1/angstrom', y_unit='meV'),
        targets='Lorentzian area',
    )
    param_analysis = edyn.ParameterAnalysis(
        parameters=analysis,
        bindings=[binding, area_binding],
    )
    param_analysis.fit()
    ```
    """

    def __init__(
        self,
        parameters: sc.Dataset | Analysis | None = None,
        bindings: FitBinding | list[FitBinding] | None = None,
        display_name: str | None = 'ParameterAnalysis',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the ParameterAnalysis.

        Parameters
        ----------
        parameters : sc.Dataset | Analysis | None, default=None
            The parameters to analyze. Can be provided as a sc.Dataset or as an Analysis (in which
            case the parameters will be extracted from the Analysis).
        bindings : FitBinding | list[FitBinding] | None, default=None
            The fit bindings to use for fitting the parameters. Can be a single FitBinding or a
            list of FitBindings. If None, no fit bindings are provided.
        display_name : str | None, default='ParameterAnalysis'
            Display name of the analysis.
        unique_name : str | None, default=None
            Unique name of the analysis. If None, a unique name is automatically generated. By
            default, None.
        """

        self._fitter = None
        self._fitter_is_dirty = True
        self._bayesian = None
        # Which targets the cached fitter was built for, so an in-place edit of a FitBinding is
        # noticed even though it cannot be observed directly.
        self._fitter_targets = None

        super().__init__(display_name=display_name, unique_name=unique_name)

        self._parameters = self._verify_parameters(parameters)

        self._bindings = self._verify_bindings(bindings)

    #############
    # Properties
    #############
    @property
    def parameters(self) -> sc.Dataset | None:
        """
        Get the parameters for the parameter analysis.

        Returns
        -------
        sc.Dataset | None
            The parameters for the parameter analysis.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: sc.Dataset | Analysis | None) -> None:
        """
        Set the parameter dataset for the parameter analysis.

        Parameters
        ----------
        value : sc.Dataset | Analysis | None
            The new parameter dataset for the parameter analysis.
        """
        self._parameters = self._verify_parameters(value)
        self._invalidate_fitter()

    @property
    def bindings(self) -> list[FitBinding]:
        """
        Get the fit bindings for the parameter analysis.

        Returns
        -------
        list[FitBinding]
            The fit bindings for the parameter analysis.
        """
        return self._bindings

    @bindings.setter
    def bindings(self, value: FitBinding | list[FitBinding] | None) -> None:
        """
        Set the fit bindings for the parameter analysis.

        Parameters
        ----------
        value : FitBinding | list[FitBinding] | None
            The new fit bindings for the parameter analysis.
        """
        self._bindings = self._verify_bindings(value)
        self._invalidate_fitter()

    @property
    def fitter(self) -> MultiFitter:
        """
        The EasyScience MultiFitter over the binding models, built on first use.

        Returns
        -------
        MultiFitter
            The cached MultiFitter.
        """
        if self._fitter_is_dirty or self._fitter is None:
            self._fitter = self._build_fitter()
            self._fitter_is_dirty = False
        return self._fitter

    @property
    def bayesian(self) -> PosteriorSampler:
        """
        Bayesian posterior sampling for this analysis, created on first use.

        Returns
        -------
        PosteriorSampler
            The sampler, which holds any chain that has been run.
        """
        if self._bayesian is None:
            self._bayesian = PosteriorSampler(
                analysis=self,
                sampling_data=self._sampling_data,
                chain_parameters=self._chain_parameters,
                parameter_labels=self._parameter_labels,
            )
        return self._bayesian

    def _invalidate_fitter(self) -> None:
        """Mark the MultiFitter, and the Sampler built from it, as needing a rebuild."""
        self._fitter_is_dirty = True
        if self._bayesian is not None:
            self._bayesian.invalidate()

    def _parameter_labels(self) -> ParameterLabels:
        """
        Get labels for the chain's parameters, qualified by binding model where needed.

        Two bindings can use models of the same kind, whose parameters would then share a name. The
        prefix is the model's name, matching the choice to report parameters under their name
        rather than their display name, since for several models the display name is just the class
        name. If two models share a name as well, the unique name is used: a label that does not
        disambiguate is worse than a long one.

        Returns
        -------
        ParameterLabels
            Labels over the free parameters of the binding models.
        """
        models = {binding.model.unique_name: binding.model for binding in self.bindings}
        owners = {}
        for model in models.values():
            for parameter in model.get_free_parameters():
                owners.setdefault(parameter.unique_name, model)
        model_names = [getattr(m, 'name', None) or m.display_name for m in models.values()]

        def qualify(parameter: Parameter) -> str | None:
            """
            Get the model name a parameter belongs to.

            Parameters
            ----------
            parameter : Parameter
                The parameter to qualify.

            Returns
            -------
            str | None
                The owning model's name, its unique name if that name is shared, or None if the
                parameter belongs to no binding model.
            """
            owner = owners.get(parameter.unique_name)
            if owner is None:
                return None
            name = getattr(owner, 'name', None) or owner.display_name
            if name is None or model_names.count(name) > 1:
                return owner.unique_name
            return name

        return ParameterLabels(self._chain_parameters(), qualify=qualify)

    #############
    # Other methods
    #############

    def fit(self) -> FitResults:
        """
        Fit the parameters using the specified fit functions and settings.

        A ``ValueError`` is raised if no parameters Dataset is provided, if no fit bindings are
        provided, or if a binding names a dataset key that is not in the parameters Dataset.

        Returns
        -------
        FitResults
            The results of the fit
        """

        xs, ys, ws, _, models = self._build_fit_inputs()
        self._invalidate_fitter_if_targets_changed(models)
        return self.fitter.fit(x=xs, y=ys, weights=ws)

    def _build_fit_inputs(self) -> tuple[list, list, list, list, list]:
        """
        Resolve every binding into the per-target data, fit functions, and models.

        Shared by fitting and sampling so that both see exactly the same targets, in the same
        order, with the same unit conversions applied.

        Returns
        -------
        tuple[list, list, list, list, list]
            The ``(x, y, weights, functions, models)`` lists, one entry per fit target.

        Raises
        ------
        ValueError
            If no parameters Dataset is provided, if no fit bindings are provided, or if a binding
            names a dataset key that is not in the parameters Dataset.
        """
        if self.parameters is None:
            raise ValueError('No parameters Dataset provided.')

        if not self.bindings:
            raise ValueError('No fit bindings provided.')

        xs = []
        ys = []
        ws = []
        funcs, models = [], []

        for binding in self.bindings:
            for target in binding.get_targets():
                if target.dataset_key not in self.parameters:
                    raise ValueError(
                        f"Parameter '{target.dataset_key}' from binding "
                        f"'{binding.unique_name}' not found in parameters Dataset."
                    )

                x, y, weight = self._get_xyweight_from_dataset(target.dataset_key)
                x_factor, y_factor = self._get_unit_conversions(target)
                x = x * x_factor
                y = y * y_factor
                weight = weight / y_factor

                xs.append(x)
                ys.append(y)
                ws.append(weight)

                funcs.append(target.function)
                models.append(binding.model)

        return xs, ys, ws, funcs, models

    #############
    # Hooks for BayesianSamplingMixin
    #############

    def _build_fitter(self) -> MultiFitter:
        """
        Build the MultiFitter over the binding models.

        Unlike the other Analysis classes, the objects being fitted are the binding models rather
        than this object, so the parameters live on those models.

        Returns
        -------
        MultiFitter
            A MultiFitter over the per-target models and fit functions.
        """
        _, _, _, funcs, models = self._build_fit_inputs()
        self._fitter_targets = self._target_signature(models)
        return MultiFitter(fit_objects=models, fit_functions=funcs)

    @staticmethod
    def _target_signature(models: list) -> tuple:
        """
        Summarize which models the fitter was built for, in target order.

        Parameters
        ----------
        models : list
            The model behind each fit target.

        Returns
        -------
        tuple
            A comparable signature of the current targets.
        """
        return tuple(model.unique_name for model in models)

    def _invalidate_fitter_if_targets_changed(self, models: list) -> None:
        """
        Rebuild the cached fitter when the bindings no longer resolve to the same targets.

        A FitBinding can be edited in place -- ``binding.targets = ...`` -- which this object
        cannot observe. Doing so changes how many datasets there are, while the cached MultiFitter
        still holds the old fit functions, and the fit then dies deep inside the minimizer. Compare
        the targets the fitter was built for against the current ones instead.

        Parameters
        ----------
        models : list
            The model behind each fit target, as currently resolved.
        """
        if self._fitter is None:
            return
        if self._target_signature(models) != getattr(self, '_fitter_targets', None):
            self._invalidate_fitter()

    def _sampling_data(self) -> tuple[list, list, list]:
        """
        Get the per-target data to bind to the Sampler.

        Returns
        -------
        tuple[list, list, list]
            The ``(x, y, weights)`` triple, one entry per fit target.
        """
        xs, ys, ws, _, models = self._build_fit_inputs()
        self._invalidate_fitter_if_targets_changed(models)
        return xs, ys, ws

    def _chain_parameters(self) -> list[Parameter]:
        """
        Get the free parameters across every binding model.

        A model appears once per target it is fitted against, so the union is taken by
        ``unique_name`` to avoid counting its parameters more than once.

        Returns
        -------
        list[Parameter]
            The free parameters of the binding models, without duplicates.
        """
        parameters = {}
        for binding in self.bindings:
            for parameter in binding.model.get_free_parameters():
                parameters.setdefault(parameter.unique_name, parameter)
        return list(parameters.values())

    def plot(
        self, names: str | list[str] | None = None, **kwargs: dict[str, Any]
    ) -> InteractiveFigure:
        """
        Plot the parameters and fit results.

        Parameters
        ----------
        names : str | list[str] | None, default=None
            The names of the parameters to plot. If None, all parameters with bindings are plotted.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        InteractiveFigure
            An interactive figure containing the plots of the parameters and fit results.

        Raises
        ------
        ValueError
            If the units of the specified parameters are not consistent.
        RuntimeError
            If plot() is called outside of a Jupyter notebook environment.
        """

        if not _in_notebook():
            raise RuntimeError('plot() can only be used in a Jupyter notebook environment.')

        if self.parameters is None:
            raise ValueError('No parameters available to plot.')

        full_model_dataset = None
        if self.bindings:
            full_model_dataset = self.calculate_model_dataset(self.bindings)

        # If no names are provided, default to plot all parameters that have bindings.
        # If no bindings are provided, plot all parameters.
        if names is None:
            names = []

            if not self.bindings:
                names = list(self.parameters.keys())
            else:
                for b in self.bindings:
                    names.extend(target.dataset_key for target in b.get_targets())

        names = self._normalize_names(names)

        # Check that the units of the specified parameters are consistent.
        units = [self.parameters[name].unit for name in names]
        first_unit = units[0]
        if any(unit != first_unit for unit in units):
            raise ValueError(f'Units are not compatible, and cannot be plotted together: {units}')

        color_cycle = itertools.cycle(rcParams['axes.prop_cycle'].by_key()['color'])
        markers = itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>'])

        plot_kwargs = {
            'title': self.display_name,
            'linestyle': {},
            'marker': {},
            'color': {},
            'markerfacecolor': {},
        }

        data_arrays = {}
        model_arrays = {}

        # map dataset keys to model labels
        param_to_model = {}
        if self.bindings is not None:
            for b in self.bindings:
                param_to_model.update({
                    target.dataset_key: target.label for target in b.get_targets()
                })

        for pname in names:
            data_arrays[pname] = self.parameters[pname]
            color = next(color_cycle)
            marker = next(markers)

            # Data styling
            plot_kwargs['linestyle'][pname] = 'none'
            plot_kwargs['marker'][pname] = marker
            plot_kwargs['color'][pname] = color
            plot_kwargs['markerfacecolor'][pname] = 'none'

            if full_model_dataset is not None and pname in param_to_model:
                mname = param_to_model[pname]
                model_arrays[mname] = full_model_dataset[mname]

                # Model styling
                plot_kwargs['linestyle'][mname] = '--'
                plot_kwargs['marker'][mname] = None
                plot_kwargs['color'][mname] = color

        # Update kwargs with user provided kwargs.
        plot_kwargs.update(kwargs)

        data_and_model = sc.Dataset(data_arrays)
        data_and_model.update(model_arrays)

        return pp.plot(data_and_model, **plot_kwargs)

    def calculate_model_dataset(self, bindings: list[FitBinding]) -> sc.Dataset:
        """
        Evaluate all bindings into a sc.Dataset of model predictions.

        Parameters
        ----------
        bindings : list[FitBinding]
            The bindings to evaluate.

        Returns
        -------
        sc.Dataset
            A sc.Dataset containing the model predictions for all bindings.

        Raises
        ------
        ValueError
            If any parameter name from the bindings is not found in the parameters Dataset.

        TypeError
            If bindings is not a list of FitBinding objects.
        """

        if self.parameters is None:
            raise ValueError('No parameters Dataset provided.')

        if not bindings:
            raise ValueError('No fit bindings provided.')

        if not isinstance(bindings, list) or not all(isinstance(b, FitBinding) for b in bindings):
            raise TypeError('bindings must be a list of FitBinding objects.')

        arrays = {}

        for b in bindings:
            for target in b.get_targets():
                if target.dataset_key not in self.parameters:
                    raise ValueError(
                        f"Parameter '{target.dataset_key}' from binding '{b.unique_name}' "
                        f'not found in parameters Dataset.'
                    )
                da = self.parameters[target.dataset_key]
                x = da.coords['Q']
                x_factor, y_factor = self._get_unit_conversions(target)

                y_model = target.function(x.values * x_factor)
                y_model = y_model / y_factor

                arrays[target.label] = sc.DataArray(
                    data=sc.array(dims=['Q'], values=y_model, unit=da.unit),
                    coords={'Q': x},
                )
        return sc.Dataset(arrays)

    def append_binding(self, binding: FitBinding) -> None:
        """
        Append a FitBinding to the list of bindings for the parameter analysis.

        Parameters
        ----------
        binding : FitBinding
            The FitBinding to append.

        Raises
        ------
        TypeError
            If binding is not a FitBinding object.
        """
        if not isinstance(binding, FitBinding):
            raise TypeError('binding must be a FitBinding object.')
        self._bindings.append(binding)

    def clear_bindings(self) -> None:
        """
        Clear all FitBindings from the list of bindings for the parameter analysis.
        """
        self._bindings.clear()

    def get_all_variables(self) -> list:
        """
        Get all variables from the fit functions.

        Returns
        -------
        list
            A list of all variables from the fit functions.
        """
        variables = set()
        for b in self._bindings:
            variables.update(b.model.get_all_variables())
        return list(variables)

    #############
    # Private methods: verification and preparation
    #############

    def _verify_bindings(self, bindings: FitBinding | list[FitBinding] | None) -> list[FitBinding]:
        """
        Verify the bindings input.

        Parameters
        ----------
        bindings : FitBinding | list[FitBinding] | None
            The bindings to verify.

        Returns
        -------
        list[FitBinding]
            A list of verified FitBindings.

        Raises
        ------
        TypeError
            If bindings is not a FitBinding, a list of FitBindings, or None.
        """
        if bindings is None:
            return []
        if isinstance(bindings, FitBinding):
            return [bindings]
        if isinstance(bindings, list) and all(isinstance(b, FitBinding) for b in bindings):
            return bindings
        raise TypeError('bindings must be a FitBinding, a list of FitBindings, or None.')

    def _verify_parameters(self, parameters: sc.Dataset | Analysis | None) -> sc.Dataset | None:
        """
        Verify the parameters input and convert it to a sc.Dataset if it's an Analysis.

        Parameters
        ----------
        parameters : sc.Dataset | Analysis | None
            The parameters to verify.

        Returns
        -------
        sc.Dataset | None
            The verified parameters as a sc.Dataset, or None if no parameters were provided.

        Raises
        ------
        TypeError
            If parameters is not a sc.Dataset, an Analysis, or None.
        ValueError
            If parameters is a sc.Dataset but does not have a 'Q' coordinate.
        """
        if parameters is None:
            return None

        if not isinstance(parameters, (sc.Dataset, Analysis)):
            raise TypeError(r'parameters must be a sc.Dataset, an Analysis, or None.')

        if isinstance(parameters, Analysis):
            verified_parameters = parameters.parameters_to_dataset()
        else:
            verified_parameters = parameters

        if 'Q' not in verified_parameters.coords:
            raise ValueError(r"parameters must have a 'Q' coordinate.")
        return verified_parameters

    def _normalize_names(self, names: str | list[str] | None) -> list[str] | None:
        """
        Normalize the names input to a list of strings and verify that they exist in the parameters
        Dataset.

        Parameters
        ----------
        names : str | list[str] | None
            The names to normalize and verify.

        Returns
        -------
        list[str] | None
            The normalized list of names, or None if names was None.

        Raises
        ------
        ValueError
            If any of the specified names are not found in the parameters Dataset, or if names is a
            list that contains non-string elements.
        """
        if names is None:
            return None
        if not isinstance(names, (str, list)):
            raise ValueError('names must be a string, a list of strings, or None.')
        if isinstance(names, list):
            if not all(isinstance(name, str) for name in names):
                raise ValueError('All names in the list must be strings.')
            for name in names:
                if name not in self.parameters:
                    raise ValueError(f"Parameter name '{name}' not found in parameters Dataset.")
        if isinstance(names, str):
            if names not in self.parameters:
                raise ValueError(f"Parameter name '{names}' not found in parameters Dataset.")
            names = [names]
        return names

    #############
    # Private methods
    #############

    def _get_unit_conversions(self, target: FitTarget) -> tuple[float, float]:
        """
        Return (x_factor, y_factor) to convert dataset values into the target's declared units.

        x_factor converts Q coordinate values from their stored unit to target.x_unit (e.g.
        1/angstrom for diffusion-model predictions). y_factor converts parameter values from their
        stored unit to target.y_unit. A factor is 1.0 when the target declares the corresponding
        unit as None (its function takes raw values). ``sc.UnitError`` is raised when x or y units
        are physically incompatible (e.g. meV vs 1/angstrom).

        Parameters
        ----------
        target : FitTarget
            The fit target whose unit contract defines the target units.

        Returns
        -------
        tuple[float, float]
            ``(x_factor, y_factor)`` scale factors to apply before/after model evaluation.
        """
        da = self.parameters[target.dataset_key]

        def factor(from_unit: str, to_unit: str | None, error_message: str) -> float:
            if to_unit is None:
                return 1.0
            try:
                return convert_value_unit(1.0, from_unit, str(to_unit))
            except Exception as e:
                raise sc.UnitError(error_message) from e

        q_unit = str(da.coords['Q'].unit)
        x_factor = factor(
            q_unit,
            target.x_unit,
            f"Q coordinate unit '{q_unit}' is incompatible with "
            f"the x_unit '{target.x_unit}' of fit target '{target.label}' "
            f"for parameter '{target.dataset_key}'.",
        )

        param_unit = str(da.unit)
        y_factor = factor(
            param_unit,
            target.y_unit,
            f"Parameter '{target.dataset_key}' unit '{param_unit}' is incompatible "
            f"with the y_unit '{target.y_unit}' of fit target '{target.label}'.",
        )

        return x_factor, y_factor

    def _get_xyweight_from_dataset(
        self, parameter_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the x, y, and weight values for a given parameter name from the parameters Dataset.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to get x, y, and weight values for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the x, y, and weight values for the given parameter name.

        Raises
        ------
        ValueError
            If the parameter name is not found in the parameters Dataset. If non-finite weights are
            found for the parameter.
        """
        if self._parameters is None:
            raise ValueError('No parameters Dataset provided.')
        if parameter_name not in self._parameters:
            raise ValueError(f"Parameter name '{parameter_name}' not found in parameters Dataset.")

        variances = self._parameters[parameter_name].variances
        values = self._parameters[parameter_name].values
        q_values = self._parameters[parameter_name].coords['Q'].values

        if variances is None:
            return q_values, values, np.ones_like(values)

        # NaN variances arise when a parameter is absent for a given Q (parameters_to_dataset
        # fills np.nan for missing parameters). Filter those rows silently; other non-finite or
        # non-positive variances are errors.
        nan_mask = np.isnan(variances)
        if np.any(~nan_mask & (~np.isfinite(variances) | (variances <= 0))):
            raise ValueError(
                f"Non-finite variances found for parameter '{parameter_name}', "
                f'cannot compute weights.'
            )
        valid_mask = ~nan_mask
        if not np.any(valid_mask):
            raise ValueError(
                f"No finite positive variances found for parameter '{parameter_name}', "
                f'cannot compute weights.'
            )

        return q_values[valid_mask], values[valid_mask], 1 / np.sqrt(variances[valid_mask])

    #############
    # Dunder methods
    #############
    def __repr__(self) -> str:
        """
        Return a string representation of the ParameterAnalysis.

        Returns
        -------
        str
            A string representation of the ParameterAnalysis.
        """
        cls = self.__class__.__name__

        n_params = len(self._parameters) if isinstance(self._parameters, sc.Dataset) else 0

        param_names = (
            list(self._parameters.keys()) if isinstance(self._parameters, sc.Dataset) else None
        )

        binding_info = [
            {
                'model': b.model.display_name,
                'targets': b.targets,
            }
            for b in self._bindings
        ]

        return (
            f'{cls}(\n'
            f'    display_name={self.display_name!r},\n'
            f'    unique_name={self.unique_name!r},\n'
            f'    n_parameters={n_params},\n'
            f'    parameter_names={param_names},\n'
            f'    bindings={binding_info})'
        )
