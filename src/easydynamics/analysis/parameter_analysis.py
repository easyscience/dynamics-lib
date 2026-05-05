# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Any

import numpy as np
import plopp as pp
import scipp as sc
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.fitting.multi_fitter import MultiFitter
from matplotlib import rcParams
from plopp.backends.matplotlib.figure import InteractiveFigure

from easydynamics.analysis.analysis import Analysis
from easydynamics.analysis.fit_binding import FitBinding
from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.utils.utils import _in_notebook

FIT_FUNCTION_TYPE = ModelComponent | ComponentCollection | DiffusionModelBase


class ParameterAnalysis(EasyDynamicsModelBase):
    """
    For analysing fitted parameters.

    Can be used to fit paramters to ModelComponents, ComponentCollections, or DiffusionModelBase
    objects, and to plot the parameters and fit results. The parameters to be analyzed can be
    provided as a sc.Dataset or directly as an Analysis object. Multiple parameters can be fitted
    simultaneously, and the fit functions can be customized for each parameter. For diffusion
    models, the area and width can be fitted separately (or not at all) by specifying fit settings.
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
        Set the parameters for the parameter analysis.

        Parameters
        ----------
        value : sc.Dataset | Analysis | None
            The new parameters for the parameter analysis.
        """
        self._parameters = self._verify_parameters(value)

    @property
    def bindings(self) -> list[FitBinding] | None:
        """
        Get the fit bindings for the parameter analysis.

        Returns
        -------
        list[FitBinding] | None
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

    #############
    # Other methods
    #############

    def fit(self) -> FitResults:
        """
        Fit the parameters using the specified fit functions and settings.

        Returns
        -------
        FitResults
            The results of the fit

        Raises
        ------
        ValueError
            If no parameters Dataset is provided. If no fit functions are provided. If no parameter
            names are found for the fit functions.
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
            param_names = binding.get_parameter_names()
            callables = binding.build_callables()

            for pname, func in zip(param_names, callables, strict=True):
                if pname not in self.parameters:
                    raise ValueError(
                        f"Parameter '{pname}' from binding '{binding.unique_name}' "
                        f'not found in parameters Dataset.'
                    )

                x, y, weight = self._get_xyweight_from_dataset(pname)

                xs.append(x)
                ys.append(y)
                ws.append(weight)

                funcs.append(func)
                models.append(binding.model)

        mf = MultiFitter(
            fit_objects=models,
            fit_functions=funcs,
        )

        return mf.fit(
            x=xs,
            y=ys,
            weights=ws,
        )

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
        if self.bindings is not None:
            full_model_dataset = self.calculate_model_dataset(self.bindings)

        if names is None:
            names = []
            for b in self.bindings:
                names.extend(b.get_parameter_names())

        names = self._normalize_names(names)

        units = [self.parameters[name].unit for name in names]
        first_unit = units[0]
        if any(unit != first_unit for unit in units):
            raise ValueError(f'Units are not consistent, and cannot be plotted together: {units}')

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

        # map parameter names to model names
        param_to_model = {}
        if self.bindings is not None:
            for b in self.bindings:
                param_names = b.get_parameter_names()
                model_names = b.get_model_names()

                param_to_model.update(dict(zip(param_names, model_names, strict=True)))

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
            param_names = b.get_parameter_names()
            model_names = b.get_model_names()
            callables = b.build_callables()

            for pname, mname, func in zip(param_names, model_names, callables, strict=True):
                if pname not in self.parameters:
                    raise ValueError(
                        f"Parameter '{pname}' from binding '{b.unique_name}' "
                        f'not found in parameters Dataset.'
                    )
                da = self.parameters[pname]
                x = da.coords['Q']

                y_model = func(x.values)

                arrays[mname] = sc.DataArray(
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
        if variances is None:
            weight = np.ones_like(self._parameters[parameter_name].values)
        elif np.any(~np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError(
                f"Non-finite variances found for parameter '{parameter_name}', "
                f'cannot compute weights.'
            )
        else:
            weight = 1 / np.sqrt(variances)

        return (
            self._parameters[parameter_name].coords['Q'].values,
            self._parameters[parameter_name].values,
            weight,
        )

    #############
    # Dunder methods
    #############
    def __repr__(self) -> str:
        cls = self.__class__.__name__

        n_params = len(self._parameters) if isinstance(self._parameters, sc.Dataset) else 0

        param_names = (
            list(self._parameters.keys()) if isinstance(self._parameters, sc.Dataset) else None
        )

        binding_info = [
            {
                'parameter': b.parameter_name,
                'model': b.model.display_name,
                'modes': b.modes,
            }
            for b in self._bindings
        ]

        return (
            f'{cls}(\n'
            f'display_name={self.display_name},\n'
            f'unique_name={self.unique_name},\n'
            f'n_parameters={n_params},\n'
            f'parameter_names={param_names},\n'
            f'bindings={binding_info}\n'
            f')'
        )
