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
from easydynamics.analysis.parameter_dataclass import _PreparedFitData
from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.utils.utils import _in_notebook

FIT_FUNCTION_TYPE = ModelComponent | ComponentCollection | DiffusionModelBase


class ParameterAnalysis(EasyDynamicsModelBase):
    """
    Analysing fitted parameters.
    """

    def __init__(
        self,
        parameters: sc.Dataset | Analysis | None = None,
        fit_functions: dict[str, FIT_FUNCTION_TYPE] | None = None,
        fit_settings: dict[str, str | list[str]] | None = None,
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
        fit_functions : dict[str, FIT_FUNCTION_TYPE] | None, default=None
            Dictionary mapping parameter names to fit functions. The fit functions can be provided
            as ModelComponents, ComponentCollections, or DiffusionModelBase objects.
        fit_settings : dict[str, str | list[str]] | None, default=None
            A dictionary mapping parameter names to fit settings. The fit settings can be provided
            as strings or lists of strings. If None, default fit settings are used.
        display_name : str | None, default='ParameterAnalysis'
            Display name of the analysis.
        unique_name : str | None, default=None
            Unique name of the analysis. If None, a unique name is automatically generated. By
            default, None.
        """

        super().__init__(display_name=display_name, unique_name=unique_name)

        self._parameters = self._verify_parameters(parameters)
        self._fit_settings = self._verify_fit_settings(fit_settings)
        self._fit_functions = self._verify_fit_functions(fit_functions)

        self._prepared_fit_data = self._prepare_fit_functions_and_parameter_names()

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
        self._prepared_fit_data = self._prepare_fit_functions_and_parameter_names()

    @property
    def fit_functions(
        self,
    ) -> dict[str, ModelComponent | ComponentCollection | DiffusionModelBase]:
        """
        Get the fit functions for the parameter analysis.

        Returns
        -------
        dict[str, ModelComponent | ComponentCollection | DiffusionModelBase]
            The fit functions for the parameter analysis.
        """
        return self._fit_functions

    @fit_functions.setter
    def fit_functions(
        self,
        value: (dict[str, ModelComponent | ComponentCollection | DiffusionModelBase] | None),
    ) -> None:
        """
        Set the fit functions for the parameter analysis.

        Parameters
        ----------
        value : dict[str, ModelComponent | ComponentCollection | DiffusionModelBase] | None
            The new fit functions for the parameter analysis.
        """
        self._fit_functions = self._verify_fit_functions(value)
        self._prepared_fit_data = self._prepare_fit_functions_and_parameter_names()

    @property
    def fit_settings(self) -> dict[str, str | list[str]]:
        """
        Get the fit settings for the parameter analysis.

        Returns
        -------
        dict[str, str | list[str]]
            The fit settings for the parameter analysis.
        """
        return self._fit_settings

    @fit_settings.setter
    def fit_settings(self, value: dict[str, str | list[str]]) -> None:
        """
        Set the fit settings for the parameter analysis.

        Parameters
        ----------
        value : dict[str, str | list[str]]
            The new fit settings for the parameter analysis.
        """
        self._fit_settings = self._verify_fit_settings(value)
        self._prepared_fit_data = self._prepare_fit_functions_and_parameter_names()

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
            If no parameters DataSet is provided. If no fit functions are provided. If no parameter
            names are found for the fit functions.
        """

        if self._parameters is None:
            raise ValueError('No parameters DataSet provided.')

        if not self._prepared_fit_data.fit_function_callables:
            raise ValueError('No fit functions provided.')

        if not self._prepared_fit_data.parameter_names:
            raise ValueError('No parameter names found for fit functions.')

        xs = []
        ys = []
        ws = []

        for name in self._prepared_fit_data.expanded_parameter_names:
            (x, y, weight) = self._get_xyweight_from_dataset(name)
            xs.append(x)
            ys.append(y)
            ws.append(weight)

        mf = MultiFitter(
            fit_objects=self._prepared_fit_data.fit_objects,
            fit_functions=self._prepared_fit_data.fit_function_callables,
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
            The names of the parameters to plot. If None, all parameters are plotted.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to the plotting function.

        Returns
        -------
        InteractiveFigure
            An interactive figure containing the plots of the parameters and fit results.

        Raises
        ------
        ValueError
            If any of the specified parameter names are not found in the parameters DataSet. If the
            units of the specified parameters are not consistent.
        RuntimeError
            If plot_data() is called outside of a Jupyter notebook environment.
        """

        if not _in_notebook():
            raise RuntimeError('plot_data() can only be used in a Jupyter notebook environment.')

        if self._parameters is None:
            raise ValueError('No parameters available to plot.')

        if names is None:
            names = self._prepared_fit_data.expanded_parameter_names
        elif isinstance(names, str):
            names = [names]

        for name in names:
            if name not in self._parameters:
                raise ValueError(f"Parameter name '{name}' not found in parameters DataSet.")

        # Make a new Dataset with only the desired parameters.
        data = sc.Dataset(coords=self._parameters.coords)

        units = [self._parameters[name].unit for name in names]
        if len(set(units)) != 1:
            raise ValueError(f'Units are not consistent, and cannot be plotted together: {units}')

        for name in names:
            data[name] = self._parameters[name]

        # Create fit arrays and plot kwargs for each parameter and fit function
        color_cycle = itertools.cycle(rcParams['axes.prop_cycle'].by_key()['color'])
        markers = ['o', 's', 'D', '^', 'v', '<', '>']
        marker_cycle = itertools.cycle(markers)

        fit_arrays = {}

        plot_kwargs_defaults = {
            'title': self.display_name,
            'linestyle': {},
            'marker': {},
            'color': {},
            'markerfacecolor': {},
        }
        x = self._parameters.coords['Q']

        for name, func, display_name in zip(
            names,
            self._prepared_fit_data.fit_function_callables,
            self._prepared_fit_data.fit_function_display_names,
            strict=True,
        ):
            fit_values = func(x.values)

            # Units need to be handled better here. See issue #65
            fit_arrays[display_name] = sc.DataArray(
                data=sc.array(dims=['Q'], values=fit_values, unit=self._parameters[name].unit),
                coords={'Q': x},
            )

            # Default plot kwargs
            color = next(color_cycle)
            marker = next(marker_cycle)

            # Data
            plot_kwargs_defaults['linestyle'][name] = 'none'
            plot_kwargs_defaults['marker'][name] = marker
            plot_kwargs_defaults['color'][name] = color
            plot_kwargs_defaults['markerfacecolor'][name] = 'none'

            # Fit
            plot_kwargs_defaults['linestyle'][display_name] = '--'
            plot_kwargs_defaults['marker'][display_name] = None
            plot_kwargs_defaults['color'][display_name] = color
            plot_kwargs_defaults['markerfacecolor'][display_name] = 'none'

        # Update kwargs with user provided kwargs.
        plot_kwargs_defaults.update(kwargs)

        fit_dataset = sc.Dataset(fit_arrays)

        data_and_model = sc.merge(fit_dataset, data)

        return pp.plot(data_and_model, **plot_kwargs_defaults)

    def get_all_variables(self) -> list:
        """
        Get all variables from the fit functions.

        Returns
        -------
        list
            A list of all variables from the fit functions.
        """
        variables = []
        for fit_funcs in self.fit_functions.values():
            variables.extend(fit_funcs.get_all_variables())
        return variables

    #############
    # Private methods: verification and preparation
    #############

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
        """
        if parameters is not None and not isinstance(parameters, (sc.Dataset, Analysis)):
            raise TypeError('parameters must be an sc.Dataset, an Analysis, or None.')

        if isinstance(parameters, Analysis):
            verified_parameters = parameters.parameters_to_dataset()
        else:
            verified_parameters = parameters
        return verified_parameters

    def _verify_fit_settings(
        self, fit_settings: dict[str, str | list[str]] | None
    ) -> dict[str, str | list[str]]:
        """
        Verify the fit settings input.

        Parameters
        ----------
        fit_settings : dict[str, str | list[str]] | None
            The fit settings to verify.

        Returns
        -------
        dict[str, str | list[str]]
            The verified fit settings.

        Raises
        ------
        TypeError
            If fit_settings is not a dictionary or None. If any key in fit_settings is not a
            string. If any value in fit_settings is not a string or a list of strings. If any item
            in any list in fit_settings is not a string.
        """
        if fit_settings is None:
            fit_settings = {}

        if not isinstance(fit_settings, dict):
            raise TypeError('fit_settings must be a dictionary of fit settings or None.')

        for key, value in fit_settings.items():
            if not isinstance(key, str):
                raise TypeError('All keys in fit_settings must be strings.')
            if not isinstance(value, (str, list)):
                raise TypeError('All values in fit_settings must be strings or lists of strings.')
            if isinstance(value, list) and not all(isinstance(item, str) for item in value):
                raise TypeError('All items in lists in fit_settings must be strings.')
        return fit_settings

    def _verify_fit_functions(
        self,
        fit_functions: (
            dict[str, ModelComponent | ComponentCollection | DiffusionModelBase] | None
        ),
    ) -> dict[str, ModelComponent | ComponentCollection | DiffusionModelBase]:
        """
        Verify the fit functions input.

        Parameters
        ----------
        fit_functions : dict[str, ModelComponent | ComponentCollection | DiffusionModelBase] | None
            The fit functions to verify.

        Returns
        -------
        dict[str, ModelComponent | ComponentCollection | DiffusionModelBase]
            The verified fit functions.

        Raises
        ------
        TypeError
            If fit_functions is not a dictionary or None.
        """

        if fit_functions is None:
            fit_functions = {}

        if not isinstance(fit_functions, dict):
            raise TypeError(
                'fit_functions must be a dictionary mapping parameter names to fit functions.'
            )

        for name, func in fit_functions.items():
            if not isinstance(name, str):
                raise TypeError('All keys in fit_functions must be strings.')
            if not isinstance(
                func,
                (
                    ModelComponent,
                    ComponentCollection,
                    DiffusionModelBase,
                ),
            ):
                raise TypeError(
                    'All values in fit_functions must be a ModelComponent, a ComponentCollection, '
                    'or a DiffusionModelBase.'
                )
        return fit_functions

    def _prepare_fit_functions_and_parameter_names(
        self,
    ) -> _PreparedFitData:
        """
        Prepare the fit functions and parameter names for fitting.


        Returns
        -------
        _PreparedFitData
            A _PreparedFitData object containing the list of fit function callables, the list of
            fit objects, the list of fit function display names, the list of original parameter
            names, and the list of expanded parameter names.

        Raises
        ------
        ValueError
            If any parameter name in fit_functions is not found in the parameters DataSet.
        """
        fit_function_callables = []
        fit_objects = []
        expanded_parameter_names = []
        fit_function_display_names = []
        for name, func in self._fit_functions.items():
            if isinstance(func, DiffusionModelBase):
                fit_funcs, fit_objs, display_names = self._diffusion_model_to_fit_functions(
                    name, func
                )
                fit_function_callables.extend(fit_funcs)
                fit_objects.extend(fit_objs)
                fit_function_display_names.extend(display_names)
                expanded_parameter_names.extend(self._get_modelcomponent_parameter_names(name))
            elif isinstance(func, (ModelComponent, ComponentCollection)):
                fit_function_callables.append(self._components_to_fit_function(func))
                fit_objects.append(func)
                fit_function_display_names.append(func.display_name)
                expanded_parameter_names.append(name)
        parameter_names = list(self._fit_functions.keys())

        # Check that all names are in the DataSet
        if self._parameters is not None:
            for name in expanded_parameter_names:
                if name not in self._parameters:
                    raise ValueError(f"Parameter name '{name}' not found in parameters DataSet.")

        return _PreparedFitData(
            fit_function_callables=fit_function_callables,
            fit_objects=fit_objects,
            fit_function_display_names=fit_function_display_names,
            parameter_names=parameter_names,
            expanded_parameter_names=expanded_parameter_names,
        )

    #############
    # Private methods
    #############

    def _diffusion_model_to_fit_functions(
        self,
        parameter_name: str,
        diffusion_model: DiffusionModelBase,
    ) -> tuple[list[callable], list[DiffusionModelBase], list[str]]:
        """
        Convert a DiffusionModelBase to a list of fit functions.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter.
        diffusion_model : DiffusionModelBase
            The diffusion model to convert.

        Returns
        -------
        tuple[list[callable], list[DiffusionModelBase], list[str]]
            A list of fit functions corresponding to the diffusion model, a list of the original
            diffusion model repeated for each fit function, and a list of display names for the fit
            functions.
        """

        # Currently only looks at the area and width of a Lorentzian.
        # Can and should be extended to also handle delta functions, more parameters etc.

        fit_functions = []
        fit_objects = []
        display_names = []

        if parameter_name in self.fit_settings:
            fit_setting = self.fit_settings[parameter_name]
            if isinstance(fit_setting, str):
                fit_setting = [fit_setting]

            if 'area' in fit_setting:
                fit_functions.append(self._make_area_function(diffusion_model))
                fit_objects.append(diffusion_model)
                display_names.append(diffusion_model.display_name + ' area')

            if 'width' in fit_setting:
                fit_functions.append(self._make_width_function(diffusion_model))
                fit_objects.append(diffusion_model)
                display_names.append(diffusion_model.display_name + ' width')
        else:
            # If no fit settings are provided for this parameter, fit
            # both area and width by default.
            fit_functions.append(self._make_area_function(diffusion_model))
            fit_objects.append(diffusion_model)
            display_names.append(diffusion_model.display_name + ' area')

            fit_functions.append(self._make_width_function(diffusion_model))
            fit_objects.append(diffusion_model)
            display_names.append(diffusion_model.display_name + ' width')

        return fit_functions, fit_objects, display_names

    @staticmethod
    def _make_area_function(model: DiffusionModelBase) -> callable:
        """
        Make a fit function for the area of a diffusion model.

        Parameters
        ----------
        model : DiffusionModelBase
            The diffusion model to make the fit function for.

        Returns
        -------
        callable
            A fit function corresponding to the area of the diffusion model.
        """

        def fit_function(
            x: np.ndarray,
            **kwargs: dict[str, Any],  # noqa: ARG001
        ) -> np.ndarray:
            return model.calculate_QISF(x) * model.scale.value

        return fit_function

    @staticmethod
    def _make_width_function(model: DiffusionModelBase) -> callable:
        """
        Make a fit function for the width of a diffusion model.

        Parameters
        ----------
        model : DiffusionModelBase
            The diffusion model to make the fit function for.

        Returns
        -------
        callable
            A fit function corresponding to the width of the diffusion model.
        """

        def fit_function(
            x: np.ndarray,
            **kwargs: dict[str, Any],  # noqa: ARG001
        ) -> np.ndarray:
            return model.calculate_width(x)

        return fit_function

    def _components_to_fit_function(
        self,
        components: ModelComponent | ComponentCollection | None,
    ) -> callable:
        """
        Convert a ModelComponent or ComponentCollection to a fit function.

        Parameters
        ----------
        components : ModelComponent | ComponentCollection | None
            The component(s) to convert.

        Returns
        -------
        callable
            A fit function corresponding to the component(s).
        """

        if components is None:
            return []

        return self._make_components_function(components)

    @staticmethod
    def _make_components_function(
        components: ModelComponent | ComponentCollection,
    ) -> callable:
        def fit_function(
            x: np.ndarray,
            **kwargs: dict[str, Any],  # noqa: ARG001
        ) -> np.ndarray:
            return components.evaluate(x)

        return fit_function

    def _get_modelcomponent_parameter_names(
        self,
        parameter_name: str,
    ) -> list[str]:
        """
        Get the parameter names for a model component.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to get names for.

        Returns
        -------
        list[str]
            A list of parameter names.
        """
        parameter_names = []

        if parameter_name in self.fit_settings:
            fit_setting = self.fit_settings[parameter_name]
            if isinstance(fit_setting, str):
                fit_setting = [fit_setting]

            if 'area' in fit_setting:
                parameter_names.append(parameter_name + ' area')

            if 'width' in fit_setting:
                parameter_names.append(parameter_name + ' width')
        else:
            parameter_names.append(parameter_name + ' area')
            parameter_names.append(parameter_name + ' width')

        return parameter_names

    def _get_xyweight_from_dataset(
        self, parameter_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the x, y, and weight values for a given parameter name from the parameters DataSet.

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
            If the parameter name is not found in the parameters DataSet.
        """
        if self._parameters is None:
            raise ValueError('No parameters DataSet provided.')
        if parameter_name not in self._parameters:
            raise ValueError(f"Parameter name '{parameter_name}' not found in parameters DataSet.")

        # Need to check the variances.
        return (
            self._parameters[parameter_name].coords['Q'].values,
            self._parameters[parameter_name].values,
            1 / self._parameters[parameter_name].variances ** 0.5,
        )

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

        fit_keys = list(self._fit_functions.keys())

        return (
            f'{cls}(\n'
            f'  display_name={self.display_name!r},\n'
            f'  unique_name={self.unique_name!r},\n'
            f'  n_parameters={n_params},\n'
            f'  parameter_names={param_names},\n'
            f'  fit_functions={fit_keys},\n'
            f'  expanded_parameters={self._prepared_fit_data.expanded_parameter_names}\n'
            f')'
        )
