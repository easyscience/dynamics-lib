# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import scipp as sc
from easyscience.fitting.minimizers.utils import FitResults
from easyscience.fitting.multi_fitter import MultiFitter

from easydynamics.analysis.analysis import Analysis
from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.settings.parameter_analysis_fit_settings import ParameterAnalysisFitSettings


class ParameterAnalysis(EasyDynamicsBase):
    """
    Analysing fitted parameters.
    """

    def __init__(
        self,
        parameters: sc.Dataset | Analysis | None = None,
        parameter_names: str | list[str] | None = None,
        fit_functions: (
            ModelComponent
            | ComponentCollection
            | DiffusionModelBase
            | list[ModelComponent | ComponentCollection | DiffusionModelBase]
            | None
        ) = None,
        fit_settings: ParameterAnalysisFitSettings | None = None,
        display_name: str | None = 'MyAnalysis',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the AnalysisBase.

        Parameters
        ----------
        display_name : str | None, default='MyAnalysis'
            Display name of the analysis.
        unique_name : str | None, default=None
            Unique name of the analysis. If None, a unique name is automatically generated. By
            default, None.

        Raises
        ------
        TypeError
            If experiment is not an Experiment or None or if sample_model is not a SampleModel or
            None or if instrument_model is not an InstrumentModel or None or if
            convolution_settings is not a ConvolutionSettings or None or if
            detailed_balance_settings is not a DetailedBalanceSettings or None or if
            extra_parameters is not a Parameter, a list of Parameters, or None.
        """

        super().__init__(display_name=display_name, unique_name=unique_name)

        # Check parameters
        if parameters is not None and not isinstance(parameters, (sc.Dataset, Analysis)):
            raise TypeError('parameters must be an sc.Dataset, an Analysis, or None.')

        if isinstance(parameters, Analysis):
            self._parameters = parameters.parameters_to_dataset()
        else:
            self._parameters = parameters

        # Check fit settings
        if fit_settings is not None and not isinstance(fit_settings, ParameterAnalysisFitSettings):
            raise TypeError('fit_settings must be a ParameterAnalysisFitSettings or None.')
        if fit_settings is None:
            fit_settings = ParameterAnalysisFitSettings()
        self._fit_settings = fit_settings

        # Check fit_functions
        if fit_functions is not None and not isinstance(
            fit_functions,
            (
                ModelComponent,
                ComponentCollection,
                DiffusionModelBase,
                list,
            ),
        ):
            raise TypeError(
                'fit_functions must be a ModelComponent, a ComponentCollection, a list of '
                'ModelComponent/ComponentCollection, a DiffusionModelBase, or None.'
            )

        # Make fit_functions a list if it's not already
        if isinstance(fit_functions, (ModelComponent, ComponentCollection, DiffusionModelBase)):
            fit_functions = [fit_functions]

        for func in fit_functions:
            if not isinstance(
                func,
                (ModelComponent, ComponentCollection, DiffusionModelBase),
            ):
                raise TypeError(
                    'All items in fit_functions list must be a ModelComponent, '
                    'a ComponentCollection, or a DiffusionModelBase.'
                )

        # Check parameter names
        if parameter_names is not None and not isinstance(parameter_names, (str, list)):
            raise TypeError('parameter_names must be a string, a list of strings, or None.')
        # Make parameter_names a list if it's not already
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]

        for name in parameter_names:
            if not isinstance(name, str):
                raise TypeError('All items in parameter_names list must be strings.')

        # Check that parameter_names and fit_functions have the same length
        if len(parameter_names) != len(fit_functions):
            raise ValueError('parameter_names must have the same length as fit_functions.')

        # Convert fit_functions to a list of callables and expand parameter_names if necessary
        fit_function_callables = []
        fit_objects = []
        expanded_parameter_names = []
        if fit_functions is not None:
            for name, func in zip(parameter_names, fit_functions, strict=True):
                if isinstance(func, DiffusionModelBase):
                    fit_funcs, fit_objs = self._diffusion_model_to_fit_functions(func)
                    fit_function_callables.extend(fit_funcs)
                    fit_objects.extend(fit_objs)
                    expanded_parameter_names.extend(
                        self._get_diffusion_model_parameter_names(name)
                    )
                elif isinstance(func, (ModelComponent, ComponentCollection)):
                    fit_function_callables.append(self._components_to_fit_function(func))
                    fit_objects.append(func)
                    expanded_parameter_names.append(name)
            self._fit_functions = fit_functions
            self._fit_function_callables = fit_function_callables
            self._fit_objects = fit_objects
            self._parameter_names = parameter_names
            self._expanded_parameter_names = expanded_parameter_names

        # Check that all names are in the DataSet
        if self._parameters is not None:
            for name in self._expanded_parameter_names:
                if name not in self._parameters:
                    raise ValueError(f"Parameter name '{name}' not found in parameters DataSet.")

    #############
    # Properties
    #############
    @property
    def fit_settings(self) -> ParameterAnalysisFitSettings:
        """
        Get the fit settings for the parameter analysis.

        Returns
        -------
        ParameterAnalysisFitSettings
            The fit settings for the parameter analysis.
        """
        return self._fit_settings

    @fit_settings.setter
    def fit_settings(self, value: ParameterAnalysisFitSettings) -> None:
        """
        Set the fit settings for the parameter analysis.

        Parameters
        ----------
        value : ParameterAnalysisFitSettings
            The new fit settings for the parameter analysis.

        Raises
        ------
        TypeError
            If value is not a ParameterAnalysisFitSettings.
        """
        if not isinstance(value, ParameterAnalysisFitSettings):
            raise TypeError('fit_settings must be a ParameterAnalysisFitSettings.')
        self._fit_settings = value

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
        """

        xs = []
        ys = []
        ws = []

        for name in self._expanded_parameter_names:
            (
                x,
                y,
                weight,
            ) = self._get_xyweight_from_dataset(name)
            xs.append(x)
            ys.append(y)
            ws.append(weight)

        mf = MultiFitter(
            fit_objects=self._fit_objects,
            fit_functions=self._fit_function_callables,
        )

        return mf.fit(
            x=xs,
            y=ys,
            weights=ws,
        )

    #############
    # Private methods
    #############

    def _diffusion_model_to_fit_functions(
        self,
        diffusion_model: DiffusionModelBase,
    ) -> tuple[list[callable], list[DiffusionModelBase]]:
        """
        Convert a DiffusionModelBase to a list of fit functions.

        Parameters
        ----------
        diffusion_model : DiffusionModelBase
            The diffusion model to convert.

        Returns
        -------
        tuple[list[callable], list[DiffusionModelBase]]
            A list of fit functions corresponding to the diffusion model.
        """

        # Currently only looks at the area and width of a Lorentzian.
        # Can and should be extended to also handle delta functions, more parameters etc.

        fit_functions = []
        fit_objects = []

        if self.fit_settings.fit_area:
            fit_functions.append(self._make_area_function(diffusion_model))
            fit_objects.append(diffusion_model)

        if self.fit_settings.fit_width:
            fit_functions.append(self._make_width_function(diffusion_model))
            fit_objects.append(diffusion_model)

        return fit_functions, fit_objects

    @staticmethod
    def _make_area_function(model: DiffusionModelBase) -> callable:
        """
        Make a fit function for the area of a diffusion model.

        Parameters
        ----------
        model : DiffusionModelBase
            The diffusion model to make the fit function for.
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
        """

        def fit_function(
            x: np.ndarray,
            **kwargs: dict[str, Any],  # noqa: ARG001
        ) -> np.ndarray:
            return model.calculate_width(x)

        return fit_function

    def _components_to_fit_functions(
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

    def _get_diffusion_model_parameter_names(
        self,
        parameter_name: str,
    ) -> list[str]:
        """
        Get the parameter names for a diffusion model.

        Parameters
        ----------
        diffusion_model : DiffusionModelBase
            The diffusion model to get parameter names from.

        Returns
        -------
        list[str]
            A list of parameter names.
        """
        parameter_names = []
        if self.fit_settings.fit_area:
            parameter_names.append(parameter_name + ' area')
        if self.fit_settings.fit_width:
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
        Return a string representation of the Analysis.

        Returns
        -------
        str
            A string representation of the Analysis.
        """
        return (
            f' {self.__class__.__name__} (display_name={self.display_name}, '
            f'unique_name={self.unique_name})'
        )
