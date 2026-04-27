# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase

FIT_FUNCTION_TYPE = ModelComponent | ComponentCollection | DiffusionModelBase


@dataclass
class _PreparedFitData:
    """
    Holds the prepared fit data for parameter analysis. This includes the fit function callables,
    fit objects, display names, and parameter names, both original and expanded.

    Attributes
    ----------
    fit_function_callables : list[callable]
        A list of callables corresponding to the fit functions.
    fit_objects : list[FIT_FUNCTION_TYPE]
        A list of the original fit objects corresponding to the fit functions.
    fit_function_display_names : list[str]
        A list of display names corresponding to the fit functions, where diffusion models are
        expanded into their parameters (e.g. "D area", "D width" for a diffusion model "D").
    parameter_names : list[str]
        A list of the original parameter names corresponding to the fit functions.
    expanded_parameter_names : list[str]
        A list of the expanded parameter names corresponding to the fit functions, where diffusion
        models are expanded into their parameters (e.g. "D area", "D width" for a diffusion model
        "D").
    """

    fit_function_callables: list[callable]
    fit_objects: list[FIT_FUNCTION_TYPE]
    fit_function_display_names: list[str]
    parameter_names: list[str]
    expanded_parameter_names: list[str]
