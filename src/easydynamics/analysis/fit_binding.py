# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.diffusion_model.diffusion_model_base import (
    DiffusionModelBase,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class FitBinding(EasyDynamicsBase):
    """
    Contract between dataset, model, and fit function for ParameterAnalysis. This class
    encapsulates the necessary information to bind a dataset key to a model and convert it into a
    fit function callable.
    """

    def __init__(
        self,
        parameter_name: str,
        model: ModelComponent | ComponentCollection | DiffusionModelBase,
        modes: str | list[str] | None = None,
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a FitBinding.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to fit. This should correspond to a key in the dataset.
        model : ModelComponent | ComponentCollection | DiffusionModelBase
            The model to fit. This can be a single ModelComponent, a ComponentCollection, or a
            DiffusionModelBase.
        modes : str | list[str] | None, default=None
            The modes to fit for diffusion models. This can be a single string, a list of strings,
            or None (which defaults to ["area", "width"]). Only applicable if the model is a
            DiffusionModelBase. Default is None.
        display_name : str | None, default=None
            An optional display name for the FitBinding. If None, the unique_name will be used.
            Default is None.
        unique_name : str | None, default=None
            An optional unique name for the FitBinding. If None, a unique name will be generated.
            Default is None.

        Raises
        ------
        TypeError
            If parameter_name is not a string, if model is not a ModelComponent,
            ComponentCollection or DiffusionModelBase, or if modes is not a string, list of
            strings, or None.

        Examples
        --------
        1. Basic usage with a ModelComponent:
        >>> import easydynamics.sample_model as sm
        >>> import easydynamics as edyn
        >>> fit_func = sm.Polynomial(coefficients=[3.7, -0.5], display_name='Straight line')
        >>> binding = edyn.FitBinding(parameter_name='Gaussian area', model=fit_func)
        >>> print(binding)
        FitBinding(parameter_name='Gaussian area', model=Polynomial(unique_name = Polynomial_1,
        unit = meV, coefficients = [Straight line_c0=3.7, Straight line_c1=-0.5]), modes=None)

        2. Usage with a DiffusionModelBase and specific modes:
        >>> brownian_diffusion_model = sm.BrownianTranslationalDiffusion(
        ...     display_name='Brownian Translational Diffusion',
        ...     diffusion_coefficient=2.4e-9,
        ...     scale=0.5,
        ... )
        >>> binding = edyn.FitBinding(
        ...     parameter_name='Lorentzian',
        ...     model=brownian_diffusion_model,
        ...     modes=['area', 'width'],
        ... )
        FitBinding(parameter_name=Lorentzian, model=Brownian Translational Diffusion,
        modes=['area', 'width'], display_name=FitBinding_1, unique_name=FitBinding_1)
        """

        super().__init__(display_name=display_name, unique_name=unique_name)

        if not isinstance(parameter_name, str):
            raise TypeError("parameter_name must be a string")

        if not isinstance(
            model, (ModelComponent, ComponentCollection, DiffusionModelBase)
        ):
            raise TypeError(
                "model must be a ModelComponent, ComponentCollection, or DiffusionModelBase"
            )

        if modes is not None and not isinstance(modes, (str, list)):
            raise TypeError("modes must be a string, list of strings, or None")

        if isinstance(modes, list) and not all(isinstance(mode, str) for mode in modes):
            raise TypeError("All modes in the list must be strings")

        self._parameter_name = parameter_name
        self._model = model
        self._modes = modes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parameter_name(self) -> str:
        """
        The name of the parameter to fit. This should correspond to a key in the dataset.

        Returns
        -------
        str
            The name of the parameter to fit.
        """
        return self._parameter_name

    @parameter_name.setter
    def parameter_name(self, value: str) -> None:
        """
        Set the name of the parameter to fit.

        Parameters
        ----------
        value : str
            The new name of the parameter to fit.

        Raises
        ------
        TypeError
            If the value is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("parameter_name must be a string")
        self._parameter_name = value

    @property
    def model(self) -> ModelComponent | ComponentCollection | DiffusionModelBase:
        """
        The model to fit. This can be a single ModelComponent, a ComponentCollection, or a
        DiffusionModelBase.

        Returns
        -------
        ModelComponent | ComponentCollection | DiffusionModelBase
            The model to fit.
        """
        return self._model

    @model.setter
    def model(
        self, value: ModelComponent | ComponentCollection | DiffusionModelBase
    ) -> None:
        """
        Set the model to fit.

        Parameters
        ----------
        value : ModelComponent | ComponentCollection | DiffusionModelBase
            The new model to fit.

        Raises
        ------
        TypeError
            If the value is not a ModelComponent, ComponentCollection, or DiffusionModelBase.
        """
        if not isinstance(
            value, (ModelComponent, ComponentCollection, DiffusionModelBase)
        ):
            raise TypeError(
                "model must be a ModelComponent, ComponentCollection, or DiffusionModelBase."
            )
        self._model = value

    @property
    def modes(self) -> str | list[str] | None:
        """
        The modes to fit for diffusion models. This can be a single string, a list of strings, or
        None (which defaults to ["area", "width"]).

        Returns
        -------
        str | list[str] | None
            The modes to fit for diffusion models.
        """
        return self._modes

    @modes.setter
    def modes(self, value: str | list[str] | None) -> None:
        """
        Set the modes to fit for diffusion models.

        Parameters
        ----------
        value : str | list[str] | None
            The new modes to fit for diffusion models.

        Raises
        ------
        TypeError
            If the value is not a string, list of strings, or None.
        """
        if value is not None and not isinstance(value, (str, list)):
            raise TypeError("modes must be a string, list of strings, or None")

        if isinstance(value, str):
            value = [value]
        if isinstance(value, list) and not all(isinstance(mode, str) for mode in value):
            raise TypeError("All modes in the list must be strings")
        self._modes = value

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def build_callables(self) -> list[Callable]:
        """
        Build the callables for fitting based on the model and modes.

        Returns
        -------
        list[Callable]
            A list of callables for fitting.
        """
        modes = self._get_modes()

        if isinstance(self.model, DiffusionModelBase):
            return [self._build_diffusion_callable(mode) for mode in modes]

        return [lambda x, **_: self.model.evaluate(x)]

    def get_model_names(self) -> list[str]:
        """
        Get the names of the models based on the current modes.

        Returns
        -------
        list[str]
            A list of model names.
        """
        modes = self._get_modes()

        if isinstance(self.model, DiffusionModelBase):
            return [f"{self.model.display_name} {mode}" for mode in modes]

        return [self.model.display_name]

    def get_parameter_names(self) -> list[str]:
        """
        Get the names of the parameters based on the current modes.

        Returns
        -------
        list[str]
            A list of parameter names.
        """
        modes = self._get_modes()

        if isinstance(self.model, DiffusionModelBase):

            # HACK
            if "delta" in modes:
                return [f"{self.parameter_name} area" for mode in modes]

            return [f"{self.parameter_name} {mode}" for mode in modes]

        return [self.parameter_name]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_diffusion_callable(self, mode: str) -> Callable:
        """
        Build a callable for a specific diffusion mode.

        Parameters
        ----------
        mode : str
            The diffusion mode ("area" or "width").

        Returns
        -------
        Callable
            A callable for the specified diffusion mode.

        Raises
        ------
        ValueError
            If the mode is unknown.
        """
        model = self.model

        if mode == "area":
            return lambda x, **_: model.calculate_QISF(x) * model.scale.value

        if mode == "width":
            return lambda x, **_: model.calculate_width(x)

        if mode == "delta":
            return lambda x, **_: model.calculate_EISF(x) * model.scale.value

        raise ValueError(f"Unknown diffusion mode: {mode}")

    def _get_modes(self) -> list[str]:
        """
        Get the modes to fit for diffusion models, defaulting to ["area", "width"] if not set.

        Returns
        -------
        list[str]
            The modes to fit for diffusion models.
        """
        return ["area", "width"] if self.modes is None else self.modes

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a string representation of the FitBinding.

        Returns
        -------
        str
            A string representation of the FitBinding.
        """
        return (
            f"FitBinding(parameter_name={self.parameter_name},\n "
            f"model={self.model.display_name},\n "
            f"modes={self.modes},\n "
            f"display_name={self.display_name},\n "
            f"unique_name={self.unique_name})"
        )
