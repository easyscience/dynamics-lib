# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase

if TYPE_CHECKING:
    from easydynamics.utils.fit_target import FitTarget


class FitBinding(EasyDynamicsBase):
    """
    Contract between dataset, model, and fit functions for ParameterAnalysis. A binding maps the
    model's fittable predictions (its FitTargets) onto keys of the parameters Dataset they should
    be fitted against.

    Examples
    --------
    **Fitting a component model to one parameter**

    Component models (e.g. a Polynomial) have a single prediction — their ``evaluate`` — so
    ``targets`` is simply the dataset key to fit against. The model's x_unit/y_unit declare the
    units its evaluate expects: here x is the dataset's Q coordinate and y the fitted parameter, so
    construct the model with matching units (or pass ``x_unit=None`` / ``y_unit=None`` to fit raw
    values):
    ```python
    import easydynamics as edyn
    import easydynamics.sample_model as sm

    fit_func = sm.Polynomial(
        coefficients=[3.7, -0.5],
        x_unit='1/angstrom',
        y_unit='meV',
        display_name='Straight line',
    )
    binding = edyn.FitBinding(model=fit_func, targets='Gaussian area')
    ```

    **Fitting a diffusion model with default dataset keys**

    Diffusion models declare their predictions (``'area'``, ``'width'``, and for DeltaLorentz also
    ``'delta_area'``). With ``targets=None`` all predictions are fitted against default dataset
    keys derived from the model's component names:
    ```python
    brownian = sm.BrownianTranslationalDiffusion(
        diffusion_coefficient=2.4e-9,
        scale=0.5,
        lorentzian_name='Lorentzian',
    )
    binding = edyn.FitBinding(model=brownian)  # fits 'Lorentzian area' and 'Lorentzian width'
    ```

    **Selecting predictions or mapping them to custom dataset keys**

    Pass a list of prediction names, or a dict mapping prediction names to dataset keys:
    ```python
    binding = edyn.FitBinding(model=brownian, targets=['width'])

    delta_lorentz = sm.DeltaLorentz(A_0=0.5, lorentzian_width=0.1)
    binding = edyn.FitBinding(
        model=delta_lorentz,
        targets={
            'width': 'Lorentzian width',
            'area': 'Lorentzian area',
            'delta_area': 'Elastic area',
        },
    )
    ```
    """

    def __init__(
        self,
        model: ModelComponent | ComponentCollection | DiffusionModelBase,
        targets: str | list[str] | dict[str, str] | None = None,
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a FitBinding.

        Validation raises ``TypeError`` if model or targets have an invalid type, and
        ``ValueError`` if targets names a prediction the model does not declare.

        Parameters
        ----------
        model : ModelComponent | ComponentCollection | DiffusionModelBase
            The model to fit. This can be a single ModelComponent, a ComponentCollection, or a
            DiffusionModelBase.
        targets : str | list[str] | dict[str, str] | None, default=None
            Which predictions of the model to fit, and against which dataset keys. For component
            models this must be a string: the dataset key to fit the model's ``evaluate`` against.
            For diffusion models: None fits all predictions against their default dataset keys; a
            string or list of strings selects predictions by name (default keys); a dict maps
            prediction names to custom dataset keys.
        display_name : str | None, default=None
            An optional display name for the FitBinding. If None, the unique_name will be used.
            Default is None.
        unique_name : str | None, default=None
            An optional unique name for the FitBinding. If None, a unique name will be generated.
            Default is None.
        """

        super().__init__(display_name=display_name, unique_name=unique_name)

        self._validate_model(model)
        self._normalize_targets(model, targets)
        self._model = model
        self._targets = targets

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    def model(self, value: ModelComponent | ComponentCollection | DiffusionModelBase) -> None:
        """
        Set the model to fit.

        Validation raises ``TypeError`` if the value has an invalid type, and ``ValueError`` if the
        current targets name a prediction the new model does not declare.

        Parameters
        ----------
        value : ModelComponent | ComponentCollection | DiffusionModelBase
            The new model to fit.
        """
        self._validate_model(value)
        self._normalize_targets(value, self._targets)
        self._model = value

    @property
    def targets(self) -> str | list[str] | dict[str, str] | None:
        """
        Which predictions of the model to fit, and against which dataset keys.

        Returns
        -------
        str | list[str] | dict[str, str] | None
            The targets specification (see ``__init__``).
        """
        return self._targets

    @targets.setter
    def targets(self, value: str | list[str] | dict[str, str] | None) -> None:
        """
        Set which predictions of the model to fit, and against which dataset keys.

        Validation raises ``TypeError`` if the value has an invalid type for the current model, and
        ``ValueError`` if it names a prediction the model does not declare.

        Parameters
        ----------
        value : str | list[str] | dict[str, str] | None
            The new targets specification (see ``__init__``).
        """
        self._normalize_targets(self._model, value)
        self._targets = value

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def get_targets(self) -> list[FitTarget]:
        """
        Get the FitTargets this binding fits, with dataset keys resolved.

        Targets are built from the model at call time, so their units and default dataset keys
        reflect the model's current state.

        Returns
        -------
        list[FitTarget]
            The resolved fit targets.
        """
        available = {target.name: target for target in self.model.get_fit_targets()}
        normalized = self._normalize_targets(self.model, self._targets)
        return [
            available[name]
            if dataset_key is None
            else replace(available[name], dataset_key=dataset_key)
            for name, dataset_key in normalized.items()
        ]

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_model(
        model: ModelComponent | ComponentCollection | DiffusionModelBase,
    ) -> None:
        """
        Validate the model type.

        Parameters
        ----------
        model : ModelComponent | ComponentCollection | DiffusionModelBase
            The model to validate.

        Raises
        ------
        TypeError
            If model is not a ModelComponent, ComponentCollection, or DiffusionModelBase.
        """
        if not isinstance(model, (ModelComponent, ComponentCollection, DiffusionModelBase)):
            raise TypeError(
                'model must be a ModelComponent, ComponentCollection, or DiffusionModelBase'
            )

    @staticmethod
    def _normalize_targets(
        model: ModelComponent | ComponentCollection | DiffusionModelBase,
        targets: str | list[str] | dict[str, str] | None,
    ) -> dict[str, str | None]:
        """
        Validate a targets specification and normalize it to a prediction-name mapping.

        This is the one place that knows the two spec dialects: component models take the dataset
        key as a plain string for their single ``'value'`` prediction, while diffusion models
        select predictions by name (None selects all) with optional dataset-key overrides.

        Parameters
        ----------
        model : ModelComponent | ComponentCollection | DiffusionModelBase
            The model the targets apply to.
        targets : str | list[str] | dict[str, str] | None
            The targets specification to validate and normalize.

        Returns
        -------
        dict[str, str | None]
            Mapping of prediction name to dataset-key override (None means the prediction's default
            dataset key is used).

        Raises
        ------
        TypeError
            If targets has an invalid type for the given model.
        ValueError
            If targets names a prediction the model does not declare.
        """
        if not isinstance(model, DiffusionModelBase):
            if not isinstance(targets, str):
                raise TypeError(
                    'For component models, targets must be the dataset key (a string) to fit '
                    "the model's evaluate against."
                )
            return {'value': targets}

        available = [target.name for target in model.get_fit_targets()]
        if targets is None:
            return dict.fromkeys(available)
        if isinstance(targets, str):
            requested = [targets]
        elif isinstance(targets, list):
            requested = targets
        elif isinstance(targets, dict):
            requested = list(targets.keys())
            if not all(isinstance(key, str) for key in targets.values()):
                raise TypeError('targets dict values must be dataset keys (strings)')
        else:
            raise TypeError(
                'targets must be None, a prediction name, a list of prediction names, '
                'or a dict mapping prediction names to dataset keys'
            )
        if not all(isinstance(name, str) for name in requested):
            raise TypeError('prediction names in targets must be strings')

        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(
                f'Unknown prediction(s) {", ".join(unknown)} for '
                f'{model.__class__.__name__}. Available predictions: '
                f'{", ".join(available)}.'
            )
        return dict(targets) if isinstance(targets, dict) else dict.fromkeys(requested)

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
            f'{self.__class__.__name__}(\n'
            f'    model={self.model.display_name},\n'
            f'    targets={self.targets},\n'
            f')'
        )
