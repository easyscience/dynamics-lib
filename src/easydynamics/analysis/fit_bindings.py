from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from easydynamics.sample_model.component_collection import ComponentCollection
    from easydynamics.sample_model.components.model_component import ModelComponent

# -----------------------------
# Binding layer (KEY IDEA)
# -----------------------------


@dataclass
class FitBinding:
    """
    Contract between dataset, model, and fit function for ParameterAnalysis. This class
    encapsulates the necessary information to bind a dataset key to a model and convert it into a
    fit function callable.
    """

    parameter_name: str
    model: ModelComponent | ComponentCollection | DiffusionModelBase
    modes: str | list[str] | None = None

    def build_callables(self) -> list[Callable]:
        if isinstance(self.modes, str):
            modes = [self.modes]
        elif self.modes is None:
            modes = ['area', 'width']  # default
        else:
            modes = self.modes

        if isinstance(self.model, DiffusionModelBase):
            return [self._build_diffusion_callable(mode) for mode in modes]

        return [lambda x, **_: self.model.evaluate(x)]

    def get_model_names(self) -> list[str]:
        if isinstance(self.modes, str):
            modes = [self.modes]
        elif self.modes is None:
            modes = ['area', 'width']
        else:
            modes = self.modes

        if isinstance(self.model, DiffusionModelBase):
            return [f'{self.model.display_name} {mode}' for mode in modes]

        return [self.model.display_name]

    def get_parameter_names(self) -> list[str]:
        if isinstance(self.modes, str):
            modes = [self.modes]
        elif self.modes is None:
            modes = ['area', 'width']
        else:
            modes = self.modes

        if len(modes) == 1:
            return [self.parameter_name]

        if isinstance(self.model, DiffusionModelBase):
            return [f'{self.parameter_name} {mode}' for mode in modes]

        return [self.parameter_name]

    def _build_diffusion_callable(self, mode: str) -> Callable:
        model = self.model

        if mode == 'area' or mode == 'auto':
            return lambda x, **_: model.calculate_QISF(x) * model.scale.value

        if mode == 'width':
            return lambda x, **_: model.calculate_width(x)

        raise ValueError(f'Unknown diffusion mode: {mode}')
