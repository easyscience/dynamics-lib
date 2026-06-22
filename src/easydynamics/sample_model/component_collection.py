# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipp as sc

from easydynamics.base_classes.easydynamics_list import EasyDynamicsList
from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.components.model_component import ModelComponent

if TYPE_CHECKING:
    from easyscience.variable import DescriptorBase

    from easydynamics.utils.utils import Numeric


class ComponentCollection(EasyDynamicsList, EasyDynamicsModelBase):
    """
    Collection of model components whose evaluate() results are summed.
    """

    def __init__(
        self,
        components: ModelComponent | list[ModelComponent] | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'ComponentCollection',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        if components is None:
            components = []
        if isinstance(components, ModelComponent):
            components = [components]
        elif not isinstance(components, list):
            raise TypeError(
                f'components must be a ModelComponent or a list of ModelComponent, '
                f'got {type(components).__name__} instead.'
            )
        for comp in components:
            if not isinstance(comp, ModelComponent):
                raise TypeError(
                    f'All items in components must be instances of ModelComponent, '
                    f'got {type(comp).__name__} instead.'
                )

        EasyDynamicsList.__init__(
            self,
            *components,
            protected_types=ModelComponent,
        )

        EasyDynamicsModelBase.__init__(
            self,
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        self._warn_if_duplicate_names()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not self

    @is_empty.setter
    def is_empty(self, _value: bool) -> None:
        raise AttributeError('is_empty is a read-only property.')

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """Convert x-axis unit on all contained components."""
        if not isinstance(new_x_unit, (str, sc.Unit)):
            raise TypeError(f'x_unit must be a string or sc.Unit, got {type(new_x_unit).__name__}')

        old_unit = self._x_unit
        try:
            for component in self:
                component.convert_x_unit(new_x_unit)
            self._x_unit = str(new_x_unit) if isinstance(new_x_unit, sc.Unit) else new_x_unit
        except Exception as e:
            try:
                for component in self:
                    component.convert_x_unit(old_unit)
            except Exception:  # noqa: S110
                pass
            raise e

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """Convert y-axis unit on all contained components."""
        if not isinstance(new_y_unit, (str, sc.Unit)):
            raise TypeError(f'y_unit must be a string or sc.Unit, got {type(new_y_unit).__name__}')

        old_unit = self._y_unit
        try:
            for component in self:
                component.convert_y_unit(new_y_unit)
            self._y_unit = str(new_y_unit) if isinstance(new_y_unit, sc.Unit) else new_y_unit
        except Exception as e:
            try:
                for component in self:
                    component.convert_y_unit(old_unit)
            except Exception:  # noqa: S110
                pass
            raise e

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        if isinstance(component, ComponentCollection):
            self.extend(component)
        else:
            self.append(component)
        self._warn_if_duplicate_names()

    def list_component_names(self) -> list[str]:
        return [component.name for component in self]

    def normalize_area(self) -> None:
        """Normalize areas of all components so they sum to 1."""
        if not self:
            raise ValueError('No components in the model to normalize.')

        area_params = []

        for component in self:
            if hasattr(component, 'area'):
                area_params.append(component.area)
            else:
                warnings.warn(
                    f"Component '{component.name}' does not have an 'area' attribute "
                    'and will be skipped in normalization.',
                    UserWarning,
                    stacklevel=2,
                )

        total_area_value = sum(p.value for p in area_params)

        if total_area_value == 0:
            raise ValueError('Total area is zero; cannot normalize.')

        if not np.isfinite(total_area_value):
            raise ValueError('Total area is not finite; cannot normalize.')

        for param in area_params:
            param.value /= total_area_value

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def get_all_variables(self) -> list[DescriptorBase]:
        return [var for component in self for var in component.get_all_variables()]

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        """Evaluate the sum of all component outputs at x."""
        if not self:
            return np.zeros_like(x)
        return sum(component.evaluate(x, output=output) for component in self)

    def evaluate_component(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        name: str,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        """Evaluate a single component by name."""
        if not self:
            raise ValueError('No components in the model to evaluate.')
        if not isinstance(name, str):
            raise TypeError(f'Component name must be a string, got {type(name)} instead.')
        matches = [comp for comp in self if comp.name == name]
        if not matches:
            raise KeyError(f"No component named '{name}' exists.")
        return matches[0].evaluate(x, output=output)

    def fix_all_parameters(self) -> None:
        for param in self.get_fittable_parameters():
            param.fixed = True

    def free_all_parameters(self) -> None:
        for param in self.get_fittable_parameters():
            param.fixed = False

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _warn_if_duplicate_names(self) -> None:
        names = [c.name for c in self]
        seen: set[str] = set()
        dups: set[str] = set()
        for name in names:
            if name in seen:
                dups.add(name)
            seen.add(name)
        if dups:
            warnings.warn(
                f'Duplicate component names in ComponentCollection: {sorted(dups)}. '
                'Components with the same name will produce duplicate parameter names.',
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        comp_names = ', '.join(c.name for c in self) or 'No components'
        return (
            f"ComponentCollection(name='{self.name}', "
            f"x_unit='{self.x_unit}', y_unit='{self.y_unit}',\n"
            f'Components: {comp_names})'
        )

    def to_dict(self) -> dict:
        return {
            '@module': self.__class__.__module__,
            '@class': self.__class__.__name__,
            'x_unit': str(self.x_unit),
            'y_unit': str(self.y_unit),
            'name': self.name,
            'display_name': self.display_name,
            'components': [c.to_dict() for c in self._data],
        }

    @classmethod
    def from_dict(cls, obj_dict: dict) -> ComponentCollection:
        def deserialise_component(d: dict) -> ModelComponent:
            module = importlib.import_module(d['@module'])
            klass = getattr(module, d['@class'])
            return klass.from_dict(d)

        components = [deserialise_component(c) for c in obj_dict.get('components', [])]

        return cls(
            components=components,
            x_unit=obj_dict.get('x_unit', 'meV'),
            y_unit=obj_dict.get('y_unit', 'dimensionless'),
            name=obj_dict.get('name', 'ComponentCollection'),
            display_name=obj_dict.get('display_name'),
        )

    def __copy__(self) -> ComponentCollection:
        return self.from_dict(self.to_dict())
