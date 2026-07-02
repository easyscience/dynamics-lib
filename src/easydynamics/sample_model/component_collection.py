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
    Collection of model components.

    Examples
    --------
    **Creating a ComponentCollection with multiple components**

    ```python
    import numpy as np
    import easydynamics.sample_model as sm

    component1 = sm.Gaussian(name='Gaussian1', area=1.0, width=1.0)
    component2 = sm.Lorentzian(name='Lorentzian1', area=2.0, width=0.5)
    collection = sm.ComponentCollection(components=[component1, component2])
    ```

    **Evaluating, appending, and removing components**

    ```python
    x = np.linspace(-5, 5, 100)
    values = collection.evaluate(x)

    component3 = sm.Gaussian(name='Gaussian2', area=0.5, width=0.8)
    collection.append(component3)

    collection.remove('Gaussian1')
    collection.list_component_names()  # ['Lorentzian1', 'Gaussian2']
    ```
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
        """
        Initialize a new ComponentCollection.

        Parameters
        ----------
        components : ModelComponent | list[ModelComponent] | None, default=None
            Initial model components to add to the ComponentCollection.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy, Q, etc.).
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity).
        name : str, default='ComponentCollection'
            Name of the collection.
        display_name : str | None, default=None
            Display name of the collection.
        unique_name : str | None, default=None
            Unique name of the collection.

        Raises
        ------
        TypeError
            If components is not a list of ModelComponent.
        """
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
        """
        Check if the ComponentCollection has no components.

        Returns
        -------
        bool
            True if the collection has no components, False otherwise.
        """
        return not self

    @is_empty.setter
    def is_empty(self, _value: bool) -> None:
        """
        Is_empty is a read-only property that indicates whether the collection has components.

        Parameters
        ----------
        _value : bool
            The value to set (ignored).

        Raises
        ------
        AttributeError
            Always raised since is_empty is read-only.
        """
        raise AttributeError(
            'is_empty is a read-only property that indicates '
            'whether the collection has components.'
        )

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert the x-axis unit of the ComponentCollection and all its components.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            The target x-axis unit to convert to.
        """
        self._convert_axis_unit(new_x_unit, axis='x')

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit of the ComponentCollection and all its components.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            The target y-axis unit to convert to.
        """
        self._convert_axis_unit(new_y_unit, axis='y')

    def _convert_axis_unit(self, unit: str | sc.Unit, axis: str) -> None:
        """
        Convert one axis unit on all components in the collection.

        Converts every component via its ``convert_<axis>_unit`` method and updates the
        collection's own unit attribute. On failure, attempts a best-effort rollback of all
        components to the old unit before re-raising.

        Parameters
        ----------
        unit : str | sc.Unit
            The new unit to convert to.
        axis : str
            Which axis to convert: ``'x'`` or ``'y'``.

        Raises
        ------
        TypeError
            If the provided unit is not a string or sc.Unit.
        Exception
            If any component cannot be converted to the specified unit.
        """
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError(f'{axis}_unit must be a string or sc.Unit, got {type(unit).__name__}')

        method = f'convert_{axis}_unit'
        old_unit = self.x_unit if axis == 'x' else self.y_unit
        try:
            for component in self:
                getattr(component, method)(unit)
            unit_str = str(unit) if isinstance(unit, sc.Unit) else unit
            if axis == 'x':
                self._x_unit = unit_str
            else:
                self._y_unit = unit_str
        except Exception as e:
            if old_unit is not None:
                try:
                    for component in self:
                        getattr(component, method)(old_unit)
                except Exception:  # noqa: S110
                    pass
            raise e

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """
        Append a model component or the components from another ComponentCollection to this
        ComponentCollection.

        Parameters
        ----------
        component : ModelComponent | ComponentCollection
            The component to append. If a ComponentCollection is provided, all of its components
            will be appended.
        """
        if isinstance(component, ComponentCollection):
            self.extend(component)
        else:
            self.append(component)
        self._warn_if_duplicate_names()

    def list_component_names(self) -> list[str]:
        """
        List the names of all components in the model.

        Returns
        -------
        list[str]
            List of names of the components in the collection.
        """
        return [component.name for component in self]

    def normalize_area(self) -> None:
        """
        Normalize the areas of all components so they sum to 1.

        This is useful for convolutions.

        Raises
        ------
        ValueError
            If there are no components in the model or if the total area is zero or not finite,
            which would prevent normalization.
        """
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

        if not area_params:
            raise ValueError('No components with an area attribute; cannot normalize.')

        # Sum the areas in a common unit so components with different (but compatible)
        # units normalize correctly. Dividing each value by the total expressed in the
        # reference unit makes the areas sum to 1 in that unit.
        reference_unit = str(area_params[0].unit)
        total_area_value = sum(
            sc.to_unit(sc.scalar(p.value, unit=str(p.unit)), reference_unit).value
            for p in area_params
        )

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
        """
        Get all parameters from all model components.

        Returns
        -------
        list[DescriptorBase]
            List of parameters in the collection.
        """
        return [var for component in self for var in component.get_all_variables()]

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        """
        Evaluate the sum of all components.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Energy axis.
        output : str, default='numpy'
            'numpy' returns np.ndarray; 'scipp' returns sc.Variable with y_unit.

        Returns
        -------
        np.ndarray | sc.Variable
            Evaluated model values.
        """
        if not self:
            if isinstance(x, (sc.Variable, sc.DataArray)):
                values = np.zeros_like(x.values, dtype=float)
                dim = x.dims[0] if x.dims else 'x'
            else:
                values = np.zeros_like(x, dtype=float)
                dim = 'x'
            if output == 'scipp':
                return sc.array(dims=[dim], values=values, unit=self.y_unit)
            return values
        # This is needed to handle both scipp and numpy output - a normal call to sum does not work
        gen = (component.evaluate(x, output=output) for component in self)
        first = next(gen)
        return sum(gen, first)

    def evaluate_component(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        name: str,
        output: str = 'numpy',
    ) -> np.ndarray | sc.Variable:
        """
        Evaluate a single component by name.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Energy axis.
        name : str
            Component name.
        output : str, default='numpy'
            'numpy' returns np.ndarray; 'scipp' returns sc.Variable with y_unit.

        Raises
        ------
        ValueError
            If there are no components in the model.
        TypeError
            If name is not a string.
        KeyError
            If no component with the given name exists in the collection.

        Returns
        -------
        np.ndarray | sc.Variable
            Evaluated values for the specified component.
        """
        if not self:
            raise ValueError('No components in the model to evaluate.')
        if not isinstance(name, str):
            raise TypeError(f'Component name must be a string, got {type(name)} instead.')
        matches = [comp for comp in self if comp.name == name]
        if not matches:
            raise KeyError(f"No component named '{name}' exists.")
        return matches[0].evaluate(x, output=output)

    def fix_all_parameters(self) -> None:
        """Fix all free parameters in the model."""
        for param in self.get_fittable_parameters():
            param.fixed = True

    def free_all_parameters(self) -> None:
        """Free all fixed parameters in the model."""
        for param in self.get_fittable_parameters():
            param.fixed = False

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _warn_if_duplicate_names(self) -> None:
        """Warn if any two components share the same name."""
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
        """
        Return a string representation of the ComponentCollection.

        Returns
        -------
        str
            String representation of the ComponentCollection.
        """
        comp_names = ', '.join(c.name for c in self) or 'No components'
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"x_unit='{self.x_unit}', y_unit='{self.y_unit}',\n"
            f'Components: {comp_names})'
        )

    def to_dict(self) -> dict:
        """
        Serialise the ComponentCollection to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the ComponentCollection.
        """
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
        """
        Deserialise a ComponentCollection from its dictionary representation.

        Parameters
        ----------
        obj_dict : dict
            Dictionary representation of the ComponentCollection, as produced by to_dict().

        Returns
        -------
        ComponentCollection
            The deserialised ComponentCollection.
        """

        def deserialise_component(d: dict) -> ModelComponent:
            """
            Deserialise a component from its dictionary representation.
            Parameters
            ----------
            d : dict
                The dictionary representation of the component.
            Returns
            -------
            ModelComponent
                The deserialised component.
            """
            module = importlib.import_module(d['@module'])
            cls = getattr(module, d['@class'])
            return cls.from_dict(d)

        components = [deserialise_component(c) for c in obj_dict['components']]

        return cls(
            components=components,
            x_unit=obj_dict['x_unit'],
            y_unit=obj_dict['y_unit'],
            name=obj_dict['name'],
            display_name=obj_dict['display_name'],
        )

    def __copy__(self) -> ComponentCollection:
        """
        Create a deep copy of the ComponentCollection.

        Returns
        -------
        ComponentCollection
            A deep copy of the ComponentCollection.
        """
        return self.from_dict(self.to_dict())
