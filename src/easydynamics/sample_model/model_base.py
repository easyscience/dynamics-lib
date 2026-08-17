# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import convert_units_with_rollback
from easydynamics.utils.utils import verify_Q_index


class ModelBase(EasyDynamicsModelBase):
    """
    Base class for Sample Models.

    Contains common functionality for models with components and Q dependence.
    """

    def __init__(
        self,
        display_name: str = 'MyModelBase',
        unique_name: str | None = None,
        x_unit: str | sc.Unit | None = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        """
        Initialize the ModelBase.

        Parameters
        ----------
        display_name : str, default='MyModelBase'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        x_unit : str | sc.Unit | None, default='meV'
            Unit of the x-axis (energy, Q, etc.).
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity).
        components : ModelComponent | ComponentCollection | None, default=None
            Template components of the model. If None, no components are added. These components
            are copied into ComponentCollections for each Q value.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.

        Raises
        ------
        TypeError
            If components is not a ModelComponent or ComponentCollection.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            display_name=display_name,
            unique_name=unique_name,
        )
        self._Q = _validate_and_convert_Q(Q)

        if components is not None and not isinstance(
            components, (ModelComponent, ComponentCollection)
        ):
            raise TypeError(
                f'Components must be a ModelComponent, a ComponentCollection or None, '
                f'got {type(components).__name__}'
            )

        self._components = ComponentCollection(x_unit=self.x_unit, y_unit=self.y_unit)
        self._component_collections: list[ComponentCollection] = []
        # Counter part of state_version: bumped whenever the dirty flag is raised.
        self._state_counter = 0
        # Template-collection version the per-Q collections were last built from. Compared
        # against self._components.version so in-place mutations of the live template
        # collection (reachable via the `components` property) are detected without
        # callbacks.
        self._built_components_version = self._components.version
        self._component_collections_is_dirty = True
        if isinstance(components, (ModelComponent, ComponentCollection)):
            self.append_component(components)

    def evaluate(
        self,
        x: Numeric | list | np.ndarray | sc.Variable | sc.DataArray,
        output: str = 'numpy',
    ) -> list[np.ndarray] | list[sc.Variable]:
        """
        Evaluate the sample model at all Q for the given x values.

        Parameters
        ----------
        x : Numeric | list | np.ndarray | sc.Variable | sc.DataArray
            Energy axis values to evaluate the model at.
        output : str, default='numpy'
            'numpy' returns np.ndarray per Q; 'scipp' returns sc.Variable per Q.

        Raises
        ------
        ValueError
            If Q is not set on the model, or if there are no components in the model to evaluate.

        Returns
        -------
        list[np.ndarray] | list[sc.Variable]
            A list of arrays containing the evaluated model values for each Q. The length of the
            list will match the number of Q values in the model.
        """
        self._ensure_component_collections_current()
        if not self._component_collections:
            if self.Q is None:
                raise ValueError(
                    'Q is not set on the model, so there are no per-Q component collections '
                    'to evaluate. Set Q before evaluating.'
                )
            raise ValueError('No components in the model to evaluate.')
        return [
            collection.evaluate(x, output=output) for collection in self._component_collections
        ]

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------
    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """
        Append a ModelComponent or ComponentCollection to the SampleModel.

        Parameters
        ----------
        component : ModelComponent | ComponentCollection
            The ModelComponent or ComponentCollection to append.
        """
        self._components.append_component(component)
        self._on_components_change()

    def remove_component(self, name: str) -> None:
        """
        Remove a ModelComponent from the SampleModel by its name.

        Parameters
        ----------
        name : str
            The name of the ModelComponent to remove.
        """
        self._components.pop(name)
        self._on_components_change()

    def clear_components(self) -> None:
        """Clear all ModelComponents from the SampleModel."""
        self._components.clear()
        self._on_components_change()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self) -> ComponentCollection:
        """
        Get the template ComponentCollection of the SampleModel.

        This is the live template collection: mutating it in place (e.g. via ``append_component``)
        is detected through its ``version`` and triggers a rebuild of the per-Q collections on next
        use.

        Returns
        -------
        ComponentCollection
            The template component collection of the SampleModel.
        """
        return self._components

    @components.setter
    def components(self, value: ModelComponent | ComponentCollection | None) -> None:
        """
        Set the components of the SampleModel.

        Parameters
        ----------
        value : ModelComponent | ComponentCollection | None
            The new components to set. If None, all components will be cleared.

        Raises
        ------
        TypeError
            If value is not a ModelComponent, ComponentCollection, or None.
        """
        if not isinstance(value, (ModelComponent, ComponentCollection, type(None))):
            raise TypeError('Components must be a ModelComponent or a ComponentCollection')

        self.clear_components()
        if value is not None:
            self.append_component(value)

    @property
    def component_collections_is_dirty(self) -> bool:
        """
        Return whether component collections need to be rebuilt before use.

        Collections are stale when the dirty flag was raised (Q or component changes through the
        model's methods) or when the live template collection was mutated in place since the
        collections were last built.

        Returns
        -------
        bool
            ``True`` if component collections have not been built yet or are stale.
        """
        return (
            self._component_collections_is_dirty
            or self._built_components_version != self._components.version
        )

    @property
    def _component_collections_is_dirty(self) -> bool:
        """
        Get the dirty flag for the per-Q component collections.

        Implemented as a property so every write is intercepted: raising the flag bumps the state
        counter (making ``state_version`` change), and clearing it records the template collection
        version the collections were built from.

        Returns
        -------
        bool
            The raw dirty flag (does not account for in-place template mutations; use
            ``component_collections_is_dirty`` for the full staleness check).
        """
        return self._component_collections_dirty_flag

    @_component_collections_is_dirty.setter
    def _component_collections_is_dirty(self, value: bool) -> None:
        """
        Set the dirty flag for the per-Q component collections.

        Parameters
        ----------
        value : bool
            ``True`` marks the collections stale and bumps the state counter. ``False`` marks them
            current and records the template collection version they now correspond to.
        """
        value = bool(value)
        if value:
            self._state_counter += 1
        else:
            self._built_components_version = self._components.version
        self._component_collections_dirty_flag = value

    @property
    def state_version(self) -> int:
        """
        Get a monotonic version of everything affecting the per-Q component collections.

        The value changes whenever Q changes, components are added/removed/replaced through the
        model's methods, or the live template collection (``components``) is mutated in place.
        Implemented as an internal counter plus the template collection's mutation version, so it
        only ever increases. Reading never rebuilds, clears or mutates anything; equal values mean
        the collections' inputs are unchanged.

        Returns
        -------
        int
            The current state version.
        """
        return self._state_counter + self._components.version

    @property
    def Q(self) -> sc.Variable | None:
        """
        Get the Q values of the SampleModel.

        Returns
        -------
        sc.Variable | None
            The Q values of the SampleModel in 1/angstrom, or None if not set.
        """
        return self._Q

    @Q.setter
    def Q(self, value: Q_type | None) -> None:
        """
        Set the Q values of the SampleModel.

        If Q is already set, it throws an error if the new Q values are not similar to the old
        ones. To change Q values, first run clear_Q().

        Parameters
        ----------
        value : Q_type | None
            The new Q values to set. If None, Q values are not changed.

        Raises
        ------
        ValueError
            If the new Q values are not similar to the old ones when Q is already set.
        """
        if value is None:
            return
        old_Q = self._Q
        new_Q = _validate_and_convert_Q(value)

        if old_Q is None:
            self._Q = new_Q
            self._on_Q_change()
            return

        if len(old_Q) != len(new_Q) or not sc.allclose(old_Q, new_Q):
            raise ValueError(
                'New Q values are not similar to the old ones. '
                'To change Q values, first run clear_Q().'
            )

    def clear_Q(self, confirm: bool = False) -> None:
        """
        Clear the Q values of the SampleModel, removing all component collections and their
        associated Parameters.

        Parameters
        ----------
        confirm : bool, default=False
            Confirmation to clear Q values.

        Raises
        ------
        ValueError
            If confirm is not True.
        """
        if not confirm:
            raise ValueError(
                'Clearing Q values requires confirmation. Set confirm=True to proceed.'
            )
        self._Q = None
        self._on_Q_change()

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def convert_x_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the x-axis unit of all components in the model.

        Parameters
        ----------
        unit : str | sc.Unit
            The new x-axis unit to convert to.
        """
        self._convert_axis_unit(unit, axis='x')

    def convert_y_unit(self, unit: str | sc.Unit) -> None:
        """
        Convert the y-axis unit of all components in the model.

        Parameters
        ----------
        unit : str | sc.Unit
            The new y-axis unit to convert to.
        """
        self._convert_axis_unit(unit, axis='y')

    def _convert_axis_unit(self, unit: str | sc.Unit, axis: str) -> None:
        """
        Convert one axis unit on all template components and per-Q collections.

        Converts every child via its ``convert_<axis>_unit`` method and updates the model's own
        unit attribute. On failure, attempts a best-effort rollback of all children to the old unit
        before re-raising the failing conversion's exception.

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
        """
        if not isinstance(unit, (str, sc.Unit)):
            raise TypeError(f'Unit must be a string or sc.Unit, got {type(unit).__name__}')

        method = f'convert_{axis}_unit'
        old_unit = self.x_unit if axis == 'x' else self.y_unit
        # Convert the template collection as a whole (not its unpacked components) so its own
        # unit attribute is updated too; regenerated per-Q collections copy that attribute.
        children = [self._components, *self._component_collections]
        convert_units_with_rollback([
            (getattr(child, method), unit, old_unit) for child in children
        ])
        unit_str = str(unit) if isinstance(unit, sc.Unit) else unit
        if axis == 'x':
            self._x_unit = unit_str
        else:
            self._y_unit = unit_str

    def fix_all_parameters(self) -> None:
        """Fix all Parameters in all ComponentCollections."""
        for par in self.get_all_variables():
            par.fixed = True

    def free_all_parameters(self) -> None:
        """Free all Parameters in all ComponentCollections."""
        for par in self.get_all_variables():
            par.fixed = False

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all Parameters and Descriptors from all ComponentCollections in the ModelBase.

        Ignores the Parameters and Descriptors in self._components as these are just templates.

        Parameters
        ----------
        Q_index : int | None, default=None
            If None, get variables for all ComponentCollections. If int, get variables for the
            ComponentCollection at this index.

        Returns
        -------
        list[Parameter]
            A list of all Parameters and Descriptors from the ComponentCollections in the
            ModelBase.
        """
        self._ensure_component_collections_current()
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)
        if Q_index is None:
            all_vars = [
                var
                for collection in self._component_collections
                for var in collection.get_all_variables()
            ]
        else:
            all_vars = self._component_collections[Q_index].get_all_variables()
        return all_vars

    def get_component_collection(self, Q_index: int) -> ComponentCollection:
        """
        Get the ComponentCollection at the given Q index.

        Parameters
        ----------
        Q_index : int
            The index of the desired ComponentCollection.

        Returns
        -------
        ComponentCollection
            The ComponentCollection at the given Q index.
        """
        self._ensure_component_collections_current()
        verify_Q_index(Q_index=Q_index, Q=self.Q)
        return self._component_collections[Q_index]

    def normalize_area(self) -> None:
        """Normalize the area of the model across all Q values."""
        self._ensure_component_collections_current()
        for collection in self._component_collections:
            collection.normalize_area()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _ensure_component_collections_current(self) -> None:
        """
        Rebuild component collections if any dependency has changed since they were last built.

        Uses the full staleness check, so both flag-raising changes (Q, component methods) and
        in-place mutations of the live template collection trigger a rebuild.
        """
        if self.component_collections_is_dirty:
            self._generate_component_collections()
            self._component_collections_is_dirty = False

    def _generate_component_collections(self) -> None:
        """Generate ComponentCollections for each Q value."""
        if self.Q is None:
            self._component_collections = []
            return

        self._component_collections = []
        for _ in range(len(self.Q)):
            self._component_collections.append(copy(self._components))

    def _on_Q_change(self) -> None:
        """Handle changes to the Q values."""
        self._component_collections_is_dirty = True

    def _on_components_change(self) -> None:
        """Handle changes to the components."""
        self._component_collections_is_dirty = True

    # ------------------------------------------------------------------
    # dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a string representation of the ModelBase.

        Returns
        -------
        str
            A string representation of the ModelBase.
        """
        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'x_unit={self.x_unit}, '
            f'y_unit={self.y_unit}, '
            f'Q={None if self.Q is None else self.Q.values}, '
            f'components={self.components})'
        )
