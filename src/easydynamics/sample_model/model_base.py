# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import plopp as pp
import scipp as sc
from plopp.backends.matplotlib.figure import WidgetFigure

from easydynamics.base_classes.easydynamics_modelbase import EasyDynamicsModelBase
from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.dependency_functions import create_dependency_functions
from easydynamics.utils.plotting import relabel_lines
from easydynamics.utils.utils import CANONICAL_Q_UNIT
from easydynamics.utils.utils import Q_TOLERANCE
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type
from easydynamics.utils.utils import _Q_values_match
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import _validate_Q_tolerance
from easydynamics.utils.utils import convert_units_with_rollback
from easydynamics.utils.utils import hbar
from easydynamics.utils.utils import kb
from easydynamics.utils.utils import verify_Q_index
from easydynamics.variable import DescriptorNumber
from easydynamics.variable import Parameter


@dataclass
class _QDependence:
    """
    The Q dependence of one Parameter of a component.

    Attributes
    ----------
    expression : str
        The expression giving the value of the Parameter as a function of ``Q``, e.g.
        ``'hbar * D * Q**2'``. For a Q-independent Parameter it is just the symbol of the one
        shared Parameter, e.g. ``'width'``.
    parameters : dict[str, Parameter]
        The Parameters used in the expression, keyed by their symbol in it, e.g.
        ``{'D': <Parameter 'Lorentzian width D'>}``. They are shared by all Q, and are what a fit
        varies.
    """

    expression: str
    parameters: dict[str, Parameter]


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
        self._Q_tolerance = Q_TOLERANCE

        if components is not None and not isinstance(
            components, (ModelComponent, ComponentCollection)
        ):
            raise TypeError(
                f'Components must be a ModelComponent, a ComponentCollection or None, '
                f'got {type(components).__name__}'
            )

        self._components = ComponentCollection(x_unit=self.x_unit, y_unit=self.y_unit)
        self._component_collections: list[ComponentCollection] = []
        # The Q dependence of every Parameter that has one, including a Q-independent one, keyed
        # by (component name, parameter name), e.g. ('Lorentzian', 'width'). Stored so the
        # dependences can be reapplied when the per-Q collections are rebuilt.
        self._Q_dependences: dict[tuple[str, str], _QDependence] = {}
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

        If the per-Q collections are current, a copy of the new component(s) is inserted into each
        of them in place, so existing per-Q parameters (e.g. fitted values) are kept.

        Parameters
        ----------
        component : ModelComponent | ComponentCollection
            The ModelComponent or ComponentCollection to append.
        """
        can_patch = self._can_patch_component_collections()
        n_before = len(self._components)
        self._components.append_component(component)
        if not can_patch:
            self._on_components_change()
            return
        added = self._components[n_before:]
        for collection in self._component_collections:
            for offset, new_component in enumerate(self._copy_templates(added)):
                collection.insert(n_before + offset, new_component)
        self._mark_component_collections_patched()

    def remove_component(self, name: str) -> None:
        """
        Remove a ModelComponent from the SampleModel by its name.

        If the per-Q collections are current, the component is removed from each of them in place,
        so the per-Q parameters of the remaining components are kept.

        Parameters
        ----------
        name : str
            The name of the ModelComponent to remove.
        """
        can_patch = self._can_patch_component_collections()
        names = self._components.list_component_names()
        self._components.pop(name)
        if not can_patch:
            self._on_components_change()
            return
        index = names.index(name)
        for collection in self._component_collections:
            collection.pop(index)
        self._mark_component_collections_patched()

    def clear_components(self) -> None:
        """Clear all ModelComponents from the SampleModel, and their Q constraints."""
        self._components.clear()
        self._Q_dependences = {}
        self._on_components_change()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self) -> ComponentCollection:
        """
        Get the template ComponentCollection of the SampleModel.

        This is the live template collection: mutating it in place (e.g.
        ``model.components.append_component(...)``, bypassing the model's own methods) is detected
        through its ``version`` and triggers a full rebuild of the per-Q collections on next use.

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

        If Q is already set, the old Q values are kept, and it throws an error if the new ones
        differ from them by more than Q_tolerance. To change Q values, first run clear_Q().

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

        if not _Q_values_match(old_Q, new_Q, self._Q_tolerance):
            raise ValueError(
                'New Q values are not similar to the old ones. '
                'To accept a larger difference, increase Q_tolerance. '
                'To change Q values, first run clear_Q().'
            )

    @property
    def Q_tolerance(self) -> float:
        """
        Get the largest difference at which new Q values count as the same as the model's own.

        Returns
        -------
        float
            The tolerance in 1/angstrom.
        """
        return self._Q_tolerance

    @Q_tolerance.setter
    def Q_tolerance(self, value: float) -> None:
        """
        Set the largest difference at which new Q values count as the same as the model's own.

        Increase it to use the model with data whose Q values differ slightly from the ones the
        model was made with. The model keeps its own Q values.

        Parameters
        ----------
        value : float
            The tolerance in 1/angstrom. Must be non-negative and finite.
        """
        self._Q_tolerance = _validate_Q_tolerance(value)

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
        """
        Fix all Parameters in all ComponentCollections.

        Dependent Parameters, such as those generated by a diffusion model, cannot be fixed and are
        skipped; they follow the Parameters they depend on.

        The template components are fixed too, so the call also takes effect on per-Q collections
        built later, e.g. when Q is set after this call.
        """
        self._components.fix_all_parameters()
        for par in self.get_fittable_parameters():
            par.fixed = True

    def free_all_parameters(self) -> None:
        """
        Free all Parameters in all ComponentCollections.

        Dependent Parameters, such as those generated by a diffusion model, are skipped.

        The template components are freed too, so the call also takes effect on per-Q collections
        built later, e.g. when Q is set after this call.
        """
        self._components.free_all_parameters()
        for par in self.get_fittable_parameters():
            par.fixed = False

    def get_parameters(
        self,
        component_name: str,
        parameter_name: str,
        Q_index: int | None = None,
    ) -> list[Parameter]:
        """
        Get one Parameter of a component at each Q.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component, e.g. ``'width'``.
        Q_index : int | None, default=None
            If None, get the Parameter at every Q. If int, get only the Parameter at this Q index.

        Returns
        -------
        list[Parameter]
            The Parameter at each requested Q, in Q order. Empty if Q is not set.
        """
        self._ensure_component_collections_current()
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)
        if Q_index is None:
            collections = self._component_collections
        else:
            collections = [self._component_collections[Q_index]]
        return [
            self._get_parameter(collection, component_name, parameter_name)
            for collection in collections
        ]

    def fix_parameter(
        self,
        component_name: str,
        parameter_name: str,
        Q_index: int | None = None,
        value: Numeric | None = None,
    ) -> None:
        """
        Fix one Parameter of a component, at every Q or at a single Q.

        When fixing at every Q, the template component is fixed too, so the call also takes effect
        on per-Q collections built later.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component, e.g. ``'width'``.
        Q_index : int | None, default=None
            If None, fix the Parameter at every Q. If int, fix it only at this Q index.
        value : Numeric | None, default=None
            If given, the Parameter is set to this value (in its own unit) before it is fixed.
        """
        for parameter in self._get_parameters_with_template(
            component_name, parameter_name, Q_index
        ):
            if value is not None:
                parameter.value = value
            parameter.fixed = True

    def free_parameter(
        self,
        component_name: str,
        parameter_name: str,
        Q_index: int | None = None,
    ) -> None:
        """
        Free one Parameter of a component, at every Q or at a single Q.

        When freeing at every Q, the template component is freed too, so the call also takes effect
        on per-Q collections built later.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component, e.g. ``'width'``.
        Q_index : int | None, default=None
            If None, free the Parameter at every Q. If int, free it only at this Q index.
        """
        for parameter in self._get_parameters_with_template(
            component_name, parameter_name, Q_index
        ):
            parameter.fixed = False

    def make_parameter_Q_independent(self, component_name: str, parameter_name: str) -> Parameter:
        """
        Make one Parameter of a component take the same value at every Q.

        This is the Q dependence where the Parameter equals a constant: the Parameter at every Q is
        made dependent on one new Parameter owned by the model, named
        ``'<component_name> <parameter_name> (all Q)'``, which is what a fit varies. It starts with
        the value, unit, bounds and fixed flag of the Parameter at Q index 0, or of the template
        when Q is not set. Use ``fit_method='simultaneous'``. Any other Q constraint on the
        Parameter is replaced, and calling this again returns the same Parameter.

        The constraint is reapplied whenever the per-Q collections are rebuilt, and applies when Q
        is set later. Use ``remove_parameter_Q_constraint`` to undo it.

        A Parameter that already depends on other Parameters, e.g. because it is generated by a
        diffusion model, is rejected with a ValueError.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component, e.g. ``'width'``.

        Returns
        -------
        Parameter
            The new Parameter that sets the value at every Q.
        """
        dependence = self._Q_dependences.get((component_name, parameter_name))
        if dependence is not None and dependence.expression == parameter_name:
            return dependence.parameters[parameter_name]
        self.remove_parameter_Q_constraint(component_name, parameter_name)
        source = self._get_unconstrained_parameters(component_name, parameter_name)[0]
        shared = Parameter(
            name=f'{component_name} {parameter_name} (all Q)',
            value=source.value,
            unit=source.unit,
            min=source.min,
            max=source.max,
            fixed=source.fixed,
        )
        self._set_Q_dependence(
            component_name, parameter_name, parameter_name, {parameter_name: shared}
        )
        return shared

    def set_parameter_Q_dependence(
        self,
        component_name: str,
        parameter_name: str,
        expression: str,
        parameters: dict[str, Numeric],
        parameter_units: dict[str, str | sc.Unit] | None = None,
    ) -> list[Parameter]:
        """
        Make one Parameter of a component follow an expression in Q.

        The Parameter at every Q is made dependent on new Parameters owned by the model, which are
        shared by all Q and are what a fit varies. Use ``fit_method='simultaneous'``. Any existing
        Q constraint on the Parameter is replaced.

        The expression may use ``Q`` (in 1/angstrom), the constants ``hbar`` and ``kb``, and the
        symbols given in *parameters*, combined with ``+ - * / **`` and the functions ``exp``,
        ``sin``, ``cos`` and ``sinc`` (sin(x)/x) of a dimensionless argument. Units are carried
        through the expression, and its result must be convertible to the unit of the Parameter:
        ```python
        (D,) = sample_model.set_parameter_Q_dependence(
            'Lorentzian',
            'width',
            expression='hbar * D * Q**2',
            parameters={'D': 2.4e-9},
            parameter_units={'D': 'm**2/s'},
        )
        ```

        The dependence is reapplied whenever the per-Q collections are rebuilt, and applies when Q
        is set later. Use ``remove_parameter_Q_constraint`` to undo it.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component, e.g. ``'width'``.
        expression : str
            The expression giving the value of the Parameter as a function of ``Q``.
        parameters : dict[str, Numeric]
            Initial values of the new Parameters, keyed by their symbol in the expression.
        parameter_units : dict[str, str | sc.Unit] | None, default=None
            Units of the new Parameters, keyed by symbol. Symbols without an entry are
            dimensionless.

        Raises
        ------
        ValueError
            If *parameters* uses a reserved symbol (``Q``, a constant or a function), if
            *parameter_units* names a symbol that is not in *parameters*, or if the Parameter
            already depends on other Parameters through something other than a Q constraint.

        Returns
        -------
        list[Parameter]
            The new Parameters, in the order of *parameters*.
        """
        parameter_units = parameter_units or {}
        reserved = sorted({'Q', 'hbar', 'kb', *create_dependency_functions()} & set(parameters))
        if reserved:
            raise ValueError(f'{reserved} are reserved symbols and cannot be used as parameters.')
        unknown = sorted(set(parameter_units) - set(parameters))
        if unknown:
            raise ValueError(f'parameter_units given for unknown parameter(s): {unknown}.')

        new_parameters = {
            symbol: Parameter(
                name=f'{component_name} {parameter_name} {symbol}',
                value=value,
                unit=parameter_units.get(symbol, 'dimensionless'),
            )
            for symbol, value in parameters.items()
        }
        self._set_Q_dependence(component_name, parameter_name, expression, new_parameters)
        return list(new_parameters.values())

    def remove_parameter_Q_constraint(self, component_name: str, parameter_name: str) -> None:
        """
        Undo ``make_parameter_Q_independent`` or ``set_parameter_Q_dependence``, so the Parameter
        varies freely with Q again.

        Each per-Q Parameter keeps its current value and comes back free, with the bounds of the
        template. Does nothing if the Parameter has no constraint.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component, e.g. ``'width'``.
        """
        if self._Q_dependences.pop((component_name, parameter_name), None) is None:
            return
        template = None
        if component_name in self._components.list_component_names():
            template = self._get_parameter(self._components, component_name, parameter_name)
        for parameter in self.get_parameters(component_name, parameter_name):
            if not parameter.independent:
                parameter.make_independent()
                if template is not None:
                    # A dependent Parameter takes the bounds of its expression.
                    parameter.min, parameter.max = template.min, template.max

    def get_all_variables(self, Q_index: int | None = None) -> list[Parameter]:
        """
        Get all Parameters and Descriptors from all ComponentCollections in the ModelBase.

        Ignores the Parameters and Descriptors in self._components as these are just templates.
        Includes the Parameters created by ``set_parameter_Q_dependence``, which are shared by all
        Q.

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
        for dependence in self._Q_dependences.values():
            all_vars.extend(dependence.parameters.values())
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

    def to_dict(self, skip: list[str] | None = None) -> dict:
        """
        Convert the model to a dictionary, including its Q constraints and Q dependences.

        Parameters
        ----------
        skip : list[str] | None, default=None
            Names of fields to leave out.

        Returns
        -------
        dict
            Dictionary representation of the model.
        """
        obj_dict = super().to_dict(skip=skip)
        if self._Q_dependences:
            obj_dict['Q_dependences'] = [
                {
                    'component_name': component_name,
                    'parameter_name': parameter_name,
                    'expression': dependence.expression,
                    'parameters': {
                        symbol: parameter.to_dict(skip=skip)
                        for symbol, parameter in dependence.parameters.items()
                    },
                }
                for (component_name, parameter_name), dependence in self._Q_dependences.items()
            ]
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'ModelBase':
        """
        Re-create a model from a dictionary, including its Q constraints and Q dependences.

        Parameters
        ----------
        obj_dict : dict
            Dictionary representation of the model, as produced by to_dict().

        Returns
        -------
        'ModelBase'
            The deserialised model.
        """
        obj_dict = dict(obj_dict)
        Q_dependences = obj_dict.pop('Q_dependences', [])
        model = super().from_dict(obj_dict)
        for dependence in Q_dependences:
            parameters = {
                symbol: Parameter.from_dict(parameter_dict)
                for symbol, parameter_dict in dependence['parameters'].items()
            }
            key = (dependence['component_name'], dependence['parameter_name'])
            model._Q_dependences[key] = _QDependence(  # ruff: ignore[private-member-access]
                dependence['expression'], parameters
            )
        model._apply_Q_constraints()  # ruff: ignore[private-member-access]
        return model

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        energy: sc.Variable,
        Q_index: int | None = None,
        plot_components: bool = True,
        **kwargs: dict[str, Any],
    ) -> WidgetFigure:
        """
        Plot the model and, optionally, its individual components on an energy grid.

        Without Q_index, a slider selects the Q value to show. This is how to look at the
        components of a resolution model, e.g.
        ``analysis.instrument_model.resolution_model.plot(energy)``. Energy offsets are not
        applied, since they belong to the instrument model, not to this model.

        Uses Plopp for plotting: https://scipp.github.io/plopp/

        Parameters
        ----------
        energy : sc.Variable
            1D energy grid to evaluate the model on.
        Q_index : int | None, default=None
            Index of the Q value to plot. If None, plot all Q values.
        plot_components : bool, default=True
            Whether to plot the individual components as well as their sum. Each component needs a
            unique name, and is labelled by its display name in the legend.
        **kwargs : dict[str, Any]
            Additional keyword arguments passed to plopp for customizing the plot.

        Raises
        ------
        TypeError
            If energy is not a 1D sc.Variable, or if plot_components is not a bool.
        ValueError
            If Q is not set or the model has no components.

        Returns
        -------
        WidgetFigure
            A Plopp WidgetFigure of the model.
        """
        if not isinstance(energy, sc.Variable) or energy.ndim != 1:
            raise TypeError('energy must be a 1D sc.Variable.')
        if not isinstance(plot_components, bool):
            raise TypeError(
                f'plot_components must be True or False, not {type(plot_components).__name__}'
            )
        verify_Q_index(Q_index=Q_index, Q=self.Q, allow_none=True)
        self._ensure_component_collections_current()
        collections = self._component_collections
        if not collections or collections[0].is_empty:
            raise ValueError('Set Q and add components to the model before plotting it.')

        # One list of per-Q curves for the total and for each component, stacked along Q.
        curves = {'Total': [c.evaluate(energy, output='scipp') for c in collections]}
        if plot_components:
            for name in collections[0].list_component_names():
                curves[name] = [
                    c.evaluate_component(energy, name, output='scipp') for c in collections
                ]
        coords = {'Q': self.Q, energy.dim: energy}
        model = sc.DataGroup({
            key: sc.DataArray(sc.concat(values, dim='Q'), coords=coords)
            for key, values in curves.items()
        })

        linestyle = dict.fromkeys(model, '--')
        linestyle['Total'] = '-'
        plot_kwargs = {
            'title': self.display_name,
            'linestyle': linestyle,
            'marker': dict.fromkeys(model),
            'color': {'Total': 'red'},
            **kwargs,
        }

        if Q_index is None:
            fig = pp.slicer(model, **{'keep': energy.dim, **plot_kwargs})
            for widget in fig.bottom_bar[0].controls.values():
                widget.slider_toggler.value = '-o-'
        else:
            fig = pp.plot(model['Q', Q_index], **plot_kwargs)

        if plot_components:
            relabel_lines(
                fig, {component.name: component.display_name for component in collections[0]}
            )
        fig.autoscale()
        return fig

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _get_parameter(
        collection: ComponentCollection,
        component_name: str,
        parameter_name: str,
    ) -> Parameter:
        """
        Get a Parameter of a named component in a collection.

        Parameters
        ----------
        collection : ComponentCollection
            The collection holding the component.
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component.

        Raises
        ------
        ValueError
            If the component has no Parameter with the given attribute name.

        Returns
        -------
        Parameter
            The Parameter.
        """
        parameter = getattr(collection[component_name], parameter_name, None)
        if not isinstance(parameter, Parameter):
            raise ValueError(f'Component {component_name!r} has no Parameter {parameter_name!r}.')
        return parameter

    def _get_unconstrained_parameters(
        self,
        component_name: str,
        parameter_name: str,
    ) -> list[Parameter]:
        """
        Get one Parameter of a component at each Q and in the template, for adding a Q constraint.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component.

        Raises
        ------
        ValueError
            If the Parameter already depends on other Parameters at some Q.

        Returns
        -------
        list[Parameter]
            The matching Parameters.
        """
        parameters = self._get_parameters_with_template(component_name, parameter_name, None)
        if not all(parameter.independent for parameter in parameters):
            raise ValueError(
                f'Parameter {parameter_name!r} of component {component_name!r} already '
                'depends on other Parameters, so it cannot be given a Q constraint.'
            )
        return parameters

    def _set_Q_dependence(
        self,
        component_name: str,
        parameter_name: str,
        expression: str,
        parameters: dict[str, Parameter],
    ) -> None:
        """
        Replace any Q constraint on one Parameter of a component with a Q dependence.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component.
        expression : str
            The expression giving the value of the Parameter as a function of ``Q``.
        parameters : dict[str, Parameter]
            The Parameters used in the expression, keyed by symbol.
        """
        targets = self._get_parameters_with_template(component_name, parameter_name, None)
        # Try the expression on a throwaway Parameter first, so that an invalid expression or
        # unit raises before any Parameter of the model is changed, also when Q is not set yet.
        probe = Parameter(name='probe', value=0.0, unit=str(targets[0].unit))
        self._make_Q_dependent(probe, expression, parameters, Q=1.0)
        probe.make_independent()

        self.remove_parameter_Q_constraint(component_name, parameter_name)
        self._get_unconstrained_parameters(component_name, parameter_name)
        self._Q_dependences[component_name, parameter_name] = _QDependence(expression, parameters)
        self._apply_Q_constraints()

    @staticmethod
    def _make_Q_dependent(
        parameter: Parameter,
        expression: str,
        parameters: dict[str, Parameter],
        Q: float,
    ) -> None:
        """
        Make a Parameter dependent on an expression in Q, keeping its unit.

        Parameters
        ----------
        parameter : Parameter
            The Parameter to make dependent.
        expression : str
            The expression giving the value of the Parameter.
        parameters : dict[str, Parameter]
            The Parameters used in the expression, keyed by symbol.
        Q : float
            The value of ``Q`` in the expression, in 1/angstrom.
        """
        # The constants and functions are private copies: make_dependent_on attaches the
        # Parameter as an observer of every mapped variable, which must not accumulate on
        # shared objects.
        dependency_map = {
            **parameters,
            'Q': DescriptorNumber(name='Q', value=float(Q), unit=CANONICAL_Q_UNIT),
            'hbar': DescriptorNumber(name='hbar', value=hbar.value, unit=str(hbar.unit)),
            'kb': DescriptorNumber(name='kb', value=kb.value, unit=str(kb.unit)),
            # Temporary, until EasyScience supports functions in dependency expressions.
            **create_dependency_functions(),
        }
        # easyscience propagates inf bounds through arithmetic, producing inf/inf=nan as a
        # transient intermediate; suppress the spurious numpy RuntimeWarning.
        with np.errstate(invalid='ignore'):
            parameter.make_dependent_on(
                dependency_expression=expression,
                dependency_map=dependency_map,
                desired_unit=str(parameter.unit),
            )

    def _get_parameters_with_template(
        self,
        component_name: str,
        parameter_name: str,
        Q_index: int | None,
    ) -> list[Parameter]:
        """
        Get one Parameter of a component at each Q, plus the template's when all Q are requested.

        Including the template makes a change to every Q survive a rebuild of the per-Q
        collections. Components that are not in the template (those generated by subclasses) only
        exist per Q.

        Parameters
        ----------
        component_name : str
            The name of the component the Parameter belongs to.
        parameter_name : str
            The attribute name of the Parameter on the component.
        Q_index : int | None
            If None, get the Parameter at every Q and in the template. If int, get only the
            Parameter at this Q index.

        Returns
        -------
        list[Parameter]
            The matching Parameters.
        """
        parameters = self.get_parameters(component_name, parameter_name, Q_index)
        in_template = component_name in self._components.list_component_names()
        if Q_index is None and (in_template or not parameters):
            parameters.append(
                self._get_parameter(self._components, component_name, parameter_name)
            )
        return parameters

    def _ensure_component_collections_current(self) -> None:
        """
        Rebuild component collections if any dependency has changed since they were last built.

        Uses the full staleness check, so both flag-raising changes (Q, component methods) and
        in-place mutations of the live template collection trigger a rebuild.
        """
        if self.component_collections_is_dirty:
            self._generate_component_collections()
            self._component_collections_is_dirty = False
            self._apply_Q_constraints()

    def _apply_Q_constraints(self) -> None:
        """
        Apply the stored Q constraints to the per-Q collections.

        Makes every Parameter with a Q dependence follow its expression. Called after the per-Q
        collections are rebuilt or patched. Parameters that are already dependent are left alone,
        and constraints on components that no longer exist are dropped.
        """
        if self.component_collections_is_dirty or not self._component_collections:
            return
        names = self._component_collections[0].list_component_names()
        self._Q_dependences = {
            (component_name, parameter_name): dependence
            for (component_name, parameter_name), dependence in self._Q_dependences.items()
            if component_name in names
        }
        for (component_name, parameter_name), dependence in self._Q_dependences.items():
            targets = self.get_parameters(component_name, parameter_name)
            for parameter, Q in zip(targets, self.Q.values, strict=True):
                if parameter.independent:
                    self._make_Q_dependent(
                        parameter, dependence.expression, dependence.parameters, Q=Q
                    )

    def _generate_component_collections(self) -> None:
        """Generate ComponentCollections for each Q value."""
        if self.Q is None:
            self._component_collections = []
            return

        self._component_collections = []
        for _ in range(len(self.Q)):
            self._component_collections.append(self._copy_templates(self._components))

    @staticmethod
    def _copy_templates(templates: ComponentCollection) -> ComponentCollection:
        """
        Copy template components for one Q value, recording each copy's template.

        Parameters
        ----------
        templates : ComponentCollection
            The template components to copy.

        Returns
        -------
        ComponentCollection
            The copies, in the same order, each returning its template from ``get_template``.
        """
        copies = copy(templates)
        for template, component in zip(templates, copies, strict=True):
            component._set_origin(template=template)  # ruff: ignore[private-member-access]
        return copies

    def _can_patch_component_collections(self) -> bool:
        """
        Check whether the per-Q collections can be updated in place instead of rebuilt.

        They can if they are current, there is one per Q, and each holds exactly the components a
        rebuild would produce, by name and in order. Per-Q collections edited directly (e.g. a
        component popped from one Q) fail the check, so a full rebuild restores them rather than
        patching at the wrong position.

        Returns
        -------
        bool
            ``True`` if the per-Q collections can be patched in place.
        """
        if self.component_collections_is_dirty:
            return False
        if len(self._component_collections) != (0 if self.Q is None else len(self.Q)):
            return False
        names = self._components.list_component_names()
        return all(
            collection.list_component_names() == names + self._generated_component_names(Q_index)
            for Q_index, collection in enumerate(self._component_collections)
        )

    def _generated_component_names(
        self,
        Q_index: int,  # ruff: ignore[unused-method-argument]
    ) -> list[str]:
        """
        Get the names of components added to a per-Q collection after the template's.

        Parameters
        ----------
        Q_index : int
            The Q index of the collection.

        Returns
        -------
        list[str]
            Empty for a plain model; subclasses that add generated components override this.
        """
        return []

    def _mark_component_collections_patched(self) -> None:
        """
        Record that the per-Q collections were updated in place to match the template.

        The state counter is bumped so consumers keyed on ``state_version`` (e.g. convolvers) see
        the structural change, while the collections themselves are marked current.
        """
        self._state_counter += 1
        self._component_collections_is_dirty = False
        self._apply_Q_constraints()

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
