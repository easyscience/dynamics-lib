# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from copy import copy

import scipp as sc

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Polynomial
from easydynamics.sample_model.components.exponential import Exponential
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.sample_model.model_base import ModelBase
from easydynamics.sample_model.sample_model import SampleModel
from easydynamics.utils.utils import Q_type


class ResolutionModel(ModelBase):
    """
    ResolutionModel represents a model of the instrument resolution in an experiment at various Q.

    Examples
    --------
    **Creating a Gaussian resolution model**

    A single Gaussian is the most common resolution model. Note that ``DeltaFunction``,
    ``Polynomial``, and ``Exponential`` components are not allowed in a ResolutionModel:
    ```python
    import numpy as np
    import easydynamics as edyn

    Q = np.linspace(0.5, 2, 7)
    resolution_model = edyn.ResolutionModel(
        components=edyn.Gaussian(width=0.05, area=1.0),
        Q=Q,
    )
    energy = np.linspace(-2, 2, 100)
    resolution = resolution_model.evaluate(energy)
    ```

    **Building a resolution model from a fitted SampleModel**

    After fitting vanadium data with a SampleModel, use ``from_sample_model`` to convert it
    directly into a ResolutionModel:
    ```python
    resolution_model = edyn.ResolutionModel.from_sample_model(fitted_sample_model)
    ```
    """

    def __init__(
        self,
        display_name: str = 'MyResolutionModel',
        unique_name: str | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        components: ModelComponent | ComponentCollection | None = None,
        Q: Q_type | None = None,
    ) -> None:
        """
        Initialize the ResolutionModel.

        Parameters
        ----------
        display_name : str, default='MyResolutionModel'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        components : ModelComponent | ComponentCollection | None, default=None
            Template components. DeltaFunction, Polynomial, and Exponential are not allowed.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        """
        # Set before super().__init__, which may call append_component (overridden below).
        self._calibrated = False
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
            x_unit=x_unit,
            y_unit=y_unit,
            components=components,
            Q=Q,
        )

    def append_component(self, component: ModelComponent | ComponentCollection) -> None:
        """
        Append a component to the ResolutionModel.

        Does not allow DeltaFunction, Polynomial, or Exponential components, as these are not
        physical resolution components.

        Parameters
        ----------
        component : ModelComponent | ComponentCollection
            Component(s) to append.

        Raises
        ------
        TypeError
            If the component is a DeltaFunction, Polynomial, or Exponential.

        Notes
        -----
        A ``RuntimeError`` propagates from the calibration guard if the model holds calibrated
        per-Q collections from ``from_sample_model``; a template change would schedule a rebuild
        that silently discards them.
        """
        self._assert_not_calibrated('append a component')
        components = component if isinstance(component, ComponentCollection) else (component,)

        for comp in components:
            if isinstance(comp, (DeltaFunction, Polynomial, Exponential)):
                raise TypeError(
                    f'Component in ResolutionModel cannot be a {comp.__class__.__name__}'
                )

        super().append_component(component)

    def remove_component(self, name: str) -> None:
        """
        Remove a component from the ResolutionModel by its name.

        Parameters
        ----------
        name : str
            The name of the component to remove.

        Notes
        -----
        A ``RuntimeError`` propagates from the calibration guard if the model holds calibrated
        per-Q collections from ``from_sample_model``; a template change would schedule a rebuild
        that silently discards them.
        """
        self._assert_not_calibrated('remove a component')
        super().remove_component(name)

    def clear_components(self) -> None:
        """
        Clear all components from the ResolutionModel.

        Notes
        -----
        A ``RuntimeError`` propagates from the calibration guard if the model holds calibrated
        per-Q collections from ``from_sample_model``; a template change would schedule a rebuild
        that silently discards them.
        """
        self._assert_not_calibrated('clear the components')
        super().clear_components()

    def clear_Q(self, confirm: bool = False) -> None:
        """
        Clear the Q values of the ResolutionModel, removing all component collections and their
        associated Parameters.

        Parameters
        ----------
        confirm : bool, default=False
            Confirmation to clear Q values.

        Notes
        -----
        A ``ValueError`` propagates from the base implementation if confirm is not True, and a
        ``RuntimeError`` propagates from the calibration guard if the model holds calibrated per-Q
        collections from ``from_sample_model``; clearing Q would discard them.
        """
        self._assert_not_calibrated('clear Q')
        super().clear_Q(confirm=confirm)

    def _assert_not_calibrated(self, action: str) -> None:
        """
        Raise if this model holds calibrated per-Q collections installed by from_sample_model.

        The per-Q collections installed by ``from_sample_model`` hold the fitted (calibrated)
        resolution, but the template components do not. Any mutation that schedules a rebuild would
        silently replace the calibrated collections with unfitted template copies, so such
        mutations fail loudly instead.

        Parameters
        ----------
        action : str
            Description of the attempted mutation, used in the error message.

        Raises
        ------
        RuntimeError
            If the model is calibrated.
        """
        if self._calibrated:
            raise RuntimeError(
                f'Cannot {action} on a ResolutionModel created by from_sample_model: its per-Q '
                f'collections hold the fitted (calibrated) resolution, and this change would '
                f'rebuild them from the unfitted template, silently discarding the calibration. '
                f'Create a new ResolutionModel (or rerun from_sample_model on an updated '
                f'SampleModel) instead.'
            )

    @classmethod
    def from_sample_model(
        cls,
        sample_model: SampleModel,
        normalize_area: bool = True,
        fix_parameters: bool = True,
    ) -> 'ResolutionModel':
        """
        Create a ResolutionModel from a SampleModel.

        DeltaFunction components (the standard QENS elastic line) are stripped from both the
        template and the per-Q collections, with a warning: a delta carries no resolution
        broadening (it is the identity under convolution), so the fitted broadened components are
        the resolution. Polynomial and Exponential components are rejected, as backgrounds do not
        belong in a resolution model.

        When the SampleModel has Q values, the fitted per-Q collections are installed as the
        calibrated resolution and the model is locked: mutations that would rebuild the collections
        from the (unfitted) template — ``append_component``, ``remove_component``,
        ``clear_components``, ``clear_Q`` — raise a RuntimeError instead of silently discarding the
        calibration.

        Parameters
        ----------
        sample_model : SampleModel
            SampleModel to create the ResolutionModel from.
        normalize_area : bool, default=True
            Whether to normalize the components in the ResolutionModel to have area 1.
        fix_parameters : bool, default=True
            Whether to fix the parameters in the ResolutionModel.

        Returns
        -------
        'ResolutionModel'
            ResolutionModel created from the SampleModel.

        Raises
        ------
        TypeError
            If sample_model is not a SampleModel, if normalize_area or fix_parameters are not bool,
            or if the SampleModel contains Polynomial or Exponential components.
        ValueError
            If a per-Q collection contains only DeltaFunction components, leaving no resolution
            shape after stripping.
        """
        if not isinstance(sample_model, SampleModel):
            raise TypeError(
                f'sample_model must be an instance of SampleModel. Got {type(sample_model)}.'
            )

        if not isinstance(normalize_area, bool):
            raise TypeError('normalize_area must be True or False.')

        if not isinstance(fix_parameters, bool):
            raise TypeError('fix_parameters must be True or False.')

        template = ComponentCollection(
            x_unit=sample_model.x_unit,
            y_unit=sample_model.y_unit,
        )
        stripped_deltas = 0
        for component in sample_model.components:
            if isinstance(component, DeltaFunction):
                stripped_deltas += 1
                continue
            template.append_component(component)

        resolution_model = cls(
            display_name=sample_model.display_name,
            x_unit=sample_model.x_unit,
            y_unit=sample_model.y_unit,
            components=template,
            Q=sample_model.Q,
        )

        if sample_model.Q is not None:
            # Prepare the per-Q collections detached from the model so no EasyScience
            # callback can schedule a rebuild halfway through, then install them and
            # clear the dirty flag in one final step.
            collections = []
            for index in range(len(sample_model.Q)):
                source = copy(sample_model.get_component_collection(Q_index=index))
                filtered = ComponentCollection(
                    name=source.name,
                    display_name=source.display_name,
                    x_unit=source.x_unit,
                    y_unit=source.y_unit,
                )
                for component in source:
                    if isinstance(component, DeltaFunction):
                        stripped_deltas += 1
                        continue
                    if isinstance(component, (Polynomial, Exponential)):
                        raise TypeError(
                            f'Component in ResolutionModel cannot be a '
                            f'{component.__class__.__name__}'
                        )
                    filtered.append_component(component)
                if len(filtered) == 0:
                    raise ValueError(
                        f'The SampleModel collection at Q index {index} contains only '
                        f'DeltaFunction components; after stripping them no resolution shape '
                        f'is left. Fit the resolution data with at least one broadened '
                        f'component (e.g. a Gaussian).'
                    )
                collections.append(filtered)
            for collection in collections:
                if normalize_area:
                    collection.normalize_area()
                if fix_parameters:
                    collection.fix_all_parameters()
            resolution_model._component_collections = collections
            resolution_model._component_collections_is_dirty = False
            resolution_model._calibrated = True

        if stripped_deltas:
            warnings.warn(
                f'Stripped {stripped_deltas} DeltaFunction component(s) from the SampleModel '
                f'when building the ResolutionModel: a delta function carries no resolution '
                f'broadening (it is the identity under convolution).',
                UserWarning,
                stacklevel=2,
            )

        return resolution_model

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'unique_name={self.unique_name!r}, '
            f'x_unit={self.x_unit}, '
            f'y_unit={self.y_unit}, '
            f'Q_len={None if self._Q is None else len(self._Q)}, '
            f'components={self.components})'
        )
