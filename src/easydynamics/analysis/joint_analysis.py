# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import scipp as sc
from easyscience.fitting.multi_fitter import MultiFitter

from easydynamics.analysis.analysis import Analysis
from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.utils.utils import CANONICAL_Q_UNIT

if TYPE_CHECKING:
    from easyscience.fitting.minimizers.utils import FitResults

    from easydynamics.sample_model.components.model_component import ModelComponent
    from easydynamics.sample_model.model_base import ModelBase
    from easydynamics.variable import Parameter


class JointAnalysis(EasyDynamicsBase):
    """
    Fit several Analysis objects together, sharing the components they have in common.

    Each Analysis keeps its own experiment, sample model and instrument model, so the data sets may
    have different resolutions, energy ranges, backgrounds and Q values. A component object that
    appears in the template of more than one of the models is *shared*. Its per-Q copies are
    grouped by Q value, taking the analyses in order: a copy joins the group whose owner is nearest
    in Q, provided that owner is within ``Q_tolerance`` and no other Q value of the copy's own
    model is nearer to it, and otherwise starts a new group, which it owns. The other copies in a
    group are made dependent on the owner, so one set of values describes all the data sets at that
    Q, and a copy alone in its group stays independent. With a ``Q_tolerance`` of 0, Q values are
    matched exactly (to six decimals). If no Q values match at all, the component is fitted
    independently in each analysis and a warning is given.

    After linking, each analysis can still be fitted or sampled on its own, but a shared parameter
    then behaves differently in the owning analysis and in a following one:

    - A following copy is a dependent parameter, which a fit does not vary. Fitting a following
      analysis alone therefore fits only the parameters that are not shared, with the shared ones
      held at the owner's current values.
    - Fitting the owning analysis alone varies the shared parameters against its own data only.
      The following copies take the new values, without their data having been used.

    Linking replaces the values and ``fixed`` flags of the following copies, so set those on the
    owner (``fix_all_parameters`` on a following model skips them). The models holding a shared
    component must use the same units. Use one JointAnalysis for a set of analyses: a second one
    listing them in another order would try to make each copy follow the other.

    A diffusion model object used by several sample models is shared as well: its Q grid becomes
    the union of their Q values, each sample model uses the components at its own Q values, and the
    global parameters (e.g. the diffusion coefficient) are fitted against all the data sets.

    Examples
    --------
    **Two measurements of the same sample with different resolutions**

    The narrow Lorentzian ``L1`` is only resolved in the first data set and the broad ``L3`` only
    fits inside the energy window of the second, while ``L2`` is seen in both. Appending the same
    ``L2`` object to both sample models shares it:
    ```python
    import easydynamics as edyn

    L1 = edyn.Lorentzian(name='L1', width=0.02)
    L2 = edyn.Lorentzian(name='L2', width=0.15)
    L3 = edyn.Lorentzian(name='L3', width=2.0)

    model1 = edyn.SampleModel(components=edyn.ComponentCollection(components=[L1, L2]))
    model2 = edyn.SampleModel(components=edyn.ComponentCollection(components=[L2, L3]))

    analysis1 = edyn.Analysis(experiment=experiment1, sample_model=model1)
    analysis2 = edyn.Analysis(experiment=experiment2, sample_model=model2)

    joint = edyn.JointAnalysis(analyses=[analysis1, analysis2])
    joint.fit()
    joint.parameters_to_dataset()  # the L2 parameters over the Q values of both data sets
    ```
    """

    def __init__(
        self,
        analyses: list[Analysis],
        Q_tolerance: float = 0.01,
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize a JointAnalysis.

        Parameters
        ----------
        analyses : list[Analysis]
            The analyses to fit together. Where Q values match, shared components are owned by the
            earliest analysis in the list.
        Q_tolerance : float, default=0.01
            How far apart, in 1/angstrom, the Q values of two analyses may be for a shared
            component to have the same parameters at both.
        display_name : str | None, default=None
            An optional display name. If None, the unique_name will be used.
        unique_name : str | None, default=None
            An optional unique name. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If analyses is not a list of Analysis objects, or Q_tolerance is not a number.
        ValueError
            If analyses is empty or holds the same Analysis twice, or Q_tolerance is negative or
            not finite.
        """
        super().__init__(display_name=display_name, unique_name=unique_name)

        if not isinstance(analyses, list) or not all(
            isinstance(analysis, Analysis) for analysis in analyses
        ):
            raise TypeError('analyses must be a list of Analysis objects.')
        if not analyses:
            raise ValueError('analyses must contain at least one Analysis.')
        if len({id(analysis) for analysis in analyses}) != len(analyses):
            raise ValueError('analyses must not contain the same Analysis more than once.')

        self._analyses = list(analyses)
        self.Q_tolerance = Q_tolerance
        self._fitter: MultiFitter | None = None
        self._fitter_object_ids: list[int] = []
        # The parameters made dependent by link_shared_components and the fixed flag each had
        # before, to restore them before relinking.
        self._linked: list[tuple[Parameter, bool]] = []

    #############
    # Properties
    #############

    @property
    def analyses(self) -> list[Analysis]:
        """
        The analyses fitted together.

        Returns
        -------
        list[Analysis]
            A copy of the list of analyses, in the order they were added.
        """
        return list(self._analyses)

    @analyses.setter
    def analyses(self, _value: list[Analysis]) -> None:
        """
        Analyses is read-only.

        Use ``append_analysis`` to add an Analysis, or create a new JointAnalysis.

        Parameters
        ----------
        _value : list[Analysis]
            The new list of analyses. This argument is ignored, as analyses is read-only.

        Raises
        ------
        AttributeError
            Always raised, since analyses is read-only.
        """
        raise AttributeError(
            'analyses is read-only. Use append_analysis to add an Analysis, or create a new '
            'JointAnalysis.'
        )

    @property
    def Q_tolerance(self) -> float:
        """
        How far apart the Q values of two analyses may be for them to share a component there.

        Returns
        -------
        float
            The tolerance, in 1/angstrom.
        """
        return self._Q_tolerance

    @Q_tolerance.setter
    def Q_tolerance(self, value: float) -> None:
        """
        Set the Q tolerance. It is used by the next ``fit`` or ``link_shared_components``.

        Parameters
        ----------
        value : float
            The tolerance, in 1/angstrom.

        Raises
        ------
        TypeError
            If value is not a number.
        ValueError
            If value is negative or not finite.
        """
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError('Q_tolerance must be a number.')
        if not 0 <= value < np.inf:
            raise ValueError('Q_tolerance must be a non-negative finite number.')
        self._Q_tolerance = float(value)

    @property
    def shared_components(self) -> list[ModelComponent]:
        """
        The template components that appear in more than one of the analyses' models.

        Returns
        -------
        list[ModelComponent]
            The shared components, in order of first appearance, each listed once.
        """
        return [component for component, _ in self._shared_components()]

    @shared_components.setter
    def shared_components(self, _value: list[ModelComponent]) -> None:
        """
        Shared_components is read-only.

        To share a component, add the same component object to the models of several analyses.

        Parameters
        ----------
        _value : list[ModelComponent]
            The new list of components. This argument is ignored, as shared_components is
            read-only.

        Raises
        ------
        AttributeError
            Always raised, since shared_components is read-only.
        """
        raise AttributeError(
            'shared_components is read-only. To share a component, add the same component object '
            'to the models of several analyses.'
        )

    @property
    def scales(self) -> list[Parameter]:
        """
        The scale of every analysis: the factor its model is multiplied by.

        Data sets of the same sample are rarely on the same intensity scale. Keep one scale fixed
        as the reference and free the others, e.g. ``joint.scales[1].fixed = False``, so that the
        shared components can describe all the data sets.

        Returns
        -------
        list[Parameter]
            The scale Parameter of each analysis' InstrumentModel, in the order of the analyses.
        """
        return [analysis.scale for analysis in self._analyses]

    @scales.setter
    def scales(self, _value: list[Parameter]) -> None:
        """
        Scales is read-only.

        Set the value of a scale, or free it, on the Parameter itself.

        Parameters
        ----------
        _value : list[Parameter]
            The new scales. This argument is ignored, as scales is read-only.

        Raises
        ------
        AttributeError
            Always raised, since scales is read-only.
        """
        raise AttributeError(
            'scales is read-only. Set the value of a scale, or free it, on the Parameter itself, '
            'e.g. scales[1].value = 2.0 or scales[1].fixed = False.'
        )

    @property
    def fitter(self) -> MultiFitter:
        """
        The EasyScience MultiFitter covering every Q index of every analysis, built on first use.

        It is rebuilt whenever an analysis replaces its per-Q Analysis1d objects.

        Returns
        -------
        MultiFitter
            The cached MultiFitter.
        """
        fit_objects = [
            analysis1d for analysis in self._analyses for analysis1d in analysis.analysis_list
        ]
        object_ids = [id(fit_object) for fit_object in fit_objects]
        if self._fitter is None or object_ids != self._fitter_object_ids:
            self._fitter = MultiFitter(
                fit_objects=fit_objects,
                fit_functions=[fit_object.as_fit_function() for fit_object in fit_objects],
            )
            self._fitter_object_ids = object_ids
        return self._fitter

    @fitter.setter
    def fitter(self, _value: MultiFitter) -> None:
        """
        Fitter is read-only.

        It is built from the analyses. Change its settings on the fitter itself, e.g. with
        ``fitter.switch_minimizer``.

        Parameters
        ----------
        _value : MultiFitter
            The new fitter. This argument is ignored, as fitter is read-only.

        Raises
        ------
        AttributeError
            Always raised, since fitter is read-only.
        """
        raise AttributeError(
            'fitter is read-only. It is built from the analyses; change its settings on the '
            'fitter itself.'
        )

    #############
    # Other methods
    #############

    def append_analysis(self, analysis: Analysis) -> None:
        """
        Add an Analysis to fit together with the others.

        It is added last, so where its Q values match those of an earlier analysis its copies of
        the shared components follow that analysis. The copies are linked by the next ``fit`` or
        ``link_shared_components``.

        Parameters
        ----------
        analysis : Analysis
            The Analysis to add.

        Raises
        ------
        TypeError
            If analysis is not an Analysis.
        ValueError
            If analysis is already one of the analyses.
        """
        if not isinstance(analysis, Analysis):
            raise TypeError('analysis must be an Analysis.')
        if any(analysis is existing for existing in self._analyses):
            raise ValueError('analysis is already one of the analyses.')
        self._analyses.append(analysis)

    def fit(self) -> list[FitResults]:
        """
        Fit all Q indices of all analyses simultaneously.

        Shared components are linked first, so that their parameters are fitted once and applied to
        every data set.

        Returns
        -------
        list[FitResults]
            One FitResults per data set: every Q index of the first analysis, then of the second,
            and so on.

        Notes
        -----
        A ``ValueError`` propagates from ``link_shared_components``, e.g. if an analysis has no Q
        values.
        """
        self.link_shared_components()

        xs, ys, ws = [], [], []
        for analysis in self._analyses:
            analysis._prepare_for_sampling()  # ruff: ignore[private-member-access]
            x, y, weights = analysis._sampling_data()  # ruff: ignore[private-member-access]
            xs.extend(x)
            ys.extend(y)
            ws.extend(weights)
        return self.fitter.fit(x=xs, y=ys, weights=ws)

    def link_shared_components(self) -> None:
        """
        Make the per-Q copies of every shared component follow the owner at their Q value.

        Called by ``fit``; call it directly to inspect or plot the linked models before fitting.
        The links made by an earlier call are released first, so it is safe to call again after a
        model or its Q values changed. Parameters of a copy that are already dependent, such as the
        width of a StretchedExponential, are left alone: they follow parameters that are linked.

        Raises
        ------
        ValueError
            If a copy and its owner have parameters in different units.

        Notes
        -----
        A ``ValueError`` propagates if an analysis has no Q values.
        """
        # A copy linked earlier may no longer be shared, or may have another owner by now.
        # Releasing a parameter leaves it free, so its fixed flag is put back.
        for parameter, fixed in self._linked:
            if not parameter.independent:
                parameter.make_independent()
                parameter.fixed = fixed
        self._linked = []

        for component, groups in self._shared_component_groups():
            if all(len(copies) == 1 for copies in groups.values()):
                warnings.warn(
                    f'Component {component.name!r} is shared, but the analyses holding it have '
                    f'no Q values within {self._Q_tolerance} 1/angstrom of each other, so it is '
                    f'fitted independently in each. Rebin the data onto a common Q grid or '
                    f'increase Q_tolerance to share it.',
                    UserWarning,
                    stacklevel=2,
                )
            # Every parameter of the other copies follows the owner's parameter of that name.
            for owner, *followers in groups.values():
                owner_parameters = {
                    parameter.name: parameter for parameter in owner.get_all_parameters()
                }
                for follower in followers:
                    for parameter in follower.get_fittable_parameters():
                        owner_parameter = owner_parameters[parameter.name]
                        if parameter.unit != owner_parameter.unit:
                            raise ValueError(
                                f'Cannot share {parameter.name!r}: it is in {parameter.unit} in '
                                f'one model and in {owner_parameter.unit} in another. Convert the '
                                f'models to the same units first.'
                            )
                        self._linked.append((parameter, parameter.fixed))
                        parameter.make_dependent_on('shared', {'shared': owner_parameter})

    def parameters_to_dataset(self) -> sc.Dataset:
        """
        Create a scipp dataset of the shared components' parameters over the Q values of all data.

        There is one row per Q value of a copy that owns its parameters, holding those (fitted)
        parameters, and NaN for a component with no owner at that Q value. A copy that follows an
        owner has no row of its own, also where its Q value differs from the owner's within
        ``Q_tolerance``. The dataset can be plotted or passed to a ``ParameterAnalysis`` like the
        dataset from ``Analysis.parameters_to_dataset``. The parameters that are not shared are
        available from each analysis' own dataset.

        Raises
        ------
        ValueError
            If two shared components have parameters with the same name.

        Returns
        -------
        sc.Dataset
            A dataset where each entry is a parameter, with dimension "Q" sorted in ascending Q.
        """
        # For every parameter of a shared component: the owner's Parameter at each Q value.
        columns: dict[str, dict[float, Parameter]] = {}
        for _, groups in self._shared_component_groups():
            for Q_value, (owner, *_followers) in groups.items():
                for parameter in owner.get_all_parameters():
                    column = columns.setdefault(parameter.name, {})
                    if Q_value in column:
                        raise ValueError(
                            f'Duplicate parameter name {parameter.name!r} at Q = {Q_value}. '
                            f'Rename the shared components so all parameters have unique names.'
                        )
                    column[Q_value] = parameter

        Q_values = sorted({Q_value for column in columns.values() for Q_value in column})
        ds = sc.Dataset(coords={'Q': sc.array(dims=['Q'], values=Q_values, unit=CANONICAL_Q_UNIT)})
        for name, column in columns.items():
            parameters = [column.get(Q_value) for Q_value in Q_values]
            ds[name] = sc.array(
                dims=['Q'],
                values=[np.nan if p is None else p.value for p in parameters],
                variances=[np.nan if p is None else p.variance for p in parameters],
                unit=next(iter(column.values())).unit,
                dtype=float,
            )
        return ds

    #############
    # Private methods
    #############

    def _shared_components(self) -> list[tuple[ModelComponent, list[tuple[ModelBase, int]]]]:
        """
        Find every template component held by more than one model.

        The models are the sample, resolution and background models of every analysis. A model
        object used by several analyses counts once.

        Returns
        -------
        list[tuple[ModelComponent, list[tuple[ModelBase, int]]]]
            For each shared component, in order of first appearance: the component and its holders
            as ``(model, template index)``, in the order of the analyses.
        """
        holders: dict[int, tuple[ModelComponent, list[tuple[ModelBase, int]]]] = {}
        seen_models: set[int] = set()
        for analysis in self._analyses:
            for model in (
                analysis.sample_model,
                analysis.instrument_model.resolution_model,
                analysis.instrument_model.background_model,
            ):
                if id(model) in seen_models:
                    continue
                seen_models.add(id(model))
                for index, component in enumerate(model.components):
                    holders.setdefault(id(component), (component, []))[1].append((model, index))
        return [entry for entry in holders.values() if len(entry[1]) > 1]

    def _shared_component_groups(
        self,
    ) -> list[tuple[ModelComponent, dict[float, list[ModelComponent]]]]:
        """
        Group the per-Q copies of every shared component by Q value.

        Returns
        -------
        list[tuple[ModelComponent, dict[float, list[ModelComponent]]]]
            For each shared component: its groups of copies, keyed by the Q value of the owner (in
            1/angstrom). The copies are in the order of the analyses, so the first is the owner.

        Raises
        ------
        ValueError
            If an analysis has no Q values.
        """
        for analysis in self._analyses:
            if analysis.Q is None:
                raise ValueError(
                    f'Analysis {analysis.display_name!r} has no Q values. Load data into its '
                    f'experiment first.'
                )
            # Building the per-Q analyses gives the models the Q values of the experiment, which
            # they lose in a rebin until the analysis is next used.
            _ = analysis.analysis_list

        result = []
        for component, holders in self._shared_components():
            groups: dict[float, list[ModelComponent]] = {}
            for model, index in holders:
                # Rounded, so that Q values that differ only by floating point noise match.
                Q_values = [round(float(Q_value), 6) for Q_value in model.Q.values]
                # The groups a copy of this model can join: those owned by the earlier models.
                owner_Q_values = list(groups)
                for Q_index, Q_value in enumerate(Q_values):
                    copy = model.get_component_collection(Q_index)[index]
                    owner_Q = min(owner_Q_values, key=lambda Q: abs(Q - Q_value), default=Q_value)
                    # Of the Q values of one model only the nearest joins a group, so that two
                    # copies of one model are never linked. The distance is rounded like the Q
                    # values, so that Q values exactly the tolerance apart match.
                    if (
                        round(abs(owner_Q - Q_value), 6) > self._Q_tolerance
                        or min(Q_values, key=lambda Q: abs(Q - owner_Q)) != Q_value
                    ):
                        owner_Q = Q_value
                    groups.setdefault(owner_Q, []).append(copy)
            result.append((component, groups))
        return result

    #############
    # Dunder methods
    #############

    def __repr__(self) -> str:
        """
        Return a string representation of the JointAnalysis.

        Returns
        -------
        str
            A string representation of the JointAnalysis.
        """
        names = ', '.join(repr(analysis.display_name) for analysis in self._analyses)
        shared = ', '.join(repr(component.name) for component in self.shared_components)
        return (
            f'{self.__class__.__name__}(\n'
            f'    analyses=[{names}],\n'
            f'    shared_components=[{shared}],\n'
            f')'
        )
