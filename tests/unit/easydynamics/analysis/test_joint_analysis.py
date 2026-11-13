# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.fitting.multi_fitter import MultiFitter

from easydynamics.analysis.analysis import Analysis
from easydynamics.analysis.joint_analysis import JointAnalysis
from easydynamics.experiment import Experiment
from easydynamics.sample_model import BackgroundModel
from easydynamics.sample_model import BrownianTranslationalDiffusion
from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model import ResolutionModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model import StretchedExponential

Q_FIRST = [0.5, 1.0, 1.5]
Q_SECOND = [1.0, 1.5, 2.0]


def make_experiment(Q, energy=None, values=None):
    energy = np.linspace(-2.0, 2.0, 5) if energy is None else energy
    values = np.ones((len(Q), len(energy))) if values is None else values
    data = sc.DataArray(
        data=sc.array(dims=['Q', 'energy'], values=values, variances=np.full(values.shape, 0.1)),
        coords={
            'Q': sc.array(dims=['Q'], values=Q, unit='1/angstrom'),
            'energy': sc.array(dims=['energy'], values=energy, unit='meV'),
        },
    )
    return Experiment(data=data)


def make_analysis(name, Q, components, background=None, experiment=None):
    background_model = (
        BackgroundModel() if background is None else BackgroundModel(components=background)
    )
    return Analysis(
        display_name=name,
        experiment=make_experiment(Q) if experiment is None else experiment,
        sample_model=SampleModel(components=ComponentCollection(components=components)),
        instrument_model=InstrumentModel(background_model=background_model),
    )


class TestJointAnalysis:
    @pytest.fixture
    def shared(self):
        return Lorentzian(name='L2', area=1.0, width=0.3)

    @pytest.fixture
    def first(self, shared):
        return make_analysis('first', Q_FIRST, [Lorentzian(name='L1', width=0.05), shared])

    @pytest.fixture
    def second(self, shared):
        return make_analysis('second', Q_SECOND, [shared, Lorentzian(name='L3', width=2.0)])

    @pytest.fixture
    def joint(self, first, second):
        return JointAnalysis(analyses=[first, second])

    @staticmethod
    def l2_width(analysis, Q_index):
        return analysis.sample_model.get_component_collection(Q_index)['L2'].width

    #############
    # Initialization and validation
    #############

    def test_init(self, joint, first, second):
        # WHEN THEN EXPECT
        assert joint.analyses == [first, second]

    def test_analyses_returns_a_copy(self, joint, first, second):
        # WHEN THEN
        joint.analyses.append('intruder')

        # EXPECT
        assert joint.analyses == [first, second]

    @pytest.mark.parametrize('name', ['analyses', 'shared_components', 'fitter'])
    def test_read_only_properties_raise(self, joint, name):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError, match=f'{name} is read-only'):
            setattr(joint, name, [])

    def test_append_analysis(self, joint, shared, first, second):
        # WHEN
        third = make_analysis('third', Q_SECOND, [shared])

        # THEN
        joint.append_analysis(third)
        joint.link_shared_components()

        # EXPECT it is last, and its copies follow the earlier analyses
        assert joint.analyses == [first, second, third]
        assert self.l2_width(third, 0).independent is False
        assert len(joint.fitter.fit_functions) == 9

    def test_append_analysis_invalid_type_raises(self, joint):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='analysis must be an Analysis'):
            joint.append_analysis('not an analysis')

    def test_append_analysis_twice_raises(self, joint, first):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='already one of the analyses'):
            joint.append_analysis(first)

    @pytest.mark.parametrize('analyses', ['not a list', ('tuple',), ['not an analysis']])
    def test_init_invalid_analyses_type_raises(self, analyses):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='list of Analysis'):
            JointAnalysis(analyses=analyses)

    def test_init_duplicate_analyses_raise(self, first):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='same Analysis more than once'):
            JointAnalysis(analyses=[first, first])

    def test_init_empty_analyses_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='at least one'):
            JointAnalysis(analyses=[])

    #############
    # Shared components
    #############

    def test_shared_components_found_by_identity(self, joint, shared):
        # WHEN THEN EXPECT
        assert joint.shared_components == [shared]

    def test_shared_components_ignores_equal_names(self):
        # WHEN two analyses with distinct components that happen to share a name
        first = make_analysis('first', Q_FIRST, [Lorentzian(name='L2')])
        second = make_analysis('second', Q_FIRST, [Lorentzian(name='L2')])

        # THEN
        joint = JointAnalysis(analyses=[first, second])

        # EXPECT
        assert joint.shared_components == []

    def test_shared_components_includes_background_components(self):
        # WHEN the same background component is in both instrument models
        background = Polynomial(name='Flat', coefficients=[0.01])
        first = make_analysis('first', Q_FIRST, [Lorentzian(name='L1')], background=background)
        second = make_analysis('second', Q_FIRST, [Lorentzian(name='L3')], background=background)

        # THEN
        joint = JointAnalysis(analyses=[first, second])

        # EXPECT
        assert joint.shared_components == [background]

    def test_shared_components_ignores_a_model_used_by_two_analyses(self):
        "Regression: its components were shared with themselves, which raised a Q spacing error"
        # WHEN two analyses on the same instrument use one ResolutionModel object
        resolution_model = ResolutionModel(components=Gaussian(name='R', width=0.1))
        analyses = [
            Analysis(
                display_name=name,
                experiment=make_experiment(Q_FIRST),
                sample_model=SampleModel(components=Lorentzian(name=name)),
                instrument_model=InstrumentModel(resolution_model=resolution_model),
            )
            for name in ('first', 'second')
        ]
        joint = JointAnalysis(analyses=analyses)

        # THEN
        joint.link_shared_components()

        # EXPECT its components are simply the same objects in both, with nothing to link
        assert joint.shared_components == []
        assert resolution_model.get_component_collection(0)['R'].width.independent is True

    def test_shared_components_listed_once_for_three_analyses(self, shared, first, second):
        # WHEN a third analysis also holds the shared component
        third = make_analysis('third', Q_SECOND, [shared])

        # THEN
        joint = JointAnalysis(analyses=[first, second, third])

        # EXPECT
        assert joint.shared_components == [shared]

    #############
    # Linking
    #############

    def test_link_shared_components_links_matching_Q_only(self, joint, first, second):
        # WHEN the second analysis has Q = 1.0 and 1.5 in common with the first, and 2.0 of its own

        # THEN
        joint.link_shared_components()

        # EXPECT the common Q follow the first analysis, the other stays independent
        assert self.l2_width(second, 0).independent is False
        assert self.l2_width(second, 1).independent is False
        assert self.l2_width(second, 2).independent is True
        self.l2_width(first, 1).value = 0.123
        assert self.l2_width(second, 0).value == pytest.approx(0.123)
        self.l2_width(first, 2).value = 0.456
        assert self.l2_width(second, 1).value == pytest.approx(0.456)
        # EXPECT the owner and the unshared components are untouched
        for Q_index in range(3):
            assert self.l2_width(first, Q_index).independent is True
            L1 = first.sample_model.get_component_collection(Q_index)['L1']
            L3 = second.sample_model.get_component_collection(Q_index)['L3']
            assert all(parameter.independent for parameter in L1.get_all_parameters())
            assert all(parameter.independent for parameter in L3.get_all_parameters())

    def test_link_shared_components_links_every_parameter(self, joint, first, second):
        # WHEN THEN
        joint.link_shared_components()

        # EXPECT area, center and width of the shared component all follow the owner
        owner = first.sample_model.get_component_collection(1)['L2']
        follower = second.sample_model.get_component_collection(0)['L2']
        for parameter in follower.get_all_parameters():
            assert parameter.independent is False
        owner.area.value = 2.5
        assert follower.area.value == pytest.approx(2.5)

    def test_link_shared_components_warns_when_no_Q_match(self, shared):
        # WHEN interleaved Q grids
        first = make_analysis('first', [0.5, 1.0, 1.5], [shared])
        second = make_analysis('second', [0.75, 1.25, 1.75], [shared])
        joint = JointAnalysis(analyses=[first, second])

        # THEN EXPECT
        with pytest.warns(UserWarning, match="'L2' is shared, but the analyses holding it"):
            joint.link_shared_components()
        for Q_index in range(3):
            assert self.l2_width(second, Q_index).independent is True

    def test_link_shared_components_raises_without_Q(self, shared):
        # WHEN one analysis has no data yet
        first = make_analysis('first', Q_FIRST, [shared])
        second = Analysis(
            display_name='second',
            experiment=Experiment(),
            sample_model=SampleModel(components=shared),
        )

        # THEN EXPECT
        with pytest.raises(ValueError, match="'second' has no Q values"):
            JointAnalysis(analyses=[first, second]).link_shared_components()

    def test_link_shared_components_releases_a_copy_that_is_no_longer_shared(
        self, joint, first, second
    ):
        "Regression: the copy stayed dependent on the discarded owner and was no longer fitted"
        # WHEN linked models, and the shared component is then removed from the owner's model
        joint.link_shared_components()
        first.sample_model.remove_component('L2')

        # THEN
        joint.link_shared_components()

        # EXPECT the former follower has its own parameters again at every Q, fixed as before:
        # the width free, the center fixed
        assert joint.shared_components == []
        for Q_index in range(3):
            assert self.l2_width(second, Q_index).independent is True
            assert self.l2_width(second, Q_index).fixed is False
            center = second.sample_model.get_component_collection(Q_index)['L2'].center
            assert center.independent is True
            assert center.fixed is True

    def test_link_shared_components_leaves_dependent_parameters_alone(self):
        "Regression: the width of a StretchedExponential was re-pointed, and came back free"
        # WHEN a shared component with a parameter that depends on its other parameters
        shared = StretchedExponential(name='SE')
        first = make_analysis('first', Q_FIRST, [shared])
        second = make_analysis('second', Q_FIRST, [shared])
        joint = JointAnalysis(analyses=[first, second])
        width = second.sample_model.get_component_collection(0)['SE'].width
        assert width.independent is False

        # THEN linking, and releasing the links again
        joint.link_shared_components()
        first.sample_model.remove_component('SE')
        joint.link_shared_components()

        # EXPECT the width still follows the parameters of its own component
        assert second.sample_model.get_component_collection(0)['SE'].width is width
        assert width.independent is False

    def test_link_shared_components_moves_the_owner_when_the_first_drops_out(
        self, shared, first, second
    ):
        # WHEN three analyses share the component, and the first one then drops it
        third = make_analysis('third', Q_SECOND, [shared])
        joint = JointAnalysis(analyses=[first, second, third])
        joint.link_shared_components()
        first.sample_model.remove_component('L2')

        # THEN
        joint.link_shared_components()

        # EXPECT the second owns the parameters now, and the third follows it
        for Q_index in range(3):
            assert self.l2_width(second, Q_index).independent is True
            assert self.l2_width(third, Q_index).independent is False
        self.l2_width(second, 0).value = 0.321
        assert self.l2_width(third, 0).value == pytest.approx(0.321)

    def test_link_shared_components_right_after_a_rebin(self, shared):
        "Regression: the models have no Q until the analysis is next used, so nothing was linked"
        # WHEN both analyses are rebinned onto a common Q grid, and not used since
        Q = [0.5, 1.0, 1.5, 2.0]
        first = make_analysis('first', Q, [shared])
        second = make_analysis('second', Q, [shared])
        joint = JointAnalysis(analyses=[first, second])
        for analysis in (first, second):
            analysis.rebin({'Q': 2}, confirm=True)

        # THEN
        joint.link_shared_components()

        # EXPECT the copies are linked at both new Q values
        assert len(first.Q) == 2
        for Q_index in range(2):
            assert self.l2_width(first, Q_index).independent is True
            assert self.l2_width(second, Q_index).independent is False

    def test_link_shared_components_different_units_raise(self, joint, first, second):
        # WHEN the per-Q copies exist in both models, and one model is converted to another unit
        for analysis in (first, second):
            _ = analysis.sample_model.get_component_collection(0)
        second.sample_model.convert_x_unit('ueV')

        # THEN EXPECT
        with pytest.raises(ValueError, match=r"Cannot share 'L2 .*': it is in"):
            joint.link_shared_components()

    def test_link_shared_components_reapplied_after_rebuild(self, joint, second):
        # WHEN linked models whose follower is then rebuilt from its template
        joint.link_shared_components()
        second.sample_model.components.append_component(Gaussian(name='Extra'))
        assert self.l2_width(second, 0).independent is True

        # THEN
        joint.link_shared_components()

        # EXPECT
        assert self.l2_width(second, 0).independent is False

    def test_link_shared_components_three_analyses(self, shared, first):
        # WHEN a third analysis with the same Q values as the second
        second = make_analysis('second', [1.0, 2.0], [shared])
        third = make_analysis('third', [1.0, 2.0], [shared])
        joint = JointAnalysis(analyses=[first, second, third])

        # THEN
        joint.link_shared_components()

        # EXPECT at Q = 1.0 both follow the first; at Q = 2.0 the third follows the second
        assert self.l2_width(second, 0).independent is False
        assert self.l2_width(third, 0).independent is False
        assert self.l2_width(second, 1).independent is True
        assert self.l2_width(third, 1).independent is False
        self.l2_width(first, 1).value = 0.111
        assert self.l2_width(third, 0).value == pytest.approx(0.111)
        self.l2_width(second, 1).value = 0.222
        assert self.l2_width(third, 1).value == pytest.approx(0.222)

    def test_link_shared_components_after_replacing_a_following_model(
        self, joint, first, second, shared
    ):
        # WHEN linked models, after which the following analysis gets a new sample model
        joint.link_shared_components()
        second.sample_model = SampleModel(
            components=ComponentCollection(components=[shared, Lorentzian(name='L4')])
        )
        assert self.l2_width(second, 0).independent is True

        # THEN
        joint.link_shared_components()

        # EXPECT the new copies follow the first analysis
        assert self.l2_width(second, 0).independent is False
        self.l2_width(first, 1).value = 0.321
        assert self.l2_width(second, 0).value == pytest.approx(0.321)

    def test_link_shared_components_after_replacing_the_owner_model(
        self, joint, first, second, shared
    ):
        # WHEN linked models, after which the owning analysis gets a new sample model
        joint.link_shared_components()
        first.sample_model = SampleModel(components=shared)

        # THEN
        joint.link_shared_components()

        # EXPECT the follower tracks the new owner, not the discarded one
        self.l2_width(first, 1).value = 0.321
        assert self.l2_width(second, 0).value == pytest.approx(0.321)

    #############
    # Fitting
    #############

    def test_fit_raises_without_Q(self, shared):
        # WHEN
        first = make_analysis('first', Q_FIRST, [shared])
        second = Analysis(
            display_name='second',
            experiment=Experiment(),
            sample_model=SampleModel(components=shared),
        )

        # THEN EXPECT
        with pytest.raises(ValueError, match="'second' has no Q values"):
            JointAnalysis(analyses=[first, second]).fit()

    def test_fit_links_and_passes_all_data_in_order(self, shared):
        # WHEN two analyses whose data differ at every Q
        values = np.arange(15.0).reshape(3, 5)
        first = make_analysis(
            'first', Q_FIRST, [shared], experiment=make_experiment(Q_FIRST, values=values + 1)
        )
        second = make_analysis(
            'second', Q_SECOND, [shared], experiment=make_experiment(Q_SECOND, values=values + 101)
        )
        joint = JointAnalysis(analyses=[first, second])
        fitter = MagicMock()
        fitter.fit.return_value = 'result'

        # THEN
        with patch.object(JointAnalysis, 'fitter', new_callable=PropertyMock, return_value=fitter):
            result = joint.fit()

        # EXPECT the shared component is linked, and the data of every Q of the first analysis
        # is followed by that of the second
        assert result == 'result'
        assert self.l2_width(second, 0).independent is False
        _, kwargs = fitter.fit.call_args
        assert len(kwargs['x']) == len(kwargs['y']) == len(kwargs['weights']) == 6
        for Q_index in range(3):
            np.testing.assert_array_equal(kwargs['y'][Q_index], values[Q_index] + 1)
            np.testing.assert_array_equal(kwargs['y'][3 + Q_index], values[Q_index] + 101)

    def test_fitter_covers_every_Q_of_every_analysis(self, joint, first, second):
        # WHEN THEN
        fitter = joint.fitter

        # EXPECT
        assert isinstance(fitter, MultiFitter)
        assert list(fitter._fit_objects) == first.analysis_list + second.analysis_list
        assert len(fitter.fit_functions) == 6

    def test_linking_leaves_one_free_parameter_per_Q_group(self, joint, first, second):
        # WHEN THEN
        joint.link_shared_components()

        # EXPECT the free parameters hold one L2 width per Q group: 3 owned by the first analysis
        # and the second analysis' own copy at Q = 2.0, instead of 3 + 3
        names = [
            parameter.name
            for analysis in (first, second)
            for analysis1d in analysis.analysis_list
            for parameter in analysis1d.get_free_parameters()
        ]
        assert names.count('L2 width') == 4
        assert names.count('L1 width') == 3
        assert names.count('L3 width') == 3

    def test_fitter_is_cached_and_rebuilt_when_analysis_list_changes(self, joint, second):
        # WHEN
        fitter = joint.fitter

        # THEN EXPECT reading again gives the same fitter
        assert joint.fitter is fitter

        # THEN the second analysis gets a new experiment, which rebuilds its Analysis1d objects
        second.experiment = make_experiment(Q_SECOND)

        # EXPECT a new fitter
        assert joint.fitter is not fitter

    def test_fit_recovers_shared_width_from_both_data_sets(self):
        # WHEN two data sets of the same Lorentzian at overlapping Q, with different resolutions
        rng = np.random.default_rng(0)
        energy = np.linspace(-3.0, 3.0, 61)
        true_width = 0.4

        def data(Q):
            # The resolution is narrow enough to be neglected when generating the truth.
            truth = np.array([Lorentzian(area=1.0, width=true_width).evaluate(energy) for _ in Q])
            noisy = truth + rng.normal(0.0, 0.02, truth.shape)
            return make_experiment(Q, energy=energy, values=noisy)

        def analysis(name, Q, resolution_width):
            result = Analysis(
                display_name=name,
                experiment=data(Q),
                sample_model=SampleModel(components=shared),
                instrument_model=InstrumentModel(
                    resolution_model=ResolutionModel(components=Gaussian(width=resolution_width))
                ),
            )
            result.instrument_model.resolution_model.fix_all_parameters()
            result.fix_energy_offset()
            return result

        shared = Lorentzian(name='L2', area=0.7, width=0.6)
        first = analysis('first', Q_FIRST, 0.01)
        second = analysis('second', Q_SECOND, 0.02)
        joint = JointAnalysis(analyses=[first, second])

        # THEN
        results = joint.fit()

        # EXPECT one width per common Q, close to the truth, reported by both analyses
        assert len(results) == 6
        assert results[0].success
        for Q_index in range(3):
            assert self.l2_width(first, Q_index).value == pytest.approx(true_width, rel=0.15)
        assert self.l2_width(second, 0).value == self.l2_width(first, 1).value
        assert self.l2_width(second, 1).value == self.l2_width(first, 2).value
        assert self.l2_width(second, 2).value == pytest.approx(true_width, rel=0.15)

    def test_fit_recovers_shared_diffusion_coefficient(self):
        # WHEN two data sets of Brownian diffusion at partly overlapping Q
        rng = np.random.default_rng(0)
        energy = np.linspace(-3.0, 3.0, 61)
        true_D = 2.0e-9

        def analysis(name, Q):
            truth_model = BrownianTranslationalDiffusion(diffusion_coefficient=true_D, Q=Q)
            truth = np.array([
                truth_model.get_component_collections(Q_index).evaluate(energy)
                for Q_index in range(len(Q))
            ])
            noisy = truth + rng.normal(0.0, 0.02, truth.shape)
            result = Analysis(
                display_name=name,
                experiment=make_experiment(Q, energy=energy, values=noisy),
                sample_model=SampleModel(diffusion_models=shared),
                instrument_model=InstrumentModel(
                    resolution_model=ResolutionModel(components=Gaussian(width=0.01))
                ),
            )
            result.instrument_model.resolution_model.fix_all_parameters()
            result.fix_energy_offset()
            return result

        shared = BrownianTranslationalDiffusion(diffusion_coefficient=1.0e-9)
        joint = JointAnalysis(analyses=[analysis('first', Q_FIRST), analysis('second', Q_SECOND)])

        # THEN
        results = joint.fit()

        # EXPECT one diffusion coefficient, close to the truth
        assert results[0].success
        assert shared.diffusion_coefficient.value == pytest.approx(true_D, rel=0.1)

    #############
    # Parameters to dataset
    #############

    def test_parameters_to_dataset_covers_union_of_Q(self, joint, first, second):
        # WHEN linked models with distinct per-Q values on the shared component
        joint.link_shared_components()
        for Q_index, width in enumerate([0.11, 0.12, 0.13]):
            self.l2_width(first, Q_index).value = width
        self.l2_width(second, 2).value = 0.14

        # THEN
        ds = joint.parameters_to_dataset()

        # EXPECT the Q values of both analyses, the owner's values where they overlap
        np.testing.assert_allclose(ds.coords['Q'].values, [0.5, 1.0, 1.5, 2.0])
        assert ds.coords['Q'].unit == sc.Unit('1/angstrom')
        assert set(ds.keys()) == {'L2 area', 'L2 center', 'L2 width'}
        np.testing.assert_allclose(ds['L2 width'].values, [0.11, 0.12, 0.13, 0.14])
        assert ds['L2 width'].unit == sc.Unit('meV')
        assert ds['L2 width'].variances is not None

    def test_parameters_to_dataset_is_nan_where_a_component_is_absent(self, shared, first):
        # WHEN a second shared component, held at other Q values than the first one
        other = Lorentzian(name='L4', width=0.7)
        second = make_analysis('second', Q_SECOND, [shared, other])
        third = make_analysis('third', [2.0, 2.5], [other])

        # THEN
        ds = JointAnalysis(analyses=[first, second, third]).parameters_to_dataset()

        # EXPECT one row per Q value of any analysis, NaN where a component is not held
        np.testing.assert_allclose(ds.coords['Q'].values, [0.5, 1.0, 1.5, 2.0, 2.5])
        np.testing.assert_allclose(ds['L2 width'].values, [0.3, 0.3, 0.3, 0.3, np.nan])
        np.testing.assert_allclose(ds['L4 width'].values, [np.nan, 0.7, 0.7, 0.7, 0.7])

    def test_parameters_to_dataset_without_shared_components_is_empty(self):
        # WHEN
        first = make_analysis('first', Q_FIRST, [Lorentzian(name='L1')])
        second = make_analysis('second', Q_FIRST, [Lorentzian(name='L3')])

        # THEN
        ds = JointAnalysis(analyses=[first, second]).parameters_to_dataset()

        # EXPECT
        assert len(ds.keys()) == 0
        assert ds.coords['Q'].shape == (0,)

    def test_parameters_to_dataset_duplicate_names_raise(self):
        # WHEN two shared components whose parameters have the same names
        shared_a = Lorentzian(name='L')
        shared_b = Lorentzian(name='L')
        with pytest.warns(UserWarning, match='Duplicate component names'):
            first = make_analysis('first', Q_FIRST, [shared_a, shared_b])
            second = make_analysis('second', Q_FIRST, [shared_a, shared_b])

        # THEN EXPECT
        with pytest.raises(ValueError, match='Duplicate parameter name'):
            JointAnalysis(analyses=[first, second]).parameters_to_dataset()

    #############
    # Dunder methods
    #############

    def test_repr(self, joint):
        # WHEN THEN
        text = repr(joint)

        # EXPECT
        assert text == (
            "JointAnalysis(\n    analyses=['first', 'second'],\n    shared_components=['L2'],\n)"
        )
