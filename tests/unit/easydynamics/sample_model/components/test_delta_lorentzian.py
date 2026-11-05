# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.components import DeltaLorentzian


class TestDeltaLorentzian:
    @pytest.fixture
    def component(self):
        # delta weight 0.25, one Lorentzian of weight 0.75 and HWHM 0.6 meV
        return DeltaLorentzian(
            name='DLName',
            display_name='TestDL',
            area=2.0,
            center=0.5,
            weights=[0.25, 0.75],
            widths=[0.6],
            x_unit='meV',
        )

    def test_init_no_inputs(self):
        # WHEN THEN
        component = DeltaLorentzian()

        # EXPECT: a delta plus one Lorentzian, centered at 0 and fixed, like Lorentzian
        assert component.display_name == 'DeltaLorentzian'
        assert component.number_of_lorentzians == 1
        assert component.area.value == pytest.approx(1.0)
        assert component.center.value == pytest.approx(0.0)
        assert component.center.fixed is True
        assert component.x_unit == 'meV'
        assert component.y_unit == 'dimensionless'

    def test_initialization(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        assert component.display_name == 'TestDL'
        assert component.number_of_lorentzians == 1
        assert component.area.value == pytest.approx(2.0)
        assert component.center.value == pytest.approx(0.5)
        assert [w.value for w in component.weights] == pytest.approx([0.25, 0.75])
        assert [w.value for w in component.widths] == pytest.approx([0.6])

    def test_number_of_lorentzians_follows_widths(self):
        # WHEN
        component = DeltaLorentzian(weights=[0.5, 0.3, 0.2], widths=[0.1, 0.4])

        # THEN EXPECT
        assert component.number_of_lorentzians == 2
        assert len(component.weights) == 3
        assert len(component.widths) == 2

    def test_units_of_parameters(self):
        # WHEN
        component = DeltaLorentzian(x_unit='meV', y_unit='1/s')

        # THEN EXPECT: area = x_unit * y_unit, widths = x_unit, weights are pure fractions
        assert sc.Unit(str(component.area.unit)) == sc.Unit('meV/s')
        assert sc.Unit(str(component.center.unit)) == sc.Unit('meV')
        assert sc.Unit(str(component.widths[0].unit)) == sc.Unit('meV')
        assert sc.Unit(str(component.weights[0].unit)) == sc.Unit('dimensionless')

    @pytest.mark.parametrize(
        'kwargs, expected_message, expected_error',
        [
            ({'area': 'invalid'}, 'area must be a number', TypeError),
            ({'center': 'invalid'}, 'center must be None or a number', TypeError),
            ({'weights': 0.5, 'widths': [0.6]}, 'weights must be a sequence', TypeError),
            ({'weights': [0.5, 0.5], 'widths': 0.6}, 'widths must be a sequence', TypeError),
            (
                {'weights': [0.5, 'a'], 'widths': [0.6]},
                'weights must contain numbers or Parameters',
                TypeError,
            ),
            (
                {'weights': [0.5, 0.5], 'widths': ['a']},
                'widths must contain numbers or Parameters',
                TypeError,
            ),
            (
                {'weights': [0.5, 0.5], 'widths': []},
                'At least one width must be provided',
                ValueError,
            ),
            (
                {'weights': [0.5], 'widths': [0.6]},
                'weights must have one more element than widths',
                ValueError,
            ),
            (
                {'weights': [0.5, 0.3, 0.2], 'widths': [0.6]},
                'weights must have one more element than widths',
                ValueError,
            ),
            ({'weights': [-0.1, 0.5], 'widths': [0.6]}, 'non-negative', ValueError),
            ({'weights': [0.5, 0.5], 'widths': [0.0]}, 'greater than zero', ValueError),
            ({'weights': [0.5, 0.5], 'widths': [-1.0]}, 'greater than zero', ValueError),
        ],
    )
    def test_init_invalid_inputs(self, kwargs, expected_message, expected_error):
        # WHEN THEN EXPECT
        with pytest.raises(expected_error, match=expected_message):
            DeltaLorentzian(**kwargs)

    def test_parameter_instances_are_used_as_they_are(self):
        # WHEN: Parameter objects (e.g. dependent ones from a model) are passed in
        weight0 = Parameter(name='w0', value=0.4, unit='dimensionless')
        weight1 = Parameter(name='w1', value=0.6, unit='dimensionless')
        width = Parameter(name='g', value=0.3, unit='meV', min=1e-10)

        component = DeltaLorentzian(weights=[weight0, weight1], widths=[width])

        # THEN EXPECT: the very same objects, so a dependency keeps driving the component
        assert component.weights[0] is weight0
        assert component.weights[1] is weight1
        assert component.widths[0] is width
        weight1.value = 0.7
        assert component.weights[1].value == pytest.approx(0.7)

    def test_evaluate_lorentzian_part_hand_derived(self, component: DeltaLorentzian):
        # GIVEN: away from the delta bin only the Lorentzian contributes
        x = np.linspace(-10, 10, 2001)
        i = int(np.argmin(np.abs(x - 1.1)))  # far from the delta bin at x = 0.5

        # WHEN
        result = component.evaluate(x)

        # EXPECT
        expected = 1.5 * (0.6 / np.pi) / ((x[i] - 0.5) ** 2 + 0.6**2)
        assert result[i] == pytest.approx(expected, rel=1e-12)

    def test_evaluate_equals_sum_of_separate_components(self):
        # GIVEN: the component must be exactly a delta plus Lorentzians
        area, center = 3.0, -0.2
        weights, widths = [0.2, 0.5, 0.3], [0.15, 0.9]
        component = DeltaLorentzian(area=area, center=center, weights=weights, widths=widths)
        x = np.linspace(-5, 5, 1001)

        reference = DeltaFunction(area=area * weights[0], center=center).evaluate(x)
        for weight, width in zip(weights[1:], widths, strict=True):
            reference += Lorentzian(area=area * weight, center=center, width=width).evaluate(x)

        # WHEN
        result = component.evaluate(x)

        # THEN EXPECT
        np.testing.assert_allclose(result, reference, rtol=1e-12, atol=0)

    def test_evaluate_integral_is_area_times_weights(self):
        # GIVEN: a Lorentzian of width 0.05 on a grid out to +-200, integrated by trapezoid
        component = DeltaLorentzian(area=2.0, weights=[0.0, 1.0], widths=[0.05])
        x = np.linspace(-200, 200, 400001)

        # WHEN
        integral = np.trapezoid(component.evaluate(x), x)

        # EXPECT: area * w1, up to the Lorentzian tail outside the grid
        assert integral == pytest.approx(2.0, rel=2e-3)

    def test_evaluate_delta_carries_the_delta_weight(self):
        # GIVEN
        component = DeltaLorentzian(area=4.0, weights=[0.5, 0.5], widths=[0.1])
        x = np.linspace(-1, 1, 201)  # bin width 0.01, delta bin at x = 0

        # WHEN
        with_delta = component.evaluate(x)
        component.weights[0].value = 0.0
        without_delta = component.evaluate(x)

        # EXPECT: the delta adds area * w0 / bin_width in the bin at the center, nothing elsewhere
        difference = with_delta - without_delta
        assert difference[100] == pytest.approx(4.0 * 0.5 / 0.01)
        assert np.count_nonzero(difference) == 1

    def test_evaluate_output_scipp(self, component: DeltaLorentzian):
        # WHEN
        result = component.evaluate(np.linspace(-1, 1, 11), output='scipp')

        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == 'dimensionless'

    def test_evaluate_with_x_in_other_unit(self, component: DeltaLorentzian):
        # GIVEN: the same energies in meV and in microeV. The Lorentzian height goes as
        # area / width, and both are 1000 times larger in microeV, so the values are equal.
        # The delta bin is left out, as its height depends on the grid spacing.
        component.weights[0].value = 0.0
        x_mev = np.linspace(-2, 2, 401)
        x_uev = sc.array(dims=['x'], values=x_mev * 1000.0, unit='microeV')

        # WHEN
        in_mev = component.evaluate(x_mev)
        in_uev = component.evaluate(x_uev)

        # EXPECT: same values, and the parameters themselves are left untouched
        np.testing.assert_allclose(in_uev, in_mev, rtol=1e-12)
        assert component.widths[0].unit == 'meV'
        assert component.widths[0].value == pytest.approx(0.6)

    def test_evaluate_with_dependent_parameters_follows_the_dependency(self):
        # GIVEN: a width that is 2 * tau, as a dependent Parameter
        tau = Parameter(name='tau', value=0.5, unit='meV', min=1e-10)
        width = Parameter.from_dependency(
            name='width',
            dependency_expression='2 * tau',
            dependency_map={'tau': tau},
        )
        component = DeltaLorentzian(weights=[0.0, 1.0], widths=[width])
        x = np.linspace(-5, 5, 1001)

        # WHEN
        before = component.evaluate(x)
        tau.value = 1.0
        after = component.evaluate(x)

        # EXPECT: the evaluation reflects the new width of 2.0
        expected = Lorentzian(area=1.0, width=2.0).evaluate(x)
        np.testing.assert_allclose(after, expected, rtol=1e-12)
        assert not np.allclose(before, after)

    @pytest.mark.parametrize('bad_value', [-0.5, np.nan, np.inf])
    def test_evaluate_with_invalid_weight_parameter_raises(self, bad_value):
        # GIVEN: a weight Parameter given as it is, so the constructor does not check its value
        weight = Parameter(name='w', value=bad_value, unit='dimensionless')
        component = DeltaLorentzian(weights=[weight, 1.0], widths=[1.0])

        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='weights'):
            component.evaluate(np.linspace(-2, 2, 101))

    @pytest.mark.parametrize('bad_value', [-1.0, 0.0, np.nan])
    def test_evaluate_with_invalid_width_parameter_raises(self, bad_value):
        # GIVEN
        width = Parameter(name='g', value=bad_value, unit='meV')
        component = DeltaLorentzian(weights=[0.5, 0.5], widths=[width])

        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='widths'):
            component.evaluate(np.linspace(-2, 2, 101))

    def test_evaluate_raises_when_a_dependent_weight_becomes_negative(self):
        # GIVEN: a weight that is 1 - s, valid at first
        s = Parameter(name='s', value=0.25, unit='dimensionless')
        weight = Parameter.from_dependency(
            name='w', dependency_expression='1 - s', dependency_map={'s': s}
        )
        component = DeltaLorentzian(weights=[weight, 0.5], widths=[1.0])
        x = np.linspace(-2, 2, 101)
        component.evaluate(x)

        # WHEN: the dependency pushes the weight below zero
        s.value = 1.5

        # EXPECT
        with pytest.raises(ValueError, match='weights'):
            component.evaluate(x)

    def test_evaluate_accepts_zero_weights(self):
        # GIVEN: zero is a valid weight, for example when a component is switched off
        component = DeltaLorentzian(weights=[0.0, 0.0], widths=[1.0])

        # WHEN THEN EXPECT
        np.testing.assert_array_equal(component.evaluate(np.linspace(-2, 2, 101)), 0.0)

    def test_evaluate_no_delta_spike_when_center_is_outside_the_grid(self):
        # GIVEN: the delta sits at 5 meV, outside the grid
        component = DeltaLorentzian(area=1.0, center=5.0, weights=[1.0, 0.0], widths=[0.1])

        # WHEN
        result = component.evaluate(np.linspace(-1, 1, 101))

        # EXPECT: all weight is on the delta and the Lorentzian has zero weight
        np.testing.assert_array_equal(result, 0.0)

    def test_evaluate_single_point_raises(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT: the delta part defines no bin width on one point, as DeltaFunction
        with pytest.raises(ValueError, match='single x value'):
            component.evaluate(0.0)

    def test_evaluate_invalid_output_raises(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match="output must be 'numpy' or 'scipp'"):
            component.evaluate(np.linspace(-1, 1, 11), output='invalid')

    def test_weights_and_widths_accept_numbers_and_parameters_mixed(self):
        # WHEN
        weight = Parameter(name='w1', value=0.6, unit='dimensionless', min=0.0)
        component = DeltaLorentzian(weights=[0.4, weight], widths=[0.3])

        # EXPECT
        assert component.weights[0].value == pytest.approx(0.4)
        assert component.weights[1] is weight

    def test_weights_and_widths_accessors_return_copies(self, component: DeltaLorentzian):
        # WHEN: the returned lists are modified
        component.weights.clear()
        component.widths.append(Parameter(name='extra', value=1.0))

        # EXPECT: the component is unchanged
        assert len(component.weights) == 2
        assert len(component.widths) == 1
        assert component.number_of_lorentzians == 1

    def test_get_all_parameters(self, component: DeltaLorentzian):
        # WHEN
        parameters = component.get_all_parameters()

        # EXPECT: area, center, 2 weights, 1 width
        assert len(parameters) == 5
        assert component.area in parameters
        assert component.center in parameters
        assert all(w in parameters for w in component.weights)
        assert all(w in parameters for w in component.widths)

    def test_fittable_parameters_exclude_dependent_ones(self):
        # GIVEN: a width that depends on another parameter
        tau = Parameter(name='tau', value=0.5, unit='meV', min=1e-10)
        width = Parameter.from_dependency(
            name='width', dependency_expression='2 * tau', dependency_map={'tau': tau}
        )
        component = DeltaLorentzian(widths=[width])

        # WHEN
        fittable = component.get_fittable_parameters()

        # EXPECT: the dependent width is listed as a parameter but is not fittable
        assert width in component.get_all_parameters()
        assert width not in fittable
        assert component.area in fittable

    def test_free_parameters_exclude_fixed_center(self):
        # WHEN
        component = DeltaLorentzian()  # center None -> fixed

        # EXPECT
        free = component.get_free_parameters()
        assert component.center not in free
        assert component.area in free

    def test_fix_and_free_all_parameters(self, component: DeltaLorentzian):
        # WHEN
        component.fix_all_parameters()

        # THEN
        assert all(p.fixed for p in component.get_all_parameters())

        # WHEN
        component.free_all_parameters()

        # THEN
        assert not any(p.fixed for p in component.get_all_parameters())

    def test_parameters_are_named_after_the_component(self):
        # WHEN
        component = DeltaLorentzian(name='Water', weights=[0.5, 0.3, 0.2], widths=[0.1, 0.4])

        # EXPECT: names are distinct, so a fit report can tell them apart
        names = [p.name for p in component.get_all_parameters()]
        assert len(set(names)) == len(names)
        assert all(n.startswith('Water') for n in names)

    def test_area_setter_rejects_negative(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(ValueError, match='violates the parameter bounds'):
            component.area = -1.0

    def test_area_setter_type_error(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='area must be a number'):
            component.area = 'a'

    def test_center_setter_number(self, component: DeltaLorentzian):
        # WHEN
        component.center = -0.3

        # EXPECT
        assert component.center.value == pytest.approx(-0.3)

    def test_center_setter_type_error(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='center must be a number'):
            component.center = 'a'

    def test_center_setter_none_resets_and_fixes(self, component: DeltaLorentzian):
        # WHEN
        component.center = None

        # THEN EXPECT
        assert component.center.value == pytest.approx(0.0)
        assert component.center.fixed is True

    def test_convert_x_unit(self, component: DeltaLorentzian):
        # WHEN
        component.convert_x_unit('microeV')

        # THEN EXPECT: center, widths and area follow the unit, the weights do not
        assert component.x_unit == 'microeV'
        assert component.center.value == pytest.approx(500.0)
        assert component.widths[0].value == pytest.approx(600.0)
        assert component.area.value == pytest.approx(2000.0)
        assert [w.value for w in component.weights] == pytest.approx([0.25, 0.75])

    def test_convert_x_unit_keeps_the_evaluated_curve(self, component: DeltaLorentzian):
        # GIVEN
        x = sc.array(dims=['x'], values=np.linspace(-2000, 2000, 401), unit='microeV')
        before = component.evaluate(x)

        # WHEN
        component.convert_x_unit('microeV')
        after = component.evaluate(x)

        # EXPECT
        np.testing.assert_allclose(after, before, rtol=1e-10)

    def test_convert_x_unit_type_error(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'x_unit must be a string or sc\.Unit'):
            component.convert_x_unit(1)

    def test_convert_y_unit_rescales_area_only(self):
        # GIVEN: x_unit='meV', y_unit='1/meV', so the area is dimensionless
        component = DeltaLorentzian(
            area=1.0, weights=[0.25, 0.75], widths=[0.6], x_unit='meV', y_unit='1/meV'
        )

        # WHEN: same dimension, different scale
        component.convert_y_unit('1/eV')

        # THEN EXPECT
        assert component.y_unit == '1/eV'
        assert component.area.value == pytest.approx(1e3)
        assert [w.value for w in component.weights] == pytest.approx([0.25, 0.75])
        assert component.widths[0].value == pytest.approx(0.6)

    def test_convert_y_unit_invalid_type_raises(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'y_unit must be a string or sc\.Unit'):
            component.convert_y_unit(123)

    def test_convert_x_unit_rollback_on_failure(self, component: DeltaLorentzian):
        # WHEN THEN
        with pytest.raises(sc.UnitError):
            component.convert_x_unit('m')

        # EXPECT: state rolled back
        assert component.x_unit == 'meV'
        assert component.area.value == pytest.approx(2.0)
        assert component.center.value == pytest.approx(0.5)
        assert component.widths[0].value == pytest.approx(0.6)

    def test_convert_y_unit_rollback_on_failure(self):
        # WHEN
        component = DeltaLorentzian(area=1.0)

        # THEN
        with pytest.raises(sc.UnitError):
            component.convert_y_unit('K')

        # EXPECT: state rolled back
        assert component.y_unit == 'dimensionless'
        assert component.area.value == pytest.approx(1.0)

    def test_y_unit_setter_raises(self, component: DeltaLorentzian):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            component.y_unit = '1/meV'

    def test_is_exported(self):
        # WHEN THEN EXPECT
        import easydynamics as edyn
        from easydynamics.sample_model import DeltaLorentzian as FromSampleModel

        assert edyn.DeltaLorentzian is DeltaLorentzian
        assert FromSampleModel is DeltaLorentzian

    def test_can_be_part_of_a_component_collection(self, component: DeltaLorentzian):
        # GIVEN
        collection = ComponentCollection()
        collection.append_component(component)
        x = np.linspace(-3, 3, 301)

        # WHEN
        result = collection.evaluate(x)

        # EXPECT
        np.testing.assert_allclose(result, component.evaluate(x))

    def test_copy(self, component: DeltaLorentzian):
        # WHEN THEN
        component_copy = copy(component)

        # EXPECT
        assert component_copy is not component
        assert component_copy.display_name == component.display_name
        assert component_copy.number_of_lorentzians == component.number_of_lorentzians
        assert component_copy.area.value == component.area.value
        assert component_copy.center.value == component.center.value
        assert component_copy.center.fixed == component.center.fixed
        assert [w.value for w in component_copy.weights] == [w.value for w in component.weights]
        assert [w.value for w in component_copy.widths] == [w.value for w in component.widths]
        assert component_copy.x_unit == component.x_unit

    def test_repr(self, component: DeltaLorentzian):
        # WHEN THEN
        repr_str = repr(component)

        # EXPECT
        assert 'DeltaLorentzian' in repr_str
        assert 'name = DLName' in repr_str
        assert 'x_unit = meV' in repr_str
        assert 'area =' in repr_str
        assert 'center =' in repr_str
        assert 'weights =' in repr_str
        assert 'widths =' in repr_str
