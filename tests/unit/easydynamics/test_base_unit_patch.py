# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import scipp as sc
from easyscience.variable import Parameter as EasyScienceParameter

from easydynamics.variable import DescriptorNumber
from easydynamics.variable import Parameter


class TestBaseUnitPatch:
    #############
    # Units with a numeric factor are kept
    #############

    def test_parameter_keeps_unit_value_and_bounds(self):
        "Regression: the unit became J*m, and the value 1.6e-33"
        # WHEN THEN
        parameter = Parameter(0.1, name='D', unit='meV*angstrom', min=0.01, max=1.0)

        # EXPECT
        assert parameter.full_value.unit == sc.Unit('meV*angstrom')
        assert parameter.value == pytest.approx(0.1)
        assert parameter.min == pytest.approx(0.01)
        assert parameter.max == pytest.approx(1.0)

    def test_value_set_later_is_read_in_the_unit(self):
        # WHEN
        parameter = Parameter(0.1, name='D', unit='meV*angstrom')

        # THEN
        parameter.value = 0.2

        # EXPECT
        assert parameter.full_value.unit == sc.Unit('meV*angstrom')
        assert parameter.value == pytest.approx(0.2)

    def test_to_dict_round_trip_keeps_the_value(self):
        # WHEN
        parameter = Parameter(0.1, name='D', unit='meV*angstrom**2', min=0.01)

        # THEN
        parameter_copy = Parameter.from_dict(parameter.to_dict())

        # EXPECT
        assert parameter_copy.full_value.unit == sc.Unit('meV*angstrom**2')
        assert parameter_copy.value == pytest.approx(0.1)
        assert parameter_copy.min == pytest.approx(0.01)

    def test_easyscience_parameter_is_patched_too(self):
        # WHEN THEN the patch changes the EasyScience class, so it applies to its objects too
        parameter = EasyScienceParameter(value=0.1, unit='meV*angstrom')

        # EXPECT
        assert parameter.value == pytest.approx(0.1)

    #############
    # Arithmetic
    #############

    def test_product_keeps_the_unit(self):
        # WHEN
        energy = Parameter(2.0, name='E', unit='meV', min=1.0, max=4.0)
        length = DescriptorNumber(3.0, name='L', unit='angstrom')

        # THEN
        product = energy * length

        # EXPECT
        assert product.full_value.unit == sc.Unit('meV*angstrom')
        assert product.value == pytest.approx(6.0)
        assert product.min == pytest.approx(3.0)
        assert product.max == pytest.approx(12.0)

    def test_chained_arithmetic_keeps_the_unit(self):
        # WHEN
        energy = Parameter(2.0, name='E', unit='meV')
        length = Parameter(3.0, name='L', unit='angstrom')

        # THEN
        result = energy * length / length

        # EXPECT
        assert result.full_value.unit == sc.Unit('meV')
        assert result.value == pytest.approx(2.0)

    def test_power_keeps_the_unit(self):
        # WHEN
        energy = Parameter(2.0, name='E', unit='meV')

        # THEN
        result = energy**2

        # EXPECT
        assert result.full_value.unit == sc.Unit('meV**2')
        assert result.value == pytest.approx(4.0)

    def test_dependency_without_desired_unit_keeps_the_unit(self):
        # WHEN
        D = Parameter(0.1, name='D', unit='meV*angstrom**2')
        Q = DescriptorNumber(2.0, name='Q', unit='1/angstrom')
        width = Parameter(0.0, name='width', unit='meV')

        # THEN
        width.make_dependent_on('D * Q**2', {'D': D, 'Q': Q})
        D.value = 0.2

        # EXPECT
        assert width.full_value.unit == sc.Unit('meV')
        assert width.value == pytest.approx(0.8)

    #############
    # A unit that is only a number is made dimensionless
    #############

    def test_parameter_with_a_unit_that_is_only_a_number(self):
        # WHEN THEN
        parameter = Parameter(1.0, name='ratio', unit='m/mm')

        # EXPECT
        assert parameter.unit == 'dimensionless'
        assert parameter.value == pytest.approx(1000.0)

    @pytest.mark.parametrize(
        ('numerator_unit', 'denominator_unit', 'expected'),
        [
            ('m', 'mm', 1000.0),
            ('mm', 'm', 0.001),
            ('meV', 'ueV', 1000.0),
            ('ueV', 'meV', 0.001),
            ('eV', 'meV', 1000.0),
            ('meV', 'J', 1.602176634e-22),
            ('angstrom', 'nm', 0.1),
            ('m', 'angstrom', 1.0e10),
            ('ps', 'ns', 0.001),
            ('K', 'mK', 1000.0),
            ('1/angstrom', '1/nm', 10.0),
            ('meV*angstrom', 'meV*nm', 0.1),
            ('meV', 'meV', 1.0),
        ],
    )
    def test_ratio_of_compatible_units_is_dimensionless(
        self, numerator_unit, denominator_unit, expected
    ):
        "Scipp gives such a ratio a unit that is only a number, e.g. 1000 for m/mm"
        # WHEN
        numerator = Parameter(1.0, name='a', unit=numerator_unit)
        denominator = Parameter(1.0, name='b', unit=denominator_unit)

        # THEN
        ratio = numerator / denominator

        # EXPECT
        assert ratio.unit == 'dimensionless'
        assert ratio.value == pytest.approx(expected)

    def test_ratio_of_unrelated_units_keeps_its_unit(self):
        # WHEN
        energy = Parameter(1.0, name='E', unit='meV')
        frequency = Parameter(1.0, name='f', unit='THz')

        # THEN
        ratio = energy / frequency

        # EXPECT
        assert ratio.full_value.unit == sc.Unit('meV/THz')
        assert ratio.value == pytest.approx(1.0)
