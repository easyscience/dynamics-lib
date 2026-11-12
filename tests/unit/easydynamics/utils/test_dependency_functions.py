# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from scipp import UnitError

from easydynamics.utils.dependency_functions import DependencyFunction
from easydynamics.utils.dependency_functions import create_dependency_functions
from easydynamics.variable import DescriptorNumber
from easydynamics.variable import Parameter


class TestDependencyFunction:
    @pytest.fixture
    def exp(self):
        return DependencyFunction('exp', np.exp, np.exp)

    def test_call_with_descriptor_number(self, exp):
        # WHEN
        argument = DescriptorNumber(name='x', value=0.5, variance=0.04)

        # THEN
        result = exp(argument)

        # EXPECT the value, and the variance propagated to first order
        assert isinstance(result, DescriptorNumber)
        assert result.unit == 'dimensionless'
        assert result.value == pytest.approx(np.exp(0.5))
        assert result.variance == pytest.approx(np.exp(0.5) ** 2 * 0.04)

    def test_call_converts_argument_to_dimensionless(self, exp):
        # WHEN the argument is dimensionless only after conversion: 0.05 nm/angstrom = 0.5
        argument = DescriptorNumber(name='x', value=0.05, unit='nm/angstrom', variance=0.0004)

        # THEN
        result = exp(argument)

        # EXPECT
        assert result.value == pytest.approx(np.exp(0.5))
        assert result.variance == pytest.approx(np.exp(0.5) ** 2 * 0.04)

    def test_call_with_unit_raises(self, exp):
        # WHEN
        argument = DescriptorNumber(name='x', value=0.5, unit='meV')

        # THEN EXPECT
        with pytest.raises(UnitError):
            exp(argument)

    def test_call_with_number(self, exp):
        # WHEN THEN EXPECT a plain number gives a plain number
        assert exp(0.5) == pytest.approx(np.exp(0.5))

    def test_in_dependency_expression(self, exp):
        # WHEN
        amplitude = Parameter(name='A', value=2.0, unit='meV')
        length = Parameter(name='L', value=0.5, unit='angstrom')
        Q = DescriptorNumber(name='Q', value=1.5, unit='1/angstrom')
        dependent = Parameter(name='w', value=1.0, unit='meV')

        # THEN
        dependent.make_dependent_on(
            'A * exp(-Q * L)',
            {'A': amplitude, 'L': length, 'Q': Q, 'exp': exp},
            desired_unit='meV',
        )

        # EXPECT the dependent Parameter follows the function, also when a dependency changes
        assert dependent.value == pytest.approx(2.0 * np.exp(-0.75))
        length.value = 1.0
        assert dependent.value == pytest.approx(2.0 * np.exp(-1.5))


class TestCreateDependencyFunctions:
    @pytest.mark.parametrize(
        ('name', 'function', 'derivative'),
        [
            ('exp', np.exp, np.exp),
            ('sin', np.sin, np.cos),
            ('cos', np.cos, lambda x: -np.sin(x)),
            ('sinc', lambda x: np.sin(x) / x, lambda x: (x * np.cos(x) - np.sin(x)) / x**2),
        ],
    )
    def test_functions(self, name, function, derivative):
        # WHEN
        argument = DescriptorNumber(name='x', value=0.7, variance=0.01)

        # THEN
        result = create_dependency_functions()[name](argument)

        # EXPECT
        assert result.value == pytest.approx(function(0.7))
        assert result.variance == pytest.approx(derivative(0.7) ** 2 * 0.01)

    def test_sinc_at_zero(self):
        # WHEN
        argument = DescriptorNumber(name='x', value=0.0, variance=0.01)

        # THEN
        result = create_dependency_functions()['sinc'](argument)

        # EXPECT the limit sin(x)/x -> 1, with zero slope
        assert result.value == pytest.approx(1.0)
        assert result.variance == pytest.approx(0.0)

    def test_creates_new_functions_on_each_call(self):
        # WHEN THEN
        first = create_dependency_functions()
        second = create_dependency_functions()

        # EXPECT
        assert first.keys() == second.keys()
        assert all(first[name] is not second[name] for name in first)
