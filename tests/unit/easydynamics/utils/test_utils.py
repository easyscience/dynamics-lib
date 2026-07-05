# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.utils.utils import _in_notebook
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import _validate_unit
from easydynamics.utils.utils import convert_parameter_unit
from easydynamics.utils.utils import convert_units_with_rollback
from easydynamics.utils.utils import verify_Q_index


class TestVerifyQIndex:
    def test_valid_index(self):
        # WHEN THEN EXPECT: no exception
        verify_Q_index(1, sc.array(dims=['Q'], values=[0.5, 1.0], unit='1/angstrom'))

    def test_none_allowed(self):
        # WHEN THEN EXPECT: no exception
        verify_Q_index(None, None, allow_none=True)

    def test_none_not_allowed_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q_index must be an int'):
            verify_Q_index(None, None)

    def test_non_int_raises(self):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q_index must be an int'):
            verify_Q_index('0', None)

    def test_negative_raises_even_when_Q_is_none(self):
        # WHEN THEN EXPECT
        with pytest.raises(IndexError, match='Q_index must be non-negative'):
            verify_Q_index(-1, None)

    def test_out_of_bounds_raises(self):
        # WHEN THEN EXPECT
        Q = sc.array(dims=['Q'], values=[0.5, 1.0], unit='1/angstrom')
        with pytest.raises(IndexError, match='Q_index 2 is out of bounds'):
            verify_Q_index(2, Q)

    def test_upper_bound_deferred_when_Q_is_none(self):
        # WHEN: Q is None (no data loaded yet)
        # THEN EXPECT: a non-negative index is accepted; the bound check is deferred
        verify_Q_index(100, None)


class TestConvertUnitsWithRollback:
    def test_applies_all_conversions(self):
        # WHEN
        p1 = Parameter(name='p1', value=1.0, unit='meV')
        p2 = Parameter(name='p2', value=2.0, unit='meV')

        # THEN
        convert_units_with_rollback([
            (lambda u, p=p1: convert_parameter_unit(p, u), 'ueV', 'meV'),
            (lambda u, p=p2: convert_parameter_unit(p, u), 'ueV', 'meV'),
        ])

        # EXPECT
        assert sc.Unit(str(p1.unit)) == sc.Unit('ueV')
        assert p1.value == pytest.approx(1000.0)
        assert sc.Unit(str(p2.unit)) == sc.Unit('ueV')
        assert p2.value == pytest.approx(2000.0)

    def test_rolls_back_on_failure_and_reraises(self):
        # WHEN: the second conversion fails (meV -> K is not valid)
        p1 = Parameter(name='p1', value=1.0, unit='meV')
        p2 = Parameter(name='p2', value=2.0, unit='meV')

        def fail(_unit):
            raise RuntimeError('conversion failed')

        # THEN EXPECT: the original exception propagates and p1 is rolled back
        with pytest.raises(RuntimeError, match='conversion failed'):
            convert_units_with_rollback([
                (lambda u, p=p1: convert_parameter_unit(p, u), 'ueV', 'meV'),
                (fail, 'ueV', 'meV'),
            ])

        assert str(p1.unit) == 'meV'
        assert p1.value == pytest.approx(1.0)
        assert str(p2.unit) == 'meV'
        assert p2.value == pytest.approx(2.0)


class TestValidateAndConvertQ:
    @pytest.mark.parametrize(
        'Q_input, expected',
        [
            (1.0, np.array([1.0])),
            (2, np.array([2])),
            ([1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0])),
            (np.array([4.0, 5.0]), np.array([4.0, 5.0])),
        ],
    )
    def test_validate_and_convert_Q_numeric_and_array(self, Q_input, expected):
        # WHEN THEN: numbers, lists, and numpy arrays are assumed to be in 1/angstrom
        result = _validate_and_convert_Q(Q_input)

        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.dims == ('Q',)
        assert result.unit == sc.Unit('1/angstrom')
        np.testing.assert_allclose(result.values, expected)

    def test_validate_and_convert_Q_scipp_variable(self):
        # WHEN
        Q = sc.array(dims=['Q'], values=[1.0, 2.0], unit='1/angstrom')

        # THEN
        result = _validate_and_convert_Q(Q)

        # EXPECT
        assert isinstance(result, sc.Variable)
        assert result.unit == sc.Unit('1/angstrom')
        np.testing.assert_allclose(result.values, [1.0, 2.0])

    def test_validate_and_convert_Q_scipp_variable_other_unit(self):
        # WHEN: a scipp Q in 1/nm
        Q = sc.array(dims=['Q'], values=[10.0, 20.0], unit='1/nm')

        # THEN
        result = _validate_and_convert_Q(Q)

        # EXPECT: converted to 1/angstrom
        assert result.unit == sc.Unit('1/angstrom')
        np.testing.assert_allclose(result.values, [1.0, 2.0])

    def test_validate_and_convert_Q_none(self):
        # WHEN THEN EXPECT
        assert _validate_and_convert_Q(None) is None

    @pytest.mark.parametrize(
        'Q_input',
        [
            'invalid',
            {'a': 1},
            (1, 2),
            object(),
        ],
    )
    def test_validate_and_convert_Q_invalid_type(self, Q_input):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Q must be a number'):
            _validate_and_convert_Q(Q_input)

    def test_validate_and_convert_Q_ndarray_wrong_dim(self):
        # WHEN THEN
        Q = np.array([[1.0, 2.0]])
        # EXPECT
        with pytest.raises(ValueError, match='Q must be a 1-dimensional array'):
            _validate_and_convert_Q(Q)

    def test_validate_and_convert_Q_scipp_wrong_dims(self):
        # WHEN THEN
        Q = sc.array(dims=['x'], values=[1.0, 2.0], unit='1/angstrom')

        # EXPECT
        with pytest.raises(ValueError, match="single dimension named 'Q'"):
            _validate_and_convert_Q(Q)


# --------------------------------------------------------------------


class TestValidateUnit:
    @pytest.mark.parametrize(
        'unit_input',
        [
            None,
            '1/angstrom',
            'meV',
            sc.Unit('meV'),
        ],
    )
    def test_validate_unit_valid(self, unit_input):
        # WHEN

        # THEN
        unit = _validate_unit(unit_input)

        # EXPECT
        if unit_input is None:
            assert unit is None
        else:
            assert isinstance(unit, str)

    def test_validate_unit_string_conversion(self):
        # WHEN

        # THEN
        unit = _validate_unit(sc.Unit('meV'))

        # EXPECT
        assert isinstance(unit, str)
        assert unit == 'meV'

    @pytest.mark.parametrize(
        'unit_input',
        [
            123,
            45.6,
            [],
            {},
            object(),
        ],
    )
    def test_validate_unit_invalid_type(self, unit_input):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='unit must be None, a string, or a scipp Unit'):
            _validate_unit(unit_input)


# -----------------------------


class TestInNotebook:
    def test_in_notebook_returns_true_for_jupyter(self, monkeypatch):
        """Should return True when IPython shell is
        ZMQInteractiveShell (Jupyter)."""

        # WHEN
        class ZMQInteractiveShell:
            __name__ = 'ZMQInteractiveShell'

        # THEN
        monkeypatch.setattr('IPython.get_ipython', lambda: ZMQInteractiveShell())

        # EXPECT
        assert _in_notebook() is True

    def test_in_notebook_returns_false_for_terminal_ipython(self, monkeypatch):
        """Should return False when IPython shell is
        TerminalInteractiveShell."""

        # WHEN
        class TerminalInteractiveShell:
            __name__ = 'TerminalInteractiveShell'

        # THEN

        monkeypatch.setattr('IPython.get_ipython', lambda: TerminalInteractiveShell())

        # EXPECT
        assert _in_notebook() is False

    def test_in_notebook_returns_false_for_unknown_shell(self, monkeypatch):
        """Should return False when IPython shell type is
        unrecognized."""

        # WHEN
        class UnknownShell:
            __name__ = 'UnknownShell'

        # THEN
        monkeypatch.setattr('IPython.get_ipython', lambda: UnknownShell())
        # EXPECT
        assert _in_notebook() is False

    def test_in_notebook_returns_false_when_no_ipython(self, monkeypatch):
        """Should return False when IPython is not installed or
        available."""

        # WHEN
        def raise_import_error(*args, **kwargs):  # noqa: ARG001
            raise ImportError

        # THEN
        monkeypatch.setattr('builtins.__import__', raise_import_error)

        # EXPECT
        assert _in_notebook() is False
