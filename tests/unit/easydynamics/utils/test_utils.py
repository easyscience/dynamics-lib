# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc

from easydynamics.utils.utils import _in_notebook
from easydynamics.utils.utils import _validate_and_convert_Q
from easydynamics.utils.utils import _validate_unit


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
        # WHEN THEN
        result = _validate_and_convert_Q(Q_input)

        # EXPECT
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected)

    def test_validate_and_convert_Q_scipp_variable(self):
        # WHEN
        Q = sc.array(dims=['Q'], values=[1.0, 2.0], unit='1/angstrom')

        # THEN
        result = _validate_and_convert_Q(Q)

        # EXPECT
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0])

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
        unit = _validate_unit(unit_input)

        if unit_input is None:
            assert unit is None
        else:
            assert isinstance(unit, str)

    def test_validate_unit_string_conversion(self):
        unit = _validate_unit(sc.Unit('meV'))

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
