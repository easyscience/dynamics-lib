# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easydynamics.io.utils import apply_min_error


class TestApplyMinError:
    @pytest.fixture
    def values(self):
        return np.array([0.0, 1.0, 2.0, np.nan])

    @pytest.fixture
    def variances(self):
        return np.array([0.0, 0.01, 4.0, np.nan])

    def test_none_leaves_data_alone(self, values, variances):
        # THEN
        new_values, new_variances = apply_min_error(values.copy(), variances.copy(), None)

        # EXPECT
        np.testing.assert_array_equal(new_values, values)
        np.testing.assert_array_equal(new_variances, variances)

    def test_number_raises_errors_below_it(self, values, variances):
        # THEN
        new_values, new_variances = apply_min_error(values.copy(), variances.copy(), 0.5)

        # EXPECT errors of 0 and 0.1 go up to 0.5, an error of 2 and a NaN are untouched
        np.testing.assert_array_equal(new_variances, [0.25, 0.25, 4.0, np.nan])
        np.testing.assert_array_equal(new_values, values)

    def test_integer_is_accepted(self, values, variances):
        # THEN
        _, new_variances = apply_min_error(values.copy(), variances.copy(), 1)

        # EXPECT
        np.testing.assert_array_equal(new_variances, [1.0, 1.0, 4.0, np.nan])

    def test_nan_masks_only_zero_error_bins(self, values, variances):
        # THEN
        new_values, new_variances = apply_min_error(values.copy(), variances.copy(), np.nan)

        # EXPECT the small but non-zero error is kept
        np.testing.assert_array_equal(new_values, [np.nan, 1.0, 2.0, np.nan])
        np.testing.assert_array_equal(new_variances, [np.nan, 0.01, 4.0, np.nan])

    @pytest.mark.parametrize('min_error', [0.0, -1.0, np.inf])
    def test_not_positive_and_finite_raises(self, values, variances, min_error):
        # THEN EXPECT
        with pytest.raises(ValueError, match='positive and finite, or NaN'):
            apply_min_error(values, variances, min_error)

    @pytest.mark.parametrize('min_error', ['1', True])
    def test_not_a_number_raises(self, values, variances, min_error):
        # THEN EXPECT
        with pytest.raises(TypeError, match='min_error must be a number or None'):
            apply_min_error(values, variances, min_error)
