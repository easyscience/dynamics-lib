# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipp as sc

from easydynamics.io.utils import apply_min_error


class TestApplyMinError:
    @pytest.fixture
    def data(self):
        return sc.DataArray(
            sc.array(
                dims=['energy'],
                values=[0.0, 1.0, 2.0],
                variances=[0.0, 0.01, 4.0],
            )
        )

    def test_none_leaves_data_alone(self, data):
        # WHEN
        original = data.copy()

        # THEN
        apply_min_error(data, None)

        # EXPECT
        assert sc.identical(data, original)

    def test_number_raises_errors_below_it(self, data):
        # THEN
        apply_min_error(data, 0.5)

        # EXPECT errors of 0 and 0.1 go up to 0.5, an error of 2 is untouched
        np.testing.assert_array_equal(data.variances, [0.25, 0.25, 4.0])
        np.testing.assert_array_equal(data.values, [0.0, 1.0, 2.0])
        assert dict(data.masks) == {}

    def test_integer_is_accepted(self, data):
        # THEN
        apply_min_error(data, 1)

        # EXPECT
        np.testing.assert_array_equal(data.variances, [1.0, 1.0, 4.0])

    def test_nan_masks_only_zero_error_bins(self, data):
        # THEN
        apply_min_error(data, np.nan)

        # EXPECT the small but non-zero error is kept, and the data itself is untouched
        np.testing.assert_array_equal(data.masks['zero_error'].values, [True, False, False])
        np.testing.assert_array_equal(data.values, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(data.variances, [0.0, 0.01, 4.0])

    @pytest.mark.parametrize('min_error', [0.0, -1.0, np.inf])
    def test_not_positive_and_finite_raises(self, data, min_error):
        # THEN EXPECT
        with pytest.raises(ValueError, match='positive and finite, or NaN'):
            apply_min_error(data, min_error)

    @pytest.mark.parametrize('min_error', ['1', True])
    def test_not_a_number_raises(self, data, min_error):
        # THEN EXPECT
        with pytest.raises(TypeError, match='min_error must be a number or None'):
            apply_min_error(data, min_error)
