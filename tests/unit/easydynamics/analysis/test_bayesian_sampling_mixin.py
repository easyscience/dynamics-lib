# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the BayesianSamplingMixin contract itself, independent of any Analysis."""

import pytest

from easydynamics.analysis.bayesian_sampling import BayesianSamplingMixin


class Incomplete(BayesianSamplingMixin):
    """A subclass that implements none of the hooks."""


@pytest.fixture
def incomplete():
    subject = Incomplete()
    subject._init_bayesian_state()
    return subject


class TestHookContract:
    def test_building_a_fitter_must_be_implemented(self, incomplete):
        # EXPECT
        with pytest.raises(NotImplementedError, match='_build_bayesian_fitter'):
            incomplete._build_bayesian_fitter()

    def test_getting_the_data_must_be_implemented(self, incomplete):
        # EXPECT
        with pytest.raises(NotImplementedError, match='_get_sampling_data'):
            incomplete._get_sampling_data()

    def test_getting_the_chain_parameters_must_be_implemented(self, incomplete):
        # EXPECT
        with pytest.raises(NotImplementedError, match='_get_chain_parameters'):
            incomplete._get_chain_parameters()

    def test_preparing_for_sampling_is_optional(self, incomplete):
        # EXPECT the default hook is a no-op rather than a failure
        assert incomplete._prepare_for_sampling() is None


class TestInitialState:
    def test_nothing_is_cached_before_use(self, incomplete):
        # EXPECT
        assert incomplete.bayesian_sampler is None
        assert incomplete.posterior_result is None
