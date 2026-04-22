# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easydynamics.settings.detailed_balance_settings import DetailedBalanceSettings


class TestDetailedBalanceSettings:
    @pytest.fixture
    def default_detailed_balance_settings(self):
        return DetailedBalanceSettings()

    def test_init(self, default_detailed_balance_settings):
        """
        Test initialization of DetailedBalanceSettings with default
        parameters.
        """
        # WHEN THEN EXPECT
        assert isinstance(default_detailed_balance_settings, DetailedBalanceSettings)
        assert default_detailed_balance_settings.use_detailed_balance is True
        assert default_detailed_balance_settings.normalize_detailed_balance is True
        assert (
            default_detailed_balance_settings.display_name == "DetailedBalanceSettings"
        )

    def test_init_with_custom_parameters(self):
        """
        Test initialization of DetailedBalanceSettings with custom
        parameters.
        """
        # WHEN
        detailed_balance_settings = DetailedBalanceSettings(
            use_detailed_balance=False,
            normalize_detailed_balance=False,
        )

        # THEN EXPECT
        assert detailed_balance_settings.use_detailed_balance is False
        assert detailed_balance_settings.normalize_detailed_balance is False
        assert detailed_balance_settings.display_name == "DetailedBalanceSettings"

    @pytest.mark.parametrize(
        "invalid_input, expected_exception, match",
        [
            (
                {"use_detailed_balance": "not_a_boolean"},
                TypeError,
                "must be True or False",
            ),
            (
                {"normalize_detailed_balance": "not_a_boolean"},
                TypeError,
                "must be True or False",
            ),
        ],
        ids=[
            "use_detailed_balance_not_boolean",
            "normalize_detailed_balance_not_boolean",
        ],
    )
    def test_init_raises_for_invalid_input(
        self, invalid_input, expected_exception, match
    ):
        """
        Test that initialization raises appropriate exceptions for
        invalid input parameters.
        """
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=match):
            DetailedBalanceSettings(**invalid_input)

    def test_setters_valid(self, default_detailed_balance_settings):

        # WHEN
        # Ensure it's True first so we can test the reset
        assert default_detailed_balance_settings.use_detailed_balance is True
        assert default_detailed_balance_settings.normalize_detailed_balance is True

        # THEN
        default_detailed_balance_settings.use_detailed_balance = False
        default_detailed_balance_settings.normalize_detailed_balance = False

        # EXPECT
        assert default_detailed_balance_settings.use_detailed_balance is False
        assert default_detailed_balance_settings.normalize_detailed_balance is False

    @pytest.mark.parametrize(
        "value, expected_exception, match",
        [
            ("5", TypeError, "must be True or False"),
            (1, TypeError, "must be True or False"),
            (None, TypeError, "must be True or False"),
        ],
        ids=[
            "string",
            "integer",
            "none",
        ],
    )
    def test_setters_invalid(
        self,
        default_detailed_balance_settings,
        value,
        expected_exception,
        match,
    ):
        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=match):
            default_detailed_balance_settings.use_detailed_balance = value

        # WHEN THEN EXPECT
        with pytest.raises(expected_exception, match=match):
            default_detailed_balance_settings.normalize_detailed_balance = value

    def test_repr_default(self, default_detailed_balance_settings):
        # WHEN
        repr_str = repr(default_detailed_balance_settings)

        # EXPECT
        assert repr_str == (
            "DetailedBalanceSettings(use_detailed_balance=True, normalize_detailed_balance=True)"
        )

    def test_repr_reflects_updated_values(self, default_detailed_balance_settings):
        # WHEN
        default_detailed_balance_settings.use_detailed_balance = False
        default_detailed_balance_settings.normalize_detailed_balance = False

        repr_str = repr(default_detailed_balance_settings)

        # EXPECT
        assert repr_str == (
            "DetailedBalanceSettings(use_detailed_balance=False, normalize_detailed_balance=False)"
        )
