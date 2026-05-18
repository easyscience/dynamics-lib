# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easydynamics.base_classes.name_mixin import NameMixin


class TestNameMixin:
    """Tests for the NameMixin class."""

    @pytest.fixture
    def name_mixin(self):
        """Fixture for creating an instance of NameMixin."""

        return NameMixin(name="TestModel")

    def test_initialization(self, name_mixin):
        """Test that the NameMixin is initialized correctly."""

        # WHEN THEN EXPECT
        assert name_mixin.name == "TestModel"
        assert name_mixin.display_name == "TestModel"
        assert name_mixin.unique_name is not None
        assert name_mixin.is_name_locked() is False

    def test_init_raises_type_error_for_invalid_name(self):
        """Test that initializing with an invalid name raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r"Name must be a string."):
            NameMixin(name=123)  # Not a string

    def test_init_name_cannot_be_none(self):
        """Test that initializing with name as None raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r"Name must be a string."):
            NameMixin(name=None)

    def test_name_setter_and_getter(self, name_mixin):
        """Test that the name setter and getter work correctly."""
        # WHEN THEN EXPECT
        assert name_mixin.name == "TestModel"

        # THEN
        name_mixin.name = "NewName"

        # EXPECT
        assert name_mixin.name == "NewName"

        # THEN
        with pytest.raises(TypeError, match=r"Name must be a string."):
            name_mixin.name = None

    @pytest.mark.parametrize(
        "invalid_name",
        [
            123,  # Not a string
            [1, 2, 3],  # Not a string
            {"name": "Test"},  # Not a string
        ],
        ids=["integer", "list", "dict"],
    )
    def test_name_setter_invalid_type(self, name_mixin, invalid_name):
        """Test that setting the name to an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r"Name must be a string."):
            name_mixin.name = invalid_name

    def test_name_locking(self, name_mixin):
        """Test that the name locking mechanism works correctly."""
        # WHEN THEN EXPECT
        assert name_mixin.is_name_locked() is False

        # Lock and unlock the name
        # THEN
        name_mixin.lock_name()

        # EXPECT
        assert name_mixin.is_name_locked() is True

        # THEN
        name_mixin.unlock_name()

        # EXPECT
        assert name_mixin.is_name_locked() is False

        # unlock an already unlocked name should raise an error
        # THEN EXPECT
        with pytest.raises(RuntimeError, match=r"Name lock count is already zero."):
            name_mixin.unlock_name()

        # locking twice should require unlocking twice
        # THEN
        name_mixin.lock_name()
        name_mixin.lock_name()

        # THEN EXPECT
        assert name_mixin.is_name_locked() is True

        # THEN
        name_mixin.unlock_name()

        # EXPECT
        assert name_mixin.is_name_locked() is True

        # THEN
        name_mixin.unlock_name()

        # EXPECT
        assert name_mixin.is_name_locked() is False
