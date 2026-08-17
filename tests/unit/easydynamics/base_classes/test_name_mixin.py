# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easydynamics.base_classes.name_mixin import NameMixin


class TestNameMixin:
    """Tests for the NameMixin class."""

    @pytest.fixture
    def name_mixin(self):
        """Fixture for creating an instance of NameMixin."""

        return NameMixin(name='TestModel')

    def test_initialization(self, name_mixin):
        """Test that the NameMixin is initialized correctly."""

        # WHEN THEN EXPECT
        assert name_mixin.name == 'TestModel'

    def test_init_raises_type_error_for_invalid_name(self):
        """Test that initializing with an invalid name raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            NameMixin(name=123)  # Not a string

    def test_init_name_cannot_be_none(self):
        """Test that initializing with name as None raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            NameMixin(name=None)

    def test_name_setter_and_getter(self, name_mixin):
        """Test that the name setter and getter work correctly."""
        # WHEN THEN EXPECT
        assert name_mixin.name == 'TestModel'

        # THEN
        name_mixin.name = 'NewName'

        # EXPECT
        assert name_mixin.name == 'NewName'

        # THEN
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            name_mixin.name = None

    @pytest.mark.parametrize(
        'invalid_name',
        [
            123,  # Not a string
            [1, 2, 3],  # Not a string
            {'name': 'Test'},  # Not a string
        ],
        ids=['integer', 'list', 'dict'],
    )
    def test_name_setter_invalid_type(self, name_mixin, invalid_name):
        """Test that setting the name to an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            name_mixin.name = invalid_name

    def test_invalid_name_fails_before_global_registration(self):
        """Regression: name validation must run before the parent registers the object."""
        # WHEN a class whose MRO reaches the registering NewBase through NameMixin
        from easyscience import global_object
        from easyscience.base_classes.new_base import NewBase

        class _RegisteredWithName(NameMixin, NewBase):
            pass

        vertices_before = set(global_object.map.vertices())

        # THEN EXPECT construction fails on the invalid name
        with pytest.raises(TypeError, match=r'Name must be a string'):
            _RegisteredWithName(name=123)

        # EXPECT no half-constructed object was registered in the global map
        assert set(global_object.map.vertices()) == vertices_before
