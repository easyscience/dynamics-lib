# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easydynamics.base_classes import EasyDynamicsBase


class TestEasyDynamicsBase:
    """Tests for the EasyDynamicsBase class."""

    @pytest.fixture
    def easy_dynamics_base(self):
        """Fixture for creating an instance of EasyDynamicsBase."""

        return EasyDynamicsBase(name='TestModel')

    def test_initialization(self, easy_dynamics_base):
        """Test that the EasyDynamicsBase is initialized correctly."""

        # WHEN THEN EXPECT
        assert easy_dynamics_base.name == 'TestModel'
        assert easy_dynamics_base.display_name == 'TestModel'
        assert easy_dynamics_base.unique_name is not None

    def test_init_raises_type_error_for_invalid_name(self):
        """Test that initializing with an invalid name raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            EasyDynamicsBase(name=123)  # Not a string

    def test_init_name_cannot_be_none(self):
        """Test that initializing with name as None raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            EasyDynamicsBase(name=None)

    def test_name_setter_and_getter(self, easy_dynamics_base):
        """Test that the name setter and getter work correctly."""
        # WHEN THEN EXPECT
        assert easy_dynamics_base.name == 'TestModel'

        # THEN
        easy_dynamics_base.name = 'NewName'

        # EXPECT
        assert easy_dynamics_base.name == 'NewName'

        # THEN
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            easy_dynamics_base.name = None

    @pytest.mark.parametrize(
        'invalid_name',
        [
            123,  # Not a string
            [1, 2, 3],  # Not a string
            {'name': 'Test'},  # Not a string
        ],
        ids=['integer', 'list', 'dict'],
    )
    def test_name_setter_invalid_type(self, easy_dynamics_base, invalid_name):
        """Test that setting the name to an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Name must be a string.'):
            easy_dynamics_base.name = invalid_name
