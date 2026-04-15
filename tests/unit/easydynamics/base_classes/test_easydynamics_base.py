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
        assert easy_dynamics_base.display_name == 'MyEasyDynamicsModel'
        assert easy_dynamics_base.unique_name is not None

    def test_init_raises_type_error_for_invalid_name(self):
        """Test that initializing with an invalid name raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match='Name must be a string or None.'):
            EasyDynamicsBase(name=123)  # Not a string

    def test_init_name_can_be_none(self):
        """Test that initializing with name as None works correctly."""
        # WHEN THEN EXPECT
        model = EasyDynamicsBase(name=None)

        # THEN EXPECT
        assert model.name is None

    def test_name_setter_and_getter(self, easy_dynamics_base):
        """Test that the name setter and getter work correctly."""
        # WHEN THEN EXPECT
        assert easy_dynamics_base.name == 'TestModel'

        # THEN
        easy_dynamics_base.name = 'NewName'

        # EXPECT
        assert easy_dynamics_base.name == 'NewName'

        # THEN
        easy_dynamics_base.name = None

        # EXPECT
        assert easy_dynamics_base.name is None

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
        with pytest.raises(TypeError, match='Name must be a string or None.'):
            easy_dynamics_base.name = invalid_name
