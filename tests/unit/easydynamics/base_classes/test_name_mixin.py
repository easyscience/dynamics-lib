# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from easyscience import global_object
from easyscience.base_classes import EasyBase

from easydynamics.base_classes.name_mixin import NameMixin
from easydynamics.variable import Parameter


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

        # WHEN a class whose MRO reaches the registering EasyBase through NameMixin
        class _RegisteredWithName(NameMixin, EasyBase):
            pass

        names_before = set(global_object.registry.names())

        # THEN EXPECT construction fails on the invalid name
        with pytest.raises(TypeError, match=r'Name must be a string'):
            _RegisteredWithName(name=123)

        # EXPECT no half-constructed object was registered in the global registry
        assert set(global_object.registry.names()) <= names_before

    #############
    # Unique names
    #############

    def test_generated_unique_name_has_the_class_prefix(self):
        """Generated names keep EasyScience's ``Class_N`` form."""
        # WHEN THEN
        parameter = Parameter(value=1.0, name='p')

        # EXPECT
        assert parameter.unique_name.startswith('Parameter_')
        assert parameter.unique_name.rsplit('_', 1)[1].isdecimal()

    def test_generated_unique_name_counts_as_default(self):
        """A generated name is flagged as default, so copies get their own."""
        # WHEN an object whose unique name was generated
        parameter = Parameter(value=1.0, name='p')

        # THEN
        serialized = parameter.to_dict()
        copied = Parameter.from_dict(serialized)

        # EXPECT the name is left out of the serialization, so the copy gets its own
        assert parameter._default_unique_name is True
        assert 'unique_name' not in serialized
        assert copied.unique_name != parameter.unique_name

    def test_setting_a_unique_name_makes_it_explicit(self):
        """Renaming an object turns its generated name into an explicit one."""
        # WHEN
        parameter = Parameter(value=1.0, name='p')

        # THEN
        parameter.unique_name = 'TestNameMixinRenamed'

        # EXPECT
        assert parameter._default_unique_name is False
        assert parameter.to_dict()['unique_name'] == 'TestNameMixinRenamed'

    def test_subclass_not_passing_unique_name_uses_easyscience_default(self):
        """A class that does not pass unique_name on still gets a registered default name."""

        # WHEN
        class _Plain(NameMixin, EasyBase):
            def __init__(self):
                super().__init__(name='plain')

        # THEN
        plain = _Plain()

        # EXPECT
        assert plain.unique_name.startswith('_Plain_')
        assert plain._default_unique_name is True

    def test_explicit_unique_name_is_kept_and_serialized(self):
        """An explicit unique name is neither replaced nor dropped from the serialization."""
        # WHEN THEN
        parameter = Parameter(value=1.0, name='p', unique_name='TestNameMixinExplicit')

        # EXPECT
        assert parameter.unique_name == 'TestNameMixinExplicit'
        assert parameter.to_dict()['unique_name'] == 'TestNameMixinExplicit'

    def test_generated_unique_names_are_distinct(self):
        """Every generated name is different."""
        # WHEN THEN
        parameters = [Parameter(value=1.0, name='p') for _ in range(50)]

        # EXPECT
        assert len({parameter.unique_name for parameter in parameters}) == 50
