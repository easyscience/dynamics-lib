# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easydynamics.base_classes.easydynamics_list import EasyDynamicsList
from easydynamics.exceptions import AmbiguousNameError
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.components.model_component import ModelComponent


class TestEasyDynamicsList:
    """Tests for the EasyDynamicsList class."""

    @pytest.fixture
    def easy_dynamics_list(self):
        """Fixture for creating an instance of EasyDynamicsList."""
        gaussian = Gaussian(name='Gaussian')
        lorentzian = Lorentzian(name='Lorentzian')
        return EasyDynamicsList(
            gaussian,
            lorentzian,
            protected_types=ModelComponent,
            display_name='TestList',
        )

    def test_initialization(self, easy_dynamics_list):
        """Test that the EasyDynamicsList is initialized correctly."""
        # WHEN THEN EXPECT
        assert easy_dynamics_list.display_name == 'TestList'
        assert len(easy_dynamics_list) == 2
        assert isinstance(easy_dynamics_list[0], Gaussian)
        assert isinstance(easy_dynamics_list[1], Lorentzian)

    def test_initialization_invalid_type(self):
        """Test that initializing with an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            EasyDynamicsList('Not a ModelComponent', protected_types=ModelComponent)

    def test_initialization_invalid_type_in_list(self):
        """Test that initializing with a list containing an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            EasyDynamicsList(
                [Gaussian(name='Gaussian'), 'Not a ModelComponent'],
                protected_types=ModelComponent,
            )

    def test_insert(self, easy_dynamics_list):
        """Test that the insert method works correctly."""
        # WHEN
        new_gaussian = Gaussian(name='NewGaussian')

        # THEN
        easy_dynamics_list.insert(1, new_gaussian)

        # EXPECT
        assert len(easy_dynamics_list) == 3
        assert isinstance(easy_dynamics_list[1], Gaussian)
        assert easy_dynamics_list[1].name == 'NewGaussian'

    def test_insert_invalid_type(self, easy_dynamics_list):
        """Test that inserting an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.insert(0, 'Not a ModelComponent')

    def test_insert_repeated_name(self, easy_dynamics_list):
        """Test that inserting an item with a repeated name works"""
        # WHEN
        new_gaussian = Gaussian(name='Gaussian')

        # THEN
        easy_dynamics_list.insert(1, new_gaussian)

        # EXPECT
        assert len(easy_dynamics_list) == 3
        assert easy_dynamics_list[1] is new_gaussian

    def test_insert_repeated_component_warns(self, easy_dynamics_list):
        """Test that inserting a repeated component raises a warning."""
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match=r'already in EasyDynamicsList'):
            easy_dynamics_list.insert(1, easy_dynamics_list[0])

        assert len(easy_dynamics_list) == 2
        assert easy_dynamics_list[1] is not easy_dynamics_list[0]

    def test_append(self, easy_dynamics_list):
        """Test that the append method works correctly."""
        # WHEN
        new_lorentzian = Lorentzian(name='NewLorentzian')

        # THEN
        easy_dynamics_list.append(new_lorentzian)

        # EXPECT
        assert len(easy_dynamics_list) == 3
        assert isinstance(easy_dynamics_list[2], Lorentzian)
        assert easy_dynamics_list[2].name == 'NewLorentzian'

    def test_append_invalid_type(self, easy_dynamics_list):
        """Test that appending an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.append('Not a ModelComponent')

    def test_append_repeated_component_warns(self, easy_dynamics_list):
        """Test that appending a repeated component raises a warning."""
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match=r'already in EasyDynamicsList'):
            easy_dynamics_list.append(easy_dynamics_list[0])

        assert len(easy_dynamics_list) == 2

    def test_extend(self, easy_dynamics_list):
        """Test that the extend method works correctly."""
        # WHEN
        new_gaussian = Gaussian(name='NewGaussian')
        new_lorentzian = Lorentzian(name='NewLorentzian')

        # THEN
        easy_dynamics_list.extend([new_gaussian, new_lorentzian])

        # EXPECT
        assert len(easy_dynamics_list) == 4
        assert isinstance(easy_dynamics_list[2], Gaussian)
        assert isinstance(easy_dynamics_list[3], Lorentzian)
        assert easy_dynamics_list[2].name == 'NewGaussian'
        assert easy_dynamics_list[3].name == 'NewLorentzian'

    def test_extend_invalid_type(self, easy_dynamics_list):
        """Test that extending with an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.extend(['Not a ModelComponent'])

    def test_extend_non_iterable(self, easy_dynamics_list):
        """Test that extending with a non-iterable raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.extend('Not an iterable')

    def test_extend_repeated_component_warns(self, easy_dynamics_list):
        """Test that extending with a repeated component raises a warning."""
        # WHEN THEN EXPECT
        with pytest.warns(UserWarning, match=r'already in EasyDynamicsList'):
            easy_dynamics_list.extend([easy_dynamics_list[0]])

        assert len(easy_dynamics_list) == 2

    def test_pop(self, easy_dynamics_list):
        """Test that the pop method works correctly."""
        # WHEN THEN
        popped_item = easy_dynamics_list.pop(0)

        # EXPECT
        assert isinstance(popped_item, Gaussian)
        assert popped_item.name == 'Gaussian'
        assert len(easy_dynamics_list) == 1

        # WHEN THEN
        popped_item = easy_dynamics_list.pop('Lorentzian')

        # EXPECT
        assert isinstance(popped_item, Lorentzian)
        assert popped_item.name == 'Lorentzian'
        assert len(easy_dynamics_list) == 0

    def test_pop_invalid_index_type(self, easy_dynamics_list):
        """Test that popping with an invalid index type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.pop(1.5)

    def test_pop_nonexistent_name(self, easy_dynamics_list):
        """Test that popping with a nonexistent name raises a KeyError."""
        # WHEN THEN EXPECT
        with pytest.raises(KeyError, match=r'No item with name "Nonexistent" found'):
            easy_dynamics_list.pop('Nonexistent')

    def test_names(self, easy_dynamics_list):
        """Test that the names method returns the correct list of names."""
        # WHEN THEN EXPECT
        assert easy_dynamics_list.get_names() == ['Gaussian', 'Lorentzian']

    def test_duplicate_names_no_duplicates(self, easy_dynamics_list):
        """Test that the get_duplicate_names method returns an empty list
        when there are no duplicates."""
        # WHEN THEN EXPECT
        assert easy_dynamics_list.get_duplicate_names() == []

    def test_duplicate_names_with_duplicates(self, easy_dynamics_list):
        """Test that the get_duplicate_names method returns the correct
        list of duplicate names."""
        # WHEN
        new_gaussian = Gaussian(name='Gaussian')
        easy_dynamics_list.append(new_gaussian)

        # THEN EXPECT
        assert easy_dynamics_list.get_duplicate_names() == ['Gaussian']

    def test_getitem_int(self, easy_dynamics_list):
        """Test getting an item by integer index."""
        # WHEN

        # THEN
        item = easy_dynamics_list[0]

        # EXPECT
        assert isinstance(item, Gaussian)
        assert item.name == 'Gaussian'

    def test_getitem_slice(self, easy_dynamics_list):
        """Test getting items by slice."""
        # WHEN

        # THEN
        sliced = easy_dynamics_list[:1]

        # EXPECT
        assert isinstance(sliced, EasyDynamicsList)
        assert len(sliced) == 1
        assert isinstance(sliced[0], Gaussian)
        assert sliced[0].name == 'Gaussian'

    def test_getitem_name(self, easy_dynamics_list):
        """Test getting an item by name."""
        # WHEN

        # THEN
        item = easy_dynamics_list['Gaussian']

        # EXPECT
        assert isinstance(item, Gaussian)
        assert item.name == 'Gaussian'

    def test_getitem_nonexistent_name(self, easy_dynamics_list):
        """Test getting an item with a nonexistent name raises KeyError."""
        # WHEN THEN EXPECT
        with pytest.raises(KeyError, match=r'No item with name "Nonexistent" found'):
            easy_dynamics_list['Nonexistent']

    def test_getitem_ambiguous_name(self, easy_dynamics_list):
        """Test getting an item with duplicate names raises AmbiguousNameError."""
        # WHEN
        easy_dynamics_list.append(Gaussian(name='Gaussian'))

        # THEN EXPECT
        with pytest.raises(AmbiguousNameError):
            easy_dynamics_list['Gaussian']

    def test_getitem_invalid_type(self, easy_dynamics_list):
        """Test getting an item with an invalid index type raises TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError, match=r'Index must be an int, slice, or str'):
            easy_dynamics_list[1.5]

    #############
    # Item assignment
    #############

    def test_setitem(self, easy_dynamics_list):
        """Test assigning an item by index replaces it and nothing else."""
        # WHEN
        new_gaussian = Gaussian(name='ReplacementGaussian')

        # THEN
        easy_dynamics_list[0] = new_gaussian

        # EXPECT
        assert easy_dynamics_list[0] is new_gaussian
        assert len(easy_dynamics_list) == 2

    def test_setitem_invalid_type_raises(self, easy_dynamics_list):
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list[0] = 'Not a ModelComponent'

    def test_setitem_repeated_component_warns(self, easy_dynamics_list):
        """Test that item assignment warns and ignores like append/insert do."""
        # WHEN THEN EXPECT assigning an item already in the list warns and is ignored
        with pytest.warns(UserWarning, match=r'already in EasyDynamicsList'):
            easy_dynamics_list[1] = easy_dynamics_list[0]

        assert easy_dynamics_list[1] is not easy_dynamics_list[0]

    #############
    # Versioning
    #############

    def test_version_starts_at_zero(self, easy_dynamics_list):
        # WHEN a freshly constructed list, even with initial items
        # THEN EXPECT version is 0
        assert easy_dynamics_list.version == 0

    def test_version_is_read_only(self, easy_dynamics_list):
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            easy_dynamics_list.version = 5

    def test_version_bumps_on_every_mutator(self, easy_dynamics_list):
        """Every mutating operation increments version; reads do not."""
        # WHEN
        version = easy_dynamics_list.version

        # THEN append
        easy_dynamics_list.append(Gaussian(name='V1'))
        # EXPECT
        assert easy_dynamics_list.version == version + 1

        # THEN insert
        easy_dynamics_list.insert(0, Gaussian(name='V2'))
        # EXPECT
        assert easy_dynamics_list.version == version + 2

        # THEN extend (one bump per item)
        easy_dynamics_list.extend([Gaussian(name='V3'), Gaussian(name='V4')])
        # EXPECT
        assert easy_dynamics_list.version == version + 4

        # THEN item assignment
        easy_dynamics_list[0] = Gaussian(name='V5')
        # EXPECT
        assert easy_dynamics_list.version == version + 5

        # THEN pop by index and by name
        easy_dynamics_list.pop(0)
        easy_dynamics_list.pop('V1')
        # EXPECT
        assert easy_dynamics_list.version == version + 7

        # THEN remove and del
        item = easy_dynamics_list[0]
        easy_dynamics_list.remove(item)
        del easy_dynamics_list[0]
        # EXPECT
        assert easy_dynamics_list.version == version + 9

        # THEN sort
        easy_dynamics_list.sort(key=lambda c: c.name)
        # EXPECT
        assert easy_dynamics_list.version == version + 10

        # THEN clear
        n_items = len(easy_dynamics_list)
        easy_dynamics_list.clear()
        # EXPECT one bump per removed item, and reading version mutates nothing
        assert easy_dynamics_list.version == version + 10 + n_items
        assert easy_dynamics_list.version == version + 10 + n_items

    def test_version_does_not_bump_on_ignored_duplicate(self, easy_dynamics_list):
        # WHEN
        version = easy_dynamics_list.version

        # THEN an insert that is ignored because the item is already in the list
        with pytest.warns(UserWarning, match=r'already in EasyDynamicsList'):
            easy_dynamics_list.insert(1, easy_dynamics_list[0])

        # EXPECT no mutation happened, so no version bump
        assert easy_dynamics_list.version == version

    def test_version_does_not_bump_on_failed_mutation(self, easy_dynamics_list):
        # WHEN
        version = easy_dynamics_list.version

        # THEN EXPECT failed mutations leave the version unchanged
        with pytest.raises(TypeError):
            easy_dynamics_list.append('Not a ModelComponent')
        with pytest.raises(KeyError):
            easy_dynamics_list.pop('Nonexistent')
        assert easy_dynamics_list.version == version
