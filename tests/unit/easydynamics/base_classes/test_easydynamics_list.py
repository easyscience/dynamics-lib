import pytest

from easydynamics.base_classes.easydynamics_list import EasyDynamicsList
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model.components.model_component import ModelComponent


class TestEasyDynamicsList:
    """Tests for the EasyDynamicsList class."""

    @pytest.fixture
    def easy_dynamics_list(self):
        """Fixture for creating an instance of EasyDynamicsList."""
        gaussian = Gaussian(name="Gaussian")
        lorentzian = Lorentzian(name="Lorentzian")
        return EasyDynamicsList(
            gaussian,
            lorentzian,
            protected_types=ModelComponent,
            display_name="TestList",
        )

    def test_initialization(self, easy_dynamics_list):
        """Test that the EasyDynamicsList is initialized correctly."""
        # WHEN THEN EXPECT
        assert easy_dynamics_list.display_name == "TestList"
        assert len(easy_dynamics_list) == 2
        assert isinstance(easy_dynamics_list[0], Gaussian)
        assert isinstance(easy_dynamics_list[1], Lorentzian)

    def test_initialization_invalid_type(self):
        """Test that initializing with an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            EasyDynamicsList("Not a ModelComponent", protected_types=ModelComponent)

    def test_initialization_invalid_type_in_list(self):
        """Test that initializing with a list containing an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            EasyDynamicsList(
                [Gaussian(name="Gaussian"), "Not a ModelComponent"],
                protected_types=ModelComponent,
            )

    def test_init_locks_name(self, easy_dynamics_list):
        """Test that the name is locked."""
        # WHEN THEN EXPECT
        with pytest.raises(AttributeError):
            easy_dynamics_list[0].name = "NewName"

    def test_insert(self, easy_dynamics_list):
        """Test that the insert method works correctly."""
        # WHEN
        new_gaussian = Gaussian(name="NewGaussian")

        # THEN
        easy_dynamics_list.insert(1, new_gaussian)

        # EXPECT
        assert len(easy_dynamics_list) == 3
        assert isinstance(easy_dynamics_list[1], Gaussian)
        assert easy_dynamics_list[1].name == "NewGaussian"

    def test_insert_invalid_type(self, easy_dynamics_list):
        """Test that inserting an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.insert(0, "Not a ModelComponent")

    def test_insert_locks_name(self, easy_dynamics_list):
        """Test that the name of the inserted item is locked."""
        # WHEN
        new_gaussian = Gaussian(name="NewGaussian")

        # THEN
        easy_dynamics_list.insert(1, new_gaussian)

        # EXPECT
        with pytest.raises(AttributeError):
            easy_dynamics_list[1].name = "AnotherName"

    def test_insert_repeated_name(self, easy_dynamics_list):
        """Test that inserting an item with a repeated name raises a ValueError."""
        # WHEN
        new_gaussian = Gaussian(name="Gaussian")
        # THEN EXPECT
        with pytest.raises(ValueError):
            easy_dynamics_list.insert(1, new_gaussian)

    def test_append(self, easy_dynamics_list):
        """Test that the append method works correctly."""
        # WHEN
        new_lorentzian = Lorentzian(name="NewLorentzian")

        # THEN
        easy_dynamics_list.append(new_lorentzian)

        # EXPECT
        assert len(easy_dynamics_list) == 3
        assert isinstance(easy_dynamics_list[2], Lorentzian)
        assert easy_dynamics_list[2].name == "NewLorentzian"

    def test_append_invalid_type(self, easy_dynamics_list):
        """Test that appending an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.append("Not a ModelComponent")

    def test_append_locks_name(self, easy_dynamics_list):
        """Test that the name of the appended item is locked."""
        # WHEN
        new_lorentzian = Lorentzian(name="NewLorentzian")

        # THEN
        easy_dynamics_list.append(new_lorentzian)

        # EXPECT
        with pytest.raises(AttributeError):
            easy_dynamics_list[2].name = "AnotherName"

    def test_append_repeated_name(self, easy_dynamics_list):
        """Test that appending an item with a repeated name raises a ValueError."""
        # WHEN
        new_lorentzian = Lorentzian(name="Lorentzian")
        # THEN EXPECT
        with pytest.raises(ValueError):
            easy_dynamics_list.append(new_lorentzian)

    def test_extend(self, easy_dynamics_list):
        """Test that the extend method works correctly."""
        # WHEN
        new_gaussian = Gaussian(name="NewGaussian")
        new_lorentzian = Lorentzian(name="NewLorentzian")

        # THEN
        easy_dynamics_list.extend([new_gaussian, new_lorentzian])

        # EXPECT
        assert len(easy_dynamics_list) == 4
        assert isinstance(easy_dynamics_list[2], Gaussian)
        assert isinstance(easy_dynamics_list[3], Lorentzian)
        assert easy_dynamics_list[2].name == "NewGaussian"
        assert easy_dynamics_list[3].name == "NewLorentzian"

    def test_extend_invalid_type(self, easy_dynamics_list):
        """Test that extending with an invalid type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.extend(["Not a ModelComponent"])

    def test_extend_non_iterable(self, easy_dynamics_list):
        """Test that extending with a non-iterable raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.extend("Not an iterable")

    def test_extend_locks_names(self, easy_dynamics_list):
        """Test that the names of the extended items are locked."""
        # WHEN
        new_gaussian = Gaussian(name="NewGaussian")
        new_lorentzian = Lorentzian(name="NewLorentzian")

        # THEN
        easy_dynamics_list.extend([new_gaussian, new_lorentzian])

        # EXPECT
        with pytest.raises(AttributeError):
            easy_dynamics_list[2].name = "AnotherName"
        with pytest.raises(AttributeError):
            easy_dynamics_list[3].name = "AnotherName"

    def test_extend_repeated_names(self, easy_dynamics_list):
        """Test that extending with items that have repeated names raises a ValueError."""
        # WHEN
        new_gaussian = Gaussian(name="NewGaussian")
        new_lorentzian = Lorentzian(name="Lorentzian")
        # THEN EXPECT
        with pytest.raises(ValueError):
            easy_dynamics_list.extend([new_gaussian, new_lorentzian])

    def test_extend_repeated_names_in_values(self, easy_dynamics_list):
        """Test that extending with items that have repeated names among themselves raises a ValueError."""
        # WHEN
        new_gaussian1 = Gaussian(name="NewGaussian")
        new_gaussian2 = Gaussian(name="NewGaussian")
        # THEN EXPECT
        with pytest.raises(ValueError):
            easy_dynamics_list.extend([new_gaussian1, new_gaussian2])

    def test_pop(self, easy_dynamics_list):
        """Test that the pop method works correctly."""
        # WHEN THEN
        popped_item = easy_dynamics_list.pop(0)

        # EXPECT
        assert isinstance(popped_item, Gaussian)
        assert popped_item.name == "Gaussian"
        assert len(easy_dynamics_list) == 1
        assert popped_item.is_name_locked() is False

        # WHEN THEN
        popped_item = easy_dynamics_list.pop("Lorentzian")

        # EXPECT
        assert isinstance(popped_item, Lorentzian)
        assert popped_item.name == "Lorentzian"
        assert len(easy_dynamics_list) == 0
        assert popped_item.is_name_locked() is False

    def test_pop_invalid_index_type(self, easy_dynamics_list):
        """Test that popping with an invalid index type raises a TypeError."""
        # WHEN THEN EXPECT
        with pytest.raises(TypeError):
            easy_dynamics_list.pop(1.5)

    def test_pop_nonexistent_name(self, easy_dynamics_list):
        """Test that popping with a nonexistent name raises a KeyError."""
        # WHEN THEN EXPECT
        with pytest.raises(KeyError, match=r'No item with name "Nonexistent" found'):
            easy_dynamics_list.pop("Nonexistent")
