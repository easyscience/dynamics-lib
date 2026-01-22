import pytest
from easyscience.global_object.global_object import GlobalObject

# TODO: remove once weakref bug is fixed


@pytest.fixture(autouse=True)
def reset_global_object():
    global_object = GlobalObject()
    global_object.map._clear()
