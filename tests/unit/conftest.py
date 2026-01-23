# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from easyscience.global_object.global_object import GlobalObject

# TODO: remove once weakref bug is fixed


@pytest.fixture(autouse=True)
def reset_global_object():
    # Clear the existing global object
    global_obj = GlobalObject()
    global_obj.map._store.clear()
    global_obj.map._Map__type_dict.clear()  # private dict, needed for weakref finalizers
    yield
