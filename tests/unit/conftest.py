# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from easyscience.global_object.global_object import GlobalObject

# TODO: remove once weakref bug is fixed


@pytest.fixture(autouse=True)
def reset_global_object():
    global_object = GlobalObject()
    global_object.map._clear()
