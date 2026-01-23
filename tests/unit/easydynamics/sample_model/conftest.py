# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

# Local fixture to reset global object map for problematic test
# TODO: remove once weakref bug is fixed

import pytest
from easyscience import global_object


@pytest.fixture(autouse=True)
def reset_global_object():
    global_object.map._clear()
