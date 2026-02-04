# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

# Local fixture to reset global object map for problematic test
# TODO: remove once weakref bug is fixed


import easyscience.global_object
import pytest

# from easyscience.global_object.map import Map


# @pytest.fixture(autouse=True)
# def reset_global_object(monkeypatch):
#     # Before each test
#     monkeypatch.setattr(easyscience.global_object, 'map', Map())
#     yield
#     # After each test (cleanup)
#     monkeypatch.setattr(easyscience.global_object, 'map', Map())


@pytest.fixture(autouse=True)
def reset_global_object():
    easyscience.global_object.map._clear()
