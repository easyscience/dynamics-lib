# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

# Local fixture to reset global object map for problematic test
# TODO: remove once weakref bug is fixed


# import easyscience.global_object
# import pytest


# @pytest.fixture(autouse=True)
# def reset_global_object():
#     easyscience.global_object.map._clear()

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=False)
def patch_easyscience_map():
    """Patch the problematic Map methods."""
    from easyscience.global_object.map import Map

    # Store the original methods
    original_add_vertex = Map.add_vertex
    # original_vertices = Map.vertices

    def safe_add_vertex(self, obj: object, obj_type: str = None):
        try:
            return original_add_vertex(self, obj, obj_type)
        except KeyError:
            # Object was garbage collected during setup
            name = obj.unique_name
            # Clean up any partial state
            if hasattr(self, '_Map__type_dict') and name in self._Map__type_dict:
                del self._Map__type_dict[name]
            if name in self._store:
                del self._store[name]

    def safe_vertices(self):
        """Safe version of vertices() that handles dictionary changes
        during iteration."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return list(self._store.keys())
            except RuntimeError as e:
                if 'dictionary changed size during iteration' in str(e):
                    if attempt < max_retries - 1:
                        # Force cleanup and try again
                        import gc

                        gc.collect()
                        continue
                    else:
                        # Last attempt - return what we can get
                        try:
                            # Try to get keys in a different way
                            keys = []
                            for k in list(self._store.data.keys()):
                                if k in self._store:
                                    keys.append(k)
                            return keys
                        except:  # noqa: E722
                            return []
                else:
                    raise
        return []

    # Apply the patches
    with (
        patch.object(Map, 'add_vertex', safe_add_vertex),
        patch.object(Map, 'vertices', safe_vertices),
    ):
        yield
