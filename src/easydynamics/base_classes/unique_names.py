# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Fast generation of unique names for EasyScience objects.

EasyScience's ``global_object.generate_unique_name`` scans every registered name on each call to
find the highest number used with a prefix, so creating N objects costs O(N^2). Building the per-Q
component collections of a model creates thousands of Parameters, and that scan dominated the time.
Here each prefix keeps a counter instead, and only the candidate name is looked up in the registry,
which is a dictionary lookup.
"""

from __future__ import annotations

import itertools

from easyscience import global_object

_counters: dict[str, itertools.count] = {}


def generate_unique_name(prefix: str) -> str:
    """
    Generate a unique name of the form ``prefix_N`` that is not registered yet.

    Names are of the same form as EasyScience's own default names. Each prefix counts upwards from
    where it left off, skipping names that are already registered (e.g. ones EasyScience generated
    itself, or ones given explicitly), so a name is never handed out twice within a process.

    Parameters
    ----------
    prefix : str
        The prefix of the name, typically the class name of the object.

    Returns
    -------
    str
        A name that is not in the EasyScience object registry.
    """
    counter = _counters.setdefault(prefix, itertools.count())
    registry = global_object.registry
    while True:
        name = f'{prefix}_{next(counter)}'
        if name not in registry:
            return name
