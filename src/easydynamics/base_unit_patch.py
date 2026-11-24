# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
Temporary patch: keep units with a numeric factor, such as ``meV*angstrom``.

Scipp writes a unit it has no name for with a numeric factor, e.g. ``meV*angstrom`` as
``1.602176634e-32 J*m``. Whenever EasyScience creates a ``DescriptorNumber`` or ``Parameter``,
including the result of every arithmetic operation, it strips such a factor, so the unit becomes
``J*m`` and the value is rescaled with it. ``Parameter(value=0.1, unit='meV*angstrom')`` then has
the value ``1.6e-33``, a value set later is read in ``J*m``, and ``2 meV * 3 angstrom`` gives
``9.6e-32 J*m``.

The stripping is meant for a unit that is only a number, such as ``1000`` from ``m/mm``, which
should become dimensionless. This patch keeps that, and leaves any other unit as it is.

Importing ``easydynamics`` applies the patch. It changes the class in EasyScience, so it applies to
every EasyScience ``DescriptorNumber`` and ``Parameter``, not only to those of EasyDynamics.

Remove this module once EasyScience no longer strips the numeric factor of a unit.
"""

from easyscience.variable import DescriptorNumber as EasyScienceDescriptorNumber


def _base_unit(self: EasyScienceDescriptorNumber) -> str:
    """
    Get the unit EasyScience converts a new DescriptorNumber or Parameter to.

    Parameters
    ----------
    self : EasyScienceDescriptorNumber
        The DescriptorNumber or Parameter.

    Returns
    -------
    str
        ``'dimensionless'`` if the unit is only a number, otherwise the unit itself.
    """
    unit = str(self._scalar.unit)
    try:
        float(unit)
    except ValueError:
        return unit
    return 'dimensionless'


EasyScienceDescriptorNumber._base_unit = _base_unit  # ruff: ignore[private-member-access]
