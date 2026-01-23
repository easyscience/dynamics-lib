# SPDX-FileCopyrightText: 2025-2026 EasyDynamics contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import scipp as sc
from easyscience import Parameter
from scipp import UnitError
from scipp.constants import Boltzmann as kB

# Small and large values of x need special treatment.
# For small values of x, the denominator is close to zero,
# which can give numerical issues.
# The issues don't start until x<~1e-6, but we use a larger threshold
# to be safe.
SMALL_THRESHOLD = 0.001
# For large values of x, the exponential term becomes negligible.
# This happens around x>~10, but we use a larger threshold to be safe.
# At very large x, exp(-x) can be rounded to 0, which can give
# numerical issues.
LARGE_THRESHOLD = 100


def _detailed_balance_factor(
    energy: int | float | list | np.ndarray | sc.Variable,
    temperature: int | float | sc.Variable | Parameter,
    energy_unit: str | sc.Unit = 'meV',
    temperature_unit: str | sc.Unit = 'K',
    divide_by_temperature: bool = True,
) -> np.ndarray:
    """
    Compute the detailed balance factor (DBF):
    DBF(E, T) = E*(n(E)+1)=E / (1 - exp(-E / (kB*T))),
    where n(E) is the Bose-Einstein distribution.
    If divide_by_temperature is True,
    the result is normalized by kB*T to have value 1 at E=0.

    Args:
        energy : number, list, np.ndarray, or scipp Variable.
        If number, assumed to be in meV unless energy_unit is set.
            Energy transfer
        T : number, scipp Variable, or Parameter.
        If number, assumed to be in K unless temperature_unit is set.
            Temperature
        energy_unit : str, optional
            Unit for energy if energy is given as a number or list.
            Default is 'meV'
        temperature_unit : str, optional
            Unit for temperature if temperature is given as a number.
            Default is 'K'
        divide_by_temperature : True or False, optional
            If True, divide the result by kB*T to make it dimensionless
            and have value 1 at E=0. Default is True.

    Returns:
        DBF : np.ndarray TODO: change to sc.Variable?
            Detailed balance factor

    Examples
    --------
    >>> detailed_balance_factor(1.0, 300)  # 1 meV at 300 K
    >>> detailed_balance_factor(
    ...     energy=[1.0, 2.0],
    ...     temperature=300,
    ...     energy_unit='microeV',
    ...     temperature_unit='K',
    ...     divide_by_temperature=False,
    ... )
    """

    # Input validation
    if not isinstance(divide_by_temperature, bool):
        raise TypeError('divide_by_temperature must be True or False.')

    if not isinstance(energy_unit, (str, sc.Unit)):
        raise TypeError('energy_unit must be a string or scipp.Unit.')

    if not isinstance(temperature_unit, (str, sc.Unit)):
        raise TypeError('temperature_unit must be a string or scipp.Unit.')

    # Convert temperature and energy to sc variables
    # to make units easy to handle
    temperature = _convert_to_scipp_variable(
        value=temperature, unit=temperature_unit, name='temperature'
    )

    if temperature.value < 0:
        raise ValueError('Temperature must be non-negative.')

    energy = _convert_to_scipp_variable(value=energy, unit=energy_unit, name='energy')

    # What if people give units that don't make sense?
    try:
        sc.to_unit(energy, unit='meV')
    except Exception as e:
        raise UnitError(
            f'The unit of energy is wrong: {energy.unit}: {e} Check that energy has a valid unit.'
        )
    # We give users the option to specify the unit of the energy,
    # but if the input has a unit, they might clash
    if energy.unit != energy_unit:
        warnings.warn(
            f'Input energy has unit {energy.unit}, but energy_unit was set to {energy_unit}. '
            f'Using {energy.unit}.'
        )

    # Same for temperature
    try:
        sc.to_unit(temperature, unit='K')
    except Exception as e:
        raise UnitError(
            f'The unit of temperature is wrong: {temperature.unit}: {e} '
            f'Check that temperature has a valid unit.'
        )

    if temperature.unit != temperature_unit:
        warnings.warn(
            f'Input temperature has unit {temperature.unit}, '
            f'but temperature_unit was set to {temperature_unit}. Using {temperature.unit}.'
        )

    # Zero temperature deserves special treatment.
    # Here, DBF is 0 for energy<0 and energy for energy>0
    if temperature.value == 0:
        if divide_by_temperature:
            raise ZeroDivisionError('Cannot divide by T when T = 0.')
        DBF = sc.where(energy < 0.0 * energy.unit, 0.0 * energy.unit, energy)

        if DBF.sizes == {}:
            DBF_values = np.array([DBF.value])
        else:
            DBF_values = DBF.values
        return DBF_values

    # Now work with finite temperatures.
    # Here, it helps to work with dimensionless x = energy/(kB*T),
    # where we have divided by kB*T
    # We first check if the units are OK.

    x = energy / (kB * temperature)

    x = sc.to_unit(x, unit='1')  # Make sure the unit is 1 and not e.g. 1e3

    # Now compute DBF. First handle small and large x, then general.

    # Small x (small energy and/or high temperature): Taylor expansion.
    # Works and is needed for both positive and negative energies
    small = sc.abs(x) < SMALL_THRESHOLD

    DBF = sc.where(small, 1 + x / 2 + x**2 / 12, sc.zeros_like(x))

    # Large x (large positive energy and/or low temperature):
    # asymptotic form. Only needed for positive energies.
    large = x > LARGE_THRESHOLD
    DBF = sc.where(large, x, DBF)

    # General case: exact formula
    mid = sc.logical_not(small) & sc.logical_not(large)
    DBF = sc.where(mid, x / (1 - sc.exp(-x)), DBF)  # zeros in x are handled by SMALL_THRESHOLD

    #
    if not divide_by_temperature:
        DBF = DBF * (kB * temperature)
        DBF = sc.to_unit(DBF, unit=energy.unit)

    if DBF.sizes == {}:
        DBF_values = np.array([DBF.value])
    else:
        DBF_values = DBF.values
    return DBF_values


def _convert_to_scipp_variable(
    value: int | float | list | np.ndarray | Parameter | sc.Variable,
    name: str,
    unit: str | None = None,
) -> sc.Variable:
    """Convert various input types to a scipp Variable with proper
    units.
    """
    if isinstance(value, sc.Variable):
        return value

    # Convert to numpy array first for consistent handling
    if isinstance(value, (int, float)):
        array_value = np.array(value)
    elif isinstance(value, (list)):
        array_value = np.array(value)
    elif isinstance(value, np.ndarray):
        array_value = value
    elif isinstance(value, Parameter):
        array_value = np.array(value.value)
        unit = value.unit
    else:
        if name == 'energy':
            raise TypeError(f'{name} must be a number, list, numpy array or scipp Variable')
        else:
            raise TypeError(
                f'{name} must be a number, list, numpy array, Parameter or scipp Variable'
            )

    # Create appropriate scipp variable based on shape
    if array_value.shape == () or (array_value.shape == (1,)):
        # Scalar or single-element array
        try:
            return sc.scalar(value=float(array_value.flat[0]), unit=unit)
        except UnitError as e:
            raise UnitError(f"Invalid unit string '{unit}' for {name}: {e}")
    else:
        # Multi-element array
        try:
            return sc.array(dims=['x'], values=array_value, unit=unit)
        except UnitError as e:
            raise UnitError(f"Invalid unit string '{unit}' for {name}: {e}")
