# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
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


def detailed_balance_factor(
    energy: float | list | np.ndarray | sc.Variable | sc.DataArray,
    temperature: float | sc.Variable | Parameter,
    energy_unit: str | sc.Unit = 'meV',
    temperature_unit: str | sc.Unit = 'K',
    divide_by_temperature: bool = True,
) -> np.ndarray:
    r"""
    Compute the detailed balance factor (DBF): $$ DBF(E, T) = E(n(E)+1)=\frac{E}{(1 - e^{-E /
    (k_B*T)})}}, $$ where $n(E)$ is the Bose-Einstein distribution, $E$ is the energy transfer, and
    $T$ is the temperature. $k_B$ is the Boltzmann constant. If divide_by_temperature is True, the
    result is normalized by $k_B*T$ to have value 1 at $E=0$.

    Parameters
    ----------
    energy : float | list | np.ndarray | sc.Variable | sc.DataArray
        The energy transfer. If number, assumed to be in meV unless energy_unit is set. If a
        DataArray, its single coordinate is used as the energy axis.
    temperature : float | sc.Variable | Parameter
        The temperature. Must be a single scalar value. If number, assumed to be in K unless
        temperature_unit is set.
    energy_unit : str | sc.Unit, default='meV'
        Unit for energy if energy is given as a number or list.
    temperature_unit : str | sc.Unit, default='K'
        Unit for temperature if temperature is given as a number.
    divide_by_temperature : bool, default=True
        If True, divide the result by $k_B*T$ to make it dimensionless and have value 1 at E=0. By
        default, True.

    Raises
    ------
    TypeError
        If energy or temperature is not one of the accepted types, or if energy_unit or
        temperature_unit is not a string or scipp Unit, or if divide_by_temperature is not a
        boolean.
    ValueError
        If temperature is negative or is not a single scalar value, if energy is a list or numpy
        array with more than 1 dimension, or if energy is a scipp DataArray without exactly one
        coordinate.
    UnitError
        If the provided energy_unit or temperature_unit is invalid, or if the units of energy or
        temperature cannot be converted to the expected units.
    ZeroDivisionError
        If divide_by_temperature is True and temperature is zero.

    Returns
    -------
    np.ndarray
        Detailed balance factor evaluated at the given energy and temperature.

    Examples
    --------
    **Basic usage**

    ```python
    import easydynamics as edyn

    dbf = edyn.detailed_balance_factor(1.0, 300)  # 1 meV at 300 K
    ```

    **Specifying units and disabling temperature normalisation**

    ```python
    dbf = detailed_balance_factor(
        energy=[1.0, 2.0],
        temperature=300,
        energy_unit='microeV',
        temperature_unit='K',
        divide_by_temperature=False,
    )
    ```
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

    if temperature.sizes != {}:
        raise ValueError(
            f'temperature must be a single scalar value, '
            f'got an array with sizes {dict(temperature.sizes)}.'
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
        ) from e
    # We give users the option to specify the unit of the energy,
    # but if the input has a unit, they might clash
    if energy.unit != energy_unit:
        warnings.warn(
            f'Input energy has unit {energy.unit}, but energy_unit was set to {energy_unit}. '
            f'Using {energy.unit}.',
            stacklevel=2,
        )

    # Same for temperature
    try:
        sc.to_unit(temperature, unit='K')
    except Exception as e:
        raise UnitError(
            f'The unit of temperature is wrong: {temperature.unit}: {e} '
            f'Check that temperature has a valid unit.'
        ) from e

    if temperature.unit != temperature_unit:
        warnings.warn(
            f'Input temperature has unit {temperature.unit}, '
            f'but temperature_unit was set to {temperature_unit}. Using {temperature.unit}.',
            stacklevel=2,
        )

    # Zero temperature deserves special treatment.
    # Here, DBF is 0 for energy<0 and energy for energy>0
    if temperature.value == 0:
        if divide_by_temperature:
            raise ZeroDivisionError('Cannot divide by T when T = 0.')
        DBF = sc.where(energy < 0.0 * energy.unit, 0.0 * energy.unit, energy)

        return np.array([DBF.value]) if DBF.sizes == {} else DBF.values

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

    return np.array([DBF.value]) if DBF.sizes == {} else DBF.values


def _convert_to_scipp_variable(
    value: float | list | np.ndarray | Parameter | sc.Variable | sc.DataArray,
    name: str,
    unit: str | None = None,
) -> sc.Variable:
    """
    Convert various input types to a scipp Variable with proper units.

    Parameters
    ----------
    value : float | list | np.ndarray | Parameter | sc.Variable | sc.DataArray
        The value to convert. Can be a number, list, numpy array, Parameter, scipp Variable, or
        scipp DataArray. If a number or list, the unit must be specified in the unit argument. A
        DataArray must have exactly one coordinate, which is used as the value (consistent with
        how components treat DataArray input to ``evaluate``).
    name : str
        The name of the variable, used for error messages.
    unit : str | None, default=None
        The unit to use if value is a number or list. Must be specified if value is a number or
        list. Ignored if value is a Parameter or sc.Variable, which have their own units. By
        default, None.

    Raises
    ------
    TypeError
        If value is not one of the accepted types, or if unit is not a string when needed.
    ValueError
        If value is a list or numpy array with more than 1 dimension, or a DataArray without
        exactly one coordinate.
    UnitError
        If the provided unit is invalid.

    Returns
    -------
    sc.Variable
        The input value converted to a scipp Variable with appropriate units.
    """
    if isinstance(value, sc.DataArray):
        coords = dict(value.coords)
        if len(coords) != 1:
            coord_names = ', '.join(coords.keys())
            raise ValueError(
                f'scipp.DataArray must have exactly one coordinate to be used as {name}. '
                f'Found {len(coords)} coordinates: {coord_names}.'
            )
        value = next(iter(coords.values()))

    if isinstance(value, sc.Variable):
        return value

    # Convert to numpy array first for consistent handling
    if isinstance(value, (int, float, list)):
        array_value = np.array(value)
    elif isinstance(value, np.ndarray):
        array_value = value
    elif isinstance(value, Parameter):
        array_value = np.array(value.value)
        unit = value.unit
    else:
        if name == 'energy':
            raise TypeError(f'{name} must be a number, list, numpy array or scipp Variable')
        raise TypeError(f'{name} must be a number, list, numpy array, Parameter or scipp Variable')

    if array_value.ndim > 1:
        raise ValueError(
            f'{name} must be at most one-dimensional, got {array_value.ndim} dimensions.'
        )

    # Create appropriate scipp variable based on shape
    if array_value.shape == () or (array_value.shape == (1,)):
        # Scalar or single-element array
        try:
            return sc.scalar(value=float(array_value.flat[0]), unit=unit)
        except UnitError as e:
            raise UnitError(f"Invalid unit string '{unit}' for {name}: {e}") from e
    else:
        # Multi-element array
        try:
            return sc.array(dims=['x'], values=array_value, unit=unit)
        except UnitError as e:
            raise UnitError(f"Invalid unit string '{unit}' for {name}: {e}") from e
