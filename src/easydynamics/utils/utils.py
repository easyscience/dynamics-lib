from easyscience import Parameter

import numpy as np

import scipp as sc

from typing import Union

Numeric = Union[float, int]


def detailed_balance_factor(omega: Union[Numeric, list, np.ndarray, sc.Variable], 
                            temperature: Union[Numeric, sc.Variable, Parameter], 
                            omega_unit: str = 'meV', 
                            temperature_unit: str = 'K', divide_by_T=True) -> Union[Numeric,np.ndarray]:
    """
    Compute the detailed balance factor:
    DBF(omega, T) = omega / (1 - exp(-omega / (kB*T)))
    If divide_by_T is True, the result is normalized by kB*T to have value 1 at omega=0.
    
    Args:
        omega : 
            Energy transfer
        T :
            Temperature

    Returns:
        DBF : sc array or scalar, depending on omega
            Detailed balance factor
    """

    # Make a scipp variable of temperature for unit handling
    if not isinstance(temperature,sc.Variable):
        if isinstance(temperature,Parameter):
            temperature=temperature.value
            temperature_unit=temperature.unit
        temperature=sc.scalar(value=float(temperature), unit=temperature_unit)


    # Handle special cases first
    if temperature.value < 0:
        raise ValueError("Temperature must be non-negative.")

    if temperature.value==0:
        # At T=0, only positive omega contributes

        #TODO: decide if I want to return scipp variable instead - for now I want numpy array.

        # if isinstance(omega, Numeric):
        #     DBF = sc.scalar(value=max(omega, 0.0),unit=omega_unit)
        # elif isinstance(omega, (list, np.ndarray)):
        #     DBF = sc.array(dims=['x'], values=[max(o, 0.0) for o in omega], unit=omega_unit)
        # elif isinstance(omega, sc.Variable):
        #     DBF = sc.where(omega < 0, 0.0, omega)

        if isinstance(omega, Numeric):
            DBF = max(omega, 0.0)
        elif isinstance(omega, (list, np.ndarray)):  # Check if omega is iterable
            DBF = np.array([max(o, 0.0) for o in omega])
        elif isinstance(omega, sc.Variable):
            DBF = sc.where(omega < 0, 0.0, omega)
            DBF = DBF.values

        return DBF


    # Now handle non-zero temperature
    kB=sc.scalar(value=8.617333262145e-2, unit='meV/K')  # Boltzmann constant in meV/K

    # Convert omega to scipp variable if needed for unit handling
    if not isinstance(omega, sc.Variable):
        if isinstance(omega, Numeric): 
            omega = sc.scalar(value=float(omega), unit=omega_unit)
        elif isinstance(omega, (list, tuple, np.ndarray)):  # Check if omega is iterable
            if len(omega) == 1:
                omega = sc.scalar(value=omega[0], unit=omega_unit)
            else:
                omega = sc.array(dims=['x'], values=omega, unit=omega_unit)


    x = omega / (kB * temperature)

    # Very large and very small x need special handling to avoid numerical issues    

    DBF = sc.zeros_like(omega)

    # Small x: Taylor expansion
    small = sc.abs(x) < 0.01

    DBF = sc.where(small, kB * temperature + omega / 2 + omega**2 / (12 * kB * temperature), DBF)

    # Large x: asymptotic form
    large = x > 50
    DBF = sc.where(large, omega, DBF)

    # General case: exact formula
    mid = ~small & ~large
    DBF = sc.where(mid, omega / (1 - sc.exp(-x)), DBF)

    if divide_by_T:
    # Normalize by kB*T to get dimensionless - also makes the value 1 at omega=0
        DBF=DBF/(kB*temperature)

    DBF=DBF.values  # Return as numpy array 
    return DBF
