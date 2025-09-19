from easyscience import Parameter

import numpy as np

import scipp as sc

from typing import Union

Numeric = Union[float, int]


def detailed_balance_factor(energy: Union[Numeric, list, np.ndarray, sc.Variable], 
                            temperature: Union[Numeric, sc.Variable, Parameter], 
                            energy_unit: str = 'meV', 
                            temperature_unit: str = 'K', divide_by_T=False) -> Union[Numeric,np.ndarray]:
    """
    Compute the detailed balance factor:
    DBF(energy, T) = energy / (1 - exp(-energy / (kB*T)))
    If divide_by_T is True, the result is normalized by kB*T to have value 1 at energy=0.

    Args:
        energy : 
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
            temperature_unit=temperature.unit
            temperature=temperature.value
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

        if isinstance(energy, Numeric):
            DBF = max(energy, 0.0)
        elif isinstance(energy, (list, np.ndarray)):  # Check if omega is iterable
            DBF = np.array([max(o, 0.0) for o in energy])
        elif isinstance(energy, sc.Variable):
            DBF = sc.where(energy < 0, 0.0, energy)
            DBF = DBF.values

        return DBF


    # Now handle non-zero temperature
    kB=sc.scalar(value=8.617333262145e-2, unit='meV/K')  # Boltzmann constant in meV/K

    # Convert omega to scipp variable if needed for unit handling
    if not isinstance(energy, sc.Variable):
        if isinstance(energy, Numeric): 
            energy = sc.scalar(value=float(energy), unit=energy_unit)
        elif isinstance(energy, (list, tuple, np.ndarray)):  # Check if omega is iterable
            if len(energy) == 1:
                energy = sc.scalar(value=energy[0], unit=energy_unit)
            else:
                energy = sc.array(dims=['x'], values=energy, unit=energy_unit)


    x = energy / (kB * temperature)

    # Very large and very small x need special handling to avoid numerical issues    

    DBF = sc.zeros_like(energy)

    # Small x: Taylor expansion
    small = sc.abs(x) < 0.01

    DBF = sc.where(small, kB * temperature + energy / 2 + energy**2 / (12 * kB * temperature), DBF)

    # Large x: asymptotic form
    large = x > 50
    DBF = sc.where(large, energy, DBF)

    # General case: exact formula
    mid = ~small & ~large
    DBF = sc.where(mid, energy / (1 - sc.exp(-x)), DBF)

    if divide_by_T:
    # Normalize by kB*T to get dimensionless - also makes the value 1 at omega=0
        DBF=DBF/(kB*temperature)

    DBF=DBF.values  # Return as numpy array 
    return DBF
