import numpy as np

import scipp as sc

from typing import Union
from easyscience import Parameter

import warnings

def detailed_balance_factor(energy: Union[int,float, list, np.ndarray, sc.Variable], 
                            temperature: Union[int,float, sc.Variable, Parameter], 
                            energy_unit: str = 'meV', 
                            temperature_unit: str = 'K',divide_by_T=True) -> np.ndarray:
    """
    Compute the detailed balance factor:
    DBF(energy, T) = energy*(n(b)+1)=energy / (1 - exp(-energy / (kB*T)))
    If divide_by_T is True, the result is normalized by kB*T to have value 1 at energy=0.
    
    Args:
        energy : number, list, np.ndarray, or scipp Variable. If number, assumed to be in meV unless energy_unit is set.
            Energy transfer
        T : number, scipp Variable, or Parameter. If number, assumed to be in K unless temperature_unit is set.
            Temperature
        energy_unit : str, optional
            Unit for energy if energy is given as a number or list. Default is 'meV'
        temperature_unit : str, optional
            Unit for temperature if temperature is given as a number. Default is 'K'

    Returns:
        DBF : np.ndarray (may be changed to scipp Variable in the future)
            Detailed balance factor
    """
    # First convert temperature to sc variable to make units easy to handle
    if not isinstance(temperature,sc.Variable):
        if isinstance(temperature,Parameter):
            temperature_unit=temperature.unit
            temperature=temperature.value
        temperature=sc.scalar(value=float(temperature), unit=temperature_unit)

    if temperature.value < 0:
        raise ValueError("Temperature must be non-negative.")
    
    # Convert energy to sc variable to make units easy to handle
    if not isinstance(energy, sc.Variable):
        if isinstance(energy, (int, float)): 
            energy = sc.scalar(value=float(energy), unit=energy_unit)
        elif isinstance(energy, (list, np.ndarray)):  # Check if energy is iterable
            if len(energy) == 1:
                energy = sc.scalar(value=energy[0], unit=energy_unit)
            else:
                energy = sc.array(dims=['x'], values=energy, unit=energy_unit)

    # We give users the option to specify the energy, but if the input has a unit, they might clash
    if energy.unit != energy_unit:
        warnings.warn(f"Input energy has unit {energy.unit}, but energy_unit was set to {energy_unit}. Using {energy.unit}.")
        energy_unit=energy.unit

    # Zero temperature deserves special treatment
    if temperature.value==0:
        if divide_by_T:
            raise ValueError("Cannot divide by T when T=0.")
        DBF = sc.where(energy < 0.0*sc.Unit(energy_unit), 0.0*sc.Unit(energy_unit), energy)
            
        if DBF.sizes == {}:  
            DBF_values = DBF.value
        else:
            DBF_values = DBF.values
        return DBF_values

    kB=sc.scalar(value=8.617333262145e-2, unit='meV/K')  # Boltzmann constant in meV/K

    x = energy / (kB * temperature)
    x = sc.to_unit(x, unit='1')  # Make sure the unit is 1 and not e.g. 1e3

    # Small and large values of x need special treatment
    SMALL_THRESHOLD=0.01
    LARGE_THRESHOLD=50
    DBF = sc.zeros_like(energy)

    # Small energy: Taylor expansion
    small = sc.abs(x) < SMALL_THRESHOLD

    first_order_term_a= kB * temperature
    first_order_term_b= energy / 2
    second_order_term= energy**2 / (12 * kB * temperature)

    DBF = sc.where(small, sc.to_unit(first_order_term_a, energy_unit) + sc.to_unit(first_order_term_b, energy_unit) + sc.to_unit(second_order_term, energy_unit), DBF) # can't add terms with different units

    # Large energy: asymptotic form
    large = x > LARGE_THRESHOLD
    DBF = sc.where(large, energy, DBF)

    # General case: exact formula
    mid = ~small & ~large
    DBF = sc.where(mid, energy / (1 - sc.exp(-x)), DBF)

    DBF=sc.to_unit(DBF, unit=energy_unit)

    if divide_by_T:
    # Normalize by kB*T to get dimensionless - also makes the value 1 at energy=0
        DBF=DBF/(kB*temperature)
        DBF=sc.to_unit(DBF, unit='1')

    if DBF.sizes == {}:  
        DBF_values = DBF.value
    else:
        DBF_values = DBF.values
    return DBF_values
