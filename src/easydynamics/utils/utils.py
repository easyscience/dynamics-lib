
import numpy as np


def detailed_balance_factor(omega_meV, temperature_K):
    """
    Compute ω * (n + 1), where n is the Bose-Einstein occupation number.
    
    This expression arises in detailed balance factors in neutron and light scattering.

    Parameters
    ----------
    omega_meV : float or np.ndarray
        Energy transfer in meV.
    temperature_K : float
        Temperature in Kelvin. Must be >= 0.
    
    Returns
    -------
    result : float or np.ndarray
        The value of ω * (n + 1), safely evaluated even for T=0.
    """
    if temperature_K < 0:
        raise ValueError("Temperature must be non-negative.")

    omega_meV = np.asarray(omega_meV, dtype=np.float64)

    if temperature_K == 0:
        return  np.maximum(omega_meV, 0.0)

    k_B_meV_per_K = 8.617333262e-2  # Boltzmann constant in meV/K

    beta = 1.0 / (k_B_meV_per_K * temperature_K)
    x = beta * omega_meV

    result = np.empty_like(omega_meV)

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        exp_x = np.exp(x)
        denom = np.expm1(x)  # More stable than exp(x) - 1
        safe = denom != 0
        result[safe] = omega_meV[safe] * exp_x[safe] / denom[safe]
        result[~safe] = k_B_meV_per_K * temperature_K  # Limit as ω → 0

    return result