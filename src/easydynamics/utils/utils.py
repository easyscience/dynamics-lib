import numpy as np


def detailed_balance_factor(omega, temperature):
    """
    Compute the detailed balance factor:
    DBF(omega, T) = omega / (1 - exp(-omega / (kB*T)))
    
    Args:
        omega : float or np.ndarray
            Energy transfer (in meV)
        T : float
            Temperature in Kelvin

    Returns:
        DBF : float or np.ndarray
            Detailed balance factor
    """
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")
    kB=8.617333262145e-2 # Boltzmann constant in meV/K

    if temperature==0:
        # At T=0, only positive omega contributes
        return np.maximum(omega, 0.0)

    x = omega / (kB * temperature)

    # Use masks for different regimes
    DBF = np.empty_like(x)

    # Small omega: Taylor expansion
    small = np.abs(x) < 0.01
    DBF[small] = kB*temperature + omega[small]/2 + omega[small]**2 / (12 * kB * temperature)

    # Large omega: asymptotic form
    large = x > 50
    DBF[large] = omega[large]

    # General case: exact formula
    mid = ~small & ~large
    DBF[mid] = omega[mid] / (1 - np.exp(-x[mid]))

    return DBF
