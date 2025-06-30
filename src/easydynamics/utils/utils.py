import numpy as np

def detailed_balance_factor(omega_meV, temperature_K):
    """
    Compute ω * (n + 1), where n is the Bose-Einstein occupation number.

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
        return np.maximum(omega_meV, 0.0)

    k_B_meV_per_K = 8.617333262e-2  # Boltzmann constant in meV/K
    beta = 1.0 / (k_B_meV_per_K * temperature_K)
    x = beta * omega_meV

    result = np.empty_like(omega_meV)

    # Handle large x = beta * omega
    large_mask = x > 50
    small_mask = ~large_mask

    result[large_mask] = omega_meV[large_mask]

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        exp_x = np.exp(x[small_mask])
        denom = np.expm1(x[small_mask])
        result[small_mask] = omega_meV[small_mask] * exp_x / denom

        # Handle x ~ 0 → ω * (1 + 1/2 x + ...)
        near_zero = denom == 0
        result[small_mask][near_zero] = k_B_meV_per_K * temperature_K

    return result
