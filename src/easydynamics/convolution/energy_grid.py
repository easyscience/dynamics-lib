from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyGrid:
    """Container for the dense energy grid and related metadata.

    Attributes:
        energy_dense: the (possibly extended & upsampled) energy grid (1D).
        span_original: span of the original energy array (max-min).
        span_dense: span of the dense grid (max-min).
        energy_even_length_offset: -0.5*dE if length is even, else 0.0 — used to correct half-bin shift.
        energy_dense_centered: energy_dense recentered around zero (same length as energy_dense).
        energy_step: grid spacing (dE) of energy_dense (positive float).
    """

    energy_dense: np.ndarray
    span_original: float
    span_dense: float
    energy_even_length_offset: float
    energy_dense_centered: np.ndarray
    energy_step: float
