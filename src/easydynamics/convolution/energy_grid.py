from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyGrid:
    """Container for the dense energy grid and related metadata.

    Attributes:
        energy_dense : np.ndarray
            The upsampled and extended energy array.
        energy_dense_centered : np.ndarray
            The centered version of energy_dense (used for resolution evaluation).
        energy_dense_step : float
            The spacing of energy_dense (used for width checks and normalization).
        energy_span_dense : float
            The total span of energy_dense. (used for width checks).
        energy_even_length_offset : float
            The offset to apply if energy_dense has even length (used for convolution alignment).
    """

    energy_dense: np.ndarray
    energy_dense_centered: np.ndarray
    energy_dense_step: float
    energy_span_dense: float
    energy_even_length_offset: float
