import numpy as np
import scipp as sc
from numpy.typing import ArrayLike

Numeric = float | int

Q_type = np.ndarray | Numeric | list | ArrayLike | sc.Variable


def _validate_and_convert_Q(Q: Q_type) -> np.ndarray:
    """
    Validate and convert Q to a numpy array.
    Parameters
    ----------
    Q : Number, list, np.ndarray or sc.Variable
        Scattering vector values in 1/angstrom.
    Returns
    -------
    np.ndarray
        Q as a np.ndarray. TODO: Update to sc.array, also propagate that to diffusionmodel
    """

    if not isinstance(Q, (Numeric, list, np.ndarray, sc.Variable)):
        raise TypeError("Q must be a number, list, numpy array, or scipp array.")

    if isinstance(Q, Numeric):
        Q = np.array([Q])
    if isinstance(Q, list):
        Q = np.array(Q)
    if isinstance(Q, np.ndarray):
        if Q.ndim > 1:
            raise ValueError("Q must be a 1-dimensional array.")

        Q = sc.array(dims=["Q"], values=Q, unit="1/angstrom")

    if isinstance(Q, sc.Variable):
        if Q.dims != ("Q",):
            raise ValueError("Q must have a single dimension named 'Q'.")
        Q = Q.to(unit="1/angstrom")
    return Q.values
