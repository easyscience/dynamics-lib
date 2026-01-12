# from easyscience.variable import DescriptorBase, Parameter
# from .components.model_component import ModelComponent
import numpy as np
import scipp as sc

# from easyscience.job.theoreticalmodel import TheoreticalModelBase
from easyscience.base_classes.model_base import ModelBase
from numpy.typing import ArrayLike

Numeric = float | int

# Base class for sample models - contains a list of ComponentCollection as function of Q and probably not much else
Q_type = np.ndarray | Numeric | list | ArrayLike


class SampleModelBase(ModelBase):
    def __init__(
        self,
        unit: str | sc.Unit = "meV",
        display_name: str = "MySampleModelBase",
        unique_name: str | None = None,
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        if unit is not None and not isinstance(unit, (str, sc.Unit)):
            raise TypeError(
                f"unit must be None, a string, or a scipp Unit, got {type(unit).__name__}"
            )
        self._unit = unit

    def _validate_and_convert_Q(self, Q: Q_type) -> np.ndarray:
        """
        Validate and convert Q to a numpy array.
        Parameters
        ----------
        Q : Number, list, or np.ndarray
            Scattering vector values in 1/angstrom.
        Returns
        -------
        np.ndarray
            Q as a numpy array.
        """
        if isinstance(Q, Numeric):
            Q = np.array([Q])
        if isinstance(Q, list):
            Q = np.array(Q)
        if not isinstance(Q, np.ndarray):
            raise TypeError("Q must be a number, list, or numpy array.")

        if Q.ndim > 1:
            raise ValueError("Q must be a 1-dimensional array.")

        return Q
