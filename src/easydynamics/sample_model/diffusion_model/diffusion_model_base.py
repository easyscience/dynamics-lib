from typing import Union

import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase
from numpy.typing import ArrayLike

Numeric = Union[float, int]

Q_type = np.ndarray | Numeric | list | ArrayLike


class DiffusionModelBase(ModelBase):
    """
    Base class for constructing diffusion models.
    """

    def __init__(
        self,
        display_name="MyDiffusionModel",
        unique_name: str | None = None,
        unit: str | sc.Unit = "meV",
    ):
        """
        Initialize a new DiffusionModel.

        Parameters
        ----------
        display_name : str
            Display name of the diffusion model.
        unit : str or sc.Unit, optional
            Unit of the diffusion model. Defaults to "meV".
        """

        if not (unit is None or isinstance(unit, (str, sc.Unit))):
            raise TypeError("unit must be None, a string, or a scipp Unit")

        super().__init__(display_name=display_name, unique_name=unique_name)
        self._unit = unit

    @property
    def unit(self) -> str:
        """
        Get the unit of the DiffusionModel.

        Returns
        -------
        str or sc.Unit or None
        """
        return str(self._unit)

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )  # noqa: E501

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
