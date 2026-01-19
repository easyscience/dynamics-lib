from typing import Union

import numpy as np
import scipp as sc
from easyscience.base_classes.model_base import ModelBase
from easyscience.variable import DescriptorNumber, Parameter
from numpy.typing import ArrayLike
from scipp import UnitError

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
        scale: Numeric = 1.0,
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
        if not isinstance(scale, Numeric):
            raise TypeError("scale must be a number.")

        scale = Parameter(name="scale", value=float(scale), fixed=False, min=0.0)

        try:
            test = DescriptorNumber(name="test", value=1, unit=unit)
            test.convert_unit("meV")
        except Exception as e:
            raise UnitError(
                f"Invalid unit: {unit}. Unit must be a string or scipp Unit and convertible to meV."
            ) from e

        super().__init__(display_name=display_name, unique_name=unique_name)
        self._unit = unit
        self._scale = scale

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

    @property
    def scale(self) -> Parameter:
        """
        Get the scale parameter of the diffusion model.

        Returns
        -------
        Parameter
            Scale parameter.
        """
        return self._scale

    @scale.setter
    def scale(self, scale: Numeric) -> None:
        """
        Set the scale parameter of the diffusion model.
        """
        if not isinstance(scale, Numeric):
            raise TypeError("scale must be a number.")
        self._scale.value = scale

    def __repr__(self):
        """
        String representation of the Diffusion model.
        """
        return f"{self.__class__.__name__}(display_name={self.display_name}, unit={self.unit})"
