from typing import Optional, Union

import scipp as sc

# from .components import ModelComponent
from easyscience.base_classes import ObjBase


class DiffusionModel(ObjBase):
    """
    Base class for constructing diffusion models.
    """

    def __init__(
        self,
        name="MyDiffusionModel",
        unit: Optional[Union[str, sc.Unit]] = "meV",
        **kwargs,
    ):
        """
        Initialize a new DiffusionModel.

        Parameters
        ----------
        name : str
            Name of the diffusion model.
        unit : str or sc.Unit, optional
            Unit of the diffusion model. Defaults to "meV".
        """

        super().__init__(name=name, unit=unit, **kwargs)

    @property
    def unit(self) -> Optional[Union[str, sc.Unit]]:
        """
        Get the unit of the DiffusionModel.

        Returns
        -------
        str or sc.Unit or None
        """
        return self._unit

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f"Unit is read-only. Use convert_unit to change the unit between allowed types "
                f"or create a new {self.__class__.__name__} with the desired unit."
            )
        )  # noqa: E501
