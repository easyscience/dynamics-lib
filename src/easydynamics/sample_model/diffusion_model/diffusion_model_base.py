from typing import Optional, Union

import scipp as sc
from easyscience.base_classes.model_base import ModelBase


class DiffusionModel(ModelBase):
    """
    Base class for constructing diffusion models.
    """

    def __init__(
        self,
        display_name="MyDiffusionModel",
        unit: Optional[Union[str, sc.Unit]] = "meV",
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

        super().__init__(display_name=display_name)
        self._unit = unit

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
