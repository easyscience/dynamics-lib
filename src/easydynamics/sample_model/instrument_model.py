# instrument_model will contain resolution_model and background_model as well as offset
import numpy as np
import scipp as sc
from easyscience.base_classes.new_base import NewBase
from easyscience.variable import Parameter
from numpy.typing import ArrayLike

from easydynamics.sample_model.background_model import BackgroundModel
from easydynamics.sample_model.resolution_model import ResolutionModel

Numeric = float | int
Q_type = np.ndarray | Numeric | list | ArrayLike


class InstrumentModel(NewBase):
    """InstrumentModel represents a model of the instrument in an experiment at various Q."""

    def __init__(
        self,
        display_name: str = "MySampleModel",
        unique_name: str | None = None,
        Q: Q_type | None = None,
        resolution_model: ResolutionModel | None = None,
        background_model: BackgroundModel | None = None,
        energy_offset: Numeric | None = None,
        energy_unit: str | sc.Unit = "meV",
    ):
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        # TODO: Think very carefully about units.

        if resolution_model is None:
            self._resolution_model = ResolutionModel()
        else:
            if not isinstance(resolution_model, ResolutionModel):
                raise TypeError(
                    f"resolution_model must be a ResolutionModel or None, got {type(resolution_model).__name__}"
                )
            self._resolution_model = resolution_model

        if background_model is None:
            self._background_model = BackgroundModel()
        else:
            if not isinstance(background_model, BackgroundModel):
                raise TypeError(
                    f"background_model must be a BackgroundModel or None, got {type(background_model).__name__}"
                )
            self._background_model = background_model

        if energy_offset is None:
            self._offset = Parameter(
                name="offset", value=0.0, unit=energy_unit, fixed=True
            )
        else:
            if not isinstance(energy_offset, Numeric):
                raise TypeError("offset must be a number or None")
            self._offset = Parameter(
                name="offset", value=float(energy_offset), unit=energy_unit, fixed=False
            )

    # --------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------

    @property
    def resolution_model(self) -> ResolutionModel:
        """The resolution model of the instrument."""
        return self._resolution_model

    @resolution_model.setter
    def resolution_model(self, value: ResolutionModel):
        if not isinstance(value, ResolutionModel):
            raise TypeError(
                f"resolution_model must be a ResolutionModel, got {type(value).__name__}"
            )
        self._resolution_model = value

    @property
    def background_model(self) -> BackgroundModel:
        """The background model of the instrument."""
        return self._background_model

    @background_model.setter
    def background_model(self, value: BackgroundModel):
        if not isinstance(value, BackgroundModel):
            raise TypeError(
                f"background_model must be a BackgroundModel, got {type(value).__name__}"
            )
        self._background_model = value

    # TODO offset needs to be a list to support multiple Q values
    @property
    def offset(self) -> Parameter:
        """The offset parameter of the instrument model."""
        return self._offset

    @offset.setter
    def offset(self, value: Numeric):
        "set the offset parameter of the instrument model."
        if not isinstance(value, Numeric):
            raise TypeError(f"offset must be a number, got {type(value).__name__}")
        self._offset.value = value

    # --------------------------------------------------------------------
    # Dunder methods
    # --------------------------------------------------------------------

    def __repr__(self):
        return f"{self.__class__.__name__}(unique_name={self.unique_name}, unit={self.unit}), resolution_model = {self.resolution_model}, background_model = {self.background_model}, offset = {self.offset}"
