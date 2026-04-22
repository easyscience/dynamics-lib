# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase


class ParameterAnalysisFitSettings(EasyDynamicsBase):
    """
    Class to manage fit settings for a ParameterAnalysis.
    """

    def __init__(
        self,
        fit_area: bool = True,
        fit_width: bool = True,
        display_name: str = 'ParameterAnalysisFitSettings',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the ParameterAnalysisFitSettings.

        Parameters
        ----------
        fit_area : bool, default=True
            Whether to fit the area of the DiffusionModel. If False, the area is not fitted.
        fit_width : bool, default=True
            Whether to fit the width of the DiffusionModel. If False, the width is not fitted.
        display_name : str, default='ParameterAnalysisFitSettings'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.


        Raises
        ------
        TypeError
            If fit_area or fit_width is not a bool.
        """
        if not isinstance(fit_area, bool):
            raise TypeError('fit_area must be True or False')
        self._fit_area = fit_area

        if not isinstance(fit_width, bool):
            raise TypeError('fit_width must be True or False')
        self._fit_width = fit_width

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fit_area(self) -> bool:
        """
        Get whether to fit the area of the parameter.

        Returns
        -------
        bool
            True if the area is fitted, False otherwise.
        """
        return self._fit_area

    @property
    def fit_width(self) -> bool:
        """
        Get whether to fit the width of the parameter.

        Returns
        -------
        bool
            True if the width is fitted, False otherwise.
        """
        return self._fit_width

    @fit_area.setter
    def fit_area(self, value: bool) -> None:
        """
        Set whether to fit the area of the parameter.

        Parameters
        ----------
        value : bool
            True to fit the area, False otherwise.

        Raises
        ------
        TypeError
            If value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError('fit_area must be True or False')
        self._fit_area = value

    @fit_width.setter
    def fit_width(self, value: bool) -> None:
        """
        Set whether to fit the width of the parameter.

        Parameters
        ----------
        value : bool
            True to fit the width, False otherwise.

        Raises
        ------

        TypeError
            If value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError('fit_width must be True or False')
        self._fit_width = value

    def __repr__(self) -> str:
        """
        Return a string representation of the ParameterAnalysisFitSettings.

        Returns
        -------
        str
            A string representation of the ParameterAnalysisFitSettings.
        """
        return (
            f'ParameterAnalysisFitSettings(fit_area={self.fit_area}, fit_width={self.fit_width})'
        )
