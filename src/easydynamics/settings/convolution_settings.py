# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.utils.utils import Numeric


class ConvolutionSettings(EasyDynamicsBase):
    """
    Settings for numerical convolutions.
    """

    def __init__(
        self,
        upsample_factor: Numeric | None = 5,
        extension_factor: Numeric | None = 0.2,
        display_name: str | None = 'MyConvolutionSettings',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the ConvolutionSettings.

        Parameters
        ----------
        upsample_factor : Numeric | None, default=5
            The factor by which to upsample the input data before convolution.
        extension_factor : Numeric | None, default=0.2
            The factor by which to extend the input data range before convolution.
        display_name : str | None, default='MyConvolutionSettings'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If upsample_factor is not a number or None. If extension_factor is not a number or
            None.
        ValueError
            If upsample_factor is not greater than 1. If extension_factor is negative.
        """
        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

        if extension_factor is not None:
            if not isinstance(extension_factor, Numeric):
                raise TypeError('Extension factor must be a number.')
            extension_factor = float(extension_factor)
            if extension_factor < 0.0:
                raise ValueError('Extension factor must be non-negative.')
        self._extension_factor = extension_factor

        if upsample_factor is not None:
            if not isinstance(upsample_factor, Numeric):
                raise TypeError('Upsample factor must be a numerical value or None.')
            upsample_factor = float(upsample_factor)
            if upsample_factor <= 1.0:
                raise ValueError('Upsample factor must be greater than 1.')
        self._upsample_factor = upsample_factor

        self._convolution_plan_is_valid = False

    @property
    def upsample_factor(self) -> Numeric | None:
        """
        Get the upsample factor.

        Returns
        -------
        Numeric | None
            The upsample factor.
        """

        return self._upsample_factor

    @upsample_factor.setter
    def upsample_factor(self, factor: Numeric | None) -> None:
        """
        Set the upsample factor and recreate the dense grid.

        Parameters
        ----------
        factor : Numeric | None
            The new upsample factor.

        Raises
        ------
        TypeError
            If factor is not a number or None.
        ValueError
            If factor is not greater than 1.
        """
        if factor is None:
            self._upsample_factor = factor
            self.convolution_plan_is_valid = False
            return

        if not isinstance(factor, Numeric):
            raise TypeError('Upsample factor must be a numerical value or None.')
        factor = float(factor)
        if factor <= 1.0:
            raise ValueError('Upsample factor must be greater than 1.')

        self._upsample_factor = factor

        self.convolution_plan_is_valid = False

    @property
    def extension_factor(self) -> float:
        """
        Get the extension factor.

        The extension factor determines how much the energy range is extended on both sides before
        convolution. 0.2 means extending by 20% of the original energy span on each side

        Returns
        -------
        float
            The extension factor.
        """

        return self._extension_factor

    @extension_factor.setter
    def extension_factor(self, factor: Numeric) -> None:
        """
        Set the extension factor and recreate the dense grid.

        The extension factor determines how much the energy range is extended on both sides before
        convolution. 0.2 means extending by 20% of the original energy span on each side.

        Parameters
        ----------
        factor : Numeric
            The new extension factor.

        Raises
        ------
        TypeError
            If factor is not a number.
        ValueError
            If factor is negative.
        """

        if not isinstance(factor, Numeric):
            raise TypeError('Extension factor must be a number.')
        if factor < 0.0:
            raise ValueError('Extension factor must be non-negative.')

        self._extension_factor = float(factor)
        self.convolution_plan_is_valid = False

    @property
    def convolution_plan_is_valid(self) -> bool:
        """
        Get whether the convolution plan is valid.

        Returns
        -------
        bool
            Whether the convolution plan is valid.
        """
        return self._convolution_plan_is_valid

    @convolution_plan_is_valid.setter
    def convolution_plan_is_valid(self, is_valid: bool) -> None:
        """
        Set whether the convolution plan is valid.

        Parameters
        ----------
        is_valid : bool
            Whether the convolution plan is valid.

        Raises
        ------
        TypeError
            If is_valid is not a bool.
        """
        if not isinstance(is_valid, bool):
            raise TypeError('convolution_plan_is_valid must be True or False.')
        self._convolution_plan_is_valid = is_valid

    def __repr__(self) -> str:
        """
        Return a string representation of the ConvolutionSettings.

        Returns
        -------
        str
            A string representation of the ConvolutionSettings.
        """
        return (
            f'{self.__class__.__name__}('
            f'upsample_factor={self.upsample_factor}, '
            f'extension_factor={self.extension_factor})'
        )
