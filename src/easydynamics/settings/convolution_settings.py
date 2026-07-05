# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase
from easydynamics.utils.utils import Numeric


class ConvolutionSettings(EasyDynamicsBase):
    """
    Settings for numerical convolutions.

    These settings are passed to an ``Analysis`` to control how the numerical convolution is
    performed. Increasing ``upsample_factor`` and ``extension_factor`` improves accuracy at the
    cost of computation time.

    Examples
    --------
    **Creating custom convolution settings and passing to an Analysis**

    ```python
    import easydynamics as edyn

    settings = edyn.ConvolutionSettings(upsample_factor=10, extension_factor=0.5)
    analysis = edyn.Analysis(convolution_settings=settings)
    ```

    **Suppressing warnings about peak widths**

    ```python
    settings = edyn.ConvolutionSettings(suppress_warnings=True)
    ```
    """

    def __init__(
        self,
        upsample_factor: Numeric | None = 5,
        extension_factor: Numeric | None = 0.2,
        suppress_warnings: bool = False,
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
        suppress_warnings : bool, default=False
            Whether to suppress warnings about wide or narrow peaks in the models.
        display_name : str | None, default='MyConvolutionSettings'
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If upsample_factor is not a number or None. If extension_factor is not a number or
            None. If suppress_warnings is not a boolean.
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

        if not isinstance(suppress_warnings, bool):
            raise TypeError('suppress_warnings must be True or False.')
        self._suppress_warnings = suppress_warnings

        # Plan-invalidation bookkeeping for convolvers sharing this settings object.
        # _plan_version is bumped whenever the plan is invalidated (accuracy knob change or
        # an explicit convolution_plan_is_valid = False); each convolver records the version
        # it last rebuilt against. _blessed_plan_version records the version at the last
        # explicit convolution_plan_is_valid = True, which suppresses rebuilds for all
        # convolvers until the next invalidation.
        self._plan_version = 0
        self._blessed_plan_version = -1

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
        Get whether the current plan version was explicitly marked valid.

        This reflects explicit assignments only: it is True after ``convolution_plan_is_valid =
        True`` until the next invalidation (a knob change or an explicit False). Whether an
        individual convolver needs a rebuild is tracked by the convolver itself against the
        settings' plan version.

        Returns
        -------
        bool
            Whether the current plan version was explicitly marked valid.
        """
        return self._blessed_plan_version == self._plan_version

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
        if is_valid:
            # An explicit True blesses the current version: all convolvers sharing these
            # settings may skip rebuilding until the next invalidation.
            self._blessed_plan_version = self._plan_version
        else:
            # An explicit False (or a knob setter delegating here) invalidates the plan for
            # every convolver sharing these settings.
            self._plan_version += 1

    def _plan_valid_for(self, seen_version: int) -> bool:
        """
        Check whether a convolver that last rebuilt at seen_version can skip rebuilding.

        Parameters
        ----------
        seen_version : int
            The plan version the convolver recorded when it last rebuilt its plan.

        Returns
        -------
        bool
            True if no invalidation happened since the convolver's rebuild, or the current version
            was explicitly blessed.
        """
        return (
            seen_version == self._plan_version or self._blessed_plan_version == self._plan_version
        )

    @property
    def suppress_warnings(self) -> bool:
        """
        Get whether to suppress warnings.

        Returns
        -------
        bool
            Whether to suppress warnings.
        """
        return self._suppress_warnings

    @suppress_warnings.setter
    def suppress_warnings(self, suppress: bool) -> None:
        """
        Set whether to suppress warnings.

        Parameters
        ----------
        suppress : bool
            Whether to suppress warnings.

        Raises
        ------
        TypeError
            If suppress is not a bool.
        """
        if not isinstance(suppress, bool):
            raise TypeError('suppress_warnings must be True or False.')
        self._suppress_warnings = suppress

    def __copy__(self) -> 'ConvolutionSettings':
        """
        Return a shallow copy of the ConvolutionSettings.

        Returns
        -------
        'ConvolutionSettings'
            A new ConvolutionSettings instance with the same parameter values.
        """
        return ConvolutionSettings(
            upsample_factor=self.upsample_factor,
            extension_factor=self.extension_factor,
            suppress_warnings=self.suppress_warnings,
            display_name=self.display_name,
        )

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
            f'extension_factor={self.extension_factor}, '
            f'suppress_warnings={self.suppress_warnings})'
        )
