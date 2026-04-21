# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from easydynamics.base_classes.easydynamics_base import EasyDynamicsBase


class DetailedBalanceSettings(EasyDynamicsBase):
    """
    Class to manage detailed balance settings for a SampleModel or Analysis.
    """

    def __init__(
        self,
        use_detailed_balance: bool = True,
        normalize_detailed_balance: bool = True,
        display_name: str = 'DetailedBalanceSettings',
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the DetailedBalanceSettings.

        Parameters
        ----------
        use_detailed_balance : bool, default=True
            Whether to apply detailed balance to the model. If False, no detailed balance is
            applied.
        normalize_detailed_balance : bool, default=True
            Whether to normalize the detailed balance factor by dividing with temperature.
        display_name : str, default="DetailedBalanceSettings"
            Display name of the model.
        unique_name : str | None, default=None
            Unique name of the model. If None, a unique name will be generated.


        Raises
        ------
        TypeError
            If use_detailed_balance or normalize_detailed_balance is not a bool.
        """
        if not isinstance(use_detailed_balance, bool):
            raise TypeError('use_detailed_balance must be True or False')
        self._use_detailed_balance = use_detailed_balance

        if not isinstance(normalize_detailed_balance, bool):
            raise TypeError('normalize_detailed_balance must be True or False')
        self._normalize_detailed_balance = normalize_detailed_balance

        super().__init__(
            display_name=display_name,
            unique_name=unique_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def use_detailed_balance(self) -> bool:
        """
        Get whether to apply detailed balance to the model.

        Returns
        -------
        bool
            True if detailed balance is applied, False otherwise.
        """
        return self._use_detailed_balance

    @use_detailed_balance.setter
    def use_detailed_balance(self, value: bool) -> None:
        """
        Set whether to apply detailed balance to the model.

        Parameters
        ----------
        value : bool
            True to apply detailed balance, False otherwise.

        Raises
        ------
        TypeError
            If value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError('use_detailed_balance must be True or False')
        self._use_detailed_balance = value

    @property
    def normalize_detailed_balance(self) -> bool:
        """
        Get whether to divide the detailed balance factor by temperature.

        Returns
        -------
        bool
            True if the detailed balance factor should be normalized by dividing with temperature,
            False otherwise.
        """
        return self._normalize_detailed_balance

    @normalize_detailed_balance.setter
    def normalize_detailed_balance(self, value: bool) -> None:
        """
        Set whether to normalize the detailed balance factor by dividing with temperature.

        Parameters
        ----------
        value : bool
            True to normalize the detailed balance factor by dividing with temperature, False
            otherwise.

        Raises
        ------
        TypeError
            If value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError('normalize_detailed_balance must be True or False')
        self._normalize_detailed_balance = value

    def __repr__(self) -> str:
        """
        Return a string representation of the DetailedBalanceSettings.

        Returns
        -------
        str
            A string representation of the DetailedBalanceSettings.
        """
        return (
            f'DetailedBalanceSettings(use_detailed_balance={self.use_detailed_balance}, '
            f'normalize_detailed_balance={self.normalize_detailed_balance})'
        )
