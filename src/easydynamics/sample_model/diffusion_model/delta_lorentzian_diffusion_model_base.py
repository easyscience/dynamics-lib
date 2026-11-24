# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import scipp as sc
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easydynamics.sample_model.component_collection import ComponentCollection
from easydynamics.sample_model.components import DeltaFunction
from easydynamics.sample_model.components import Lorentzian
from easydynamics.sample_model.diffusion_model.diffusion_model_base import DiffusionModelBase
from easydynamics.utils.fit_target import FitTarget
from easydynamics.utils.utils import CANONICAL_Q_UNIT
from easydynamics.utils.utils import Numeric
from easydynamics.utils.utils import Q_type


class DeltaLorentzianDiffusionModelBase(DiffusionModelBase, metaclass=ABCMeta):
    r"""
    Base class for diffusion models made of one delta function and n Lorentzians per Q-value.

    The model is

    $$ S(Q, E) = K \left[ A_0(Q)\,\delta(E) + \sum_{j=1}^{n} A_j(Q)\, L(E, \Gamma_j(Q)) \right] $$

    where $K$ is the scale factor, $A_0(Q)$ is the Elastic Incoherent Structure Factor (EISF),
    $A_j(Q)$ is the Quasi-Elastic Incoherent Structure Factor (QISF) of order $j$, and $L(E,
    \Gamma)$ is a Lorentzian with half width at half maximum $\Gamma$. The total QISF is
    $\sum_{j=1}^{n} A_j(Q)$, and EISF + QISF = 1 unless the model truncates an infinite sum. $Q$ is
    assumed to have units of 1/angstrom. Creates ComponentCollections with one delta function and n
    Lorentzian components for given Q-values. Their areas and widths are dependent Parameters, so
    they follow the model parameters during a fit.

    The base class holds no physics. A subclass supplies $\Gamma_j$ through ``calculate_width``,
    $A_0$ through ``calculate_EISF``, $A_j$ through ``calculate_QISF_order``, the dependent
    Parameter of $A_j$ through ``_create_weight`` (order 0 is the EISF), and the dependent width
    through ``_write_width_dependency_expression`` and ``_write_width_dependency_map_expression``.
    It calls ``self.create_component_collections()`` at the end of its ``__init__``, after its
    parameters exist.

    The Lorentzian of order $j$ is named ``'<lorentzian_name>_<symbol><j>'``, where the symbol is
    the class attribute ``_order_symbol``, the conventional symbol of the subclass's order (``'l'``
    by default). ``get_fit_targets`` provides ``'elastic_area'`` and, for every order,
    ``'area_<j>'`` and ``'width_<j>'``.

    See Also
    --------
    EquivalentSitesCircle : Jumps between N equivalent sites on a circle.
    JumpSitesLogNormalDistribution : The same with a log-normal distribution of jump rates.
    GaussianModel3D : Diffusion in a three-dimensional harmonic confinement.

    Examples
    --------
    **Using a model built on this base class**

    The base class is abstract. A subclass, here EquivalentSitesCircle with N = 3 sites, has N - 1
    = 2 Lorentzians per Q-value:
    ```python
    import numpy as np
    import easydynamics as edyn

    model = edyn.EquivalentSitesCircle(
        radius=1.0,
        residence_time=2.0,
        number_of_sites=3,
        Q=np.linspace(0.5, 2, 7),
    )
    model.number_of_lorentzians  # 2
    model.lorentzian_component_name(1)  # 'EquivalentSitesCircle_l1'
    model.calculate_EISF() + model.calculate_QISF()  # ones
    ```
    """

    _order_symbol: str = 'l'

    def __init__(
        self,
        number_of_lorentzians: int,
        *,
        name: str,
        display_name: str | None,
        scale: Numeric = 1.0,
        Q: Q_type | None = None,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        lorentzian_name: str | None = None,
        lorentzian_display_name: str | None = None,
        delta_name: str = 'Delta function',
        delta_display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the shared part of a delta + n Lorentzians diffusion model.

        Builds no components; the subclass calls ``create_component_collections`` once its
        parameters exist.

        Parameters
        ----------
        number_of_lorentzians : int
            Number n of Lorentzians per Q. Must be an integer of at least 1.
        name : str
            Name of the diffusion model.
        display_name : str | None
            Display name of the diffusion model.
        scale : Numeric, default=1.0
            Scale factor for the diffusion model. Must be a non-negative number.
        Q : Q_type | None, default=None
            Q values for the model. If None, Q is not set.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis (energy/frequency). Must be convertible to meV.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the model output (intensity). Determines scale.unit = x_unit * y_unit.
        lorentzian_name : str | None, default=None
            Base name of the Lorentzian components. If None, it will be set to the name of the
            diffusion model.
        lorentzian_display_name : str | None, default=None
            Base display name of the Lorentzian components. If None, it will be set to
            lorentzian_name.
        delta_name : str, default='Delta function'
            Name of the delta function component.
        delta_display_name : str | None, default=None
            Display name of the delta function component. If None, it will be set to delta_name.
        unique_name : str | None, default=None
            Unique name of the diffusion model. If None, a unique name will be generated.

        Raises
        ------
        TypeError
            If number_of_lorentzians is not an integer, or if delta_name or delta_display_name is
            not a string (or None).
        ValueError
            If number_of_lorentzians is smaller than 1.
        """
        super().__init__(
            scale=scale,
            Q=Q,
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            lorentzian_name=lorentzian_name,
            lorentzian_display_name=lorentzian_display_name,
            unique_name=unique_name,
        )

        if isinstance(number_of_lorentzians, bool) or not isinstance(
            number_of_lorentzians, (int, np.integer)
        ):
            raise TypeError('number_of_lorentzians must be an integer.')
        if number_of_lorentzians < 1:
            raise ValueError('number_of_lorentzians must be at least 1.')
        self._number_of_lorentzians = int(number_of_lorentzians)

        if not isinstance(delta_name, str):
            raise TypeError('delta_name must be a string.')
        if delta_display_name is None:
            delta_display_name = delta_name
        if not isinstance(delta_display_name, str):
            raise TypeError('delta_display_name must be a string or None.')
        self._delta_name = delta_name
        self._delta_display_name = delta_display_name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def number_of_lorentzians(self) -> int:
        """
        Get the number of Lorentzians per Q.

        Returns
        -------
        int
            The number n of Lorentzians. Set at construction.
        """
        return self._number_of_lorentzians

    @property
    def delta_name(self) -> str:
        """
        Get the name of the delta function component.

        Returns
        -------
        str
            Name of the delta function component.
        """
        return self._delta_name

    @delta_name.setter
    def delta_name(self, delta_name: str) -> None:
        """
        Set the name of the delta function component.

        Parameters
        ----------
        delta_name : str
            The new name for the delta function component.

        Raises
        ------
        TypeError
            If delta_name is not a string.
        """
        if not isinstance(delta_name, str):
            raise TypeError('delta_name must be a string.')
        self._delta_name = delta_name

    @property
    def delta_display_name(self) -> str:
        """
        Get the display name of the delta function component.

        Returns
        -------
        str
            Display name of the delta function component.
        """
        return self._delta_display_name

    @delta_display_name.setter
    def delta_display_name(self, delta_display_name: str | None) -> None:
        """
        Set the display name of the delta function component.

        Parameters
        ----------
        delta_display_name : str | None
            The new display name for the delta function component.

        Raises
        ------
        TypeError
            If delta_display_name is not a string or None.
        """
        if not isinstance(delta_display_name, (str, type(None))):
            raise TypeError('delta_display_name must be a string or None.')
        self._delta_display_name = delta_display_name

    # ------------------------------------------------------------------
    # Physics, supplied by the subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def calculate_width(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        r"""
        Calculate the half-width at half-maximum (HWHM) $\Gamma_j(Q)$ of the Lorentzian of order
        $j$.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order j of the Lorentzian, between 1 and n.

        Returns
        -------
        np.ndarray
            HWHM values in the unit of the model (e.g., meV).
        """

    @abstractmethod
    def calculate_EISF(self, Q: Q_type | None = None) -> np.ndarray:
        """
        Calculate the Elastic Incoherent Structure Factor (EISF), $A_0(Q)$, the weight of the delta
        function.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.

        Returns
        -------
        np.ndarray
            EISF values (dimensionless).
        """

    @abstractmethod
    def calculate_QISF_order(self, Q: Q_type | None = None, order: int = 1) -> np.ndarray:
        """
        Calculate the Quasi-Elastic Incoherent Structure Factor (QISF) of one order, $A_j(Q)$, the
        weight of the Lorentzian of order $j$.

        Parameters
        ----------
        Q : Q_type | None, default=None
            Scattering vector in 1/angstrom.
        order : int, default=1
            The order j of the Lorentzian, between 1 and n.

        Returns
        -------
        np.ndarray
            QISF values of that order (dimensionless).
        """

    @abstractmethod
    def _create_weight(self, Q: float, order: int) -> Parameter:
        """
        Create the EISF (order 0) or the QISF of one order as a dimensionless dependent Parameter.

        Parameters
        ----------
        Q : float
            Scattering vector in 1/angstrom.
        order : int
            The order j of the Lorentzian; 0 for the delta function.

        Returns
        -------
        Parameter
            The EISF or QISF, which becomes the weight of the component.
        """

    @abstractmethod
    def _write_width_dependency_expression(self, order: int) -> str:
        """
        Write the dependency expression for the width of the Lorentzian of a given order.

        Parameters
        ----------
        order : int
            The order of the Lorentzian.

        Returns
        -------
        str
            Dependency expression for the width.
        """

    @abstractmethod
    def _write_width_dependency_map_expression(self) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map for the width.

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the width.
        """

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------

    def lorentzian_component_name(self, order: int) -> str:
        """
        Get the name of the Lorentzian component of a given order.

        Parameters
        ----------
        order : int
            The order of the Lorentzian, between 1 and n.

        Returns
        -------
        str
            The component name, '<lorentzian_name>_<symbol><order>'.
        """
        self._verify_order(order)
        return f'{self.lorentzian_name}_{self._order_symbol}{order}'

    def create_component_collections(self) -> list[ComponentCollection]:
        r"""
        Create ComponentCollections for the model at the current Q values.

        Each collection holds a delta function with area $K A_0(Q)$ (scale times EISF) and n
        Lorentzians with widths $\Gamma_j(Q)$ from ``calculate_width`` and areas $K A_j(Q)$ (scale
        times the QISF of order $j$). The created collections are installed on the model, so the
        returned list is the live one.

        Returns
        -------
        list[ComponentCollection]
            One ComponentCollection per Q value.
        """
        if self.Q is None:
            self._component_collections = []
            return self._component_collections

        Q = self.Q.values
        component_collection_list = [None] * len(Q)

        for i, Q_value in enumerate(Q):
            component_collection_list[i] = ComponentCollection(
                name=f'{self.name}_Q{Q_value:.2f}',
                display_name=f'{self.display_name}_Q{Q_value:.2f}',
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )

            delta_component = DeltaFunction(
                name=self.delta_name,
                display_name=self.delta_display_name,
                x_unit=self.x_unit,
                y_unit=self.y_unit,
            )
            with np.errstate(invalid='ignore'):
                delta_component.area.make_dependent_on(
                    dependency_expression=self._write_area_dependency_expression(),
                    dependency_map=self._write_area_dependency_map_expression(
                        self._create_weight(float(Q_value), 0)
                    ),
                )
            component_collection_list[i].append_component(delta_component)

            for order in range(1, self.number_of_lorentzians + 1):
                lorentzian_base_display_name = self.lorentzian_display_name or self.lorentzian_name
                lorentzian_component = Lorentzian(
                    name=self.lorentzian_component_name(order),
                    display_name=f'{lorentzian_base_display_name}_{self._order_symbol}{order}',
                    x_unit=self.x_unit,
                    y_unit=self.y_unit,
                )

                # easyscience propagates inf bounds through arithmetic, producing inf/inf=nan
                # as a transient intermediate. Python's min/max ignore nan so the final bounds
                # are correct; suppress the spurious numpy RuntimeWarning.
                with np.errstate(invalid='ignore'):
                    lorentzian_component.width.make_dependent_on(
                        dependency_expression=self._write_width_dependency_expression(order),
                        dependency_map=self._write_width_dependency_map_expression(),
                        desired_unit=self.x_unit,
                    )
                    lorentzian_component.area.make_dependent_on(
                        dependency_expression=self._write_area_dependency_expression(),
                        dependency_map=self._write_area_dependency_map_expression(
                            self._create_weight(float(Q_value), order)
                        ),
                    )
                component_collection_list[i].append_component(lorentzian_component)

        self._component_collections = component_collection_list
        return self._component_collections

    def get_fit_targets(self) -> list[FitTarget]:
        """
        Get the fittable predictions of the model as FitTargets.

        The model predicts ``'elastic_area'`` (``scale * EISF``) and, for every order from 1 to n,
        ``'area_<order>'`` (``scale`` times the QISF of that order) and ``'width_<order>'`` (its
        HWHM). The base class implementation of DiffusionModelBase is replaced because there is no
        single Lorentzian. Select the orders to fit with the ``targets`` of a FitBinding.

        Returns
        -------
        list[FitTarget]
            The fittable predictions of this model.
        """
        targets = [
            FitTarget(
                name='elastic_area',
                dataset_key=f'{self.delta_name} area',
                function=lambda Q, model=self, **_: model.calculate_EISF(Q) * model.scale.value,
                label=f'{self.display_name} elastic_area',
                x_unit=CANONICAL_Q_UNIT,
                y_unit=str(self.scale.unit),
            )
        ]
        for order in range(1, self.number_of_lorentzians + 1):
            key = self.lorentzian_component_name(order)
            targets.append(
                FitTarget(
                    name=f'area_{order}',
                    dataset_key=f'{key} area',
                    function=lambda Q, model=self, order=order, **_: (
                        model.calculate_QISF_order(Q, order) * model.scale.value
                    ),
                    label=f'{self.display_name} area_{order}',
                    x_unit=CANONICAL_Q_UNIT,
                    y_unit=str(self.scale.unit),
                )
            )
            targets.append(
                FitTarget(
                    name=f'width_{order}',
                    dataset_key=f'{key} width',
                    function=lambda Q, model=self, order=order, **_: model.calculate_width(
                        Q, order
                    ),
                    label=f'{self.display_name} width_{order}',
                    x_unit=CANONICAL_Q_UNIT,
                    y_unit=str(self.x_unit),
                )
            )
        return targets

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _write_area_dependency_expression(self) -> str:
        """
        Write the dependency expression for the area of a component.

        Returns
        -------
        str
            Dependency expression for the area, the scale times the EISF or QISF.
        """
        return 'scale * weight'

    def _write_area_dependency_map_expression(
        self, weight: Parameter
    ) -> dict[str, DescriptorNumber]:
        """
        Write the dependency map for the areas.

        Parameters
        ----------
        weight : Parameter
            The EISF or the QISF of one order, from ``_create_weight`` (dimensionless).

        Returns
        -------
        dict[str, DescriptorNumber]
            Dependency map for the area.
        """
        return {'scale': self.scale, 'weight': weight}

    def _verify_order(self, order: int) -> None:
        """
        Verify that order is a valid Lorentzian order.

        Parameters
        ----------
        order : int
            The order to verify.

        Raises
        ------
        TypeError
            If order is not an integer.
        ValueError
            If order is not between 1 and n.
        """
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise TypeError('order must be an integer.')
        if not 1 <= order <= self.number_of_lorentzians:
            raise ValueError(f'order must be between 1 and {self.number_of_lorentzians}.')

    def _on_Q_change(self) -> None:
        """Rebuild the component collections when Q changes."""
        self.create_component_collections()
