# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipp as sc
from easyscience.variable import Parameter

from easydynamics.sample_model.components.mixins import CreateParametersMixin
from easydynamics.sample_model.components.model_component import ModelComponent
from easydynamics.utils.utils import Numeric

MINIMUM_ALPHA = 1e-10  # To avoid a vanishing form parameter
MAXIMUM_ALPHA = 1.0  # Above 1 the Mittag-Leffler function is no longer a relaxation function


class DiffusionDampedMittagLeffler(CreateParametersMixin, ModelComponent):
    r"""
    Model of the diffusion-damped Mittag-Leffler relaxation spectrum, Eq. (42) of Hassani et al.

    This is the spectral (energy-domain) form of the multiscale relaxation model used for
    intrinsically disordered proteins in

    A. N. Hassani, L. Haris, M. Appel, T. Seydel, A. M. Stadler and G. R. Kneller, *Multiscale
    relaxation dynamics and diffusion of myelin basic protein in solution studied by quasielastic
    neutron scattering*, J. Chem. Phys. **156**, 025102 (2022),
    [doi:10.1063/5.0077100](https://doi.org/10.1063/5.0077100).

    In the time domain the internal dynamics is described by the Mittag-Leffler (ML) relaxation
    function $\phi_{ML}(t) = E_\alpha(-(|t|/\tau_R)^\alpha)$ (their Eq. 28), damped by global
    translational diffusion, $e^{-\epsilon |t|}$ with $\epsilon = D q^2$ (their Eqs. 38-39). The
    Fourier transform of that product is the "generalised Lorentzian" of their Eq. (42),

    $$ \tilde{\phi}^{(\epsilon)}_{ML}(\omega) = \frac{1}{\pi}
    \frac{\epsilon(\omega^2+\epsilon^2)^{\alpha/2}
    + \omega\sin(\alpha\arg(\epsilon+i|\omega|))
    + \epsilon\cos(\alpha\arg(\epsilon+i|\omega|))}
    {(\omega^2+\epsilon^2)\left(\left((\omega^2+\epsilon^2)^{\alpha}+1\right)
    (\omega^2+\epsilon^2)^{-\alpha/2} + 2\cos(\alpha\arg(\epsilon+i|\omega|))\right)} $$

    which this component evaluates as

    $$ I(x) = \frac{A}{\Gamma} \tilde{\phi}^{(\epsilon/\Gamma)}_{ML}\left(\frac{|x|}{\Gamma}\right)
    $$

    where $A$ is the scale factor (``scale``), $\alpha$ is the form parameter (``alpha``), $\Gamma
    = \hbar/\tau_R$ is the ML relaxation rate expressed as an energy (``width``) and $\epsilon =
    \hbar D q^2$ is the diffusion damping, also expressed as an energy (``damping``). scale has
    unit = x_unit * y_unit; width and damping have unit = x_unit; alpha is dimensionless.

    Two remarks on the relation to the printed Eq. (42):

    - Eq. (42) is written for $\tau_R = 1$, i.e. for $\omega$ and $\epsilon$ measured in units of
      the relaxation rate. Rescaling $\omega \to \omega/\Gamma$ and $\epsilon \to \epsilon/\Gamma$
      and dividing by $\Gamma$ restores a general relaxation time while leaving the printed
      expression intact; that is the form implemented here.
    - The $1/\pi$ prefactor is absent from the printed Eq. (42), but is required for the lineshape
      to be normalised the way Eq. (41) of the same paper assumes, i.e.
      $\int d\omega\, \tilde{\phi}^{(\epsilon)}_{ML}(\omega) = 1$. It is included here, so
      ``scale`` is the integrated area of the profile, and the $\alpha \to 1$ limit is a Lorentzian
      of area ``scale`` and half width at half maximum ``width + damping``.

    The profile is symmetric about $x = 0$, so there is no center parameter. Because a strictly
    positive ``damping`` is enforced, the profile stays regular at $x = 0$, where the undamped ML
    spectrum would diverge as $|\omega|^{\alpha - 1}$.

    Examples
    --------
    **Creating a diffusion-damped Mittag-Leffler component**

    ```python
    import numpy as np
    import easydynamics as edyn

    ml = edyn.DiffusionDampedMittagLeffler(scale=1.0, alpha=0.8, width=0.05, damping=0.01)
    x = np.linspace(-0.5, 0.5, 200)
    values = ml.evaluate(x)
    ```

    **Modifying parameters after construction**

    ```python
    import easydynamics as edyn

    ml = edyn.DiffusionDampedMittagLeffler(name='MBP internal dynamics')
    ml.scale = 0.9
    ml.alpha = 0.75
    ml.width = 0.02
    ml.damping = 0.005
    ```
    """

    def __init__(
        self,
        scale: Numeric = 1.0,
        alpha: Numeric = 1.0,
        width: Numeric = 1.0,
        damping: Numeric = 1.0,
        x_unit: str | sc.Unit = 'meV',
        y_unit: str | sc.Unit = 'dimensionless',
        name: str = 'DiffusionDampedMittagLeffler',
        display_name: str | None = None,
        unique_name: str | None = None,
    ) -> None:
        """
        Initialize the diffusion-damped Mittag-Leffler component.

        Parameters
        ----------
        scale : Numeric, default=1.0
            Scale factor in front of the normalised profile, i.e. the integrated area of the
            profile.  Unit is ``x_unit * y_unit``.  Must be non-negative.
        alpha : Numeric, default=1.0
            Form parameter of the Mittag-Leffler relaxation function.  Must lie in (0, 1]. alpha=1
            gives exponential relaxation, and hence a Lorentzian profile, while smaller values give
            a broader relaxation rate spectrum.
        width : Numeric, default=1.0
            Mittag-Leffler relaxation rate ``hbar / tau_R`` in x_unit.  Must be strictly positive.
        damping : Numeric, default=1.0
            Diffusion damping ``hbar * D * q**2`` in x_unit.  Must be strictly positive; it is what
            keeps the profile finite at x=0.
        x_unit : str | sc.Unit, default='meV'
            Unit of the x-axis.  width and damping are stored in this unit. scale_unit = x_unit *
            y_unit.
        y_unit : str | sc.Unit, default='dimensionless'
            Unit of the y-axis (output).
        name : str, default='DiffusionDampedMittagLeffler'
            Name of the component.
        display_name : str | None, default=None
            Display name shown when plotting.  Falls back to *name* if None.
        unique_name : str | None, default=None
            Globally unique identifier.  Auto-generated if None.
        """
        super().__init__(
            x_unit=x_unit,
            y_unit=y_unit,
            name=name,
            display_name=display_name,
            unique_name=unique_name,
        )

        self._scale = self._create_scale_parameter(
            scale=scale, name=name, x_unit=self.x_unit, y_unit=self.y_unit
        )
        self._alpha = self._create_alpha_parameter(alpha=alpha, name=name)
        # These methods live in CreateParametersMixin
        self._width = self._create_width_parameter(width=width, name=name, x_unit=self.x_unit)
        self._damping = self._create_width_parameter(
            width=damping, name=name, param_name='damping', x_unit=self.x_unit
        )

    @staticmethod
    def _create_scale_parameter(
        scale: Numeric,
        name: str,
        x_unit: str | sc.Unit,
        y_unit: str | sc.Unit,
    ) -> Parameter:
        """
        Create the scale Parameter with unit = x_unit * y_unit.

        Parameters
        ----------
        scale : Numeric
            Initial scale value.  Must be non-negative.
        name : str
            Base name used to label the Parameter (``name + ' scale'``).
        x_unit : str | sc.Unit
            X-axis unit.  The resulting scale unit is ``x_unit * y_unit``.
        y_unit : str | sc.Unit
            Y-axis unit.  The resulting scale unit is ``x_unit * y_unit``.

        Returns
        -------
        Parameter
            Configured scale Parameter with ``unit = x_unit * y_unit`` and ``min = 0``.

        Raises
        ------
        TypeError
            If *scale* is not a numeric type.
        ValueError
            If *scale* is not finite, or is negative.
        """
        if not isinstance(scale, Numeric):
            raise TypeError('scale must be a number.')
        if not np.isfinite(scale):
            raise ValueError('scale must be a finite number.')
        if float(scale) < 0:
            raise ValueError('scale must be non-negative.')
        return Parameter(
            name=name + ' scale',
            value=float(scale),
            unit=str(sc.Unit(x_unit) * sc.Unit(y_unit)),
            min=0.0,
        )

    @staticmethod
    def _create_alpha_parameter(alpha: Numeric, name: str) -> Parameter:
        """
        Create the dimensionless form parameter alpha, bounded to (0, 1].

        Parameters
        ----------
        alpha : Numeric
            Initial value of the form parameter.
        name : str
            Base name used to label the Parameter (``name + ' alpha'``).

        Returns
        -------
        Parameter
            Configured alpha Parameter with ``unit = 'dimensionless'``.

        Raises
        ------
        TypeError
            If *alpha* is not a numeric type.
        ValueError
            If *alpha* is not finite, or does not lie in (0, 1].
        """
        if not isinstance(alpha, Numeric):
            raise TypeError('alpha must be a number.')
        if not np.isfinite(alpha):
            raise ValueError('alpha must be a finite number.')
        if not MINIMUM_ALPHA <= float(alpha) <= MAXIMUM_ALPHA:
            raise ValueError('alpha must be greater than zero and at most one.')
        return Parameter(
            name=name + ' alpha',
            value=float(alpha),
            unit='dimensionless',
            min=MINIMUM_ALPHA,
            max=MAXIMUM_ALPHA,
        )

    @property
    def scale(self) -> Parameter:
        """
        Get the scale parameter.

        Returns
        -------
        Parameter
            The scale Parameter with unit ``x_unit * y_unit``.  It is the integrated area of the
            profile.
        """
        return self._scale

    @scale.setter
    def scale(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New scale value (in current scale unit = x_unit * y_unit).

        Notes
        -----
        A ``TypeError`` propagates from the shared value setter if *value* is not a numeric type,
        and a ``ValueError`` propagates from it if *value* violates the scale parameter's bounds,
        e.g. a negative value against its ``min=0``.
        """
        self._set_bounded_parameter_value(self._scale, value, 'scale')

    @property
    def alpha(self) -> Parameter:
        """
        Get the form parameter of the Mittag-Leffler relaxation function.

        Returns
        -------
        Parameter
            The dimensionless alpha Parameter, bounded to (0, 1].
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New form parameter.  Must lie in (0, 1].

        Notes
        -----
        A ``TypeError`` propagates from the shared value setter if *value* is not a numeric type,
        and a ``ValueError`` propagates from it if *value* falls outside (0, 1].
        """
        self._set_bounded_parameter_value(self._alpha, value, 'alpha')

    @property
    def width(self) -> Parameter:
        """
        Get the width parameter (Mittag-Leffler relaxation rate).

        Returns
        -------
        Parameter
            The relaxation rate ``hbar / tau_R`` Parameter with unit ``x_unit``.
        """
        return self._width

    @width.setter
    def width(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New relaxation rate in x_unit.  Must be strictly positive.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not positive, or violates the width parameter's bounds.
        """
        if not isinstance(value, Numeric):
            raise TypeError('width must be a number')
        if float(value) <= 0:
            raise ValueError('width must be positive')
        self._set_bounded_parameter_value(self._width, value, 'width')

    @property
    def damping(self) -> Parameter:
        """
        Get the damping parameter (the diffusion damping ``hbar * D * q**2``).

        Returns
        -------
        Parameter
            The diffusion damping Parameter with unit ``x_unit``.
        """
        return self._damping

    @damping.setter
    def damping(self, value: Numeric) -> None:
        """
        Parameters
        ----------
        value : Numeric
            New diffusion damping in x_unit.  Must be strictly positive.

        Raises
        ------
        TypeError
            If *value* is not a numeric type.
        ValueError
            If *value* is not positive, or violates the damping parameter's bounds.
        """
        if not isinstance(value, Numeric):
            raise TypeError('damping must be a number')
        if float(value) <= 0:
            raise ValueError('damping must be positive')
        self._set_bounded_parameter_value(self._damping, value, 'damping')

    def _evaluate_values(self, x_vals: np.ndarray, eval_unit: str | None) -> np.ndarray:
        r"""
        Evaluate the diffusion-damped Mittag-Leffler spectrum at x_vals.

        Eq. (42) of Hassani et al. is evaluated in the reduced variables $\omega = |x|/\Gamma$ and
        $\epsilon' = \epsilon/\Gamma$, and the result divided by $\Gamma$, which turns the printed
        $\tau_R = 1$ expression into the general one. Parameters in the model's own units are
        temporarily converted to eval_unit for the computation.

        Parameters
        ----------
        x_vals : np.ndarray
            Raw x values expressed in eval_unit.
        eval_unit : str | None
            The unit of x_vals.

        Returns
        -------
        np.ndarray
            Evaluated Mittag-Leffler spectrum values at x_vals.
        """
        width = self._resolve_param_value(self._width, eval_unit)
        damping = self._resolve_param_value(self._damping, eval_unit)
        scale = self._resolve_param_value(self._scale, self._eval_area_unit(eval_unit))
        alpha = self._alpha.value

        # Reduced (tau_R = 1) variables, so that Eq. (42) applies verbatim.
        omega = np.abs(x_vals) / width
        epsilon = damping / width

        modulus_squared = omega**2 + epsilon**2
        # arg(epsilon + i|omega|); damping > 0 is enforced, so modulus_squared is never zero.
        phase = alpha * np.arctan2(omega, epsilon)
        numerator = (
            epsilon * modulus_squared ** (alpha / 2)
            + omega * np.sin(phase)
            + epsilon * np.cos(phase)
        )
        denominator = modulus_squared * (
            (modulus_squared**alpha + 1) * modulus_squared ** (-alpha / 2) + 2 * np.cos(phase)
        )
        return scale * numerator / (np.pi * width * denominator)

    def convert_x_unit(self, new_x_unit: str | sc.Unit) -> None:
        """
        Convert x-axis parameters (width, damping) and scale to new_x_unit.

        The dimensionless alpha is unaffected.

        Parameters
        ----------
        new_x_unit : str | sc.Unit
            Target x-axis unit.  Must be dimensionally compatible with the current x_unit.
        """
        self._convert_x_unit_area_based(
            new_x_unit=new_x_unit,
            x_params=[self._width, self._damping],
            area_param=self._scale,
        )

    def convert_y_unit(self, new_y_unit: str | sc.Unit) -> None:
        """
        Convert the y-axis (output) unit by rescaling the scale parameter.

        The scale is rescaled from ``x_unit * old_y_unit`` to ``x_unit * new_y_unit``.

        Parameters
        ----------
        new_y_unit : str | sc.Unit
            Target y-axis unit.
        """
        self._convert_y_unit_area_based(new_y_unit=new_y_unit, area_param=self._scale)

    def __repr__(self) -> str:
        """
        Return a string representation of the diffusion-damped Mittag-Leffler component.

        Returns
        -------
        str
            A string representation of the diffusion-damped Mittag-Leffler component.
        """
        return (
            f'{self.__class__.__name__}(name = {self.name}, display_name = {self.display_name}, '
            f'x_unit = {self.x_unit}, y_unit = {self.y_unit},\n'
            f'    scale = {self.scale},\n'
            f'    alpha = {self.alpha},\n'
            f'    width = {self.width},\n'
            f'    damping = {self.damping})'
        )
