from easyscience.job.analysis import AnalysisBase
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting.fitter import Fitter as EasyScienceFitter

from easyscience.variable import Parameter

from easydynamics.resolution import ResolutionHandler


from easydynamics.sample import SampleModel
from easydynamics.sample import DiffusionModel

from easydynamics.Experiment import Experiment

from typing import Iterable, Dict, Tuple, Optional


import numpy as np

import scipp as sc

import matplotlib.pyplot as plt


class Analysis(AnalysisBase):
    def __init__(self, name="MyAnalysis", interface=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._theory = None
        self._experiment = None
        self._offset = Parameter(name="offset", value=0.0, unit="meV")
        self._diffusion_model = None

        self._resolution_model = None
        self._background_model = None

    def set_diffusion_model(self, diffusion_model: DiffusionModel):
        """Set the diffusion model for the analysis.
        Args:
            diffusion_model (DiffusionModel): The diffusion model to be used in the analysis.
        """
        if not isinstance(diffusion_model, DiffusionModel):
            raise TypeError(
                "The diffusion model must be an instance of DiffusionModel."
            )
        self._diffusion_model = diffusion_model

    def set_theory(self, theory: SampleModel):
        """Set the model to be fitted.
        Args:
            theory (SampleModel): The theoretical model to be used in the analysis.
        """
        if not isinstance(theory, SampleModel):
            raise TypeError("The theory must be an instance of SampleModel.")
        self._theory = theory

    def set_experiment(self, experiment: Experiment):
        """Set the experimental for the analysis.
        Args:
            experiment (Experiment): The experimental model to be used in the analysis.
        """
        if not isinstance(experiment, Experiment):
            raise TypeError("The experiment must be an instance of Experiment.")
        self._experiment = experiment

    def set_background_model(self, background: SampleModel):
        """Set the model for the background.
        Args:
            background (SampleModel): The background model.
        """
        if not isinstance(background, SampleModel):
            raise TypeError("Background model must be an instance of SampleModel.")
        self._background_model = background

    def set_resolution_model(self, resolution: SampleModel):
        """Set the resolution model for the experiment. The resolution will be normalised to have area 1.
        Args:
            resolution (SampleModel): The resolution model to be used in the experiment.
        """
        # TODO: allow resolution to be DataArray or SampleModel

        if resolution is not None and not isinstance(resolution, SampleModel):
            raise TypeError("Resolution model must be an instance of SampleModel.")
        self._resolution_model = resolution

        if self._resolution_model is not None:
            self.normalize_resolution()

    def fix_resolution_parameters(self):
        """Fix all parameters in the resolution model."""
        if self._resolution_model is not None:
            for param in self._resolution_model.get_parameters():
                param.fixed = True

    def normalize_resolution(self):
        """Normalize the resolution model to have an area of 1."""
        self._resolution_model.normalize_area()

    def set_offset(self, offset: float, unit):
        # TODO: handle units properly
        self._offset.value = offset

    def fix_offset(self, fix: bool = True):
        self._offset.fixed = fix

    def calculate_theory(self, x) -> np.ndarray:
        """
        Calculate the theoretical model by convolving the sample model with the resolution model
        and adding the background model.
        """

        if self._resolution_model is None:
            y = self._theory.evaluate(x - self._offset.value)
        else:
            resolution_handler = ResolutionHandler()
            y = resolution_handler.convolve(
                x, self._theory, self._resolution_model, self._offset
            )

        if self._background_model is not None:
            y += self._background_model.evaluate(x)

        return y

    def calculate_individual_components(self, x=None, add_background=True) -> dict:
        """
        Calculate the individual components of the theory model.

        Parameters
        ----------
        x : np.ndarray
            Energy axis (e.g., in meV).

        Returns
        -------
        dict
            A dictionary with component names as keys and evaluated values as values.
        """

        if self._theory is None:
            raise RuntimeError(
                "Theory model must be set before calculating components."
            )

        # standard: use experimental data x if not provided
        if x is None:
            if self._experiment is None or self._experiment._data is None:
                raise RuntimeError("No x values provided and no experiment data set.")
            x, _, _ = self._experiment.extract_xye_data(self._experiment._data)

        components = {}

        if self._resolution_model is not None:
            resolution_handler = ResolutionHandler()

        for name, component in self._theory.components.items():
            if self._resolution_model is None:
                components[name] = component.evaluate(x - self._offset.value)
            else:
                components[name] = resolution_handler.convolve(
                    x=x,
                    sample_model=self._theory,
                    resolution_model=self._resolution_model,
                    offset=self._offset,
                    selected_component_name=name,
                )

            if add_background and self._background_model is not None:
                components[name] += self._background_model.evaluate(
                    x - self._offset.value
                )

        # If background model is set, add its components
        if self._background_model is not None:
            background_components = self._background_model.components.items()
            for name, component in background_components:
                components[name] = component.evaluate(x - self._offset.value)

        return components

    def fit(self):
        x, y, e = self._experiment.extract_xye_data(self._experiment._data)

        def fit_func(x_vals):
            return self.calculate_theory(x_vals)

        # multi_fitter = EasyScienceMultiFitter(
        #     fit_objects=[self],
        #     fit_functions=[fit_func],
        # )

        # # Perform the fit
        # fit_result = multi_fitter.fit(x=[x], y=[y], weights=[1.0 / e])

        fitter = EasyScienceFitter(
            fit_object=self,
            fit_function=fit_func,
        )

        # Perform the fit
        fit_result = fitter.fit(x=x, y=y, weights=1.0 / e)

        # Store result
        self.fit_result = fit_result

        return fit_result

    def plot_data_and_model(self, plot_individual_components: bool = False):
        """
        Plot the experimental data and the theoretical fit.

        Args:
            plot_individual_components (bool): If True, plots individual components of the theory model.
        """
        if self._experiment is None or self._theory is None:
            raise RuntimeError("Experiment and theory must be set before plotting.")

        if self._experiment._data is None:
            raise RuntimeError("No data has been set in the experiment.")

        # Extract data
        x, y, e = self._experiment.extract_xye_data(self._experiment._data)

        # Start plot
        fig = plt.figure(figsize=(10, 6))
        plt.errorbar(
            x,
            y,
            yerr=e,
            label="Data",
            color="black",
            marker="o",
            linestyle="None",
            markerfacecolor="none",
        )

        # Compute and plot fit
        fit_y = self.calculate_theory(x)
        plt.plot(x, fit_y, label="Model", color="red")

        if plot_individual_components:
            components = self.calculate_individual_components()
            for name, y in components.items():
                plt.plot(x, y, label=f" {name}", linestyle="--")

        # Labels and legend
        plt.xlabel("Energy (meV)")  # TODO: Handle units
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return fig

    def seed_from(
        self,
        other: "Analysis",
        *,
        domains: Iterable[str] = ("theory", "background", "resolution"),
        only_unfixed: bool = True,
        strict_components: bool = True,
        strict_params: bool = True,
        include_temperature: bool = False,
        require_same_units: bool = True,
        convert_units: bool = False,
        copy_offset: bool = True,
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Copy parameter *values* from `other` into this Analysis, domain by domain.

        Parameters
        ----------
        other : Analysis
            Source analysis whose current parameter values will be used.
        domains : ('theory','background','resolution')
            Which SampleModels to seed.
        only_unfixed : bool
            If True, skip fixed params in *this* analysis.
        strict_components : bool
            If True, require identical component-name sets; else use intersection.
        strict_params : bool
            If True, require identical parameter-name sets per component; else use intersection.
        include_temperature : bool
            If True, copy model temperature (when present on both).
        require_same_units : bool
            If True, raise on unit mismatch; otherwise allow.
        convert_units : bool
            If True, convert this analysis' param units to match `other` before copying values.
        copy_offset : bool
            If True, also copy `other._offset.value` → `self._offset.value` (unless fixed here).

        Returns
        -------
        Dict[str, Dict[str, Tuple[old, new]]]
            Per-domain reports of value changes; 'offset' reported under key 'analysis'.
        """
        if not isinstance(other, Analysis):
            raise TypeError("seed_from: `other` must be an Analysis")

        report: Dict[str, Dict[str, Tuple[float, float]]] = {}

        def _maybe_update(
            domain_name: str,
            this_model: Optional[SampleModel],
            other_model: Optional[SampleModel],
        ):
            if this_model is None or other_model is None:
                return
            rep = this_model.update_values_from(
                other_model,
                only_unfixed=only_unfixed,
                strict_components=strict_components,
                strict_params=strict_params,
                include_temperature=include_temperature,
                require_same_units=require_same_units,
                convert_units=convert_units,
            )
            if rep:
                report[domain_name] = rep

        if "theory" in domains:
            _maybe_update("theory", self._theory, other._theory)
        if "background" in domains:
            _maybe_update("background", self._background_model, other._background_model)
        if "resolution" in domains:
            _maybe_update("resolution", self._resolution_model, other._resolution_model)

        # Copy analysis-level offset value
        if copy_offset and hasattr(self, "_offset") and hasattr(other, "_offset"):
            if not getattr(self._offset, "fixed", False):
                old = self._offset.value
                self._offset.value = other._offset.value
                report.setdefault("analysis", {})["offset"] = (old, self._offset.value)

        return report

    def switch_minimizer(self, minimizer: AvailableMinimizers) -> None:
        """
        Switch the minimizer for the fitting.

        :param minimizer: Minimizer to be switched to
        """
        self.easy_science_multi_fitter.switch_minimizer(minimizer)

    def get_parameters(self):
        """
        Get all parameters from the theory, resolution, background models, and experiment offset.

        Returns:
            List[Parameter]: A list of all parameters.
        """
        params = []

        if self._theory is not None:
            params.extend(self._theory.get_parameters())

        if self._diffusion_model is not None:
            params.extend(self._diffusion_model.get_parameters())

        if self._experiment is not None:
            if self._resolution_model is not None:
                params.extend(self._resolution_model.get_parameters())
            if self._background_model is not None:
                params.extend(self._background_model.get_parameters())

        params.append(self._offset)

        return params

    def get_fit_parameters(self):
        """
        Get all fit parameters from the theory, resolution, background models, and experiment offset,
        filtering out fixed parameters.

        Returns:
            List[Parameter]: A list of unfixed fit parameters.
        """
        # return [param for param in self.get_parameters() if not getattr(param, 'fixed', False)]
        return [
            param
            for param in self.get_parameters()
            if not getattr(param, "fixed", False)
            and getattr(param, "_independent", True)
        ]
