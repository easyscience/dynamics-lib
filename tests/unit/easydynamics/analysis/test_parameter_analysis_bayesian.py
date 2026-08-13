# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Bayesian sampling on ParameterAnalysis, with the Sampler mocked out."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience.fitting.multi_fitter import MultiFitter

import easydynamics as edyn
import easydynamics.sample_model as sm

SAMPLER_PATH = 'easydynamics.analysis.bayesian_sampling.Sampler'
Q = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0])


def make_dataset():
    widths = 0.10 + 0.35 * Q
    areas = 2.0 - 0.3 * Q
    return sc.Dataset({
        'Lorentzian width': sc.DataArray(
            data=sc.array(
                dims=['Q'], values=widths, variances=np.full_like(widths, 1e-4), unit='meV'
            ),
            coords={'Q': sc.array(dims=['Q'], values=Q, unit='1/angstrom')},
        ),
        'Lorentzian area': sc.DataArray(
            data=sc.array(
                dims=['Q'], values=areas, variances=np.full_like(areas, 4e-4), unit='meV'
            ),
            coords={'Q': sc.array(dims=['Q'], values=Q, unit='1/angstrom')},
        ),
    })


def make_analysis(two_bindings=True):
    bindings = [
        edyn.FitBinding(
            model=sm.Polynomial(
                coefficients=[0.1, 0.35], x_unit='1/angstrom', y_unit='meV', name='Width line'
            ),
            targets='Lorentzian width',
        )
    ]
    if two_bindings:
        bindings.append(
            edyn.FitBinding(
                model=sm.Polynomial(
                    coefficients=[2.0, -0.3], x_unit='1/angstrom', y_unit='meV', name='Area line'
                ),
                targets='Lorentzian area',
            )
        )
    return edyn.ParameterAnalysis(parameters=make_dataset(), bindings=bindings)


def bound_all(analysis, half_width=5.0):
    for parameter in analysis._get_chain_parameters():
        parameter.min = float(parameter.value) - half_width
        parameter.max = float(parameter.value) + half_width


def fake_results(parameters, n_draws=50):
    draws = np.tile([float(p.value) for p in parameters], (n_draws, 1))
    return SimpleNamespace(
        draws=draws,
        param_names=[p.unique_name for p in parameters],
        logp=np.zeros(n_draws),
        state=MagicMock(Ngen=10, Npop=4),
    )


@pytest.fixture
def analysis():
    return make_analysis()


class TestFitterExposure:
    def test_fitter_is_a_cached_multifitter(self, analysis):
        # EXPECT
        assert isinstance(analysis.fitter, MultiFitter)
        assert analysis.fitter is analysis.fitter

    def test_fit_still_returns_per_target_results(self, analysis):
        # WHEN
        results = analysis.fit()

        # EXPECT one result per fit target, as before
        assert isinstance(results, list)
        assert len(results) == 2

    def test_changing_bindings_rebuilds_the_fitter(self, analysis):
        # WHEN
        original = analysis.fitter
        analysis.bindings = analysis.bindings[:1]

        # EXPECT
        assert analysis.fitter is not original

    def test_changing_parameters_rebuilds_the_fitter(self, analysis):
        # WHEN
        original = analysis.fitter
        analysis.parameters = make_dataset()

        # EXPECT
        assert analysis.fitter is not original


class TestChainParameters:
    def test_covers_every_binding_model(self, analysis):
        # WHEN
        parameters = analysis._get_chain_parameters()

        # EXPECT both Polynomials contribute their two coefficients
        assert len(parameters) == 4
        assert len({p.unique_name for p in parameters}) == 4

    def test_labels_are_unique(self, analysis):
        # WHEN
        labels = [analysis.parameter_label(p) for p in analysis._get_chain_parameters()]

        # EXPECT
        assert len(set(labels)) == len(labels)

    def test_model_name_is_not_repeated_in_the_label(self, analysis):
        # WHEN a model already names its parameters after itself
        labels = [analysis.parameter_label(p) for p in analysis._get_chain_parameters()]

        # EXPECT no 'Width line: Width line_c0'
        assert 'Width line_c0' in labels
        assert not any(label.count('Width line') > 1 for label in labels)


class TestSampling:
    def test_refuses_unbounded_parameters(self, analysis):
        # EXPECT
        with pytest.raises(ValueError, match='finite bounds'):
            analysis.sample_posterior(samples=10)

    def test_binds_one_dataset_per_target(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.sample_posterior(samples=10)

        # EXPECT
        args, kwargs = sampler_class.call_args
        assert len(args[1]) == 2
        assert len(kwargs['weights']) == 2

    def test_summary_uses_model_qualified_labels(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()

        with patch(SAMPLER_PATH) as sampler_class:
            sampler_class.return_value.sample.return_value = fake_results(parameters)
            analysis.sample_posterior(samples=10)

        # EXPECT
        names = [entry.name for entry in analysis.posterior_summary()]
        assert len(set(names)) == len(names)
        assert 'Width line_c0' in names

    def test_restores_parameter_values(self, analysis):
        # WHEN
        bound_all(analysis)
        parameters = analysis._get_chain_parameters()
        before = [float(p.value) for p in parameters]

        with patch(SAMPLER_PATH) as sampler_class:

            def mutate(**_kwargs):
                for parameter in parameters:
                    parameter.value = float(parameter.value) + 1.0
                return fake_results(parameters)

            sampler_class.return_value.sample.side_effect = mutate
            analysis.sample_posterior(samples=10)

        # EXPECT
        assert [float(p.value) for p in parameters] == pytest.approx(before)

    def test_missing_parameters_dataset_raises(self):
        # WHEN
        analysis = edyn.ParameterAnalysis()

        # EXPECT
        with pytest.raises(ValueError, match='No parameters Dataset'):
            analysis.sample_posterior(samples=10)

    def test_missing_bindings_raises(self):
        # WHEN
        analysis = edyn.ParameterAnalysis(parameters=make_dataset())

        # EXPECT
        with pytest.raises(ValueError, match='No fit bindings'):
            analysis.sample_posterior(samples=10)


class TestParameterLabelEdgeCases:
    def test_colliding_names_are_qualified_by_model(self):
        # WHEN two bindings use models whose parameters share a name
        shared_name_model_a = sm.Polynomial(
            coefficients=[0.1, 0.35], x_unit='1/angstrom', y_unit='meV', name='Line'
        )
        shared_name_model_b = sm.Polynomial(
            coefficients=[2.0, -0.3], x_unit='1/angstrom', y_unit='meV', name='Line'
        )
        analysis = edyn.ParameterAnalysis(
            parameters=make_dataset(),
            bindings=[
                edyn.FitBinding(model=shared_name_model_a, targets='Lorentzian width'),
                edyn.FitBinding(model=shared_name_model_b, targets='Lorentzian area'),
            ],
        )

        # THEN
        parameters = analysis._get_chain_parameters()
        names = [p.name for p in parameters]
        labels = [analysis.parameter_label(p) for p in parameters]

        # EXPECT the bare names collide, and the labels resolve it
        assert len(set(names)) < len(names)
        assert len(set(labels)) == len(labels)

    def test_single_binding_keeps_plain_names(self):
        # WHEN
        analysis = make_analysis(two_bindings=False)

        # EXPECT no model prefix, since there is nothing to disambiguate
        labels = [analysis.parameter_label(p) for p in analysis._get_chain_parameters()]
        assert labels == ['Width line_c0', 'Width line_c1']

    def test_parameter_from_outside_the_analysis_keeps_its_name(self, analysis):
        # WHEN a parameter belongs to none of the binding models
        from easyscience.variable import Parameter

        stranger = Parameter(name='Width line_c0', value=1.0)

        # EXPECT it is returned unqualified rather than mislabelled
        assert analysis.parameter_label(stranger) == 'Width line_c0'

    def test_models_without_a_display_name_fall_back_to_the_unique_name(self):
        # WHEN two colliding models have no display name to tell them apart
        model_a = sm.Polynomial(coefficients=[0.1, 0.35], x_unit='1/angstrom', y_unit='meV')
        model_b = sm.Polynomial(coefficients=[2.0, -0.3], x_unit='1/angstrom', y_unit='meV')
        analysis = edyn.ParameterAnalysis(
            parameters=make_dataset(),
            bindings=[
                edyn.FitBinding(model=model_a, targets='Lorentzian width'),
                edyn.FitBinding(model=model_b, targets='Lorentzian area'),
            ],
        )

        # THEN
        labels = [analysis.parameter_label(p) for p in analysis._get_chain_parameters()]

        # EXPECT still unambiguous, which is what matters
        assert len(set(labels)) == len(labels)

    def test_colliding_names_with_distinct_models_use_the_display_name(self):
        # WHEN two diffusion models are bound to different targets. Their parameters are not named
        # after the model, so the names collide while the model names do not.
        analysis = edyn.ParameterAnalysis(
            parameters=make_dataset(),
            bindings=[
                edyn.FitBinding(
                    model=sm.BrownianTranslationalDiffusion(
                        name='Diffusion A', diffusion_coefficient=2.4e-9, scale=0.5
                    ),
                    targets={'width': 'Lorentzian width'},
                ),
                edyn.FitBinding(
                    model=sm.BrownianTranslationalDiffusion(
                        name='Diffusion B', diffusion_coefficient=2.4e-9, scale=0.5
                    ),
                    targets={'area': 'Lorentzian area'},
                ),
            ],
        )

        # THEN
        parameters = analysis._get_chain_parameters()
        labels = [analysis.parameter_label(p) for p in parameters]

        # EXPECT the model's name resolves the collision
        assert len({p.name for p in parameters}) < len(parameters)
        assert len(set(labels)) == len(labels)
        assert any(label.startswith('Diffusion A: ') for label in labels)
        assert any(label.startswith('Diffusion B: ') for label in labels)

    def test_ambiguous_name_owned_by_no_model_keeps_its_name(self):
        # WHEN a parameter shares an ambiguous name but belongs to none of the models
        from easyscience.variable import Parameter

        analysis = edyn.ParameterAnalysis(
            parameters=make_dataset(),
            bindings=[
                edyn.FitBinding(
                    model=sm.Polynomial(
                        coefficients=[0.1, 0.35], x_unit='1/angstrom', y_unit='meV', name='Line'
                    ),
                    targets='Lorentzian width',
                ),
                edyn.FitBinding(
                    model=sm.Polynomial(
                        coefficients=[2.0, -0.3], x_unit='1/angstrom', y_unit='meV', name='Line'
                    ),
                    targets='Lorentzian area',
                ),
            ],
        )
        stranger = Parameter(name='Line_c0', value=1.0)

        # EXPECT it falls back to the plain name rather than claiming an owner
        assert analysis.parameter_label(stranger) == 'Line_c0'


class TestInPlaceBindingEdits:
    def test_changing_the_number_of_targets_rebuilds_the_fitter(self):
        # WHEN a binding is edited in place so that it resolves to two targets instead of one.
        # ParameterAnalysis cannot observe this, and the cached fitter would otherwise still hold
        # one fit function against two datasets, which dies inside the minimizer.
        binding = edyn.FitBinding(
            model=sm.BrownianTranslationalDiffusion(
                name='Brownian',
                lorentzian_name='Lorentzian',
                diffusion_coefficient=2.4e-9,
                scale=0.5,
            ),
            targets={'width': 'Lorentzian width'},
        )
        analysis = edyn.ParameterAnalysis(parameters=make_dataset(), bindings=[binding])
        assert len(analysis.fit()) == 1

        binding.targets = {'width': 'Lorentzian width', 'area': 'Lorentzian area'}

        # EXPECT the fit follows the binding rather than failing on a stale fitter
        assert len(analysis.fit()) == 2

    def test_shrinking_the_targets_also_rebuilds(self):
        # WHEN
        binding = edyn.FitBinding(
            model=sm.BrownianTranslationalDiffusion(
                name='Brownian',
                lorentzian_name='Lorentzian',
                diffusion_coefficient=2.4e-9,
                scale=0.5,
            ),
            targets={'width': 'Lorentzian width', 'area': 'Lorentzian area'},
        )
        analysis = edyn.ParameterAnalysis(parameters=make_dataset(), bindings=[binding])
        assert len(analysis.fit()) == 2

        binding.targets = {'width': 'Lorentzian width'}

        # EXPECT
        assert len(analysis.fit()) == 1
