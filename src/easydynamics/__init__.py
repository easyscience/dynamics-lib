# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
EasyDynamics library.

Everything public is re-exported here, so ``import easydynamics as edyn`` reaches all of it and a
reader never has to look up which sub-package a name came from. The sub-packages remain importable
for anyone who prefers them; this is only the front door.
"""

from easydynamics.analysis import Analysis
from easydynamics.analysis import BoundsSuggestion
from easydynamics.analysis import BoundsSuggestions
from easydynamics.analysis import MultiQPosteriorSampler
from easydynamics.analysis import ParameterAnalysis
from easydynamics.analysis import ParameterLabels
from easydynamics.analysis import ParameterPosterior
from easydynamics.analysis import PosteriorSampler
from easydynamics.analysis import PosteriorSummary
from easydynamics.analysis.analysis1d import Analysis1d
from easydynamics.analysis.fit_binding import FitBinding
from easydynamics.base_classes import EasyDynamicsBase
from easydynamics.base_classes import EasyDynamicsModelBase
from easydynamics.convolution import Convolution
from easydynamics.experiment import Experiment
from easydynamics.sample_model import BackgroundModel
from easydynamics.sample_model import BrownianTranslationalDiffusion
from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import DampedHarmonicOscillator
from easydynamics.sample_model import DeltaFunction
from easydynamics.sample_model import DeltaLorentz
from easydynamics.sample_model import Exponential
from easydynamics.sample_model import ExpressionComponent
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import InstrumentModel
from easydynamics.sample_model import JumpTranslationalDiffusion
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial
from easydynamics.sample_model import ResolutionModel
from easydynamics.sample_model import SampleModel
from easydynamics.sample_model import Voigt
from easydynamics.settings import ConvolutionSettings
from easydynamics.settings import DetailedBalanceSettings
from easydynamics.utils import detailed_balance_factor
from easydynamics.utils import plot_corner
from easydynamics.utils import plot_posterior_predictive
from easydynamics.utils import plot_trace
from easydynamics.utils import slicerplot_with_residuals
from easydynamics.utils.utils import hbar

__all__ = [
    'Analysis',
    'Analysis1d',
    'BackgroundModel',
    'BoundsSuggestion',
    'BoundsSuggestions',
    'BrownianTranslationalDiffusion',
    'ComponentCollection',
    'Convolution',
    'ConvolutionSettings',
    'DampedHarmonicOscillator',
    'DeltaFunction',
    'DeltaLorentz',
    'DetailedBalanceSettings',
    'EasyDynamicsBase',
    'EasyDynamicsModelBase',
    'Experiment',
    'Exponential',
    'ExpressionComponent',
    'FitBinding',
    'Gaussian',
    'InstrumentModel',
    'JumpTranslationalDiffusion',
    'Lorentzian',
    'MultiQPosteriorSampler',
    'ParameterAnalysis',
    'ParameterLabels',
    'ParameterPosterior',
    'Polynomial',
    'PosteriorSampler',
    'PosteriorSummary',
    'ResolutionModel',
    'SampleModel',
    'Voigt',
    'detailed_balance_factor',
    'hbar',
    'plot_corner',
    'plot_posterior_predictive',
    'plot_trace',
    'slicerplot_with_residuals',
]
