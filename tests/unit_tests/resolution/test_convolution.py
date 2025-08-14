import pytest
import numpy as np
from scipy.integrate import simpson

from easyscience.variable import Parameter
from easydynamics.sample import SampleModel, GaussianComponent, LorentzianComponent
from easydynamics.sample.components import ModelComponent
from easydynamics.utils import detailed_balance_factor

# class TestConvolution:
#     @pytest.fixture
#     def sample_model(self):
#         return SampleModel(name="TestSampleModel")

 