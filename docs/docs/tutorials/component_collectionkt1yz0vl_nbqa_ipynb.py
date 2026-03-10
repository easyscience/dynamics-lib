# %%NBQA-CELL-SEPdb06d8
import matplotlib.pyplot as plt
import numpy as np

from easydynamics.sample_model import ComponentCollection
from easydynamics.sample_model import DampedHarmonicOscillator
from easydynamics.sample_model import Gaussian
from easydynamics.sample_model import Lorentzian
from easydynamics.sample_model import Polynomial

hash(0x1261347A)

# %%NBQA-CELL-SEPdb06d8
component_collection = ComponentCollection()

# Creating components
gaussian = Gaussian(display_name='Gaussian', width=0.5, area=1)
dho = DampedHarmonicOscillator(display_name='DHO', center=1.0, width=0.3, area=2.0)
lorentzian = Lorentzian(display_name='Lorentzian', center=-1.0, width=0.2, area=1.0)
polynomial = Polynomial(display_name='Polynomial', coefficients=[0.1, 0, 0.5])  # y=0.1+0.5*x^2

# Adding components to the component collection
component_collection.append_component(gaussian)
component_collection.append_component(dho)
component_collection.append_component(lorentzian)
component_collection.append_component(polynomial)

x = np.linspace(-2, 2, 100)

plt.figure()
y = component_collection.evaluate(x)
plt.plot(x, y, label='Component collection')

for component in component_collection.components:
    y = component.evaluate(x)
    plt.plot(x, y, label=component.display_name)

plt.legend()
plt.show()
