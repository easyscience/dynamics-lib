# %%NBQA-CELL-SEP5932b7
import matplotlib.pyplot as plt
import numpy as np

from easydynamics.sample_model import BrownianTranslationalDiffusion

hash(0x4D38875)

# %%NBQA-CELL-SEP5932b7
# Create Brownian Translational Diffusion model
# and plot the model for different Q values.
# Q is in Angstrom^-1 and energy in meV.

Q = np.linspace(0.5, 2, 7)
energy = np.linspace(-2, 2, 501)
scale = 1.0
diffusion_coefficient = 2.4e-9  # m^2/s

diffusion_model = BrownianTranslationalDiffusion(
    display_name='DiffusionModel',
    scale=scale,
    diffusion_coefficient=diffusion_coefficient,
)

component_collections = diffusion_model.create_component_collections(Q)


cmap = plt.cm.jet
nQ = len(component_collections)
plt.figure()
for Q_index in range(len(component_collections)):
    color = cmap(Q_index / (nQ - 1))
    y = component_collections[Q_index].evaluate(energy)
    plt.plot(energy, y, label=f'Q={Q[Q_index]} Å^-1', color=color)

plt.legend()
plt.show()
plt.xlabel('Energy (meV)')
plt.ylabel('Intensity (arb. units)')
plt.title('Brownian Translational Diffusion Model')

# %%NBQA-CELL-SEP5932b7
# Calculate and plot the half width at half maximum (HWHM) as function
# of Q
Q = np.linspace(0.1, 2, 101)
HWHM = diffusion_model.calculate_width(Q)
plt.figure()
plt.plot(Q, HWHM)
plt.xlabel('Q (Å$^{-1}$)')
plt.ylabel('HWHM (meV)')
plt.xlim(0, 2.5)
plt.ylim(0, max(HWHM) * 1.1)
plt.title('HWHM vs Q for Brownian Translational Diffusion')
