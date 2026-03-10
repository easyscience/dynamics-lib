# %%NBQA-CELL-SEPdb06d8
import matplotlib.pyplot as plt
import numpy as np

from easydynamics.utils import _detailed_balance_factor as detailed_balance_factor

hash(0x513734FC)

# %%NBQA-CELL-SEPdb06d8
temperatures = [1, 10, 100]
temperature_unit = 'K'
energy = np.linspace(-1, 1, 100)
# energy=1.0
energy_unit = 'meV'

plt.figure()
for temperature in temperatures:
    DBF = detailed_balance_factor(energy, temperature, energy_unit, temperature_unit)
    plt.plot(energy, DBF, label=f'T={temperature} K')
plt.legend()
plt.xlabel('Energy transfer (meV)')
plt.ylabel('Detailed balance factor')
plt.title(
    'Detailed balance factor for different temperatures, \n '
    'normalized to 1 at zero energy transfer'
)
plt.show()

# %%NBQA-CELL-SEPdb06d8
temperatures = [1, 10, 100]
temperature_unit = 'K'
energy = np.linspace(-1, 1, 100)
# energy=1.0
energy_unit = 'meV'

plt.figure()
for temperature in temperatures:
    DBF = detailed_balance_factor(
        energy, temperature, energy_unit, temperature_unit, divide_by_temperature=False
    )
    plt.plot(energy, DBF, label=f'T={temperature} K')
plt.legend()
plt.xlabel('Energy transfer (meV)')
plt.ylabel('Detailed balance factor')
plt.title('Detailed balance factor for different temperatures, not normalized')
plt.show()
