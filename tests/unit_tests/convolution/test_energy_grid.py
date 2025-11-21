import numpy as np

from easydynamics.convolution.energy_grid import EnergyGrid


class TestEnergyGrid:
    def test_energy_grid_attributes(self):
        energy_dense = np.array([-1.0, 0.0, 1.0])
        energy_dense_centered = np.array([-1.0, 0.0, 1.0])
        energy_dense_step = 1.0
        energy_span_dense = 2.0
        energy_even_length_offset = 0.0

        energy_grid = EnergyGrid(
            energy_dense=energy_dense,
            energy_span_dense=energy_span_dense,
            energy_even_length_offset=energy_even_length_offset,
            energy_dense_centered=energy_dense_centered,
            energy_dense_step=energy_dense_step,
        )

        assert np.array_equal(energy_grid.energy_dense, energy_dense)
        assert np.array_equal(energy_grid.energy_dense_centered, energy_dense_centered)
        assert energy_grid.energy_dense_step == energy_dense_step
        assert energy_grid.energy_span_dense == energy_span_dense
        assert energy_grid.energy_even_length_offset == energy_even_length_offset

    # energy_dense: np.ndarray
    # energy_dense_centered: np.ndarray
    # energy_dense_step: float
    # span_dense: float
    # energy_even_length_offset: float
