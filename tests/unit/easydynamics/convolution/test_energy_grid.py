# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from easydynamics.convolution.energy_grid import EnergyGrid


class TestEnergyGrid:
    def test_energy_grid_attributes(self):
        # WHEN
        energy_dense = np.array([-1.0, 0.0, 1.0])
        energy_dense_centered = np.array([-1.0, 0.0, 1.0])
        energy_dense_step = 1.0
        energy_span_dense = 2.0
        energy_even_length_offset = 0.0

        # THEN
        energy_grid = EnergyGrid(
            energy_dense=energy_dense,
            energy_span_dense=energy_span_dense,
            energy_even_length_offset=energy_even_length_offset,
            energy_dense_centered=energy_dense_centered,
            energy_dense_step=energy_dense_step,
        )

        # EXPECT
        assert np.array_equal(energy_grid.energy_dense, energy_dense)
        assert np.array_equal(energy_grid.energy_dense_centered, energy_dense_centered)
        assert energy_grid.energy_dense_step == energy_dense_step
        assert energy_grid.energy_span_dense == energy_span_dense
        assert energy_grid.energy_even_length_offset == energy_even_length_offset
