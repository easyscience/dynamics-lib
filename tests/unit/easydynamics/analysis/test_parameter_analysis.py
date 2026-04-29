# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import scipp as sc


class TestParameterAnalysis:
    @pytest.fixture
    def dataset(self):
        Q = sc.array(dims=['Q'], values=[0.1, 0.2])
        return sc.Dataset(
            data={
                'parameter1': sc.DataArray(
                    data=sc.array(dims=['Q'], values=[1.0, 2.0], variances=[0.1, 0.2], unit='meV'),
                    coords={'Q': Q},
                ),
                'parameter2': sc.DataArray(
                    data=sc.array(
                        dims=['Q'],
                        values=[1.5, 2.5],
                        variances=[0.15, 0.25],
                        unit='1/meV',
                    ),
                    coords={'Q': Q},
                ),
                'parameter3 area': sc.DataArray(
                    data=sc.array(dims=['Q'], values=[4.0, 5.0], variances=[0.3, 0.5], unit='meV'),
                    coords={'Q': Q},
                ),
                'parameter3 width': sc.DataArray(
                    data=sc.array(dims=['Q'], values=[6.0, 7.0], variances=[0.6, 0.7], unit='meV'),
                    coords={'Q': Q},
                ),
            }
        )

    # def test_parameter_property(self, parameter_analysis):
    #     # WHEN
    #     parameters = parameter_analysis.parameters

    #     # THEN EXPECT
    #     assert isinstance(parameters, sc.Dataset)
    #     assert set(parameters.keys()) == {
    #         'parameter1',
    #         'parameter2',
    #         'parameter3 area',
    #         'parameter3 width',
    #     }

    #     # WHEN
    #     Q = sc.array(dims=['Q'], values=[0.1, 0.2])
    #     new_data = sc.Dataset(
    #         data={
    #             'parameter4': sc.DataArray(
    #                 data=sc.array(
    #                     dims=['Q'],
    #                     values=[71.0, 12.0],
    #                     variances=[1.1, 2.2],
    #                     unit='meV',
    #                 ),
    #                 coords={'Q': Q},
    #             ),
    #             'parameter5': sc.DataArray(
    #                 data=sc.array(
    #                     dims=['Q'],
    #                     values=[8.5, 0.5],
    #                     variances=[2.15, 1.25],
    #                     unit='1/meV',
    #                 ),
    #                 coords={'Q': Q},
    #             ),
    #         }
    #     )

    #     # THEN
    #     parameter_analysis.parameters = new_data

    #     # EXPECT
    #     assert parameter_analysis.parameters is new_data
