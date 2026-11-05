# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import copy

from easydynamics.variable import DescriptorNumber


class TestDescriptorNumber:
    #############
    # Display name
    #############

    def test_display_name_falls_back_to_name(self):
        # WHEN THEN
        variable = DescriptorNumber(1.0, name='area')

        # EXPECT
        assert variable.display_name == 'area'

    def test_explicit_display_name_is_kept(self):
        # WHEN THEN
        variable = DescriptorNumber(1.0, name='area', display_name='Peak area')

        # EXPECT
        assert variable.name == 'area'
        assert variable.display_name == 'Peak area'

    def test_copy_keeps_name_and_display_name(self):
        # WHEN
        variable = DescriptorNumber(1.0, name='area')

        # THEN
        variable_copy = copy(variable)

        # EXPECT
        assert variable_copy.name == 'area'
        assert variable_copy.display_name == 'area'
