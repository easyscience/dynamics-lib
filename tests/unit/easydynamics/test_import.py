# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


def test_import_easydynamics():
    # WHEN THEN EXPECT
    import easydynamics

    assert easydynamics is not None
