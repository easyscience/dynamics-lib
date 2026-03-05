# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest


@pytest.mark.fast
def test_import_easydynamics():
    import easydynamics

    assert easydynamics is not None
