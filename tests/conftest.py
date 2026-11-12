# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib as mpl

# Render with the non-interactive Agg backend, whatever the machine's default is. Under an
# interactive default (e.g. TkAgg on Windows), every figure plopp creates opens a GUI window,
# which makes tests that render for real fail intermittently with Tcl errors.
mpl.use('Agg')
