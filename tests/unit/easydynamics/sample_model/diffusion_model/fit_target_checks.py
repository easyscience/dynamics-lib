# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

Q = np.array([0.3, 0.7, 1.1, 1.5, 1.9])


def undeclared_parameters_with_effect(model, Q: np.ndarray = Q) -> dict[str, list[str]]:
    """
    For each fit target, list the free parameters it does not declare but whose change alters its
    output -- i.e. declarations that are missing.

    Each undeclared parameter is moved by 10% and the target re-evaluated, then restored.
    """
    missing = {}
    for target in model.get_fit_targets():
        declared = {parameter.unique_name for parameter in target.parameters}
        missing[target.name] = []
        for parameter in model.get_free_parameters():
            if parameter.unique_name in declared:
                continue
            before = np.asarray(target.function(Q))
            original = parameter.value
            parameter.value = original * 0.9 if original else 0.1
            after = np.asarray(target.function(Q))
            parameter.value = original
            if not np.allclose(before, after):
                missing[target.name].append(parameter.name)
    return missing


def declared_names(model) -> dict[str, list[str]]:
    """Map each fit target's name to the names of the parameters it declares."""
    return {
        target.name: [parameter.name for parameter in target.parameters]
        for target in model.get_fit_targets()
    }
