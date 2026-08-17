# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the flat public namespace.

Tutorials and docstring examples reach everything through ``import easydynamics as edyn``, which
only works while the front door keeps up with the sub-packages. These check that it does.
"""

import importlib
import json
import pathlib
import re

import pytest

import easydynamics as edyn

SUB_PACKAGES = [
    'easydynamics.analysis',
    'easydynamics.base_classes',
    'easydynamics.convolution',
    'easydynamics.experiment',
    'easydynamics.sample_model',
    'easydynamics.sample_model.components',
    'easydynamics.sample_model.diffusion_model',
    'easydynamics.settings',
    'easydynamics.utils',
]

TUTORIALS = pathlib.Path(__file__).resolve().parents[3] / 'docs' / 'docs' / 'tutorials'


class TestFrontDoor:
    def test_import_easydynamics(self):
        # WHEN THEN EXPECT: importing raises no error
        import easydynamics  # ruff: ignore[unused-import]

    def test_everything_declared_is_importable(self):
        # THEN EXPECT no name in __all__ that cannot actually be reached
        missing = [name for name in edyn.__all__ if not hasattr(edyn, name)]
        assert missing == []

    @pytest.mark.parametrize('module_name', SUB_PACKAGES)
    def test_sub_package_exports_are_re_exported(self, module_name):
        # THEN
        module = importlib.import_module(module_name)

        # EXPECT anything public in a sub-package is on the front door too, so a tutorial never
        # has to import from the sub-package to reach it
        missing = [name for name in getattr(module, '__all__', []) if name not in edyn.__all__]
        assert missing == [], f'{module_name} exports not re-exported: {missing}'

    def test_re_exports_are_the_same_objects(self):
        # WHEN
        from easydynamics.sample_model import Gaussian

        # THEN EXPECT the front door is an alias, not a copy
        assert edyn.Gaussian is Gaussian

    def test_all_is_sorted_and_unique(self):
        # THEN EXPECT a list that stays easy to scan and cannot hide a duplicate
        assert edyn.__all__ == sorted(edyn.__all__)
        assert len(edyn.__all__) == len(set(edyn.__all__))


class TestTutorialImportStyle:
    @pytest.mark.parametrize('notebook', sorted(TUTORIALS.glob('*.ipynb')), ids=lambda p: p.name)
    def test_notebooks_use_only_the_flat_namespace(self, notebook):
        # THEN
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        imports = [
            line.strip()
            for cell in cells
            if cell['cell_type'] == 'code'
            for line in ''.join(cell['source']).splitlines()
            if re.match(r'^\s*(import|from)\s+easydynamics', line)
        ]

        # EXPECT one way in, so a reader never has to scroll back to find where a name came from
        assert set(imports) <= {'import easydynamics as edyn'}, (
            f'{notebook.name} imports EasyDynamics some other way: {imports}'
        )
