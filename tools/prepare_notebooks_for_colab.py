# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Replace documentation setup with a Colab installer.

Run after HTML rendering on the built notebooks and the executed copies
used by Mike's rebuild. Outputs and widget metadata stay intact.
"""

import argparse
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path

import nbformat
from packaging.version import Version

PACKAGE_NAME = 'easydynamics'
SETUP_TAG = 'docs-static-plots'


def pip_specifier():
    """Pin releases and use an unpinned package for development docs."""
    try:
        installed = Version(version(PACKAGE_NAME))
    except PackageNotFoundError:
        return PACKAGE_NAME
    if installed.is_devrelease or installed.local or installed.major == 999:
        return PACKAGE_NAME
    return f'{PACKAGE_NAME}=={installed.public}'


def main():
    """Replace the setup cell after rendering documentation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    install_comment = (
        f'# Install {PACKAGE_NAME} if needed, for example when running in Google Colab.\n'
    )
    source = install_comment + (
        'import importlib.util\n'
        '\n'
        f"if importlib.util.find_spec('{PACKAGE_NAME}') is None:\n"
        f'    %pip install {pip_specifier()}'
    )
    notebooks = set()
    for path in args.paths:
        notebooks.update([path] if path.is_file() else path.rglob('*.ipynb'))
    for path in sorted(notebooks):
        if path.suffix != '.ipynb':
            continue
        if '.ipynb_checkpoints' in path.parts:
            continue
        notebook = nbformat.read(path, as_version=4)
        authored_cells = []
        installer = None
        for cell in notebook.cells:
            tags = cell.metadata.get('tags', [])
            # The header stays the same across package versions.
            is_installer = (
                cell.cell_type == 'code'
                and 'hide-in-docs' in tags
                and cell.source.startswith(install_comment)
            )
            if SETUP_TAG in tags or is_installer:
                if installer is None:
                    installer = cell
            else:
                authored_cells.append(cell)

        if installer is None:
            installer = nbformat.v4.new_code_cell()
        installer.update(
            source=source,
            metadata={'tags': ['hide-in-docs']},
            execution_count=None,
            outputs=[],
        )
        notebook.cells = [installer, *authored_cells]
        nbformat.write(notebook, path)


if __name__ == '__main__':
    main()
