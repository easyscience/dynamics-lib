# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Add hidden widget setup before executing documentation notebooks."""

import argparse
from pathlib import Path

import nbformat

SETUP_TAG = 'docs-static-plots'
WIDGET_SETUP = """\
from base64 import b64decode
from ipywidgets import Image, Widget
from ipympl.backend_nbagg import Canvas
from matplotlib.backend_bases import Event

# Hide informational download messages if this project uses Pooch.
try:
    import pooch
except ImportError:
    pass
else:
    pooch.get_logger().setLevel('WARNING')

def snapshot_canvases(widget):
    if isinstance(widget, Canvas):
        # Include hover-only controls and preserve the figure's display dimensions.
        widget.callbacks.process('figure_enter_event', Event('figure_enter_event', widget))
        data = widget._repr_mimebundle_()
        width, height = widget._size
        return Image(value=b64decode(data['image/png']), width=int(width), height=int(height))
    if hasattr(widget, 'children'):
        widget.children = tuple(snapshot_canvases(child) for child in widget.children)
    return widget

def widget_mimebundle(widget):
    data = widget._repr_mimebundle_()
    snapshot = snapshot_canvases(widget)
    return data if snapshot is widget else snapshot._repr_mimebundle_()

get_ipython().display_formatter.mimebundle_formatter.for_type(Widget, widget_mimebundle)
"""


def main():
    """Insert one hidden setup cell without changing authored cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    notebooks = set()
    for path in args.paths:
        notebooks.update([path] if path.is_file() else path.rglob('*.ipynb'))
    notebooks = sorted(
        path
        for path in notebooks
        if path.suffix == '.ipynb' and '.ipynb_checkpoints' not in path.parts
    )
    if not notebooks:
        parser.error('No documentation notebooks found.')

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        notebook.cells = [
            cell for cell in notebook.cells if SETUP_TAG not in cell.metadata.get('tags', [])
        ]
        notebook.cells.insert(
            0,
            nbformat.v4.new_code_cell(
                WIDGET_SETUP,
                metadata={'tags': ['hide-in-docs', SETUP_TAG]},
            ),
        )
        nbformat.write(notebook, path)


if __name__ == '__main__':
    main()
