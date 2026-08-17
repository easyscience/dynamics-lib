# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Download every data file the tutorial notebooks fetch, once, before they are run.

The notebooks are executed in parallel with ``pytest -n auto``, and several of them fetch the same
file through ``pooch``. On a cold cache the workers race: one is still writing the file into the
cache while another tries to open it, which fails on Windows with a permission error. Fetching
everything up front leaves the parallel run with nothing to do but read.

Run as ``python tools/prefetch_tutorial_data.py``; it is wired into the ``notebook-tests`` task.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pooch

TUTORIALS = Path(__file__).resolve().parent.parent / 'docs' / 'docs' / 'tutorials'

# Matches one pooch.retrieve(...) call site; the notebooks keep these calls free of nested
# parentheses, so everything up to the first closing parenthesis is the argument list.
RETRIEVE_PATTERN = re.compile(r'pooch\.retrieve\s*\(([^)]*)\)')
# Matches the url=... and known_hash=... keyword arguments inside a single call. The optional
# ``f`` prefix on the URL is captured so templated URLs can be recognised and skipped.
URL_PATTERN = re.compile(r"url\s*=\s*(f?)['\"]([^'\"]+)['\"]")
HASH_PATTERN = re.compile(r"known_hash\s*=\s*['\"]([^'\"]+)['\"]")


def find_downloads() -> dict[str, str]:
    """
    Collect the ``(url, known_hash)`` pairs the notebooks fetch.

    Each ``pooch.retrieve(...)`` call site is parsed on its own, so a cell with several calls
    cannot pair one call's URL with another call's hash. Calls whose URL is an f-string, or that
    lack a literal ``known_hash``, are skipped rather than guessed at; the notebook will simply
    fetch those itself.

    Returns
    -------
    dict[str, str]
        Mapping of URL to expected hash, deduplicated across notebooks.
    """
    downloads: dict[str, str] = {}
    for notebook in sorted(TUTORIALS.rglob('*.ipynb')):
        if '.ipynb_checkpoints' in notebook.parts:
            continue
        cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
        for cell in cells:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            for call in RETRIEVE_PATTERN.finditer(source):
                arguments = call.group(1)
                url_match = URL_PATTERN.search(arguments)
                hash_match = HASH_PATTERN.search(arguments)
                if url_match is None or hash_match is None or url_match.group(1) == 'f':
                    continue
                downloads[url_match.group(2)] = hash_match.group(1)
    return downloads


def main() -> int:
    """
    Fetch every tutorial data file into the pooch cache.

    Deliberately never fails: this only warms a cache. A file that cannot be fetched here is left
    to the notebook that needs it, which reports the problem with far more context than this script
    could, and which is where the failure belongs.

    Returns
    -------
    int
        Always zero.
    """
    downloads = find_downloads()
    if not downloads:
        sys.stdout.write('No tutorial downloads found.\n')
        return 0

    failures = 0
    for url, known_hash in downloads.items():
        name = url.rsplit('/', 1)[-1]
        try:
            pooch.retrieve(url=url, known_hash=known_hash)
        except Exception as error:  # ruff: ignore[blind-except] - report and continue, the notebook will retry
            failures += 1
            sys.stdout.write(f'could not prefetch {name}, leaving it to the notebook: {error}\n')
        else:
            sys.stdout.write(f'cached  {name}\n')

    sys.stdout.write(f'{len(downloads) - failures}/{len(downloads)} tutorial data files ready.\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
