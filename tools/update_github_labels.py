"""
Set/update GitHub labels for current or specified easyscience
repository.

Requires:
  - gh CLI installed
  - gh auth login completed

Usage:
  python update_github_labels.py
  python update_github_labels.py --dry-run
  python update_github_labels.py --repo easyscience/my-repo
  python update_github_labels.py --repo easyscience/my-repo --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


EASYSCIENCE_ORG = 'easyscience'


# --- Label definitions -----------------------------------------------------------

BASIC_GITHUB_LABELS = [
    'bug',
    'documentation',
    'duplicate',
    'enhancement',
    'good first issue',
    'help wanted',
    'invalid',
    'question',
    'wontfix',
]

NEW_BASIC_LABEL_NAMES = [
    '[scope] bug',
    '[scope] documentation',
    '[maintainer] duplicate',
    '[scope] enhancement',
    '[maintainer] good first issue',
    '[maintainer] help wanted',
    '[maintainer] invalid',
    '[maintainer] question',
    '[maintainer] wontfix',
]

SCOPE_LABELS = [
    ('bug', 'Bug report or fix (major.minor.PATCH)'),
    ('documentation', 'Documentation only changes (major.minor.patch.POST)'),
    ('enhancement', 'Adds/improves features (major.MINOR.patch)'),
    ('maintenance', 'Code/tooling cleanup, no feature or bugfix (major.minor.PATCH)'),
    ('significant', 'Breaking or major changes (MAJOR.minor.patch)'),
    ('⚠️ label needed', 'Automatically added to issues and PRs without a [scope] label'),
]

MAINTAINER_LABELS = [
    ('duplicate', 'Already reported or submitted'),
    ('good first issue', 'Good entry-level issue for newcomers'),
    ('help wanted', 'Needs additional help to resolve or implement'),
    ('invalid', 'Invalid, incorrect or outdated'),
    ('question', 'Needs clarification, discussion, or more information'),
    ('wontfix', 'Will not be fixed or continued'),
]

PRIORITY_LABELS = [
    ('lowest', 'Very low urgency'),
    ('low', 'Low importance'),
    ('medium', 'Normal/default priority'),
    ('high', 'Should be prioritized soon'),
    ('highest', 'Urgent. Needs attention ASAP'),
    ('⚠️ label needed', 'Automatically added to issues without a [priority] label'),
]

BOT_LABEL = (
    '[bot] pull request',
    'Automated release PR. Excluded from changelog/versioning',
)

COLORS = {
    'scope': 'd73a4a',
    'maintainer': '0e8a16',
    'priority': 'fbca04',
    'bot': '5319e7',
}


# --- Helpers --------------------------------------------------------------------


@dataclass(frozen=True)
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def run_cmd(args: list[str], *, dry_run: bool, check: bool = True) -> CmdResult:
    """Run a command (or print it in dry-run mode)."""
    cmd_str = ' '.join(shlex.quote(a) for a in args)

    if dry_run:
        print(f'{cmd_str}')
        return CmdResult(0, '', '')

    proc = subprocess.run(
        args,
        text=True,
        capture_output=True,
    )
    res = CmdResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())

    if check and proc.returncode != 0:
        raise RuntimeError(f'Command failed ({proc.returncode}): {cmd_str}\n{res.stderr}')

    return res


def get_current_repo_name_with_owner() -> str:
    res = subprocess.run(
        ['gh', 'repo', 'view', '--json', 'nameWithOwner'],
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(res.stdout)
    nwo = data.get('nameWithOwner')
    if not nwo or '/' not in nwo:
        raise RuntimeError('Could not determine current repository name')
    return nwo


def try_rename_label(repo: str, old: str, new: str, *, dry_run: bool) -> None:
    try:
        run_cmd(
            ['gh', 'label', 'edit', old, '--name', new, '--repo', repo],
            dry_run=dry_run,
        )
        print(f'Rename: {old!r} → {new!r}')
    except Exception:
        print(f'Skip rename (label not found): {old!r}')


def upsert_label(
    repo: str,
    name: str,
    color: str,
    description: str,
    *,
    dry_run: bool,
) -> None:
    run_cmd(
        [
            'gh',
            'label',
            'create',
            name,
            '--color',
            color,
            '--description',
            description,
            '--force',
            '--repo',
            repo,
        ],
        dry_run=dry_run,
    )
    print(f'Upsert label: {name!r}')


def upsert_group(
    repo: str,
    prefix: str,
    color: str,
    items: Iterable[tuple[str, str]],
    *,
    dry_run: bool,
) -> None:
    for short, desc in items:
        upsert_label(
            repo,
            f'[{prefix}] {short}',
            color,
            desc,
            dry_run=dry_run,
        )


# --- Main -----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description='Sync GitHub labels for easyscience repos')
    parser.add_argument(
        '--repo',
        help='Target repository in the form easyscience/<name>',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print actions without applying changes',
    )
    args = parser.parse_args()

    if args.repo:
        repo = args.repo
    else:
        repo = get_current_repo_name_with_owner()

    org, _ = repo.split('/', 1)

    if org.lower() != EASYSCIENCE_ORG:
        print(
            f"Refusing to run: repository {repo!r} is not under '{EASYSCIENCE_ORG}'.",
            file=sys.stderr,
        )
        return 2

    print(f'Target repository: {repo}')
    if args.dry_run:
        print('Running in DRY-RUN mode (no changes will be made)\n')

    # 1) Rename basic labels
    for old, new in zip(BASIC_GITHUB_LABELS, NEW_BASIC_LABEL_NAMES, strict=True):
        try_rename_label(repo, old, new, dry_run=args.dry_run)

    # 2) Scope / Maintainer / Priority groups
    upsert_group(repo, 'scope', COLORS['scope'], SCOPE_LABELS, dry_run=args.dry_run)
    upsert_group(repo, 'maintainer', COLORS['maintainer'], MAINTAINER_LABELS, dry_run=args.dry_run)
    upsert_group(repo, 'priority', COLORS['priority'], PRIORITY_LABELS, dry_run=args.dry_run)

    # 3) Bot label
    upsert_label(
        repo,
        BOT_LABEL[0],
        COLORS['bot'],
        BOT_LABEL[1],
        dry_run=args.dry_run,
    )

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
