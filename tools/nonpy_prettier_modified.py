from __future__ import annotations

import shutil
import subprocess
import sys


def get_stdout_lines(cmd: list[str]) -> list[str]:
    """Execute a command and return its standard output as a list of
    non-empty lines.

    The command is executed with `check=True`, so a non-zero exit code
    will raise `subprocess.CalledProcessError`.

    Args:
        cmd: Command and arguments to execute, as a list of strings.

    Returns:
        A list of stripped, non-empty lines from the command's stdout.
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    """Run Prettier on modified or newly added non-Python files in the
    Git repository.

    The file list is constructed from:
      - tracked files that are added or modified (staged or unstaged)
      - untracked files that are not ignored by Git

    Prettier is always run with `--list-different`.
    If `--write` is provided as a command-line argument, files are fixed
    in place in addition to being listed.

    Returns:
        Exit code of the Prettier process, or 0 if there are no files to
        process.
    """
    # Tracked added/modified (staged or not)
    files = get_stdout_lines(['git', 'diff', '--name-only', '--diff-filter=AM', 'HEAD'])

    # Untracked new files
    files += get_stdout_lines(['git', 'ls-files', '--others', '--exclude-standard'])

    if not files:
        return 0  # nothing to do

    # Locate npx executable (npx.cmd on Windows)
    npx = shutil.which('npx') or shutil.which('npx.cmd')
    if not npx:
        print('ERROR: npx not found in PATH', file=sys.stderr)
        return 2

    # Windows requires running .cmd files via the shell
    need_shell = npx.lower().endswith('.cmd')

    cmd = [
        npx,
        'prettier',
        '--list-different',
    ]

    # Optionally enable fixing
    if '--write' in sys.argv:
        cmd.append('--write')

    cmd += [
        '--ignore-unknown',
        '--config=prettierrc.toml',
        '--',
        *files,
    ]

    proc = subprocess.run(cmd, shell=need_shell)
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
