"""Check and fix consistency between Parameter/Descriptor
definitions and their public property docstrings and type
annotations.

Usage::

    python param_consistency.py --check
    python param_consistency.py --fix
    python param_consistency.py src/mypackage/ --check

Template (see architecture.md §9.8 for the full spec)
------------------------------------------------------
Given ``description='Length of the a axis of the unit
cell.'``, ``units='Å'``, and type ``Parameter``:

Writable::

    @property
    def length_a(self) -> Parameter:
        \"""Length of the a axis of the unit cell (Å).

        Reading this property returns the underlying
        ``Parameter`` object. Assigning to it updates
        the parameter value.
        \"""
        return self._length_a

    @length_a.setter
    def length_a(self, value: float) -> None:
        self._length_a.value = value

Read-only::

    @property
    def length_a(self) -> Parameter:
        \"""Length of the a axis of the unit cell (Å).

        Reading this property returns the underlying
        ``Parameter`` object.
        \"""
        return self._length_a

Rules:

- ``{desc}`` = description without trailing period
  (single source of truth).
- ``{units}`` = units string; omit ``({units})`` when
  absent or empty.
- Getter summary: ``{desc} ({units}).`` or ``{desc}.``
- Getter body mentions the descriptor class and, for
  writable properties, notes that assignment updates
  the value.
- Setter has **no** docstring.
- Getter return annotation: the descriptor class name.
- Setter value annotation: ``float`` for numeric,
  ``str`` for string.
- Setter return annotation: ``None``.

Exit code 0 when all checks pass (or fix succeeds),
1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

_SRC_ROOT = (
    Path(__file__).resolve().parents[1]
    / 'src'
    / 'easydiffraction'
)

_DESCRIPTOR_TYPES = frozenset(
    {'Parameter', 'NumericDescriptor', 'StringDescriptor'}
)

# Canonical setter value annotation per descriptor family.
_SETTER_ANN: dict[str, str] = {
    'Parameter': 'float',
    'NumericDescriptor': 'float',
    'StringDescriptor': 'str',
}


# ---------------------------------------------------------
# Data structures
# ---------------------------------------------------------


@dataclass
class DescriptorInfo:
    """Descriptor definition from ``__init__``."""

    attr_name: str  # e.g. '_length_a'
    prop_name: str  # e.g. 'length_a'
    type_name: str  # 'Parameter' | …
    description: str  # e.g. 'Length of …'
    units: str | None  # e.g. 'Å', or None


@dataclass
class PropertyInfo:
    """Property getter / setter AST nodes."""

    name: str
    getter: ast.FunctionDef
    setter: ast.FunctionDef | None = None


@dataclass
class Edit:
    """A source-level edit.

    Replace ``lines[start:end]`` with *new_text*.
    When ``start == end`` the edit is an insertion
    before that line.
    """

    start: int  # 0-based inclusive
    end: int  # 0-based exclusive
    new_text: str


@dataclass
class FileResult:
    """Analysis result for one source file."""

    path: Path
    issues: list[str] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)


# ---------------------------------------------------------
# Template helpers
# ---------------------------------------------------------


def _strip_dot(s: str) -> str:
    """Remove trailing period and whitespace."""
    s = s.rstrip()
    if s.endswith('.'):
        s = s[:-1].rstrip()
    return s


def _getter_docstring(
    desc: str,
    units: str | None,
    type_name: str,
    has_setter: bool,
    indent: str,
) -> str:
    """Build the expected getter docstring."""
    d = _strip_dot(desc)
    summary = f'{d} ({units}).' if units else f'{d}.'

    if has_setter:
        body = (
            f'Reading this property returns the underlying '
            f'``{type_name}`` object. '
            f'Assigning to it updates the parameter value.'
        )
    else:
        body = (
            f'Reading this property returns the underlying '
            f'``{type_name}`` object.'
        )

    return (
        f'{indent}"""{summary}\n'
        f'\n'
        f'{indent}{body}\n'
        f'{indent}"""\n'
    )


def _normalize(text: str) -> str:
    """Collapse whitespace for comparison."""
    return ' '.join(text.split()).lower()


# ---------------------------------------------------------
# AST helpers
# ---------------------------------------------------------


def _call_name(node: ast.Call) -> str | None:
    """Return the simple name of a Call's func."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _kwarg_str(
    call: ast.Call,
    name: str,
) -> str | None:
    """Extract a string keyword argument."""
    for kw in call.keywords:
        if (
            kw.arg == name
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return None


def _ann_str(ann: ast.expr | None) -> str | None:
    """Return annotation as a source string."""
    if ann is None:
        return None
    if isinstance(ann, ast.Name):
        return ann.id
    if isinstance(ann, ast.Constant) and isinstance(
        ann.value, str
    ):
        return ann.value  # forward reference
    return ast.unparse(ann)


def _body_indent(
    func: ast.FunctionDef,
    lines: list[str],
) -> str:
    """Compute the indentation for the body."""
    def_line = lines[func.lineno - 1]
    return ' ' * (
        len(def_line) - len(def_line.lstrip()) + 4
    )


def _def_line_range(
    func: ast.FunctionDef,
    lines: list[str],
) -> tuple[int, int]:
    """Return 0-based [start, end) of the def."""
    start = func.lineno - 1
    for i in range(start, min(start + 10, len(lines))):
        if lines[i].rstrip().endswith(':'):
            return start, i + 1
        if func.body and i + 1 >= func.body[0].lineno:
            break
    return start, start + 1


def _docstring_range(
    func: ast.FunctionDef,
) -> tuple[str | None, int, int]:
    """Return (text, start_0, end_exclusive_0)."""
    if not func.body:
        return None, -1, -1
    first = func.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        # end_lineno is 1-based inclusive
        return (
            first.value.value,
            first.lineno - 1,
            first.end_lineno,
        )
    return None, -1, -1


# ---------------------------------------------------------
# Extraction
# ---------------------------------------------------------


def _extract_descriptors(
    cls: ast.ClassDef,
) -> dict[str, DescriptorInfo]:
    """Find self._xxx = DescriptorType(...) in init."""
    result: dict[str, DescriptorInfo] = {}

    init = next(
        (
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == '__init__'
        ),
        None,
    )
    if init is None:
        return result

    for stmt in ast.walk(init):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
        ):
            target = stmt.targets[0]
            value = stmt.value
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
        ):
            target = stmt.target
            value = stmt.value
        else:
            continue

        # Target must be self._xxx
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == 'self'
            and target.attr.startswith('_')
        ):
            continue

        if not isinstance(value, ast.Call):
            continue

        name = _call_name(value)
        if name not in _DESCRIPTOR_TYPES:
            continue

        desc_str = _kwarg_str(value, 'description')
        if not desc_str or not _strip_dot(desc_str):
            continue

        units = _kwarg_str(value, 'units') or None
        prop = target.attr.lstrip('_')
        result[prop] = DescriptorInfo(
            target.attr, prop, name, desc_str, units
        )

    return result


def _extract_properties(
    cls: ast.ClassDef,
) -> dict[str, PropertyInfo]:
    """Find property getters and setters."""
    result: dict[str, PropertyInfo] = {}

    for item in cls.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        for dec in item.decorator_list:
            # @property
            if (
                isinstance(dec, ast.Name)
                and dec.id == 'property'
            ):
                result[item.name] = PropertyInfo(
                    item.name, item
                )
                break
            # @xxx.setter
            if (
                isinstance(dec, ast.Attribute)
                and dec.attr == 'setter'
                and isinstance(dec.value, ast.Name)
                and dec.value.id in result
            ):
                result[dec.value.id].setter = item
                break

    return result


# ---------------------------------------------------------
# Analysis (shared by --check and --fix)
# ---------------------------------------------------------


def _analyze_file(path: Path) -> FileResult:
    """Analyze one file, return issues and edits."""
    result = FileResult(path)
    try:
        source = path.read_text(encoding='utf-8')
    except Exception:  # noqa: BLE001
        return result

    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return result

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        descriptors = _extract_descriptors(node)
        properties = _extract_properties(node)

        for prop_name, prop in properties.items():
            if prop_name not in descriptors:
                continue
            desc = descriptors[prop_name]
            _analyze_property(
                node.name, prop, desc, lines, result
            )

    return result


def _analyze_property(
    cls_name: str,
    prop: PropertyInfo,
    desc: DescriptorInfo,
    lines: list[str],
    result: FileResult,
) -> None:
    """Check one property against the template."""
    loc = f'{cls_name}.{prop.name}'
    indent = _body_indent(prop.getter, lines)
    has_setter = prop.setter is not None

    # --- Getter return annotation ---
    actual_ret = _ann_str(prop.getter.returns)
    expected_ret = desc.type_name
    if actual_ret != expected_ret:
        result.issues.append(
            f'{loc}: getter annotation '
            f'-> {actual_ret} (expected {expected_ret})'
        )
        ds, de = _def_line_range(prop.getter, lines)
        def_indent = lines[ds][
            : len(lines[ds]) - len(lines[ds].lstrip())
        ]
        new_def = (
            f'{def_indent}def {prop.name}'
            f'(self) -> {expected_ret}:\n'
        )
        result.edits.append(Edit(ds, de, new_def))

    # --- Getter docstring ---
    expected_doc = _getter_docstring(
        desc.description,
        desc.units,
        desc.type_name,
        has_setter,
        indent,
    )
    actual_doc_text, doc_s, doc_e = _docstring_range(
        prop.getter
    )

    if actual_doc_text is None:
        result.issues.append(
            f'{loc}: getter missing docstring'
        )
        _, def_end = _def_line_range(prop.getter, lines)
        result.edits.append(
            Edit(def_end, def_end, expected_doc)
        )
    else:
        actual_src = ''.join(lines[doc_s:doc_e])
        if _normalize(actual_src) != _normalize(
            expected_doc
        ):
            result.issues.append(
                f'{loc}: getter docstring '
                'does not match template'
            )
            result.edits.append(
                Edit(doc_s, doc_e, expected_doc)
            )

    # --- Setter ---
    if prop.setter is None:
        return

    # Setter def-line annotations
    setter_args = prop.setter.args.args
    setter_param = (
        setter_args[1].arg
        if len(setter_args) >= 2
        else 'value'
    )
    expected_ann = _SETTER_ANN[desc.type_name]

    actual_val_ann = None
    if (
        len(setter_args) >= 2
        and setter_args[1].annotation
    ):
        actual_val_ann = _ann_str(
            setter_args[1].annotation
        )

    actual_ret_ann = _ann_str(prop.setter.returns)

    if (
        actual_val_ann != expected_ann
        or actual_ret_ann != 'None'
    ):
        parts: list[str] = []
        if actual_val_ann != expected_ann:
            parts.append(
                f'value: {actual_val_ann} '
                f'(expected {expected_ann})'
            )
        if actual_ret_ann != 'None':
            parts.append(
                f'return: {actual_ret_ann} '
                '(expected None)'
            )
        result.issues.append(
            f'{loc}: setter annotation '
            f'— {", ".join(parts)}'
        )

        ds, de = _def_line_range(prop.setter, lines)
        def_indent = lines[ds][
            : len(lines[ds]) - len(lines[ds].lstrip())
        ]
        new_def = (
            f'{def_indent}def {prop.name}'
            f'(self, {setter_param}: '
            f'{expected_ann}) -> None:\n'
        )
        result.edits.append(Edit(ds, de, new_def))

    # Setter docstring — should not exist
    setter_doc_text, sd_s, sd_e = _docstring_range(
        prop.setter
    )
    if setter_doc_text is not None:
        result.issues.append(
            f'{loc}: setter has docstring '
            '(should have none)'
        )
        result.edits.append(Edit(sd_s, sd_e, ''))


# ---------------------------------------------------------
# Apply edits
# ---------------------------------------------------------


def _apply_edits(
    lines: list[str],
    edits: list[Edit],
) -> list[str]:
    """Apply edits bottom-up to preserve line numbers."""
    sorted_edits = sorted(
        edits, key=lambda e: e.start, reverse=True
    )
    result = list(lines)
    for edit in sorted_edits:
        new_lines = edit.new_text.splitlines(keepends=True)
        result[edit.start : edit.end] = new_lines
    return result


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------


def _collect_py_files(paths: list[str]) -> list[Path]:
    """Resolve paths to a sorted list of .py files.

    Each entry can be a directory (recursively globbed)
    or a single .py file.  When *paths* is empty,
    defaults to ``_SRC_ROOT``.
    """
    if not paths:
        return sorted(_SRC_ROOT.rglob('*.py'))

    result: list[Path] = []
    for raw in paths:
        p = Path(raw).resolve()
        if p.is_dir():
            result.extend(p.rglob('*.py'))
        elif p.is_file() and p.suffix == '.py':
            result.append(p)
    return sorted(set(result))


def main() -> int:
    """Run param-consistency check or fix."""
    parser = argparse.ArgumentParser(
        description=(
            'Parameter / property consistency: '
            'docstrings and type hints.'
        ),
    )
    parser.add_argument(
        'paths',
        nargs='*',
        help=(
            'Directories or .py files to scan '
            '(default: src/easydiffraction/)'
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--check',
        action='store_true',
        help='Validate consistency (default)',
    )
    group.add_argument(
        '--fix',
        action='store_true',
        help='Auto-fix docstrings and type hints',
    )
    args = parser.parse_args()

    py_files = _collect_py_files(args.paths)
    repo_root = Path(__file__).resolve().parents[1]
    total_issues = 0
    total_fixed = 0
    files_touched = 0

    for path in py_files:
        result = _analyze_file(path)
        if not result.issues:
            continue

        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path

        if args.fix:
            source_lines = path.read_text(
                encoding='utf-8'
            ).splitlines(keepends=True)
            fixed_lines = _apply_edits(
                source_lines, result.edits
            )
            path.write_text(
                ''.join(fixed_lines), encoding='utf-8'
            )
            count = len(result.issues)
            total_fixed += count
            files_touched += 1
            print(
                f'📝 {rel}: fixed {count} issue(s)'
            )
        else:
            for issue in result.issues:
                print(f'  ❌ {rel}: {issue}')
            total_issues += len(result.issues)

    # Summary
    print()
    if args.fix:
        print(
            f'✅ Fixed {total_fixed} issue(s) '
            f'in {files_touched} file(s).'
        )
        return 0
    if total_issues:
        print(
            f'❌ {total_issues} consistency '
            'issue(s) found.'
        )
        return 1
    print('✅ All properties match the template.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
