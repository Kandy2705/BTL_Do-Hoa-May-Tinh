"""Filesystem and config helpers."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path for student laptops
    yaml = None

from btl2.utils.constants import DEFAULT_OUTPUT_SUBDIRS


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dictionary.

    The preferred path uses PyYAML. If the package is missing, we fall back to
    a tiny indentation-based parser that supports the config style used in this repo.
    """
    path_obj = Path(path)
    if yaml is not None:
        with path_obj.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}
    return _load_yaml_fallback(path_obj)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the Path object."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_output_tree(root: str | Path) -> Path:
    """Create the expected dataset folder layout under one output root."""
    root_path = ensure_dir(root)
    for subdir in DEFAULT_OUTPUT_SUBDIRS:
        ensure_dir(root_path / subdir)
    return root_path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save JSON with indentation so generated metadata is human-readable."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    """Parse a very small YAML subset used by the bundled config files.

    Supported:
    - nested dictionaries by indentation
    - scalar values
    - inline lists like `[1, 2, 3]`
    - booleans, ints, floats, and quoted strings
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if raw_value == "":
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_yaml_scalar(raw_value)

    return root


def _parse_yaml_scalar(raw_value: str) -> Any:
    """Convert a simple YAML scalar into a Python value."""
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        pass

    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value.strip("\"'")
