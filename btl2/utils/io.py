"""Helper đọc cấu hình và ghi file cho pipeline BTL 2."""

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
    """Đọc file YAML config thành dict Python.

    Ưu tiên PyYAML nếu máy có cài. Nếu thiếu package, fallback sang parser nhỏ
    dựa trên indentation, đủ dùng cho các file config đi kèm repo.
    """
    path_obj = Path(path)
    if yaml is not None:
        with path_obj.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}
    return _load_yaml_fallback(path_obj)


def ensure_dir(path: str | Path) -> Path:
    """Tạo thư mục nếu chưa tồn tại và trả về `Path` của thư mục đó."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_output_tree(root: str | Path) -> Path:
    """Tạo layout thư mục dataset chuẩn dưới một output root."""
    root_path = ensure_dir(root)
    # Danh sách subdir nằm trong constants để CLI, UI và script kiểm tra dùng chung.
    for subdir in DEFAULT_OUTPUT_SUBDIRS:
        ensure_dir(root_path / subdir)
    return root_path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Ghi JSON có indent để metadata/annotation dễ đọc bằng mắt."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    """Parse một tập con YAML nhỏ dùng cho config đi kèm.

    Hỗ trợ:
    - dict lồng nhau theo indentation
    - scalar value
    - list inline như `[1, 2, 3]`
    - boolean, int, float và string có/không có quote
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

        # Khi indent giảm, pop stack để quay lại dict cha tương ứng.
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
    """Chuyển scalar YAML đơn giản thành kiểu Python phù hợp."""
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
