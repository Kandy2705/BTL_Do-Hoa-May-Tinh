"""Helper cho segmentation mask của BTL 2.

Mask RGB lưu màu, nhưng metadata cần nói rõ màu đó thuộc class/instance nào.
Mapping trong file JSON giúp người đọc hoặc script visualize giải mã mask mà
không cần nhớ công thức sinh màu.
"""

from __future__ import annotations

from btl2.scene.scene_object import SceneObject
from btl2.utils.colors import class_color, instance_color


def build_segmentation_mapping(objects: list[SceneObject], mask_rgb) -> dict:
    """Tạo mapping JSON-friendly từ màu mask sang class và instance id."""
    mapping = {}
    for obj in objects:
        # Road là nền nên dùng màu class cố định; object động dùng màu theo instance
        # để phân biệt hai xe/người cùng class trong một frame.
        if obj.class_name == "road":
            color = class_color("road")
        elif obj.instance_id > 0:
            color = instance_color(obj.instance_id)
        else:
            continue
        # JSON không dùng tuple làm key, nên encode màu RGB thành chuỗi "r_g_b".
        mapping[f"{color[0]}_{color[1]}_{color[2]}"] = {
            "instance_id": obj.instance_id,
            "class_name": obj.class_name,
            "class_id": obj.semantic_id,
        }
    return mapping
