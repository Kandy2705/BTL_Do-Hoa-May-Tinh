"""Xuất metadata JSON cho từng frame BTL 2."""

from __future__ import annotations

from btl2.scene.scene import Scene


def export_frame_metadata(scene: Scene, bboxes: list[dict], segmentation_map: dict) -> dict:
    """Gom mọi thông tin cần để debug, visualize hoặc tái kiểm tra một frame."""
    return {
        "frame_id": scene.frame_id,
        "seed": scene.seed,
        "split": scene.split,
        # Camera/light được lưu đầy đủ để biết frame được nhìn từ đâu và chiếu sáng thế nào.
        "camera": {
            "position": scene.camera.position.tolist(),
            "target": scene.camera.target.tolist(),
            "up": scene.camera.up.tolist(),
            "fov_y_degrees": scene.camera.fov_y_degrees,
            "near": scene.camera.near,
            "far": scene.camera.far,
            "image_width": scene.camera.image_width,
            "image_height": scene.camera.image_height,
        },
        "light": {
            "direction": scene.light.direction.tolist(),
            "intensity": scene.light.intensity,
            "ambient_strength": scene.light.ambient_strength,
        },
        # Object metadata giữ lại transform và class/instance id của toàn scene,
        # kể cả object bị lọc khỏi bbox vì quá nhỏ hoặc ngoài khung hình.
        "objects": [obj.to_metadata() for obj in scene.objects],
        "bounding_boxes": bboxes,
        "segmentation_mapping": segmentation_map,
    }
