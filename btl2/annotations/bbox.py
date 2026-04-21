"""
TÍNH TOÁN BOUNDING BOX TỪ CÁC ĐỐI TƯỢNG 3D DỰA TRÊN MÀN HÌNH.

Hàm này thực hiện pipeline hoàn chỉnh từ 3D world space → 2D screen space:
1. Transform AABB từ local → world space
2. Project từ world → NDC space
3. Convert từ NDC → screen coordinates
4. Tính bounding box và kiểm tra visibility
"""

from __future__ import annotations

import numpy as np

from btl2.renderer.camera import CameraMatrices
from btl2.scene.scene import Scene
from btl2.utils.math3d import ndc_to_screen, project_points, transform_points


def compute_bounding_boxes(scene: Scene, camera: CameraMatrices, config: dict) -> list[dict]:
    """
    TÍNH TOÁN BOUNDING BOX CHO TẤT CẢ ĐỐI TƯỢNG TRONG SCENE.

    Pipeline:
    1. Duyệt qua từng object trong scene
    2. Transform AABB từ local → world space
    3. Project từ world → NDC (Normalized Device Coordinates)
    4. Convert từ NDC → screen coordinates
    5. Tính bounding box và kiểm tra visibility

    Args:
        scene: Scene chứa tất cả objects
        camera: Camera matrices (view, projection)
        config: Configuration dict với các threshold

    Returns:
        List[dict]: Danh sách bounding boxes với metadata
    """
    results: list[dict] = []
    min_bbox_pixels = float(config["min_bbox_pixels"])  # Kích thước tối thiểu của bbox

    # DUYỆT QUA TỪNG ĐỐI TƯỢNG TRONG SCENE
    for obj in scene.objects:
        # Bỏ qua các object không hợp lệ
        if obj.instance_id <= 0 or obj.aabb_local is None:
            continue

        # BƯỚC 1: TRANSFORM AABB TỪ LOCAL → WORLD SPACE
        # AABB (Axis-Aligned Bounding Box) là khung chữ nhật bao quanh object
        # corners() trả về 8 góc của bounding box trong local coordinates
        corners_world = transform_points(obj.model_matrix, obj.aabb_local.corners())

        # BƯỚC 2: PROJECT TỪ WORLD → NDC SPACE
        # NDC: Normalized Device Coordinates [-1, 1] range
        # ndc_xy: tọa độ 2D trong [-1, 1]
        # ndc_depth: độ sâu [0, 1] (0 = near, 1 = far)
        ndc_xy, ndc_depth = project_points(corners_world, camera.view, camera.projection)

        # KIỂM TRA ĐỐI TƯỢNG CÓ NẰM TRƯỚC CAMERA KHÔNG
        # ndc_depth <= 1.0 nghĩa là object nằm trong far plane
        in_front = np.any(ndc_depth <= 1.0)
        if not in_front:
            obj.visible = False
            continue  # Object nằm sau camera, bỏ qua

        # BƯỚC 3: CONVERT TỪ NDC → SCREEN COORDINATES
        # NDC [-1, 1] → Screen [0, width/height]
        screen = ndc_to_screen(ndc_xy, camera.width, camera.height)

        # TÍNH TOÁN BOUNDING BOX TRONG SCREEN SPACE
        # Clip coordinates để nằm trong màn hình
        x_min = float(np.clip(np.min(screen[:, 0]), 0.0, camera.width - 1))
        y_min = float(np.clip(np.min(screen[:, 1]), 0.0, camera.height - 1))
        x_max = float(np.clip(np.max(screen[:, 0]), 0.0, camera.width - 1))
        y_max = float(np.clip(np.max(screen[:, 1]), 0.0, camera.height - 1))

        # Kích thước bounding box
        width = x_max - x_min
        height = y_max - y_min

        # KIỂM TRA KÍCH THƯỚC TỐI THIỂU
        # Bỏ qua bounding box quá nhỏ
        if width < min_bbox_pixels or height < min_bbox_pixels:
            obj.visible = False
            continue

        # TÍNH TOÁN TỶ LỆ VISIBLE
        # visibility = (bbox_area) / (screen_area)
        visibility = float((width * height) / (camera.width * camera.height))
        if visibility < float(config["visibility_threshold"]):
            obj.visible = False
            continue  # Object quá nhỏ so với màn hình

        # ĐÁNH DẤU OBJECT LÀ VISIBLE
        obj.visible = True

        # TẠO RESULT CHO OBJECT NÀY
        results.append(
            {
                "instance_id": obj.instance_id,        # ID duy nhất của object
                "class_name": obj.class_name,          # Tên class (VD: "car", "person")
                "class_id": obj.semantic_id,           # ID semantic cho object type
                "bbox_xyxy": [x_min, y_min, x_max, y_max],  # Top-left, bottom-right
                "bbox_xywh": [x_min, y_min, width, height],  # Top-left, width, height
                "visibility_ratio": visibility,         # Tỷ lệ visible [0, 1]
            }
        )

    return results
