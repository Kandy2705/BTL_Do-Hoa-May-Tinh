"""Service chính của BTL 2: nối dựng scene, render và xuất annotation.

Luồng tổng quát của BTL 2:
1. Đọc cấu hình YAML để biết kích thước ảnh, seed, số frame và class cần sinh.
2. Dựng scene đường phố hoặc chuyển scene hiện có từ BTL 1 sang định dạng BTL 2.
3. Render cùng một scene qua ba pass: RGB, depth và segmentation mask.
4. Từ dữ liệu render + camera, xuất bbox, YOLO, COCO, metadata và báo cáo kiểm tra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import glfw

from btl2.annotations.bbox import compute_bounding_boxes
from btl2.annotations.coco_export import CocoExporter
from btl2.annotations.custom_export import CustomCsvExporter
from btl2.annotations.dataset_consistency import validate_dataset
from btl2.annotations.depth_export import linearize_depth, save_depth_outputs
from btl2.annotations.metadata_export import export_frame_metadata
from btl2.annotations.occlusion import estimate_occlusion_ratios
from btl2.annotations.segmentation import build_segmentation_mapping
from btl2.annotations.yolo_export import export_yolo_labels, write_dataset_yaml
from btl2.renderer.camera import build_camera_matrices
from btl2.renderer.framebuffer import RenderTarget
from btl2.renderer.material import Material
from btl2.renderer.mesh import GLMesh, upload_mesh
from btl2.renderer.render_pass_depth import DepthRenderPass
from btl2.renderer.render_pass_rgb import RGBRenderPass
from btl2.renderer.render_pass_segmentation import SegmentationRenderPass
from btl2.renderer.window import OffscreenWindow
from btl2.scene.road_scene_builder import RoadSceneBuilder
from btl2.scene.scene import CameraState, DirectionalLight, Scene
from btl2.scene.scene_object import SceneObject
from btl2.scene.object_loader import MeshData
from btl2.utils.colors import class_color, color_to_float, instance_color
from btl2.utils.image import save_mask, save_rgb
from btl2.utils.io import ensure_output_tree
from btl2.utils.constants import CLASS_TO_ID


@dataclass
class FrameArtifacts:
    """Tất cả kết quả trung gian/cuối cùng sinh ra từ một frame đã render.

    `rgb`, `depth_*` và `mask_rgb` là dữ liệu ảnh; `bboxes`, `metadata`,
    `yolo_lines` và `segmentation_map` là phần nhãn dùng cho training/evaluate.
    Gom các trường này vào một dataclass giúp các bước render và export trao
    đổi rõ ràng, tránh truyền quá nhiều biến rời rạc.
    """

    rgb: np.ndarray
    depth_linear: np.ndarray
    depth_gray: np.ndarray
    mask_rgb: np.ndarray
    bboxes: list[dict]
    metadata: dict
    yolo_lines: list[str]
    segmentation_map: dict


class SyntheticRoadApp:
    """Pipeline cấp cao của BTL 2.

    Lớp này không tự sinh geometry chi tiết và cũng không tự viết shader; nó
    đóng vai trò điều phối:
    - `RoadSceneBuilder` tạo `Scene` và `MeshData`.
    - Các `RenderPass` biến scene thành ảnh RGB/depth/mask.
    - Các module `annotations` chuyển kết quả thành format dataset.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        # Tạo đủ cây thư mục output ngay từ đầu để các bước sau chỉ cần ghi file.
        self.output_root = ensure_output_tree(config["output_dir"])
        self.builder = RoadSceneBuilder(config)
        width = int(config["image_width"])
        height = int(config["image_height"])
        # Nếu đang được gọi từ app BTL 1 thì đã có GLFW/OpenGL context sẵn.
        # Khi đó phải tái sử dụng context hiện tại, nếu không việc terminate GLFW
        # ở app offscreen sẽ làm hỏng luôn context chính của editor.
        self._shared_context = glfw.get_current_context()
        self._owns_window = self._shared_context is None
        self.window = OffscreenWindow(width, height) if self._owns_window else None
        if self._owns_window and self.window is not None:
            self._shared_context = glfw.get_current_context()
        # Mỗi render pass dùng một framebuffer riêng. Việc tách ra như vậy giúp
        # đọc lại ảnh RGB, depth và mask độc lập, không bị pass sau ghi đè pass trước.
        self.rgb_target = RenderTarget(width, height)
        self.seg_target = RenderTarget(width, height)
        self.depth_target = RenderTarget(width, height)
        # Ba shader pipeline dùng cùng scene/camera nhưng mục tiêu khác nhau:
        # RGB để nhìn giống ảnh thật, depth để lấy khoảng cách, segmentation để mã màu object.
        self.rgb_pass = RGBRenderPass("shaders/btl2")
        self.depth_pass = DepthRenderPass("shaders/btl2")
        self.seg_pass = SegmentationRenderPass("shaders/btl2")
        # COCO có thể lưu segmentation dưới dạng polygon hoặc RLE. Nếu config sai
        # thì fallback về polygon vì dễ đọc và dễ kiểm tra bằng mắt hơn.
        coco_segmentation_mode = str(self.config.get("annotations", {}).get("coco_segmentation_mode", "polygon")).strip().lower()
        if coco_segmentation_mode not in {"polygon", "rle"}:
            coco_segmentation_mode = "polygon"
        self.coco_exporter = CocoExporter(segmentation_mode=coco_segmentation_mode)
        self.custom_csv_exporter = CustomCsvExporter()

    def close(self) -> None:
        """Giải phóng context OpenGL offscreen nếu BTL 2 tự tạo window."""
        if self._owns_window and self.window is not None:
            self.window.destroy()

    def generate_dataset(self, num_frames: int | None = None) -> list[dict]:
        """Sinh và ghi một dataset hoàn chỉnh, đã chia train/val."""
        total = int(num_frames or self.config["num_frames"])
        summaries: list[dict] = []
        for frame_index in range(total):
            # Mỗi frame dùng seed = base_seed + frame_index nên dataset lặp lại
            # được chính xác, nhưng vẫn khác nhau giữa các frame.
            scene, mesh_registry = self.builder.build_scene(frame_index)
            frame = self.render_frame(scene, mesh_registry)
            paths = self.export_frame(scene, frame)
            # Hai exporter này gom dữ liệu toàn bộ dataset trước, rồi cuối vòng
            # lặp mới ghi JSON/CSV để có đủ danh sách images/annotations.
            self.coco_exporter.add_frame(scene.frame_id, scene.split, scene.camera.image_width, scene.camera.image_height, frame.bboxes, paths)
            self.custom_csv_exporter.add_frame(scene.frame_id, scene.split, frame.bboxes, paths)
            summaries.append(paths)

        self.coco_exporter.write(self.output_root / "annotations_coco")
        self.custom_csv_exporter.write(self.output_root / "annotations_custom")
        write_dataset_yaml(self.output_root, list(self.coco_exporter.categories.values()))
        self._validate_exported_dataset()
        return summaries

    def preview_scene(self, seed_override: int | None = None) -> FrameArtifacts:
        """Render thử một scene mà không ghi toàn bộ dataset ra đĩa."""
        if seed_override is not None:
            # Override seed trực tiếp vào config để builder dùng cùng cơ chế với generate.
            self.config["seed"] = int(seed_override)
        scene, mesh_registry = self.builder.build_scene(0)
        return self.render_frame(scene, mesh_registry)

    def generate_from_btl1_scene(self, btl1_objects: list, num_frames: int, base_seed: int = 42) -> list[dict]:
        """Xuất dataset từ scene đang dựng trong BTL 1 và các camera đã đặt.

        Chỉ các camera là object trong scene mới được dùng. Camera mặc định của
        viewer không nằm trong `btl1_objects`, nên tự nhiên không bị đưa vào dataset.
        """
        # Người dùng có thể đặt nhiều camera trong scene BTL 1. Khi xuất nhiều
        # frame, pipeline sẽ quay vòng qua danh sách camera đó.
        placed_cameras = [obj for obj in btl1_objects if hasattr(obj, "camera_fov")]
        if not placed_cameras:
            raise RuntimeError("Khong co camera nao trong scene. Hay them Camera trong Hierarchy truoc khi xuat BTL 2.")

        # Renderable ở đây là object có `drawable`; các node điều khiển, camera,
        # đèn hoặc helper không có mesh sẽ không được xuất.
        renderables = [obj for obj in btl1_objects if hasattr(obj, "drawable") and getattr(obj, "drawable", None) is not None]
        if not renderables:
            raise RuntimeError("Khong co object nao co the render trong scene hien tai.")

        summaries: list[dict] = []
        for frame_index in range(num_frames):
            camera_obj = placed_cameras[frame_index % len(placed_cameras)]
            scene, mesh_registry = self._build_scene_from_btl1(
                all_objects=btl1_objects,
                renderables=renderables,
                camera_obj=camera_obj,
                frame_index=frame_index,
                seed=base_seed + frame_index,
            )
            frame = self.render_frame(scene, mesh_registry)
            paths = self.export_frame(scene, frame)
            self.coco_exporter.add_frame(scene.frame_id, scene.split, scene.camera.image_width, scene.camera.image_height, frame.bboxes, paths)
            self.custom_csv_exporter.add_frame(scene.frame_id, scene.split, frame.bboxes, paths)
            summaries.append(paths)

        self.coco_exporter.write(self.output_root / "annotations_coco")
        self.custom_csv_exporter.write(self.output_root / "annotations_custom")
        write_dataset_yaml(self.output_root, list(self.coco_exporter.categories.values()))
        self._validate_exported_dataset()
        return summaries

    def _validate_exported_dataset(self) -> None:
        """Kiểm tra dataset sau khi xuất để phát hiện lỗi sớm."""
        validation_cfg = self.config.get("validation", {})
        enabled = bool(validation_cfg.get("enabled", True))
        if not enabled:
            return

        # Validator kiểm tra nhiều mối liên hệ chéo: ảnh có nhãn tương ứng,
        # mask có pixel cho instance, COCO/YOLO không lệch class, depth có đủ file...
        report = validate_dataset(
            self.output_root,
            fix=bool(validation_cfg.get("auto_fix", True)),
            require_depth=bool(validation_cfg.get("require_depth", True)),
            require_depth_npy=bool(validation_cfg.get("require_depth_npy", self.config.get("save_depth_npy", False))),
            require_coco=bool(validation_cfg.get("require_coco", True)),
            require_mask_pixels=bool(validation_cfg.get("require_mask_pixels", True)),
        )
        report_path = self.output_root / "quality_report.json"
        report_path.write_text(__import__("json").dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        issue_count = int(report.get("summary", {}).get("total_issues", 0))
        warning_count = int(report.get("summary", {}).get("total_warnings", 0))
        if issue_count:
            preview = "; ".join(report.get("issues", [])[:5])
            raise RuntimeError(
                f"Dataset validation failed with {issue_count} issue(s). "
                f"Report: {report_path}. First issues: {preview}"
            )
        if warning_count:
            print(f"BTL2 dataset validation warnings: {warning_count}. Report: {report_path}")

    def render_frame(self, scene, mesh_registry) -> FrameArtifacts:
        """Render một frame qua RGB/depth/segmentation rồi suy ra annotation."""
        if self.window is not None:
            self.window.make_current()
        elif self._shared_context is not None:
            glfw.make_context_current(self._shared_context)
        # CameraMatrices chứa view/projection dùng chung cho shader và phép chiếu bbox.
        camera = build_camera_matrices(scene.camera)
        # MeshData còn nằm ở CPU; `upload_mesh` tạo VAO/VBO/EBO/texture để GPU vẽ được.
        meshes: dict[str, GLMesh] = {name: upload_mesh(mesh) for name, mesh in mesh_registry.items()}
        materials = self._build_materials(scene)

        # Pass 1: ảnh RGB cuối cùng, có texture và ánh sáng để nhìn như ảnh camera.
        self.rgb_pass.render(scene, camera, self.rgb_target, meshes, materials)
        rgb = self.rgb_target.read_rgb()

        # Pass 2: depth buffer của OpenGL là phi tuyến, nên cần linearize để dùng
        # như bản đồ khoảng cách và tạo thêm PNG xem nhanh.
        self.depth_pass.render(scene, camera, self.depth_target, meshes)
        depth_buffer = self.depth_target.read_depth()
        depth_linear, depth_gray = linearize_depth(depth_buffer, camera.near, camera.far)

        # Pass 3: segmentation gán màu duy nhất cho từng instance. Ảnh mask này là
        # nguồn để tính occlusion và segmentation polygon/RLE cho COCO.
        self.seg_pass.render(scene, camera, self.seg_target, meshes, materials)
        mask_rgb = self.seg_target.read_rgb()

        # BBox được tính bằng cách chiếu AABB 3D qua view/projection; sau đó mask
        # giúp ước lượng object bị che bao nhiêu bởi object khác.
        bboxes = compute_bounding_boxes(scene, camera, self.config["annotations"])
        occlusion = estimate_occlusion_ratios(mask_rgb, scene.objects)
        for bbox in bboxes:
            bbox["occlusion_ratio"] = occlusion.get(bbox["instance_id"], 0.0)

        # Metadata giữ lại đủ thông tin để debug một frame: camera, light, object,
        # bbox và mapping màu segmentation.
        segmentation_map = build_segmentation_mapping(scene.objects, mask_rgb)
        metadata = export_frame_metadata(scene, bboxes, segmentation_map)
        yolo_lines = export_yolo_labels(bboxes, scene.camera.image_width, scene.camera.image_height)

        # Trả framebuffer về mặc định để vòng render chính của BTL 1 tiếp tục an toàn.
        from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return FrameArtifacts(
            rgb=rgb,
            depth_linear=depth_linear,
            depth_gray=depth_gray,
            mask_rgb=mask_rgb,
            bboxes=bboxes,
            metadata=metadata,
            yolo_lines=yolo_lines,
            segmentation_map=segmentation_map,
        )

    def export_frame(self, scene, frame: FrameArtifacts) -> dict[str, str]:
        """Ghi ảnh, depth, mask, label và metadata của một frame ra cây dataset."""
        split = scene.split
        frame_id = scene.frame_id
        # Tên file dùng cùng `frame_id` trên mọi thư mục để dễ truy ngược:
        # images/train/frame_000001.png -> labels_yolo/train/frame_000001.txt...
        image_path = self.output_root / "images" / split / f"{frame_id}.png"
        depth_png_path = self.output_root / "depth" / split / f"{frame_id}_depth.png"
        depth_npy_path = self.output_root / "depth" / split / f"{frame_id}_depth.npy"
        mask_path = self.output_root / "masks" / split / f"{frame_id}_mask.png"
        yolo_path = self.output_root / "labels_yolo" / split / f"{frame_id}.txt"
        metadata_path = self.output_root / "metadata" / split / f"{frame_id}.json"

        save_rgb(image_path, frame.rgb)
        save_mask(mask_path, frame.mask_rgb)
        save_depth_outputs(depth_png_path, depth_npy_path, frame.depth_gray, frame.depth_linear, bool(self.config["save_depth_npy"]))
        yolo_path.write_text("\n".join(frame.yolo_lines), encoding="utf-8")
        # Metadata cũng lưu lại path output để các script kiểm tra/visualize không
        # phải tự đoán vị trí file liên quan.
        frame.metadata["paths"] = {
            "rgb": str(image_path),
            "depth_png": str(depth_png_path),
            "depth_npy": str(depth_npy_path) if self.config["save_depth_npy"] else None,
            "mask": str(mask_path),
            "yolo": str(yolo_path),
        }
        metadata_path.write_text(__import__("json").dumps(frame.metadata, indent=2), encoding="utf-8")
        return {
            "frame_id": frame_id,
            "split": split,
            "rgb": str(image_path),
            "depth_png": str(depth_png_path),
            "mask": str(mask_path),
            "yolo": str(yolo_path),
            "metadata": str(metadata_path),
        }

    @staticmethod
    def _build_materials(scene) -> dict[int, Material]:
        """Tạo material cho từng object, gồm màu RGB và màu mã hóa segmentation."""
        materials: dict[int, Material] = {}
        for obj in scene.objects:
            # Road là nền, không phải instance training riêng, nên dùng màu theo class.
            # Object động dùng màu theo instance_id để phân biệt từng xe/người/biển báo.
            if obj.class_name == "road":
                seg_color = color_to_float(class_color("road"))
            else:
                seg_color = color_to_float(instance_color(obj.instance_id if obj.instance_id > 0 else 0))
            materials[obj.instance_id] = Material(base_color=obj.base_color, segmentation_color=seg_color)
        return materials

    def _build_scene_from_btl1(self, all_objects: list, renderables: list, camera_obj, frame_index: int, seed: int) -> tuple[Scene, dict[str, MeshData]]:
        """Chuyển scene BTL 1 hiện tại sang `Scene`/`MeshData` của BTL 2."""
        split = "train" if frame_index < int(self.config["num_frames"] * float(self.config.get("train_split", 0.8))) else "val"
        camera = self._camera_from_btl1(camera_obj)
        light = self._light_from_btl1(all_objects)
        scene = Scene(
            frame_id=f"frame_{frame_index + 1:06d}",
            seed=seed,
            split=split,
            camera=camera,
            light=light,
            background_color=np.array([0.58, 0.72, 0.92], dtype=np.float32),
        )

        mesh_registry: dict[str, MeshData] = {}
        instance_id = 1
        for obj in renderables:
            drawable = getattr(obj, "drawable", None)
            # BTL 1 không bắt buộc object có class dataset, nên BTL 2 suy luận class
            # từ tên object. Người dùng có thể đặt tên chứa car/bus/person/sign...
            class_name = self._infer_class_name(obj.name)
            if class_name is None:
                continue
            is_road = class_name == "road"
            # Road/city mesh có thể nhiều material/texture; tách theo material giúp
            # render offscreen giữ màu gần giống viewport BTL 1.
            meshes = self._meshes_from_drawable(drawable, split_by_material=is_road)
            if not meshes:
                continue
            # Road dùng instance_id=0 và semantic_id=255 để không sinh bbox YOLO,
            # nhưng vẫn có thể xuất mask nền nếu cần kiểm tra segmentation.
            current_instance_id = 0 if is_road else instance_id
            semantic_id = 255 if is_road else CLASS_TO_ID[class_name]

            for mesh_suffix, mesh in meshes:
                mesh_key = f"btl1_mesh_{obj.id}_{mesh_suffix}"
                mesh_registry[mesh_key] = mesh
                scene.add_object(
                    SceneObject(
                        name=obj.name if len(meshes) == 1 else f"{obj.name}_{mesh_suffix}",
                        class_name=class_name,
                        mesh_key=mesh_key,
                        position=np.asarray(obj.position, dtype=np.float32),
                        rotation_degrees=np.asarray(obj.rotation, dtype=np.float32),
                        scale=np.asarray(obj.scale, dtype=np.float32),
                        base_color=self._base_color_from_btl1(obj, class_name),
                        instance_id=current_instance_id,
                        semantic_id=semantic_id,
                        metadata={"source": "btl1_scene", "original_name": obj.name},
                        aabb_local=mesh.aabb,
                    )
                )
            if not is_road:
                instance_id += 1

        if not scene.objects:
            raise RuntimeError("Scene hien tai khong co drawable hop le de xuat dataset.")
        return scene, mesh_registry

    def _camera_from_btl1(self, camera_obj) -> CameraState:
        """Tạo camera BTL 2 từ camera object đã đặt trong scene BTL 1."""
        position = np.asarray(camera_obj.position, dtype=np.float32)
        rotation = np.radians(np.asarray(camera_obj.rotation, dtype=np.float32))
        pitch = float(rotation[0])
        yaw = float(rotation[1])
        # BTL 1 lưu rotation theo Euler; ở đây đổi pitch/yaw thành vector forward
        # để tạo target cho ma trận look-at của pipeline BTL 2.
        forward = np.array(
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float32,
        )
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        forward = forward / np.linalg.norm(forward)
        return CameraState(
            position=position,
            target=position + forward * 10.0,
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            fov_y_degrees=float(camera_obj.camera_fov),
            near=float(camera_obj.camera_near),
            far=float(camera_obj.camera_far),
            image_width=int(self.config["image_width"]),
            image_height=int(self.config["image_height"]),
        )

    @staticmethod
    def _light_from_btl1(scene_objects: list) -> DirectionalLight:
        """Xấp xỉ directional light từ đèn BTL 1 đầu tiên, hoặc dùng mặc định."""
        for obj in scene_objects:
            if hasattr(obj, "light_intensity") and getattr(obj, "visible", True):
                # Với directional light, hướng quan trọng hơn vị trí tuyệt đối.
                # Ta lấy vector vị trí object làm hướng ánh sáng đơn giản, dễ dự đoán.
                direction = np.asarray(obj.position, dtype=np.float32)
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([0.4, -1.0, 0.2], dtype=np.float32)
                direction = direction / np.linalg.norm(direction)
                color = np.asarray(getattr(obj, "light_color", [1.0, 1.0, 1.0]), dtype=np.float32)
                return DirectionalLight(
                    direction=direction,
                    color=color,
                    intensity=max(1.15, float(getattr(obj, "light_intensity", 1.0))),
                    ambient_strength=0.52,
                )
        return DirectionalLight(
            direction=np.array([0.4, -1.0, 0.2], dtype=np.float32) / np.linalg.norm(np.array([0.4, -1.0, 0.2], dtype=np.float32)),
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            intensity=1.15,
            ambient_strength=0.52,
        )

    @classmethod
    def _meshes_from_drawable(cls, drawable, split_by_material: bool = False) -> list[tuple[str, MeshData]]:
        """Chuyển drawable của BTL 1 thành một hoặc nhiều mesh BTL 2.

        Preview OBJ của BTL 1 có thể vẽ nhiều material bằng các đoạn index khác
        nhau. Renderer offscreen của BTL 2 đơn giản hơn: một mesh thường gắn một
        texture chính. Vì vậy static city/road mesh cần tách theo material để ảnh
        RGB xuất ra vẫn giữ texture mapping giống viewport.
        """
        if drawable is None or not hasattr(drawable, "vertices"):
            return []
        vertices = np.asarray(drawable.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            return []

        normals = np.asarray(getattr(drawable, "normals", np.tile([[0.0, 1.0, 0.0]], (len(vertices), 1))), dtype=np.float32)
        if normals.shape != vertices.shape:
            normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(vertices), 1))
        texcoords = np.asarray(getattr(drawable, "texcoords", np.zeros((len(vertices), 2))), dtype=np.float32)
        if texcoords.shape != (len(vertices), 2):
            texcoords = np.zeros((len(vertices), 2), dtype=np.float32)

        if hasattr(drawable, "indices") and getattr(drawable, "indices") is not None and len(drawable.indices) >= 3:
            indices = np.asarray(drawable.indices, dtype=np.uint32).reshape(-1)
        else:
            # Một số primitive BTL 1 không có index buffer. Khi đó coi mỗi 3 vertex
            # liên tiếp là một tam giác độc lập.
            usable = (len(vertices) // 3) * 3
            if usable < 3:
                return []
            indices = np.arange(usable, dtype=np.uint32)
            vertices = vertices[:usable]
            normals = normals[:usable]
            texcoords = texcoords[:usable]

        if split_by_material and getattr(drawable, "material_groups", None):
            material_texture_paths = getattr(drawable, "material_texture_paths", {}) or {}
            meshes: list[tuple[str, MeshData]] = []
            for group_index, group in enumerate(drawable.material_groups):
                start = int(group.get("start", 0))
                count = int(group.get("count", 0))
                if count <= 0 or start < 0 or start + count > len(indices):
                    continue
                group_indices = indices[start:start + count]
                material_name = group.get("material")
                texture_path = cls._valid_texture_path(material_texture_paths.get(material_name))
                suffix = f"mat{group_index:03d}"
                mesh = cls._mesh_subset(vertices, normals, texcoords, group_indices, texture_path)
                if mesh is not None:
                    meshes.append((suffix, mesh))
            if meshes:
                return meshes

        texture_path = cls._valid_texture_path(getattr(drawable, "material_texture_path", None))
        if texture_path is None and getattr(drawable, "material_texture_paths", None):
            first_texture = next(iter(drawable.material_texture_paths.values()), None)
            texture_path = cls._valid_texture_path(first_texture)

        mesh = cls._mesh_from_arrays(vertices, normals, texcoords, indices, texture_path)
        return [("full", mesh)] if mesh is not None else []

    @staticmethod
    def _valid_texture_path(path_value) -> Path | None:
        if not path_value:
            return None
        texture_path = Path(path_value)
        return texture_path if texture_path.exists() else None

    @staticmethod
    def _mesh_subset(
        vertices: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
        group_indices: np.ndarray,
        texture_path: Path | None,
    ) -> MeshData | None:
        if group_indices.size < 3:
            return None
        unique_indices, remapped = np.unique(group_indices.astype(np.uint32), return_inverse=True)
        return SyntheticRoadApp._mesh_from_arrays(
            vertices[unique_indices],
            normals[unique_indices],
            texcoords[unique_indices],
            remapped.astype(np.uint32),
            texture_path,
        )

    @staticmethod
    def _mesh_from_arrays(
        vertices: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
        indices: np.ndarray,
        texture_path: Path | None,
    ) -> MeshData | None:
        if len(vertices) == 0 or len(indices) < 3:
            return None

        from btl2.utils.math3d import AABB

        return MeshData(
            vertices=vertices,
            normals=normals,
            indices=indices,
            aabb=AABB(vertices.min(axis=0), vertices.max(axis=0)),
            texcoords=texcoords,
            texture_path=texture_path,
        )

    @staticmethod
    def _infer_class_name(name: str) -> str | None:
        """Suy luận class training của BTL 2 từ tên object trong BTL 1."""
        lowered = name.lower()
        if any(token in lowered for token in ("road", "street", "lane", "ground", "floor", "terrain", "city", "intersection", "building")):
            return "road"
        if any(token in lowered for token in ("ped", "human", "person", "walker")):
            return "person"
        if any(token in lowered for token in ("motorbike", "motorcycle", "moto", "bike", "scooter")):
            return "motorbike"
        if "bus" in lowered:
            return "bus"
        if any(token in lowered for token in ("truck", "lorry")):
            return "truck"
        if "sign" in lowered:
            return "traffic_sign"
        if any(token in lowered for token in ("light", "signal")):
            return "traffic_light"
        return "car"

    @staticmethod
    def _base_color_from_btl1(obj, class_name: str) -> np.ndarray:
        """Ưu tiên màu object BTL 1, nếu không hợp lệ thì dùng màu class ổn định."""
        class_rgb = color_to_float(class_color(class_name))
        raw_color = np.asarray(getattr(obj, "color", [1.0, 1.0, 1.0])[:3], dtype=np.float32)
        if raw_color.size != 3 or not np.all(np.isfinite(raw_color)):
            return class_rgb

        alpha = float(getattr(obj, "color", [1.0, 1.0, 1.0, 1.0])[3]) if len(getattr(obj, "color", [])) >= 4 else 1.0
        is_default_white = bool(np.all(raw_color > 0.92))
        is_default_black = bool(np.all(raw_color < 0.04))
        # Trắng/đen mặc định thường không phải màu chủ ý của người dùng; dùng màu
        # theo class sẽ giúp ảnh preview và segmentation dễ phân biệt hơn.
        if class_name == "road" or alpha <= 0.05 or is_default_white or is_default_black:
            return class_rgb
        return np.clip(raw_color, 0.0, 1.0).astype(np.float32)
