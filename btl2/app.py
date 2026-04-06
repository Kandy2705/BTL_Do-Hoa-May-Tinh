"""Main application service that ties scene generation, rendering, and export together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import glfw

from btl2.annotations.bbox import compute_bounding_boxes
from btl2.annotations.coco_export import CocoExporter
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
from btl2.utils.colors import color_to_float, instance_color
from btl2.utils.image import save_mask, save_rgb
from btl2.utils.io import ensure_output_tree


@dataclass
class FrameArtifacts:
    """All arrays and annotations produced from one rendered frame."""

    rgb: np.ndarray
    depth_linear: np.ndarray
    depth_gray: np.ndarray
    mask_rgb: np.ndarray
    bboxes: list[dict]
    metadata: dict
    yolo_lines: list[str]
    segmentation_map: dict


class SyntheticRoadApp:
    """High-level pipeline: build scene, render passes, export annotations."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.output_root = ensure_output_tree(config["output_dir"])
        self.builder = RoadSceneBuilder(config)
        width = int(config["image_width"])
        height = int(config["image_height"])
        # Nếu đang được gọi từ app BTL 1 thì đã có GLFW/OpenGL context sẵn.
        # Khi đó phải tái sử dụng context hiện tại, nếu không việc terminate GLFW
        # ở app offscreen sẽ làm hỏng luôn context chính của editor.
        self._owns_window = glfw.get_current_context() is None
        self.window = OffscreenWindow(width, height) if self._owns_window else None
        self.rgb_target = RenderTarget(width, height)
        self.seg_target = RenderTarget(width, height)
        self.depth_target = RenderTarget(width, height)
        self.rgb_pass = RGBRenderPass("shaders/btl2")
        self.depth_pass = DepthRenderPass("shaders/btl2")
        self.seg_pass = SegmentationRenderPass("shaders/btl2")
        self.coco_exporter = CocoExporter()

    def close(self) -> None:
        """Release the offscreen OpenGL context."""
        if self._owns_window and self.window is not None:
            self.window.destroy()

    def generate_dataset(self, num_frames: int | None = None) -> list[dict]:
        """Generate and export a full dataset split into train and val."""
        total = int(num_frames or self.config["num_frames"])
        summaries: list[dict] = []
        for frame_index in range(total):
            scene, mesh_registry = self.builder.build_scene(frame_index)
            frame = self.render_frame(scene, mesh_registry)
            paths = self.export_frame(scene, frame)
            self.coco_exporter.add_frame(scene.frame_id, scene.split, scene.camera.image_width, scene.camera.image_height, frame.bboxes, paths)
            summaries.append(paths)

        self.coco_exporter.write(self.output_root / "annotations_coco")
        write_dataset_yaml(self.output_root, list(self.coco_exporter.categories.values()))
        return summaries

    def preview_scene(self, seed_override: int | None = None) -> FrameArtifacts:
        """Render one scene without writing a full dataset."""
        if seed_override is not None:
            self.config["seed"] = int(seed_override)
        scene, mesh_registry = self.builder.build_scene(0)
        return self.render_frame(scene, mesh_registry)

    def generate_from_btl1_scene(self, btl1_objects: list, num_frames: int, base_seed: int = 42) -> list[dict]:
        """Export dataset frames from the existing BTL1 scene and its placed cameras.

        Only scene cameras are used. The viewer default camera is not part of
        `btl1_objects`, so it is naturally excluded.
        """
        placed_cameras = [obj for obj in btl1_objects if hasattr(obj, "camera_fov")]
        if not placed_cameras:
            raise RuntimeError("Khong co camera nao trong scene. Hay them Camera trong Hierarchy truoc khi xuat BTL 2.")

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
            summaries.append(paths)

        self.coco_exporter.write(self.output_root / "annotations_coco")
        write_dataset_yaml(self.output_root, list(self.coco_exporter.categories.values()))
        return summaries

    def render_frame(self, scene, mesh_registry) -> FrameArtifacts:
        """Run RGB, depth, and segmentation passes and derive annotations."""
        if self.window is not None:
            self.window.make_current()
        camera = build_camera_matrices(scene.camera)
        meshes: dict[str, GLMesh] = {name: upload_mesh(mesh) for name, mesh in mesh_registry.items()}
        materials = self._build_materials(scene)

        self.rgb_pass.render(scene, camera, self.rgb_target, meshes, materials)
        rgb = self.rgb_target.read_rgb()

        self.depth_pass.render(scene, camera, self.depth_target, meshes)
        depth_buffer = self.depth_target.read_depth()
        depth_linear, depth_gray = linearize_depth(depth_buffer, camera.near, camera.far)

        self.seg_pass.render(scene, camera, self.seg_target, meshes, materials)
        mask_rgb = self.seg_target.read_rgb()

        bboxes = compute_bounding_boxes(scene, camera, self.config["annotations"])
        occlusion = estimate_occlusion_ratios(mask_rgb, scene.objects)
        for bbox in bboxes:
            bbox["occlusion_ratio"] = occlusion.get(bbox["instance_id"], 0.0)

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
        """Write one frame worth of images and labels to the dataset tree."""
        split = scene.split
        frame_id = scene.frame_id
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
        """Create per-object materials for RGB and segmentation passes."""
        materials: dict[int, Material] = {}
        for obj in scene.objects:
            seg_color = color_to_float(instance_color(obj.instance_id if obj.instance_id > 0 else 0))
            materials[obj.instance_id] = Material(base_color=obj.base_color, segmentation_color=seg_color)
        return materials

    def _build_scene_from_btl1(self, all_objects: list, renderables: list, camera_obj, frame_index: int, seed: int) -> tuple[Scene, dict[str, MeshData]]:
        """Convert the current BTL1 scene representation into the BTL2 scene format."""
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
            mesh = self._mesh_from_drawable(drawable)
            if mesh is None:
                continue
            mesh_key = f"btl1_mesh_{obj.id}"
            mesh_registry[mesh_key] = mesh
            class_name = self._infer_class_name(obj.name)
            scene.add_object(
                SceneObject(
                    name=obj.name,
                    class_name=class_name,
                    mesh_key=mesh_key,
                    position=np.asarray(obj.position, dtype=np.float32),
                    rotation_degrees=np.asarray(obj.rotation, dtype=np.float32),
                    scale=np.asarray(obj.scale, dtype=np.float32),
                    base_color=np.asarray(obj.color[:3], dtype=np.float32),
                    instance_id=instance_id,
                    semantic_id={"car": 0, "pedestrian": 1, "traffic_sign": 2, "traffic_light": 3}[class_name],
                    metadata={"source": "btl1_scene", "original_name": obj.name},
                    aabb_local=mesh.aabb,
                )
            )
            instance_id += 1

        if not scene.objects:
            raise RuntimeError("Scene hien tai khong co drawable hop le de xuat dataset.")
        return scene, mesh_registry

    def _camera_from_btl1(self, camera_obj) -> CameraState:
        """Create a BTL2 camera from a camera placed in the BTL1 scene."""
        position = np.asarray(camera_obj.position, dtype=np.float32)
        rotation = np.radians(np.asarray(camera_obj.rotation, dtype=np.float32))
        pitch = float(rotation[0])
        yaw = float(rotation[1])
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
        """Approximate a directional light from the first visible BTL1 light, or use a default."""
        for obj in scene_objects:
            if hasattr(obj, "light_intensity") and getattr(obj, "visible", True):
                direction = np.asarray(obj.position, dtype=np.float32)
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([0.4, -1.0, 0.2], dtype=np.float32)
                direction = direction / np.linalg.norm(direction)
                color = np.asarray(getattr(obj, "light_color", [1.0, 1.0, 1.0]), dtype=np.float32)
                return DirectionalLight(
                    direction=direction,
                    color=color,
                    intensity=float(getattr(obj, "light_intensity", 1.0)),
                    ambient_strength=0.35,
                )
        return DirectionalLight(
            direction=np.array([0.4, -1.0, 0.2], dtype=np.float32) / np.linalg.norm(np.array([0.4, -1.0, 0.2], dtype=np.float32)),
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            intensity=1.0,
            ambient_strength=0.35,
        )

    @staticmethod
    def _mesh_from_drawable(drawable) -> MeshData | None:
        """Convert a BTL1 drawable into CPU-side mesh buffers expected by BTL2."""
        if drawable is None or not hasattr(drawable, "vertices"):
            return None
        vertices = np.asarray(drawable.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            return None

        normals = np.asarray(getattr(drawable, "normals", np.tile([[0.0, 1.0, 0.0]], (len(vertices), 1))), dtype=np.float32)
        if normals.shape != vertices.shape:
            normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(vertices), 1))

        if hasattr(drawable, "indices") and getattr(drawable, "indices") is not None and len(drawable.indices) >= 3:
            indices = np.asarray(drawable.indices, dtype=np.uint32).reshape(-1)
        else:
            usable = (len(vertices) // 3) * 3
            if usable < 3:
                return None
            indices = np.arange(usable, dtype=np.uint32)
            vertices = vertices[:usable]
            normals = normals[:usable]

        from btl2.utils.math3d import AABB

        return MeshData(
            vertices=vertices,
            normals=normals,
            indices=indices,
            aabb=AABB(vertices.min(axis=0), vertices.max(axis=0)),
        )

    @staticmethod
    def _infer_class_name(name: str) -> str:
        """Infer one of the BTL2 training classes from an object name."""
        lowered = name.lower()
        if any(token in lowered for token in ("ped", "human", "person", "walker")):
            return "pedestrian"
        if "sign" in lowered:
            return "traffic_sign"
        if any(token in lowered for token in ("light", "signal")):
            return "traffic_light"
        return "car"
