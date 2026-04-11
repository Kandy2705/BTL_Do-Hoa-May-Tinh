from __future__ import annotations

import importlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Import Scene class
from core.GameObject import GameObject, GameObjectOBJ, GameObjectLight, GameObjectCamera, GameObjectMath


ShaderPaths = Tuple[str, str]


def _default_shader_paths() -> ShaderPaths:
    return ("./shaders/standard.vert", "./shaders/standard.frag")


from libs.transform import Trackball, quaternion_from_axis_angle, quaternion_slerp
from libs.loss_functions import LOSS_FUNCTIONS
class AppModel:
    SGD_PRESETS: Dict[str, Dict[str, Any]] = {
        "Stable Classroom": {
            "learning_rate": 0.003,
            "momentum": 0.60,
            "batch_size": 16,
            "max_iterations": 8000,
            "simulation_speed": 1,
            "trail_width": 1.0,
        },
        "Fast Convergence": {
            "learning_rate": 0.008,
            "momentum": 0.80,
            "batch_size": 24,
            "max_iterations": 5000,
            "simulation_speed": 2,
            "trail_width": 1.1,
        },
        "Oscillation Demo": {
            "learning_rate": 0.025,
            "momentum": 0.15,
            "batch_size": 1,
            "max_iterations": 4000,
            "simulation_speed": 1,
            "trail_width": 1.3,
        },
        "Divergence Demo": {
            "learning_rate": 0.10,
            "momentum": 0.00,
            "batch_size": 1,
            "max_iterations": 1200,
            "simulation_speed": 1,
            "trail_width": 1.4,
        },
    }

    def __init__(self) -> None:
        from components.scene import Scene
        # Khối state này mô tả "ứng dụng đang ở đâu":
        # đang chọn category nào, shape nào, shader nào, object nào...
        self.selected_idx: int = -1  # -1 means no shape selected
        self.selected_category: int = 5  # 5: Normal mode (default)
        self.selected_shader: int = 0

        self.active_drawable: Optional[Any] = None
        self.drawables: List[Any] = []
        
        self.math_function: str = "(x**2 + y - 11)**2 + (x + y**2 - 7)**2"  # mac dinh
        self.model_filename: str = ""  # For .obj/.ply files
        self.texture_filename: str = ""  # For texture files
        self.object_color: Tuple[float, float, float] = (1.0, 0.5, 0.0)  # Default orange color
        
        # Object type for conditional components
        self.object_type: str = "mesh"  # "mesh", "light", "camera"
        self.selected_hierarchy_idx = -1  # -1 means no hierarchy object selected

        self.object_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Default white color

        self.active_tool = 'select'
        
        # === SGD Visualization State ===
        # Toàn bộ phần BTL 1 - Phần 2 được giữ state tập trung ở đây
        # để controller và viewer đều đọc cùng một nguồn dữ liệu.
        self.sgd_visualizer = None
        self.sgd_loss_function = "Himmelblau"
        self.sgd_learning_rate = 0.001
        self.sgd_momentum = 0.01
        self.sgd_batch_size = 1
        self.sgd_max_iterations = 10000
        self.sgd_simulation_speed = 1
        self.sgd_show_trajectory = True
        self.sgd_show_projected_trajectory = True
        self.sgd_show_drop_lines = True
        self.sgd_show_contours = True
        self.sgd_view_mode = "combined"  # surface / contour / combined
        self.sgd_trail_width = 1.0
        self.sgd_colorblind_mode = False
        self.sgd_replay_enabled = False
        self.sgd_replay_step = 0
        self.sgd_selected_preset = "Custom"
        self.sgd_hover_enabled = True
        self.sgd_hover_info = None
        self.sgd_wireframe_mode = 0  # 0: fill, 1: wireframe, 2: point
        self.sgd_optimizers_enabled = {
            'GD': True,
            'SGD': True,
            'MiniBatch': True,
            'Momentum': True,
            'Nesterov': True,
            'Adam': True,
        }
        self.sgd_chart_visible = {
            'GD': True,
            'SGD': True,
            'MiniBatch': True,
            'Momentum': True,
            'Nesterov': True,
            'Adam': True,
        }
        # Himmelblau minima at (3,2), (-2.8,3.1), (-3.8,-3.3), (3.6,-1.8)
        self.sgd_initial_positions = {
            'GD': [4.5, 4.0],        # Đỏ
            'SGD': [-4.0, 4.0],      # Xanh lá
            'MiniBatch': [4.0, -4.0], # Xanh dương
            'Momentum': [-4.0, -4.0], # Vàng
            'Nesterov': [4.5, -4.0], # Cam
            'Adam': [0.0, 4.5],      # Hồng
        }
        self.sgd_simulation_running = False
        self.sgd_step_count = 0
        
        # --- THÊM DÒNG NÀY: Công tắc chuyển đổi RGB / Depth Map ---
        self.display_mode = 0  # 0: RGB tiêu chuẩn, 1: Depth Map

        # === BTL 2 Synthetic Dataset Bridge State ===
        # Các state này là chiếc cầu nối giữa app tương tác của BTL 1
        # và pipeline sinh dataset tự động của BTL 2 trong thư mục btl2/.
        self.btl2_config_path: str = "configs/btl2/showcase_full.yaml"
        self.btl2_output_dir: str = "outputs/btl2/showcase_dataset"
        self.btl2_num_frames: int = 20
        self.btl2_seed: int = 42
        self.btl2_source_mode: str = "current_scene"
        self.btl2_last_status: str = "Idle: chua chay BTL2."
        self.btl2_last_result: Dict[str, Any] = {}
        self.btl2_scene_camera_count: int = 0
        self.btl2_scene_renderable_count: int = 0
        self.btl2_preview_camera_state: Dict[str, Any] = {}
        self.btl2_detector_weight_path: str = self._find_latest_yolo_weight()
        self.btl2_detector_loaded_path: str = ""
        self.btl2_detector_loaded_mtime: float = 0.0
        self.btl2_preview_mode: str = "rgb"
        self.btl2_preview_source_image_path: str = self._find_default_preview_image()
        self.btl2_preview_path: str = ""
        self.btl2_preview_status: str = "Idle: chua co dataset preview."
        self.btl2_inference_image_path: str = self._find_default_preview_image()
        self.btl2_inference_preview_path: str = ""
        self.btl2_inference_status: str = "Idle: chua load detector."
        self.btl2_inference_summary: str = ""
        self.btl2_inference_device: str = self._detect_btl2_inference_device()
        self.btl2_inference_last_result: Dict[str, Any] = {}
        self._btl2_detector: Optional[Any] = None
        self.refresh_btl2_preview()

        # === Lab: Sphere quaternion SLERP animation ===
        self.lab_slerp_enabled: bool = False                    # Bật/tắt animation quay tròn
        self.lab_slerp_loop_seconds: float = 2.6            # Thời gian 1 vòng quay (0°→180°)
        self.lab_slerp_radius: float = 2.0                   # Bán kính đường tròn quay
        self.lab_slerp_center_xy: Tuple[float, float] = (0.0, 0.0)  # Tâm quay trong mặt phẳng XY
        self.lab_slerp_start_time: float = time.perf_counter()        # Thời điểm bắt đầu animation
        self.lab_slerp_targets: Dict[Any, Dict[str, Any]] = {}   # Danh sách các sphere cần animate
        self.lab_slerp_active_info: Optional[Dict[str, float]] = None # Info của sphere đang active
        
        # Initialize Scene
        self.scene = Scene()
        
        # Hierarchy objects list - REFACTORED: Each object is now a GameObject with proper structure
        self.hierarchy_objects = []
        
        # Component data for mesh objects
        self.mesh_components = {
            "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
            "mesh_renderer": {"shader": 0, "texture": "", "color": [1.0, 0.5, 0.0]}
        }

    def menu_options(self) -> List[str]:
        if self.selected_category == 0:  # 2D
            return [
                "Triangle",
                "Rectangle", 
                "Pentagon",
                "Hexagon",
                "Circle",
                "Ellipse",
                "Trapezoid",
                "Star",
                "Arrow",
            ]
        elif self.selected_category == 1:  # 3D
            return [
                "Cube",
                "Sphere (Tetrahedron)",
                "Sphere (Grid)",
                "Sphere (Lat-Long)",
                "Cylinder",
                "Cone",
                "Truncated Cone",
                "Tetrahedron",
                "Torus",
                "Prism",
            ]
        elif self.selected_category == 2:
            return [
                "Mathematical Surface z=f(x,y)",
            ]
        elif self.selected_category == 3: 
            return [
                "Model from .obj/.ply file",
            ]
        elif self.selected_category == 6:
            return ["Synthetic Road Scene Generator"]
        else:
            return ["SGD Visualization"]

    def shader_names(self) -> List[str]:
        return ["Solid Color", "Gouraud", "Phong", "Rainbow Interpolation"]

    def _shape_factories(self) -> List[Tuple[str, str]]:
        if self.selected_category == 0:  # 2D
            return [
                ("geometry.2d.triangle2d", "Triangle"),
                ("geometry.2d.rectangle2d", "Rectangle"),
                ("geometry.2d.pentagon2d", "Pentagon"),
                ("geometry.2d.hexagon2d", "Hexagon"),
                ("geometry.2d.circle2d", "Circle"),
                ("geometry.2d.ellipse2d", "Ellipse"),
                ("geometry.2d.trapezoid2d", "Trapezoid"),
                ("geometry.2d.star2d", "Star"),
                ("geometry.2d.arrow2d", "Arrow"),
            ]
        elif self.selected_category == 1:  # 3D
            return [
                ("geometry.3d.cube3d", "Cube"),
                ("geometry.3d.sphere_tetrahedron3d", "SphereTetrahedron"),
                ("geometry.3d.sphere_grid3d", "SphereGrid"),
                ("geometry.3d.sphere_latlong3d", "SphereLatLong"),
                ("geometry.3d.cylinder3d", "Cylinder"),
                ("geometry.3d.cone3d", "Cone"),
                ("geometry.3d.truncated_cone3d", "TruncatedCone"),
                ("geometry.3d.tetrahedron3d", "Tetrahedron"),
                ("geometry.3d.torus3d", "Torus"),
                ("geometry.3d.prism3d", "Prism"),
            ]
        elif self.selected_category == 2:  # Mathematical Surface
            return [
                ("geometry.math_surface3d", "MathematicalSurface"),
            ]
        elif self.selected_category == 3:  # Model from file
            return [
                ("geometry.model_loader3d", "ModelLoader"),
            ]
        elif self.selected_category == 6:  # BTL 2 bridge mode
            return [("", "")]
        else:  # SGD
            return [("", "")]

    def _shader_paths(self) -> ShaderPaths:
        # Hàm này ánh xạ shader mode đang chọn sang cặp file shader tương ứng.
        if self.selected_shader == 0:
            return ("./shaders/color_interp.vert", "./shaders/color_interp.frag")
        if self.selected_shader == 1:
            return ("./shaders/gouraud.vert", "./shaders/gouraud.frag")
        if self.selected_shader == 2:
            return ("./shaders/phong.vert", "./shaders/phong.frag")
        return _default_shader_paths()

    def load_active_drawable(self) -> None:
        # Đây là hàm tải lại shape đang được preview ở menu bên trái.
        # Ý tưởng là: người dùng đổi shape/shader/category -> tạo lại drawable mới cho sạch state.
        self.active_drawable = None
        self.drawables: List[Any] = []

        if self.selected_idx == -1:  # No shape selected
            return

        if not (0 <= self.selected_idx < len(self._shape_factories())):
            return

        module_name, class_name = self._shape_factories()[self.selected_idx]
        if not module_name or not class_name:
            return

        try:
            # Import động giúp cùng một code có thể tải nhiều shape khác nhau
            # chỉ dựa trên tên module và tên class.
            module = importlib.import_module(module_name)
            shape_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"[AppModel] failed to load {module_name}.{class_name}: {e}")
            return

        vert_shader, frag_shader = self._shader_paths()
        
        if class_name == "MathematicalSurface":
            try:
                # Với MathematicalSurface, người dùng nhập chuỗi công thức.
                # Ở đây chuỗi đó được biến thành hàm f(x, y) để generate mesh.
                import numpy as np
                safe_dict = {
                    'x': None, 'y': None,
                    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                    'pi': np.pi, 'e': np.e,
                    'abs': np.abs, 'min': np.minimum, 'max': np.maximum
                }
                func_str = f"def f(x, y): return {self.math_function}"
                exec(func_str, safe_dict)
                func = safe_dict['f']
                drawable = shape_cls(vert_shader, frag_shader, func=func)
            except Exception as e:
                print(f"Error parsing math function: {e}")
                print("Using default function instead")
                drawable = shape_cls(vert_shader, frag_shader)
        elif class_name == "ModelLoader":
            if self.model_filename:
                drawable = shape_cls(vert_shader, frag_shader, filename=self.model_filename)
            else:
                print("No model file specified, using default cube")
                drawable = shape_cls(vert_shader, frag_shader)
        else:
            drawable = shape_cls(vert_shader, frag_shader)
            
        drawable.setup()
        # Mode 3 đang là Rainbow Interpolation trên standard shader.
        if hasattr(drawable, 'render_mode') and self.selected_shader == 3:
            drawable.render_mode = 3
        self.drawables.append(drawable)
        self.active_drawable = drawable

    def set_selected(self, idx: int) -> None:
        if idx == self.selected_idx:
            return
        self.selected_idx = idx
        self.load_active_drawable()

    def set_category(self, category: int) -> None:
        if category == self.selected_category:
            return
        self.selected_category = category
        self.selected_idx = -1  # Don't auto-select first shape
        self.load_active_drawable()

    def set_shader(self, shader_idx: int) -> None:
        if shader_idx == self.selected_shader:
            return
        self.selected_shader = shader_idx
        self.load_active_drawable()

    def set_math_function(self, func_str: str) -> None:
        if func_str != self.math_function:
            self.math_function = func_str

    def set_model_filename(self, filename: str) -> None:
        if filename != self.model_filename:
            self.model_filename = filename

    def set_texture_filename(self, filename: str) -> None:
        """Set texture file path"""
        if filename != self.texture_filename:
            self.texture_filename = filename
            # Update drawable texture if active drawable exists
            if self.active_drawable and hasattr(self.active_drawable, 'set_texture'):
                self.active_drawable.set_texture(filename)

    def sync_btl2_config(self) -> None:
        """Đọc config BTL 2 để panel trong app cũ phản ánh đúng file YAML hiện tại."""
        from btl2.utils.io import load_yaml

        cfg = load_yaml(self.btl2_config_path)
        if cfg:
            self.btl2_output_dir = cfg.get("output_dir", self.btl2_output_dir)
            self.btl2_num_frames = int(cfg.get("num_frames", self.btl2_num_frames))
            self.btl2_seed = int(cfg.get("seed", self.btl2_seed))
        self.refresh_btl2_scene_summary()
        self._refresh_btl2_inference_defaults()

    def refresh_btl2_scene_summary(self) -> None:
        """Đếm nhanh scene hiện tại có bao nhiêu camera và object render được cho BTL 2."""
        self.btl2_scene_camera_count = len([obj for obj in self.scene.objects if hasattr(obj, 'camera_fov')])
        self.btl2_scene_renderable_count = len(
            [obj for obj in self.scene.objects if hasattr(obj, 'drawable') and getattr(obj, 'drawable', None) is not None]
        )

    def run_btl2_generator(self) -> dict[str, Any]:
        """Gọi pipeline BTL 2 ngay từ app BTL 1 để hai phần liên thông với nhau."""
        from btl2.app import SyntheticRoadApp
        from btl2.utils.io import load_yaml

        cfg = load_yaml(self.btl2_config_path)
        cfg["output_dir"] = self.btl2_output_dir
        cfg["num_frames"] = int(self.btl2_num_frames)
        cfg["seed"] = int(self.btl2_seed)

        app = SyntheticRoadApp(cfg)
        try:
            summaries = app.generate_dataset(int(self.btl2_num_frames))
        finally:
            app.close()

        self.btl2_last_result = {
            "mode": "procedural_demo",
            "config_path": self.btl2_config_path,
            "output_dir": self.btl2_output_dir,
            "num_frames": int(self.btl2_num_frames),
            "seed": int(self.btl2_seed),
            "generated_frames": len(summaries),
            "first_frame": summaries[0] if summaries else None,
        }
        self.btl2_last_status = f"Done: generated {len(summaries)} frames -> {self.btl2_output_dir}"
        self._refresh_btl2_inference_defaults()
        return self.btl2_last_result

    def run_btl2_from_current_scene(self) -> dict[str, Any]:
        """Xuất dataset trực tiếp từ scene đang dựng trong BTL 1 qua các camera đã đặt."""
        from btl2.app import SyntheticRoadApp
        from btl2.utils.io import load_yaml

        self.refresh_btl2_scene_summary()
        cfg = load_yaml(self.btl2_config_path)
        cfg["output_dir"] = self.btl2_output_dir
        cfg["num_frames"] = int(self.btl2_num_frames)
        cfg["seed"] = int(self.btl2_seed)

        app = SyntheticRoadApp(cfg)
        try:
            summaries = app.generate_from_btl1_scene(
                self.scene.objects,
                num_frames=int(self.btl2_num_frames),
                base_seed=int(self.btl2_seed),
            )
        finally:
            app.close()

        self.btl2_last_result = {
            "mode": "current_scene",
            "config_path": self.btl2_config_path,
            "output_dir": self.btl2_output_dir,
            "num_frames": len(summaries),
            "seed": int(self.btl2_seed),
            "generated_frames": len(summaries),
            "first_frame": summaries[0] if summaries else None,
        }
        self.btl2_last_status = f"Done: exported {len(summaries)} frames from current BTL1 scene -> {self.btl2_output_dir}"
        self._refresh_btl2_inference_defaults()
        return self.btl2_last_result

    @staticmethod
    def _detect_btl2_inference_device() -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _image_candidates(image_dir: Path) -> List[Path]:
        image_paths: List[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(sorted(image_dir.glob(pattern)))
        return sorted(image_paths)

    @staticmethod
    def _yolo_label_for_image(image_path: Path) -> Optional[Path]:
        """Return the matching YOLO label path for a dataset image, if any."""
        try:
            split = image_path.parent.name
            dataset_root = image_path.parents[2]
        except IndexError:
            return None

        for labels_dirname in ("labels_yolo", "labels"):
            candidate = dataset_root / labels_dirname / split / f"{image_path.stem}.txt"
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def _image_has_yolo_annotations(cls, image_path: Path) -> bool:
        label_path = cls._yolo_label_for_image(image_path)
        if label_path is None:
            return False
        try:
            return bool(label_path.read_text(encoding="utf-8").strip())
        except OSError:
            return False

    def _find_latest_yolo_weight(self) -> str:
        weights_root = Path("outputs/training/yolo")
        if not weights_root.exists():
            return ""

        candidates = sorted(
            weights_root.glob("*/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return str(candidates[0]) if candidates else ""

    def _find_default_inference_image(self) -> str:
        candidate_roots = [
            Path(self.btl2_output_dir),
            Path("outputs/btl2/showcase_finetune_dataset"),
            Path("outputs/btl2/showcase_dataset"),
            Path("outputs/btl2/unity_dataset"),
            Path("outputs/btl2/demo_dataset"),
        ]
        seen: set[str] = set()
        fallback_image = ""
        for root in candidate_roots:
            if not root:
                continue
            root_resolved = str(root)
            if root_resolved in seen:
                continue
            seen.add(root_resolved)
            image_paths = self._image_candidates(root / "images" / "train")
            if not image_paths:
                continue
            if not fallback_image:
                fallback_image = str(image_paths[0])
            for image_path in image_paths:
                if self._image_has_yolo_annotations(image_path):
                    return str(image_path)
        return fallback_image

    def _find_current_output_image(self) -> str:
        """Return the first RGB frame from the currently selected BTL2 output dir."""
        output_root = Path(self.btl2_output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root

        for split in ("train", "val"):
            image_dir = output_root / "images" / split
            preferred = image_dir / "frame_000001.png"
            if preferred.exists():
                return str(preferred)

            image_paths = self._image_candidates(image_dir)
            if image_paths:
                return str(image_paths[0])

        return ""

    def _find_default_preview_image(self) -> str:
        """Preview should follow the latest generated scene, not the fine-tune pool."""
        return self._find_current_output_image() or self._find_default_inference_image()

    def _preview_source_is_current_output(self) -> bool:
        if not self.btl2_preview_source_image_path:
            return False

        try:
            preview_path = Path(self.btl2_preview_source_image_path).expanduser()
            output_root = Path(self.btl2_output_dir).expanduser()
            if not preview_path.is_absolute():
                preview_path = (Path.cwd() / preview_path).resolve()
            else:
                preview_path = preview_path.resolve()
            if not output_root.is_absolute():
                output_root = (Path.cwd() / output_root).resolve()
            else:
                output_root = output_root.resolve()
            preview_path.relative_to(output_root)
            return True
        except (OSError, ValueError):
            return False

    def _refresh_btl2_inference_defaults(self) -> None:
        latest_weight = self._find_latest_yolo_weight()
        if latest_weight and not self.btl2_detector_weight_path:
            self.btl2_detector_weight_path = latest_weight

        preview_image = self._find_default_preview_image()
        if preview_image:
            self.btl2_preview_source_image_path = preview_image
            self.btl2_inference_image_path = preview_image
        self.refresh_btl2_preview()

    @staticmethod
    def _resolve_btl2_mask_path(dataset_root: Path, split: str, stem: str) -> Optional[Path]:
        mask_dir = dataset_root / "masks" / split
        for candidate in (
            mask_dir / f"{stem}_mask.png",
            mask_dir / f"{stem}_layer.png",
            mask_dir / f"{stem}.png",
        ):
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _render_btl2_boxes_preview(image_path: Path, metadata_path: Path, output_path: Path) -> None:
        import json
        from PIL import Image, ImageDraw

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        boxes = metadata.get("bounding_boxes", [])
        if not boxes and metadata.get("objects"):
            boxes = [
                {
                    "bbox_xywh": obj.get("bbox_pixels"),
                    "class_name": obj.get("class_name", "unknown"),
                }
                for obj in metadata.get("objects", [])
                if obj.get("bbox_pixels")
            ]

        for bbox in boxes:
            bbox_xywh = bbox.get("bbox_xywh")
            if not bbox_xywh:
                continue
            x, y, w, h = bbox_xywh
            draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 0), width=2)
            draw.text((x + 4, y + 4), bbox.get("class_name", "unknown"), fill=(255, 255, 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    @staticmethod
    def _render_btl2_depth_proxy(image_path: Path, output_path: Path) -> None:
        """Create a lightweight depth preview for augmented datasets without real depth files."""
        from PIL import Image, ImageDraw

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        depth = Image.new("L", (width, height), 220)
        draw = ImageDraw.Draw(depth)

        # Background gradient: soft depth preview, not full black/white contrast.
        for y in range(height):
            value = int(235 - 45 * (y / max(height - 1, 1)))
            draw.line((0, y, width, y), fill=value)

        label_path = AppModel._yolo_label_for_image(image_path)
        if label_path is not None and label_path.exists():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    _, x_center, y_center, box_width, box_height = [float(part) for part in parts]
                except ValueError:
                    continue
                x = (x_center - box_width * 0.5) * width
                y = (y_center - box_height * 0.5) * height
                w = box_width * width
                h = box_height * height
                # Larger/lower boxes are usually closer in this dashcam-style scene.
                closeness = min(1.0, max(0.0, y_center + box_height * 0.45))
                value = int(220 - 80 * closeness)
                draw.rectangle((x, y, x + w, y + h), fill=value)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth.save(output_path)

    @staticmethod
    def _render_btl2_mask_preview(mask_path: Path, metadata_path: Path, output_path: Path) -> None:
        """Render a human-readable color preview for low-valued instance masks."""
        import json
        from PIL import Image
        from btl2.utils.colors import class_color

        mask = Image.open(mask_path).convert("RGB")
        mask_np = np.asarray(mask, dtype=np.uint8)
        preview_np = np.zeros_like(mask_np)

        mapping: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                mapping = payload.get("segmentation_mapping", {}) or {}
            except (OSError, json.JSONDecodeError):
                mapping = {}

        fallback_palette = [
            (70, 200, 120),
            (220, 70, 70),
            (255, 140, 70),
            (160, 110, 80),
            (70, 150, 240),
            (90, 150, 255),
            (255, 210, 60),
        ]

        flat_mask = mask_np.reshape(-1, 3)
        flat_preview = preview_np.reshape(-1, 3)
        unique_colors = np.unique(flat_mask, axis=0)
        palette_idx = 0
        for color in unique_colors:
            if not np.any(color):
                continue
            key = f"{int(color[0])}_{int(color[1])}_{int(color[2])}"
            class_name = mapping.get(key, {}).get("class_name")
            if class_name:
                draw_color = class_color(class_name)
            else:
                draw_color = fallback_palette[palette_idx % len(fallback_palette)]
                palette_idx += 1
            matches = np.all(flat_mask == color, axis=1)
            flat_preview[matches] = np.asarray(draw_color, dtype=np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview_np, mode="RGB").save(output_path)

    def set_btl2_preview_mode(self, mode: str) -> None:
        if mode not in {"rgb", "depth", "mask", "boxes"}:
            return
        if mode == self.btl2_preview_mode:
            return
        self.btl2_preview_mode = mode
        self.refresh_btl2_preview()

    def refresh_btl2_preview(self) -> None:
        image_path = Path(self.btl2_preview_source_image_path).expanduser() if self.btl2_preview_source_image_path else None
        current_output_image = self._find_current_output_image()
        if current_output_image and not self._preview_source_is_current_output():
            image_path = Path(current_output_image)
        elif image_path is None or not str(image_path):
            default_image = self._find_default_preview_image()
            image_path = Path(default_image).expanduser() if default_image else None

        if image_path is None:
            self.btl2_preview_path = ""
            self.btl2_preview_status = "Idle: chua tim thay frame mau de preview."
            return

        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        if not image_path.exists():
            self.btl2_preview_path = ""
            self.btl2_preview_status = f"Missing preview source image: {image_path}"
            return

        self.btl2_preview_source_image_path = str(image_path)
        try:
            split = image_path.parent.name
            dataset_root = image_path.parents[2]
        except IndexError:
            self.btl2_preview_path = ""
            self.btl2_preview_status = f"Invalid preview image path: {image_path}"
            return

        stem = image_path.stem
        preview_path: Optional[Path] = None
        mode_label = self.btl2_preview_mode.upper()

        if self.btl2_preview_mode == "rgb":
            preview_path = image_path
        elif self.btl2_preview_mode == "depth":
            candidate = dataset_root / "depth" / split / f"{stem}_depth.png"
            if candidate.exists():
                preview_path = candidate
            else:
                candidate = dataset_root / "previews" / f"{stem}_depth_proxy.png"
                if not candidate.exists() or candidate.stat().st_mtime < image_path.stat().st_mtime:
                    self._render_btl2_depth_proxy(image_path, candidate)
                preview_path = candidate
        elif self.btl2_preview_mode == "mask":
            mask_path = self._resolve_btl2_mask_path(dataset_root, split, stem)
            metadata_path = dataset_root / "metadata" / split / f"{stem}.json"
            if mask_path is not None:
                candidate = dataset_root / "previews" / f"{stem}_mask_preview.png"
                source_mtime = mask_path.stat().st_mtime
                if metadata_path.exists():
                    source_mtime = max(source_mtime, metadata_path.stat().st_mtime)
                if not candidate.exists() or candidate.stat().st_mtime < source_mtime:
                    self._render_btl2_mask_preview(mask_path, metadata_path, candidate)
                preview_path = candidate
        elif self.btl2_preview_mode == "boxes":
            metadata_path = dataset_root / "metadata" / split / f"{stem}.json"
            candidate = dataset_root / "previews" / f"{stem}_boxes.png"
            if metadata_path.exists():
                if not candidate.exists() or candidate.stat().st_mtime < max(image_path.stat().st_mtime, metadata_path.stat().st_mtime):
                    self._render_btl2_boxes_preview(image_path, metadata_path, candidate)
                preview_path = candidate

        if preview_path is None or not preview_path.exists():
            self.btl2_preview_path = ""
            self.btl2_preview_status = f"Missing {mode_label} preview for {stem}."
            return

        self.btl2_preview_path = str(preview_path)
        self.btl2_preview_status = f"Previewing {mode_label}: {preview_path.name}"

    def load_btl2_detector(self) -> Dict[str, Any]:
        from ultralytics import YOLO

        weight_path = Path(self.btl2_detector_weight_path).expanduser()
        if not weight_path.is_absolute():
            weight_path = Path.cwd() / weight_path
        if not weight_path.exists():
            raise FileNotFoundError(f"Khong tim thay weight file: {weight_path}")

        current_mtime = float(weight_path.stat().st_mtime)
        needs_reload = (
            self._btl2_detector is None
            or self.btl2_detector_loaded_path != str(weight_path)
            or abs(self.btl2_detector_loaded_mtime - current_mtime) > 1e-6
        )

        if needs_reload:
            self._btl2_detector = YOLO(str(weight_path))
            self.btl2_detector_loaded_path = str(weight_path)
            self.btl2_detector_loaded_mtime = current_mtime

        self.btl2_inference_status = f"Loaded detector: {weight_path.name} ({self.btl2_inference_device})"
        return {
            "weights": self.btl2_detector_loaded_path,
            "device": self.btl2_inference_device,
        }

    def run_btl2_inference(self) -> Dict[str, Any]:
        image_path = Path(self.btl2_inference_image_path).expanduser()
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Khong tim thay anh de inference: {image_path}")

        self.load_btl2_detector()
        assert self._btl2_detector is not None

        results = self._btl2_detector.predict(
            source=str(image_path),
            device=self.btl2_inference_device,
            verbose=False,
        )
        if not results:
            raise RuntimeError("YOLO khong tra ve ket qua nao.")

        result = results[0]
        output_dir = Path("outputs/inference/ui")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_pred.jpg"

        if hasattr(result, "save"):
            result.save(filename=str(output_path))
        else:
            plotted = result.plot()
            if hasattr(plotted, "save"):
                plotted.save(output_path)
            else:
                from PIL import Image

                Image.fromarray(plotted).save(output_path)

        class_names = getattr(result, "names", None) or getattr(self._btl2_detector, "names", {})
        detections = 0
        summary = "No detections."
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_ids = [int(v) for v in boxes.cls.tolist()]
            detections = len(cls_ids)
            if detections:
                counts = Counter(class_names.get(idx, str(idx)) for idx in cls_ids)
                summary = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))
            elif not self._image_has_yolo_annotations(image_path):
                summary = "No detections. Selected image also has no YOLO labels; choose Use Sample Image again."

        self.btl2_inference_preview_path = str(output_path)
        self.btl2_inference_summary = summary
        self.btl2_inference_last_result = {
            "image": str(image_path),
            "preview": str(output_path),
            "weights": self.btl2_detector_loaded_path,
            "detections": detections,
            "summary": summary,
            "device": self.btl2_inference_device,
        }
        self.btl2_inference_status = f"Done: inferred {image_path.name} with {detections} detection(s)."
        return self.btl2_inference_last_result

    def _clear_scene_objects_for_btl2_preview(self) -> None:
        # Dọn scene hiện tại để preview procedural xuất hiện rõ ràng, không chồng dữ liệu cũ.
        self.scene.objects.clear()
        self.scene.selected_objects.clear()
        self.hierarchy_objects.clear()
        self.selected_hierarchy_idx = -1

    @staticmethod
    def _map_btl2_class_to_btl1_shape(class_name: str) -> str:
        mapping = {
            "road": "Cube",
            "car": "Cube",
            "bus": "Cube",
            "truck": "Cube",
            "motorbike": "Cube",
            "person": "Cylinder",
            "traffic_sign": "Cube",
            "traffic_light": "Cube",
        }
        return mapping.get(class_name, "Cube")

    @staticmethod
    def _resolve_btl2_preview_model_path(scene_obj: Any) -> Optional[str]:
        metadata = getattr(scene_obj, "metadata", {}) or {}
        source_asset = metadata.get("source_asset")
        if not source_asset:
            return None

        asset_path = (Path("assets/models") / source_asset).resolve()
        if asset_path.suffix.lower() not in {".obj", ".ply"}:
            return None
        if not asset_path.exists():
            return None
        return str(asset_path)

    def _build_btl2_preview_object_spec(self, scene_obj: Any) -> Dict[str, str]:
        model_filename = self._resolve_btl2_preview_model_path(scene_obj)
        if model_filename:
            return {
                "obj_type": "custom_model",
                "shape_name": "Cube",
                "model_filename": model_filename,
            }
        return {
            "obj_type": "3d",
            "shape_name": self._map_btl2_class_to_btl1_shape(scene_obj.class_name),
            "model_filename": "",
        }

    @staticmethod
    def _match_btl2_preview_scale(source_obj: Any, created_obj: Any) -> List[float]:
        source_scale = np.asarray(source_obj.scale, dtype=np.float32)
        source_aabb = getattr(source_obj, "aabb_local", None)
        if source_aabb is not None:
            source_extent = np.asarray(source_aabb.max_corner - source_aabb.min_corner, dtype=np.float32)
            target_world_size = [float(v) for v in (source_extent * source_scale).tolist()]
        else:
            target_world_size = [float(v) for v in source_scale.tolist()]
        drawable = getattr(created_obj, "drawable", None)
        if drawable is None or not hasattr(drawable, "vertices"):
            return target_world_size

        drawable_vertices = getattr(drawable, "vertices", None)
        if drawable_vertices is None or len(drawable_vertices) == 0:
            return target_world_size

        preview_extent = np.max(drawable_vertices, axis=0) - np.min(drawable_vertices, axis=0)

        matched_scale: List[float] = []
        for idx, value in enumerate(target_world_size):
            dst_extent = float(preview_extent[idx]) if idx < len(preview_extent) else 0.0
            if dst_extent > 1e-6:
                matched_scale.append(float(value / dst_extent))
            else:
                matched_scale.append(float(value))
        return matched_scale

    def load_btl2_procedural_preview_into_scene(self) -> dict[str, Any]:
        """Nạp 1 frame procedural từ BTL2 vào scene BTL1 để xem trực tiếp trong viewport."""
        from btl2.scene.road_scene_builder import RoadSceneBuilder
        from btl2.utils.io import load_yaml
        import numpy as np

        cfg = load_yaml(self.btl2_config_path)
        cfg["seed"] = int(self.btl2_seed)
        cfg["num_frames"] = max(1, int(cfg.get("num_frames", self.btl2_num_frames)))
        cfg["output_dir"] = self.btl2_output_dir

        builder = RoadSceneBuilder(cfg)
        preview_scene, _ = builder.build_scene(frame_index=0)
        self.btl2_preview_camera_state = {
            "position": preview_scene.camera.position.tolist(),
            "target": preview_scene.camera.target.tolist(),
            "up": preview_scene.camera.up.tolist(),
            "fov": float(preview_scene.camera.fov_y_degrees),
            "near": float(preview_scene.camera.near),
            "far": float(preview_scene.camera.far),
        }

        self._clear_scene_objects_for_btl2_preview()

        # 1) Add procedural objects as BTL1 mesh objects.
        added_mesh = 0
        for obj in preview_scene.objects:
            preview_spec = self._build_btl2_preview_object_spec(obj)
            self.add_hierarchy_object(
                obj.name,
                preview_spec["obj_type"],
                preview_spec["shape_name"],
                model_filename=preview_spec["model_filename"],
            )
            created = self.scene.objects[-1]

            created.position = [float(v) for v in obj.position.tolist()]
            created.rotation = [float(v) for v in obj.rotation_degrees.tolist()]
            created.scale = self._match_btl2_preview_scale(obj, created)
            if preview_spec["obj_type"] == "custom_model":
                created.color = [1.0, 1.0, 1.0, 1.0]
            else:
                base_color = obj.base_color.tolist()
                created.color = [float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0]
            created.shader = 2  # Phong
            self._sync_scene_object_visuals(created)
            added_mesh += 1

        # 2) Add one light from BTL2 directional light (approximate for BTL1 point-light model).
        self.add_hierarchy_object("BTL2 Light", "light")
        light_obj = self.scene.objects[-1]
        light_dir = np.asarray(preview_scene.light.direction, dtype=np.float32)
        light_obj.position = [
            float(-light_dir[0] * 8.0),
            float(max(2.0, abs(light_dir[1]) * 8.0)),
            float(-light_dir[2] * 8.0),
        ]
        light_obj.light_color = [float(v) for v in preview_scene.light.color.tolist()]
        light_obj.light_intensity = float(preview_scene.light.intensity)

        # 3) Add one camera object for dataset context (viewer vẫn dùng Scene Camera mặc định khi ở BTL2 tab).
        self.add_hierarchy_object("BTL2 Dashcam", "camera")
        cam_obj = self.scene.objects[-1]
        cam_obj.camera_fov = float(preview_scene.camera.fov_y_degrees)
        cam_obj.camera_near = float(preview_scene.camera.near)
        cam_obj.camera_far = float(preview_scene.camera.far)
        cam_pos = np.asarray(preview_scene.camera.position, dtype=np.float32)
        cam_target = np.asarray(preview_scene.camera.target, dtype=np.float32)
        cam_obj.position = [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])]

        if hasattr(cam_obj, "trackball"):
            cam_obj.trackball.fov = cam_obj.camera_fov
            cam_obj.trackball.near = cam_obj.camera_near
            cam_obj.trackball.far = cam_obj.camera_far
            # Đặt khoảng cách gần tương đương khoảng cách eye-target để inspector đọc hợp lý hơn.
            cam_obj.trackball.distance = float(max(np.linalg.norm(cam_pos - cam_target), 0.2))
            if hasattr(cam_obj.trackball, "pos2d"):
                cam_obj.trackball.pos2d[0] = float(cam_pos[0])
                cam_obj.trackball.pos2d[1] = float(cam_pos[1])

        self.refresh_btl2_scene_summary()
        self.btl2_last_status = (
            f"Loaded procedural preview to BTL1 scene: "
            f"{added_mesh} mesh object(s), 1 light, 1 camera."
        )
        self.btl2_last_result = {
            "mode": "procedural_preview_to_btl1",
            "generated_frames": 1,
            "output_dir": self.btl2_output_dir,
            "mesh_objects": added_mesh,
            "cameras": self.btl2_scene_camera_count,
            "renderables": self.btl2_scene_renderable_count,
        }
        return self.btl2_last_result

    @staticmethod
    def _is_sphere_drawable(drawable: Any) -> bool:
        if drawable is None:
            return False
        return "Sphere" in drawable.__class__.__name__

    def _capture_lab_slerp_targets(self) -> None:
        self.lab_slerp_targets = {}
        for obj in self.scene.objects:
            drawable = getattr(obj, "drawable", None)
            if not self._is_sphere_drawable(drawable):
                continue
            pos = list(getattr(obj, "position", [0.0, 0.0, 0.0]))
            rot = list(getattr(obj, "rotation", [0.0, 0.0, 0.0]))
            x = float(pos[0]) if len(pos) >= 1 else 0.0
            y = float(pos[1]) if len(pos) >= 2 else 0.0
            z = float(pos[2]) if len(pos) >= 3 else 0.0
            theta = math.atan2(y, x) if abs(x) + abs(y) > 1e-6 else 0.0
            rot_z = float(rot[2]) if len(rot) >= 3 else 0.0
            self.lab_slerp_targets[obj.id] = {
                "base_theta": theta,
                "base_pos_z": z,
                "base_rot_z": rot_z,
            }

        if self._is_sphere_drawable(self.active_drawable):
            active_pos = list(getattr(self.active_drawable, "position", [0.0, 0.0, 0.0]))
            active_rot = list(getattr(self.active_drawable, "rotation", [0.0, 0.0, 0.0]))
            ax = float(active_pos[0]) if len(active_pos) >= 1 else 0.0
            ay = float(active_pos[1]) if len(active_pos) >= 2 else 0.0
            az = float(active_pos[2]) if len(active_pos) >= 3 else 0.0
            atheta = math.atan2(ay, ax) if abs(ax) + abs(ay) > 1e-6 else 0.0
            arot_z = float(active_rot[2]) if len(active_rot) >= 3 else 0.0
            self.lab_slerp_active_info = {
                "base_theta": atheta,
                "base_pos_z": az,
                "base_rot_z": arot_z,
            }
        else:
            self.lab_slerp_active_info = None

        self.lab_slerp_start_time = time.perf_counter()

    def set_lab_slerp_enabled(self, enabled: bool) -> None:
        """
        BẬT/TẮT CHẾ ĐỘ ANIMATION SLERP.
        
        Args:
            enabled: True để bật, False để tắt
        """
        enabled = bool(enabled)
        if self.lab_slerp_enabled == enabled:
            return  # Tránh thay đổi không cần thiết
        self.lab_slerp_enabled = enabled
        if enabled:
            self._capture_lab_slerp_targets()  # Bật thì capture targets

    def refresh_lab_slerp_targets(self) -> None:
        """
        LÀM MỚI LẠI CÁC TARGETS CHO SLERP.
        
        Gọi lại hàm capture để update lại danh sách targets
        khi có thay đổi trong scene (thêm/xóa sphere).
        """
        self._capture_lab_slerp_targets()

    def update_lab_slerp_animation(self) -> None:
        if not self.lab_slerp_enabled:
            return 

        if not self.lab_slerp_targets and self.lab_slerp_active_info is None:
            self._capture_lab_slerp_targets()
            if not self.lab_slerp_targets and self.lab_slerp_active_info is None:
                return

        loop_s = max(float(self.lab_slerp_loop_seconds), 0.2)
        elapsed = time.perf_counter() - self.lab_slerp_start_time
        phase = (elapsed / loop_s) % 1.0
        t = phase * 2.0 if phase <= 0.5 else (1.0 - phase) * 2.0 

        q_start = quaternion_from_axis_angle((0.0, 0.0, 1.0), degrees=0.0)  # Góc bắt đầu
        q_end = quaternion_from_axis_angle((0.0, 0.0, 1.0), degrees=180.0) # Góc kết thúc
        q_delta = quaternion_slerp(q_start, q_end, t)  # Nội suy quaternion

        # Tính góc quay quanh trục Z
        delta_z_deg = math.degrees(2.0 * math.atan2(float(q_delta[3]), float(q_delta[0]))) # 3 là z; 0 là w
        delta_z_rad = math.radians(delta_z_deg)
        cx, cy = self.lab_slerp_center_xy
        radius = float(self.lab_slerp_radius)

        # Cập nhật vị trí cho tất cả targets
        by_id = {obj.id: obj for obj in self.scene.objects}
        for obj_id, info in self.lab_slerp_targets.items():
            obj = by_id.get(obj_id)
            if obj is None:
                continue
            if not hasattr(obj, "position") or len(obj.position) < 3:
                continue
            if not hasattr(obj, "rotation") or len(obj.rotation) < 3:
                continue
                
            # Tính vị trí mới trên đường tròn
            theta = float(info["base_theta"]) + delta_z_rad
            obj.position[0] = float(cx + radius * math.cos(theta))  # X = tâm + R*cos(θ)
            obj.position[1] = float(cy + radius * math.sin(theta))  # Y = tâm + R*sin(θ)
            obj.position[2] = float(info["base_pos_z"])
            obj.rotation[2] = float(info["base_rot_z"] + delta_z_deg)  # Cập nhật góc Z

        if self.lab_slerp_active_info is not None and self._is_sphere_drawable(self.active_drawable):
            pos = getattr(self.active_drawable, "position", [0.0, 0.0, 0.0])
            rot = getattr(self.active_drawable, "rotation", [0.0, 0.0, 0.0])
            if len(pos) >= 3 and len(rot) >= 3:
                theta = float(self.lab_slerp_active_info["base_theta"]) + delta_z_rad
                pos[0] = float(cx + radius * math.cos(theta))
                pos[1] = float(cy + radius * math.sin(theta))
                pos[2] = float(self.lab_slerp_active_info["base_pos_z"])
                rot[2] = float(self.lab_slerp_active_info["base_rot_z"] + delta_z_deg)

    def set_object_type(self, obj_type: str) -> None:
        """Set object type: 'mesh', 'light', or 'camera'"""
        if obj_type in ["mesh", "light", "camera"]:
            self.object_type = obj_type

    def _sync_scene_object_visuals(self, scene_obj) -> None:
        # Hàm này đồng bộ dữ liệu "mức object" xuống "mức drawable":
        # shader, màu, texture, flat shading...
        drawable = getattr(scene_obj, 'drawable', None)
        if drawable is None:
            return

        if hasattr(drawable, 'render_mode') and hasattr(scene_obj, 'shader'):
            drawable.render_mode = scene_obj.shader

        if hasattr(scene_obj, 'color') and hasattr(drawable, 'set_color'):
            if getattr(scene_obj, 'shader', 0) == 3 and hasattr(drawable, 'restore_auto_colors'):
                drawable.restore_auto_colors()
            else:
                drawable.set_color(scene_obj.color[:3])

        if getattr(scene_obj, 'texture_filename', "") and hasattr(drawable, 'set_texture'):
            drawable.set_texture(scene_obj.texture_filename)

        if hasattr(drawable, 'use_flat_color') and getattr(self, 'global_flat_color_enabled', False):
            drawable.use_flat_color = True
            if hasattr(drawable, 'set_solid_color') and hasattr(scene_obj, 'color'):
                drawable.set_solid_color(scene_obj.color[:3])

    def add_hierarchy_object(
        self,
        name: str,
        obj_type: str,
        shape_name: str = "Cube",
        model_filename: str = "",
    ) -> None:
        from core.GameObject import GameObject, GameObjectOBJ, GameObjectLight, GameObjectCamera, GameObjectMath
        
        # 1. Tạo đúng loại GameObject theo nhu cầu của người dùng.
        # Mesh, light, camera đều được đưa về cùng mô hình scene object để quản lý thống nhất.
        if obj_type == "3d":
            new_obj = GameObjectOBJ(name)
            
            # Tạo drawable theo shape_name
            vert_shader = "./shaders/standard.vert"
            frag_shader = "./shaders/standard.frag"
            
            # Import các class shape 3D
            import sys
            import os
            shape3d_path = os.path.join(os.path.dirname(__file__), 'geometry', '3d')
            if shape3d_path not in sys.path:
                sys.path.insert(0, shape3d_path)
            
            # Map tên hiển thị trên UI sang class thực tế bên thư mục geometry/3d.
            shape_classes = {
                "Cube": "cube3d.Cube",
                "Sphere (Tetrahedron)": "sphere_tetrahedron3d.SphereTetrahedron",
                "Sphere (Grid)": "sphere_grid3d.SphereGrid", 
                "Sphere (Lat-Long)": "sphere_latlong3d.SphereLatLong",
                "Cylinder": "cylinder3d.Cylinder",
                "Cone": "cone3d.Cone",
                "Truncated Cone": "truncated_cone3d.TruncatedCone",
                "Tetrahedron": "tetrahedron3d.Tetrahedron",
                "Torus": "torus3d.Torus",
                "Prism": "prism3d.Prism"
            }
            
            shape_class_name = shape_classes.get(shape_name, "cube3d.Cube")
            module_name, class_name = shape_class_name.split('.')
            
            # Import và tạo instance
            shape_module = __import__(module_name, fromlist=[class_name])
            shape_class = getattr(shape_module, class_name)
            new_obj.drawable = shape_class(vert_shader, frag_shader)
            new_obj.drawable.setup()
            
        elif obj_type == "custom_model":
            new_obj = GameObjectOBJ(name)
            vert_shader = "./shaders/standard.vert"
            frag_shader = "./shaders/standard.frag"

            import sys
            import os
            model_path = os.path.join(os.path.dirname(__file__), 'geometry')
            if model_path not in sys.path:
                sys.path.insert(0, model_path)

            resolved_model = model_filename or self.model_filename
            new_obj.obj_ply_file = resolved_model
            if resolved_model:
                self.model_filename = resolved_model

            from model_loader3d import ModelLoader
            new_obj.drawable = ModelLoader(vert_shader, frag_shader, filename=resolved_model or None)
            new_obj.drawable.setup()
            
        elif obj_type == "2d":
            new_obj = GameObjectOBJ(name)
            
            # Tạo drawable 2D theo shape_name
            vert_shader = "./shaders/standard.vert"
            frag_shader = "./shaders/standard.frag"
            
            # Import các class shape 2D
            import sys
            import os
            shape2d_path = os.path.join(os.path.dirname(__file__), 'geometry', '2d')
            if shape2d_path not in sys.path:
                sys.path.insert(0, shape2d_path)
            
            # Map 2D shape names to classes
            shape_classes_2d = {
                "Triangle": "triangle2d.Triangle",
                "Rectangle": "rectangle2d.Rectangle",
                "Pentagon": "pentagon2d.Pentagon",
                "Hexagon": "hexagon2d.Hexagon",
                "Circle": "circle2d.Circle",
                "Ellipse": "ellipse2d.Ellipse",
                "Trapezoid": "trapezoid2d.Trapezoid",
                "Star": "star2d.Star",
                "Arrow": "arrow2d.Arrow"
            }
            
            shape_class_name = shape_classes_2d.get(shape_name, "triangle2d.Triangle")
            module_name, class_name = shape_class_name.split('.')
            
            # Import và tạo instance
            shape_module = __import__(module_name, fromlist=[class_name])
            shape_class = getattr(shape_module, class_name)
            new_obj.drawable = shape_class(vert_shader, frag_shader)
            new_obj.drawable.setup()
            
        elif obj_type == "math":
            new_obj = GameObjectMath(name)
            # MathematicalSurface cũng là một object trong scene,
            # chỉ khác ở chỗ mesh của nó được sinh từ hàm z = f(x, y).
            vert_shader = "./shaders/standard.vert"
            frag_shader = "./shaders/standard.frag"
            
            # Import MathematicalSurface
            import sys
            import os
            math_path = os.path.join(os.path.dirname(__file__), 'geometry')
            if math_path not in sys.path:
                sys.path.insert(0, math_path)
            
            from math_surface3d import MathematicalSurface
            new_obj.drawable = MathematicalSurface(vert_shader, frag_shader)
            new_obj.drawable.setup()
        elif obj_type == "light":
            new_obj = GameObjectLight(name)
            new_obj.drawable = None # Đèn không cần vẽ lưới (hoặc vẽ icon sau)
        elif obj_type == "camera":
            new_obj = GameObjectCamera(name)
            new_obj.drawable = None
            
            from libs.transform import Trackball
            new_obj.trackball = Trackball()
            
            # --- THÊM 3 DÒNG NÀY ĐỂ NẠP GIÁ TRỊ GỐC CHO CAMERA ---
            new_obj.trackball.fov = new_obj.camera_fov
            new_obj.trackball.near = new_obj.camera_near
            new_obj.trackball.far = new_obj.camera_far
        else:
            new_obj = GameObject(name)
            new_obj.drawable = None

        if getattr(new_obj, 'drawable', None) is not None:
            if hasattr(new_obj.drawable, 'render_mode'):
                new_obj.shader = new_obj.drawable.render_mode
            self._sync_scene_object_visuals(new_obj)
            
        # 2. Thêm vào Scene
        self.scene.add_object(new_obj)
        self.scene.select_object(new_obj)
        
        # 3. Thêm vào hierarchy_objects cho UI compatibility
        hierarchy_obj = {
            "id": new_obj.id,
            "name": new_obj.name,
            "type": obj_type,
            "selected": True,
            "visible": new_obj.visible
        }
        if obj_type == "custom_model":
            hierarchy_obj["model_data"] = {"filename": getattr(new_obj, "obj_ply_file", "")}
        self.hierarchy_objects.append(hierarchy_obj)
    
    def select_hierarchy_object(self, idx: int) -> None:
        if 0 <= idx < len(self.hierarchy_objects):
            self.selected_hierarchy_idx = idx
            for i, obj in enumerate(self.hierarchy_objects):
                obj["selected"] = (i == idx)
        else:
            self.selected_hierarchy_idx = -1

    def get_selected_hierarchy_object(self):
        if 0 <= self.selected_hierarchy_idx < len(self.hierarchy_objects):
            return self.hierarchy_objects[self.selected_hierarchy_idx]
        return None

    def update_object_data(self, obj_id: int, key_path: str, value: Any) -> None:
        object_idx = next((i for i, o in enumerate(self.scene.objects) if o.id == obj_id), -1)
        
        if object_idx != -1:
            target_obj = self.scene.objects[object_idx]

            # --- THÊM ĐOẠN NÀY ĐỂ XỬ LÝ LỆNH BẬT/TẮT CÁC CHẾ ĐỘ ---
            # Xử lý các lệnh bật/tắt state đặc biệt (bool, enum)
            if hasattr(target_obj, 'drawable') and target_obj.drawable:
                
                # Nút "Flat Color" (A)
                if key_path == "u_use_flat_color":
                    target_obj.drawable.use_flat_color = value
                    print(f"[{target_obj.name}] Flat Color: {value}")
                
                # Nút "Gouraud/Phong" (C)
                elif key_path == "u_enable_lighting":
                    target_obj.drawable.render_mode = 2 if value else 0
                    print(f"[{target_obj.name}] Lighting Enabled: {value}")
            # --------------------------------------------------------


                elif key_path == "shader":
                    if hasattr(target_obj, 'drawable') and target_obj.drawable:
                        target_obj.drawable.render_mode = value
                        print(f"[{target_obj.name}] Render Mode changed to: {value}")
    
    def update_selected_object_data(self, key: str, value) -> None:
        """Update data for the selected hierarchy object"""
        if 0 <= self.selected_hierarchy_idx < len(self.hierarchy_objects):
            # Handle nested key updates (e.g., "transform.position")
            if "." in key:
                keys = key.split(".")
                obj = self.hierarchy_objects[self.selected_hierarchy_idx]
                for k in keys[:-1]:
                    obj = obj.setdefault(k, {})
                obj[keys[-1]] = value
            else:
                # Handle direct key updates
                self.hierarchy_objects[self.selected_hierarchy_idx][key] = value
            
            # Special handling for math function or model filename changes
            selected_obj = self.hierarchy_objects[self.selected_hierarchy_idx]
            if key == "math_script" and selected_obj["type"] == "math":
                # Get the math function from the actual GameObject
                scene_obj = None
                for obj in self.scene.objects:
                    if obj.id == selected_obj["id"]:
                        scene_obj = obj
                        break
                
                if scene_obj and hasattr(scene_obj, 'math_script'):
                    math_function = scene_obj.math_script
                    print(f"[DEBUG] Reloading math object with function: {math_function}")
                    # Reload the specific math object in hierarchy
                    self._reload_hierarchy_object(selected_obj, math_function)
                    print("[DEBUG] Math object reloaded")
            elif key == "model_data.filename" and selected_obj["type"] == "custom_model":
                self.model_filename = value  # Update global model filename for compatibility
                scene_obj = next((obj for obj in self.scene.objects if obj.id == selected_obj["id"]), None)
                if scene_obj is not None and hasattr(scene_obj, "obj_ply_file"):
                    scene_obj.obj_ply_file = value
                print(f"[DEBUG] Reloading model object with filename: {value}")
                # Reload the specific model object in hierarchy
                self._reload_hierarchy_object(selected_obj)
                print("[DEBUG] Model object reloaded")
        else:
            # If no hierarchy object selected, update mesh components
            if "." in key:
                keys = key.split(".")
                comp = self.mesh_components
                for k in keys[:-1]:
                    comp = comp.setdefault(k, {})
                comp[keys[-1]] = value
            else:
                self.mesh_components[key] = value


    def set_color(self, color: Tuple[float, float, float]) -> None:
        """Set object color from RGB tuple"""
        self.object_color = color
        # Update drawable color if active drawable exists
        if self.active_drawable and hasattr(self.active_drawable, 'set_color'):
            self.active_drawable.set_color(color)

    def _reload_hierarchy_object(self, hierarchy_obj, math_function=None):
        """Reload a specific hierarchy object with updated parameters"""
        obj_id = hierarchy_obj["id"]
        obj_type = hierarchy_obj["type"]
        obj_name = hierarchy_obj["name"]
        
        print(f"[DEBUG] Reloading object: {obj_name} (type: {obj_type}, id: {obj_id})")
        
        # Find the actual GameObject in scene
        scene_obj = None
        for obj in self.scene.objects:
            if obj.id == obj_id:
                scene_obj = obj
                break
        
        if not scene_obj:
            print(f"[DEBUG] Scene object not found for id: {obj_id}")
            return
        
        print(f"[DEBUG] Found scene object: {scene_obj.name}")
        
        # Recreate the drawable with updated parameters
        if obj_type == "math":
            # Create new MathematicalSurface with updated function
            vert_shader = "./shaders/standard.vert"
            frag_shader = "./shaders/standard.frag"
            
            import sys
            import os
            math_path = os.path.join(os.path.dirname(__file__), 'geometry')
            if math_path not in sys.path:
                sys.path.insert(0, math_path)
            
            from math_surface3d import MathematicalSurface
            
            # Use the provided math function or get from object
            if math_function is None and hasattr(scene_obj, 'math_script'):
                math_function = scene_obj.math_script
            
            # Parse the math function
            import numpy as np
            safe_dict = {
                'x': None, 'y': None,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'e': np.e,
                'abs': np.abs, 'min': np.minimum, 'max': np.maximum
            }
            func_str = f"def f(x, y): return {math_function}"
            print(f"[DEBUG] Parsing math function: {func_str}")
            exec(func_str, safe_dict)
            func = safe_dict['f']
            
            scene_obj.drawable = MathematicalSurface(vert_shader, frag_shader, func=func)
            scene_obj.drawable.setup()
            self._sync_scene_object_visuals(scene_obj)
            print("[DEBUG] MathematicalSurface recreated")
            
        elif obj_type == "custom_model":
            # Create new ModelLoader with updated filename
            vert_shader = "./shaders/standard.vert"
            frag_shader = "./shaders/standard.frag"
            
            import sys
            import os
            model_path = os.path.join(os.path.dirname(__file__), 'geometry')
            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            
            from model_loader3d import ModelLoader
            model_filename = getattr(scene_obj, "obj_ply_file", "") or hierarchy_obj.get("model_data", {}).get("filename", "") or self.model_filename
            scene_obj.obj_ply_file = model_filename
            print(f"[DEBUG] Creating ModelLoader with filename: {model_filename}")
            scene_obj.drawable = ModelLoader(vert_shader, frag_shader, filename=model_filename or None)
            scene_obj.drawable.setup()
            self._sync_scene_object_visuals(scene_obj)
            print("[DEBUG] ModelLoader recreated")
        else:
            print(f"[DEBUG] Unknown object type: {obj_type}")

    def reload_current_shape(self) -> None:
        """Reload the current shape"""
        self.load_active_drawable()

    # === SGD Visualization Methods ===
    
    def init_sgd_visualizer(self):
        """Initialize the SGD visualizer with the current loss function"""
        from geometry.sgd_visualizer import SGDVisualizer
        from libs.loss_functions import LOSS_FUNCTIONS
        
        loss_func = LOSS_FUNCTIONS.get(self.sgd_loss_function)
        if loss_func is None:
            loss_func = LOSS_FUNCTIONS["Himmelblau"]
        
        x_range = loss_func.domain_range
        y_range = loss_func.domain_range
        if self.sgd_loss_function == "Himmelblau":
            x_range = (-6.0, 6.0)
            y_range = (-6.0, 6.0)
        
        self.sgd_visualizer = SGDVisualizer(loss_func, x_range=x_range, y_range=y_range, resolution=180)
        self.sgd_visualizer.show_contours = self.sgd_show_contours
        self.sgd_visualizer.show_drop_lines = self.sgd_show_drop_lines
        self.sgd_visualizer.show_projected_trajectory = self.sgd_show_projected_trajectory
        self.sgd_visualizer.view_mode = self.sgd_view_mode
        self.sgd_visualizer.trail_width_scale = max(0.2, float(self.sgd_trail_width))
        self.sgd_visualizer.set_colorblind_palette(self.sgd_colorblind_mode)
        
        for opt_name, opt_type in [('GD', 'GD'), ('SGD', 'SGD'), ('MiniBatch', 'MiniBatch'), 
                                    ('Momentum', 'Momentum'), ('Nesterov', 'Nesterov'), ('Adam', 'Adam')]:
            if self.sgd_optimizers_enabled.get(opt_name, True):
                initial_pos = self.sgd_initial_positions.get(opt_name)
                self.sgd_visualizer.add_optimizer(opt_name, opt_type, initial_pos)
        
        self.sgd_visualizer.setup()
        self.sgd_simulation_running = False
        self.sgd_step_count = 0
        self.sgd_replay_step = 0
    
    def set_sgd_loss_function(self, loss_name):
        """Change the loss function and reinitialize optimizers"""
        if loss_name in LOSS_FUNCTIONS:
            self.sgd_loss_function = loss_name
            self.init_sgd_visualizer()
    
    def sgd_step(self):
        """Perform one optimization step for all enabled optimizers"""
        if self.sgd_visualizer is None:
            return
        
        for opt_name in ['GD', 'SGD', 'MiniBatch', 'Momentum', 'Nesterov', 'Adam']:
            if self.sgd_optimizers_enabled.get(opt_name, True) and opt_name in self.sgd_visualizer.optimizers:
                self.sgd_visualizer.step_optimizer(opt_name, self.sgd_learning_rate, self.sgd_momentum, self.sgd_batch_size)
                self.sgd_visualizer.update_trajectory(opt_name)
        
        self.sgd_step_count += 1
        if not self.sgd_replay_enabled:
            self.sgd_replay_step = self.sgd_step_count
    
    def reset_sgd(self):
        """Reset all optimizers to initial positions"""
        if self.sgd_visualizer is None:
            return
        
        for opt_name in self.sgd_visualizer.optimizers:
            initial_pos = self.sgd_initial_positions.get(opt_name)
            self.sgd_visualizer.reset_optimizer(opt_name, initial_pos)
        
        self.sgd_simulation_running = False
        self.sgd_step_count = 0
        self.sgd_replay_step = 0
    
    def get_sgd_stats(self, replay_step=None):
        """Get current optimization statistics for all optimizers"""
        if self.sgd_visualizer is None:
            return {}
        
        stats = {}
        for opt_name, opt_data in self.sgd_visualizer.optimizers.items():
            history = opt_data.get('history', [])
            if len(history) == 0:
                continue

            if replay_step is None:
                idx = len(history) - 1
            else:
                idx = min(max(int(replay_step), 0), len(history) - 1)

            pos = history[idx]
            loss_hist = opt_data.get('loss_history', [])
            grad_hist = opt_data.get('grad_history', [])
            loss_val = float(loss_hist[idx]) if idx < len(loss_hist) else float(opt_data.get('loss', 0.0))
            grad_val = float(grad_hist[idx]) if idx < len(grad_hist) else float(opt_data.get('gradient_mag', 0.0))

            stats[opt_name] = {
                'position': [float(pos[0]), float(pos[1])],
                'loss': loss_val,
                'gradient_mag': grad_val,
                'step': idx,
            }
        return stats

    def get_sgd_metric_series(self, replay_step=None, max_points=500):
        if self.sgd_visualizer is None:
            return {}

        max_points = max(20, int(max_points))
        series = {}
        for opt_name, opt_data in self.sgd_visualizer.optimizers.items():
            loss_hist = list(opt_data.get('loss_history', []))
            grad_hist = list(opt_data.get('grad_history', []))
            if len(loss_hist) == 0:
                continue

            if replay_step is None:
                end_idx = len(loss_hist) - 1
            else:
                end_idx = min(max(int(replay_step), 0), len(loss_hist) - 1)
            if end_idx < 0:
                continue

            loss_vals = loss_hist[:end_idx + 1]
            grad_vals = grad_hist[:end_idx + 1]
            if len(loss_vals) > max_points:
                stride = int(math.ceil(len(loss_vals) / max_points))
                loss_vals = loss_vals[::stride]
                grad_vals = grad_vals[::stride]

            series[opt_name] = {
                'loss': loss_vals,
                'grad': grad_vals,
            }
        return series

    def set_sgd_colorblind_mode(self, enabled):
        self.sgd_colorblind_mode = bool(enabled)
        if self.sgd_visualizer is not None:
            self.sgd_visualizer.set_colorblind_palette(self.sgd_colorblind_mode)

    def apply_sgd_preset(self, preset_name):
        preset = self.SGD_PRESETS.get(preset_name)
        if preset is None:
            return False

        self.sgd_learning_rate = float(preset['learning_rate'])
        self.sgd_momentum = float(preset['momentum'])
        self.sgd_batch_size = int(preset['batch_size'])
        self.sgd_max_iterations = int(preset['max_iterations'])
        self.sgd_simulation_speed = int(preset['simulation_speed'])
        self.sgd_trail_width = float(preset['trail_width'])
        self.sgd_selected_preset = preset_name
        self.sgd_replay_enabled = False
        self.sgd_simulation_running = False
        self.sgd_step_count = 0
        self.sgd_replay_step = 0

        if self.sgd_visualizer is not None:
            self.sgd_visualizer.trail_width_scale = max(0.2, float(self.sgd_trail_width))
            self.reset_sgd()
        return True
    
    def toggle_optimizer_enabled(self, opt_name):
        """Toggle an optimizer on/off"""
        if opt_name in self.sgd_optimizers_enabled:
            self.sgd_optimizers_enabled[opt_name] = not self.sgd_optimizers_enabled[opt_name]
            
            if self.sgd_visualizer:
                if self.sgd_optimizers_enabled[opt_name]:
                    opt_type = opt_name
                    if opt_name == 'MiniBatch':
                        opt_type = 'MiniBatch'
                    initial_pos = self.sgd_initial_positions.get(opt_name)
                    self.sgd_visualizer.add_optimizer(opt_name, opt_type, initial_pos)
                else:
                    if opt_name in self.sgd_visualizer.optimizers:
                        del self.sgd_visualizer.optimizers[opt_name]
