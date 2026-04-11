"""Procedural road-scene generator used to build each dataset frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from btl2.scene.camera_rig import build_dashcam_camera
from btl2.scene.lighting import sample_directional_light
from btl2.scene.object_loader import MeshData, ObjectLoader
from btl2.scene.randomizer import Randomizer
from btl2.scene.scene import Scene
from btl2.scene.scene_object import SceneObject
from btl2.utils.colors import color_to_float, class_color
from btl2.utils.constants import CLASS_NAMES, CLASS_TO_ID


@dataclass
class AssetSpec:
    """Asset registration entry that maps a class to an optional file path."""

    class_name: str
    mesh_key: str
    relative_path: str | None
    primitive_name: str


class RoadSceneBuilder:
    """Generate diverse forward-driving scenes with deterministic seeds."""

    def __init__(self, config: dict, asset_root: str | Path = "assets/models") -> None:
        self.config = config
        self.loader = ObjectLoader(asset_root)
        self.showcase_layout = bool(self.config.setdefault("scene", {}).get("showcase_layout", False))
        scene_classes = self.config.setdefault("scene", {}).setdefault("classes", {})
        # Backward compatibility with older configs that used "pedestrian".
        if "person" not in scene_classes and "pedestrian" in scene_classes:
            scene_classes["person"] = dict(scene_classes["pedestrian"])
        self.asset_specs = {
            "person": AssetSpec(
                "person",
                "person_mesh",
                self._find_asset(
                    ("pedestrians", "people", "Pedestrians", "People", "Person", "person"),
                    ("person", "pedestrian", "human", "mei"),
                ),
                "cylinder",
            ),
            "car": AssetSpec(
                "car",
                "car_mesh",
                self._find_asset(("vehicles", "cars", "Car", "Cars", "car"), ("car", "taxi", "sedan", "suv", "hatch")),
                "box",
            ),
            "bus": AssetSpec(
                "bus",
                "bus_mesh",
                self._find_asset(("buses", "bus", "Buses", "Bus", "vehicles"), ("bus",)),
                "box",
            ),
            "truck": AssetSpec(
                "truck",
                "truck_mesh",
                self._find_asset(("trucks", "truck", "Truck", "Trucks", "vehicles"), ("truck", "lorry", "semi", "pickup")),
                "box",
            ),
            "motorbike": AssetSpec(
                "motorbike",
                "motorbike_mesh",
                self._find_asset(
                    ("motorbikes", "motorcycles", "bikes", "vehicles", "Motorbike", "motorbike", "MotorBike"),
                    ("bike", "motor", "scooter", "mot"),
                ),
                "box",
            ),
            "traffic_sign": AssetSpec(
                "traffic_sign",
                "traffic_sign_mesh",
                self._find_asset(
                    ("traffic_signs", "trafficSigns", "signs", "TrafficSigns", "Traffic_sign", "traffic_sign"),
                    ("sign", "roadsign"),
                ),
                "box",
            ),
            "traffic_light": AssetSpec(
                "traffic_light",
                "traffic_light_mesh",
                self._find_asset(
                    ("traffic_lights", "trafficLights", "lights", "TrafficLights", "Traffic_light", "traffic_light"),
                    ("light", "signal", "stoplight"),
                ),
                "box",
            ),
        }

    def build_scene(self, frame_index: int) -> tuple[Scene, dict[str, MeshData]]:
        """Create one scene and all mesh resources needed to render it."""
        base_seed = int(self.config["seed"])
        seed = base_seed + frame_index
        randomizer = Randomizer(seed)
        image_width = int(self.config["image_width"])
        image_height = int(self.config["image_height"])
        split = "train" if frame_index < int(self.config["num_frames"] * float(self.config["train_split"])) else "val"
        camera = build_dashcam_camera(self.config["camera"], image_width, image_height, randomizer)
        light = sample_directional_light(self.config["lighting"], randomizer)
        background = np.array(self.config["scene"]["sky_top_color"], dtype=np.float32)

        scene = Scene(
            frame_id=f"frame_{frame_index + 1:06d}",
            seed=seed,
            split=split,
            camera=camera,
            light=light,
            background_color=background,
        )

        mesh_registry: dict[str, MeshData] = {}
        self._add_static_road(scene, mesh_registry)

        min_objects = int(self.config["scene"]["min_objects"])
        max_objects = int(self.config["scene"]["max_objects"])
        total_dynamic = randomizer.randint(min_objects, max_objects)
        counts = self._distribute_counts(total_dynamic, randomizer)

        instance_id = 1
        class_spawn_indices = {name: 0 for name in counts}
        for class_name, count in counts.items():
            for _ in range(count):
                spec = self.asset_specs[class_name]
                mesh = self.loader.load_or_primitive(spec.relative_path, spec.primitive_name)
                mesh_registry[spec.mesh_key] = mesh
                spawn_index = class_spawn_indices.get(class_name, 0)
                scene.add_object(self._spawn_object(class_name, spec, mesh, instance_id, spawn_index, randomizer))
                class_spawn_indices[class_name] = spawn_index + 1
                instance_id += 1

        return scene, mesh_registry

    def _add_static_road(self, scene: Scene, mesh_registry: dict[str, MeshData]) -> None:
        """Insert the road plane as a fixed scene object."""
        mesh = self.loader.load_or_primitive(None, "plane")
        mesh_registry["road_plane"] = mesh
        road_cfg = self.config["scene"]
        road = SceneObject(
            name="road",
            class_name="road",
            mesh_key="road_plane",
            position=np.array([0.0, -0.01, road_cfg["road_length"] * 0.35], dtype=np.float32),
            rotation_degrees=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            scale=np.array([road_cfg["road_width"], 1.0, road_cfg["road_length"]], dtype=np.float32),
            base_color=color_to_float(class_color("road")),
            instance_id=0,
            semantic_id=255,
            metadata={"static": True},
            aabb_local=mesh.aabb,
        )
        scene.add_object(road)

    def _spawn_object(
        self,
        class_name: str,
        spec: AssetSpec,
        mesh: MeshData,
        instance_id: int,
        spawn_index: int,
        randomizer: Randomizer,
    ) -> SceneObject:
        """Sample a plausible pose and scale for one dynamic road-scene object."""
        if self.showcase_layout:
            return self._spawn_showcase_object(class_name, spec, mesh, instance_id, spawn_index, randomizer)

        lane_count = int(self.config["scene"]["lane_count"])
        lane_width = float(self.config["scene"]["lane_width"])
        lane_center = ((randomizer.randint(0, lane_count - 1) - (lane_count - 1) / 2.0) * lane_width)
        min_forward_z = {
            "person": 9.0,
            "car": 10.5,
            "bus": 15.5,
            "truck": 13.5,
            "motorbike": 8.5,
            "traffic_sign": 10.0,
            "traffic_light": 11.0,
        }.get(class_name, 9.0)
        forward_z = randomizer.uniform(min_forward_z, self.config["scene"]["road_length"] - 6.0)

        if class_name == "person":
            lane_center += randomizer.choice([-1.8, 1.8]) + randomizer.uniform(-0.5, 0.5)
            y = 0.9
            yaw = randomizer.uniform(-180.0, 180.0)
        elif class_name == "traffic_sign":
            lane_center = randomizer.choice([-self.config["scene"]["road_width"] * 0.55, self.config["scene"]["road_width"] * 0.55])
            y = 1.5
            yaw = randomizer.choice([-90.0, 90.0])
        elif class_name == "traffic_light":
            lane_center = randomizer.choice([-self.config["scene"]["road_width"] * 0.6, self.config["scene"]["road_width"] * 0.6])
            y = 2.4
            yaw = randomizer.choice([-90.0, 90.0])
        else:
            lane_center += randomizer.uniform(-0.25, 0.25)
            y = {
                "car": 0.55,
                "bus": 0.9,
                "truck": 0.85,
                "motorbike": 0.45,
            }.get(class_name, 0.55)
            yaw = randomizer.choice([0.0, 180.0]) + randomizer.uniform(-8.0, 8.0)

        scale_cfg = self.config["scene"]["classes"][class_name]["scale_range"]
        uniform_scale = randomizer.uniform(scale_cfg[0], scale_cfg[1])
        class_scale = self._class_scale(class_name, mesh)
        scale = class_scale * uniform_scale
        y = self._grounded_center_y(class_name, mesh, scale, y)

        return SceneObject(
            name=f"{class_name}_{instance_id:03d}",
            class_name=class_name,
            mesh_key=spec.mesh_key,
            position=np.array([lane_center, y, forward_z], dtype=np.float32),
            rotation_degrees=np.array([0.0, yaw, 0.0], dtype=np.float32),
            scale=scale.astype(np.float32),
            base_color=color_to_float(class_color(class_name)),
            instance_id=instance_id,
            semantic_id=CLASS_TO_ID[class_name],
            metadata={"source_asset": spec.relative_path or spec.primitive_name},
            aabb_local=mesh.aabb,
        )

    def _spawn_showcase_object(
        self,
        class_name: str,
        spec: AssetSpec,
        mesh: MeshData,
        instance_id: int,
        spawn_index: int,
        randomizer: Randomizer,
    ) -> SceneObject:
        """Place one deterministic object per class so the preview always shows every class clearly."""
        road_width = float(self.config["scene"]["road_width"])
        showcase_positions = {
            "motorbike": (-2.2, 0.45, 8.0, 180.0),
            "car": (1.3, 0.55, 10.8, 180.0),
            "truck": (-3.2, 0.85, 15.8, 180.0),
            "bus": (3.2, 0.9, 22.0, 180.0),
            "person": (road_width * 0.23, 0.9, 12.5, 165.0),
            "traffic_sign": (-road_width * 0.31, 1.5, 7.2, 90.0),
            "traffic_light": (road_width * 0.31, 2.4, 7.8, -90.0),
        }
        default_pose = (0.0, 0.55, 15.0, 180.0)
        x, y, z, yaw = showcase_positions.get(class_name, default_pose)
        position_jitter_x = float(self.config["scene"].get("showcase_position_jitter_x", 0.0))
        position_jitter_z = float(self.config["scene"].get("showcase_position_jitter_z", 0.0))
        yaw_jitter = float(self.config["scene"].get("showcase_yaw_jitter_degrees", 0.0))
        if spawn_index:
            z += 5.0 * spawn_index
            x += 1.2 * (-1 if spawn_index % 2 else 1)
        if position_jitter_x > 0.0:
            x += randomizer.uniform(-position_jitter_x, position_jitter_x)
        if position_jitter_z > 0.0:
            z += randomizer.uniform(-position_jitter_z, position_jitter_z)
        if yaw_jitter > 0.0:
            yaw += randomizer.uniform(-yaw_jitter, yaw_jitter)

        scale_cfg = self.config["scene"]["classes"][class_name]["scale_range"]
        uniform_scale = randomizer.uniform(scale_cfg[0], scale_cfg[1])
        scale = self._class_scale(class_name, mesh) * uniform_scale
        y = self._grounded_center_y(class_name, mesh, scale, y)

        return SceneObject(
            name=f"{class_name}_{instance_id:03d}",
            class_name=class_name,
            mesh_key=spec.mesh_key,
            position=np.array([x, y, z], dtype=np.float32),
            rotation_degrees=np.array([0.0, yaw, 0.0], dtype=np.float32),
            scale=scale.astype(np.float32),
            base_color=color_to_float(class_color(class_name)),
            instance_id=instance_id,
            semantic_id=CLASS_TO_ID[class_name],
            metadata={"source_asset": spec.relative_path or spec.primitive_name},
            aabb_local=mesh.aabb,
        )

    @staticmethod
    def _class_scale(class_name: str, mesh: MeshData | None = None) -> np.ndarray:
        fallback_scale = {
            "person": np.array([1.8, 1.8, 1.8], dtype=np.float32),
            "car": np.array([4.6, 4.6, 4.6], dtype=np.float32),
            "bus": np.array([9.8, 9.8, 9.8], dtype=np.float32),
            "truck": np.array([6.2, 6.2, 6.2], dtype=np.float32),
            "motorbike": np.array([2.4, 2.4, 2.4], dtype=np.float32),
            "traffic_sign": np.array([1.7, 1.7, 1.7], dtype=np.float32),
            "traffic_light": np.array([2.8, 2.8, 2.8], dtype=np.float32),
        }[class_name]

        if mesh is None or mesh.aabb is None:
            return fallback_scale

        extent = mesh.aabb.max_corner - mesh.aabb.min_corner
        if np.any(extent <= 1e-6):
            return fallback_scale

        target_axis, target_size = {
            "person": (1, 1.8),
            "car": (2, 4.6),
            "bus": (2, 9.8),
            "truck": (2, 6.2),
            "motorbike": (0, 2.4),
            "traffic_sign": (1, 1.7),
            "traffic_light": (1, 2.8),
        }[class_name]
        uniform_scale = float(target_size / extent[target_axis])
        return np.array([uniform_scale, uniform_scale, uniform_scale], dtype=np.float32)

    @staticmethod
    def _grounded_center_y(class_name: str, mesh: MeshData, scale: np.ndarray, fallback_y: float) -> float:
        if class_name == "road" or mesh.aabb is None:
            return fallback_y
        return float(max(0.02, -float(mesh.aabb.min_corner[1]) * float(scale[1])))

    def _distribute_counts(self, total_dynamic: int, randomizer: Randomizer) -> dict[str, int]:
        """Allocate object counts per class while respecting config limits."""
        limits = self.config["scene"]["classes"]
        classes = [name for name in CLASS_NAMES if name in limits]
        counts = {name: int(limits[name]["min_count"]) for name in classes}
        remaining = max(0, total_dynamic - sum(counts.values()))
        while remaining > 0:
            candidate = randomizer.choice(classes)
            if counts[candidate] < int(limits[candidate]["max_count"]):
                counts[candidate] += 1
                remaining -= 1
            else:
                available = [name for name in classes if counts[name] < int(limits[name]["max_count"])]
                if not available:
                    break
        return counts

    def _first_available_asset(self, *category_dirs: str) -> str | None:
        """Return the first discoverable mesh path among candidate category dirs."""
        for category_dir in category_dirs:
            candidate = self._first_asset(category_dir)
            if candidate is not None:
                return candidate
        return None

    def _find_asset(self, category_dirs: tuple[str, ...], keywords: tuple[str, ...] = ()) -> str | None:
        """Find a mesh from candidate folders; prefer filenames containing keywords."""
        candidates = self._collect_assets(*category_dirs)
        if not candidates:
            candidates = self._collect_all_assets()
        if not candidates:
            return None
        if keywords:
            keyword_lc = tuple(k.lower() for k in keywords if k)
            for rel in candidates:
                rel_lc = rel.lower()
                if any(k in rel_lc for k in keyword_lc):
                    return rel
        return candidates[0]

    def _collect_assets(self, *category_dirs: str) -> list[str]:
        """Collect OBJ/PLY candidates from candidate dirs (recursive)."""
        asset_root = self.loader.asset_root
        collected: list[str] = []
        for category_dir in category_dirs:
            category_path = asset_root / category_dir
            if not category_path.exists():
                continue
            for pattern in ("*.obj", "*.ply"):
                for match in sorted(category_path.rglob(pattern)):
                    if match.is_file():
                        collected.append(str(match.relative_to(asset_root)))
        return collected

    def _collect_all_assets(self) -> list[str]:
        """Collect every OBJ/PLY asset from the models root as a keyword-search fallback."""
        asset_root = self.loader.asset_root
        collected: list[str] = []
        for pattern in ("*.obj", "*.ply"):
            for match in sorted(asset_root.rglob(pattern)):
                if match.is_file():
                    collected.append(str(match.relative_to(asset_root)))
        return collected

    def _first_asset(self, category_dir: str) -> str | None:
        """Return the first OBJ or PLY path relative to asset root if one exists."""
        asset_root = self.loader.asset_root
        category_path = asset_root / category_dir
        if not category_path.exists():
            return None
        for pattern in ("*.obj", "*.ply"):
            matches = sorted(category_path.rglob(pattern))
            if matches:
                return str(matches[0].relative_to(asset_root))
        return None
