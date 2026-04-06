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
from btl2.utils.constants import CLASS_TO_ID


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
        self.asset_specs = {
            "car": AssetSpec("car", "car_mesh", self._first_asset("vehicles"), "box"),
            "pedestrian": AssetSpec("pedestrian", "pedestrian_mesh", self._first_asset("pedestrians"), "cylinder"),
            "traffic_sign": AssetSpec("traffic_sign", "traffic_sign_mesh", self._first_asset("traffic_signs"), "box"),
            "traffic_light": AssetSpec("traffic_light", "traffic_light_mesh", self._first_asset("traffic_lights"), "box"),
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
        for class_name, count in counts.items():
            for _ in range(count):
                spec = self.asset_specs[class_name]
                mesh = self.loader.load_or_primitive(spec.relative_path, spec.primitive_name)
                mesh_registry[spec.mesh_key] = mesh
                scene.add_object(self._spawn_object(class_name, spec, mesh, instance_id, randomizer))
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
        randomizer: Randomizer,
    ) -> SceneObject:
        """Sample a plausible pose and scale for one dynamic road-scene object."""
        lane_count = int(self.config["scene"]["lane_count"])
        lane_width = float(self.config["scene"]["lane_width"])
        lane_center = ((randomizer.randint(0, lane_count - 1) - (lane_count - 1) / 2.0) * lane_width)
        forward_z = randomizer.uniform(8.0, self.config["scene"]["road_length"] - 6.0)

        if class_name == "pedestrian":
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
            y = 0.55
            yaw = randomizer.choice([0.0, 180.0]) + randomizer.uniform(-8.0, 8.0)

        scale_cfg = self.config["scene"]["classes"][class_name]["scale_range"]
        uniform_scale = randomizer.uniform(scale_cfg[0], scale_cfg[1])
        class_scale = {
            "car": np.array([1.8, 1.2, 4.0], dtype=np.float32),
            "pedestrian": np.array([0.6, 1.8, 0.6], dtype=np.float32),
            "traffic_sign": np.array([0.35, 1.2, 0.15], dtype=np.float32),
            "traffic_light": np.array([0.45, 2.6, 0.45], dtype=np.float32),
        }[class_name]
        scale = class_scale * uniform_scale

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

    def _distribute_counts(self, total_dynamic: int, randomizer: Randomizer) -> dict[str, int]:
        """Allocate object counts per class while respecting config limits."""
        classes = ["car", "pedestrian", "traffic_sign", "traffic_light"]
        limits = self.config["scene"]["classes"]
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

    def _first_asset(self, category_dir: str) -> str | None:
        """Return the first OBJ or PLY path relative to asset root if one exists."""
        category_path = Path("assets/models") / category_dir
        if not category_path.exists():
            return None
        for pattern in ("*.obj", "*.ply"):
            matches = sorted(category_path.glob(pattern))
            if matches:
                return str(matches[0].relative_to("assets/models"))
        return None
