from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple


ShaderPaths = Tuple[str, str]


def _default_shader_paths() -> ShaderPaths:
    return ("./shaders/color_interp.vert", "./shaders/color_interp.frag")


class AppModel:

    menu_options: List[str] = [
        "2D: Triangle",
        "2D: Rectangle",
        "2D: Pentagon",
        "2D: Hexagon",
        "2D: Circle",
        "2D: Ellipse",
        "2D: Trapezoid",
        "2D: Star",
        "2D: Arrow",
        "3D: Cube",
        "3D: Sphere (Tetrahedron)",
        "3D: Sphere (Grid)",
        "3D: Sphere (Lat-Long)",
        "3D: Cylinder",
        "3D: Cone",
        "3D: Truncated Cone",
        "3D: Tetrahedron",
        "3D: Torus",
        "3D: Prism",
        "Part 2: SGD (Himmelblau)",
    ]

    shader_names: List[str] = ["Color Interpolation", "Gouraud", "Phong"]

    _shape_factories: List[Tuple[str, str]] = [
        ("geometry.triangle2d", "Triangle"),
        ("geometry.rectangle2d", "Rectangle"),
        ("geometry.pentagon2d", "Pentagon"),
        ("geometry.hexagon2d", "Hexagon"),
        ("geometry.circle2d", "Circle"),
        ("geometry.ellipse2d", "Ellipse"),
        ("geometry.trapezoid2d", "Trapezoid"),
        ("geometry.star2d", "Star"),
        ("geometry.arrow2d", "Arrow"),
        ("geometry.cube3d", "Cube"),
        ("geometry.sphere_tetrahedron3d", "SphereTetrahedron"),
        ("geometry.sphere_grid3d", "SphereGrid"),
        ("geometry.sphere_latlong3d", "SphereLatLong"),
        ("geometry.cylinder3d", "Cylinder"),
        ("geometry.cone3d", "Cone"),
        ("geometry.truncated_cone3d", "TruncatedCone"),
        ("geometry.tetrahedron3d", "Tetrahedron"),
        ("geometry.torus3d", "Torus"),
        ("geometry.prism3d", "Prism"),
        ("", ""),
    ]

    def __init__(self) -> None:
        self.selected_idx: int = 0
        self.selected_shader: int = 0

        self.active_drawable: Optional[Any] = None
        self.drawables: List[Any] = []

    def _shader_paths(self) -> ShaderPaths:
        if self.selected_shader == 0:
            return ("./shaders/color_interp.vert", "./shaders/color_interp.frag")
        if self.selected_shader == 1:
            return ("./shaders/gouraud.vert", "./shaders/gouraud.frag")
        if self.selected_shader == 2:
            return ("./shaders/phong.vert", "./shaders/phong.frag")
        return _default_shader_paths()

    def load_active_drawable(self) -> None:
        self.active_drawable = None
        self.drawables = []

        if not (0 <= self.selected_idx < len(self._shape_factories)):
            return

        module_name, class_name = self._shape_factories[self.selected_idx]
        if not module_name or not class_name:
            return

        try:
            module = importlib.import_module(module_name)
            shape_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"[AppModel] failed to load {module_name}.{class_name}: {e}")
            return

        vert_shader, frag_shader = self._shader_paths()
        drawable = shape_cls(vert_shader, frag_shader).setup()
        self.active_drawable = drawable
        self.drawables.append(drawable)

    def set_selected(self, idx: int) -> None:
        if idx == self.selected_idx:
            return
        self.selected_idx = idx
        self.load_active_drawable()

    def set_shader(self, shader_idx: int) -> None:
        if shader_idx == self.selected_shader:
            return
        self.selected_shader = shader_idx
        self.load_active_drawable()
