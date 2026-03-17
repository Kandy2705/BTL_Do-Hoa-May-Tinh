from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple


ShaderPaths = Tuple[str, str]


def _default_shader_paths() -> ShaderPaths:
    return ("./shaders/color_interp.vert", "./shaders/color_interp.frag")


class AppModel:

    def __init__(self) -> None:
        self.selected_idx: int = 0
        self.selected_category: int = 0  # 0: 2D, 1: 3D, 2: SGD
        self.selected_shader: int = 0

        self.active_drawable: Optional[Any] = None
        self.drawables: List[Any] = []

    @property
    def menu_options(self) -> List[str]:
        if self.selected_category == 0:  # 2D
            return [
                "2D: Triangle",
                "2D: Rectangle", 
                "2D: Pentagon",
                "2D: Hexagon",
                "2D: Circle",
                "2D: Ellipse",
                "2D: Trapezoid",
                "2D: Star",
                "2D: Arrow",
            ]
        elif self.selected_category == 1:  # 3D
            return [
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
            ]
        else:  # SGD
            return ["Part 2: SGD (Himmelblau)"]

    @property
    def shader_names(self) -> List[str]:
        return ["Color Interpolation", "Gouraud", "Phong"]

    @property
    def category_options(self) -> List[str]:
        return ["2D Shapes", "3D Shapes", "SGD"]

    def _shape_factories(self) -> List[Tuple[str, str]]:
        if self.selected_category == 0:  # 2D
            return [
                ("geometry.triangle2d", "Triangle"),
                ("geometry.rectangle2d", "Rectangle"),
                ("geometry.pentagon2d", "Pentagon"),
                ("geometry.hexagon2d", "Hexagon"),
                ("geometry.circle2d", "Circle"),
                ("geometry.ellipse2d", "Ellipse"),
                ("geometry.trapezoid2d", "Trapezoid"),
                ("geometry.star2d", "Star"),
                ("geometry.arrow2d", "Arrow"),
            ]
        elif self.selected_category == 1:  # 3D
            return [
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
            ]
        else:  # SGD
            return [("", "")]

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

        if not (0 <= self.selected_idx < len(self._shape_factories())):
            return

        module_name, class_name = self._shape_factories()[self.selected_idx]
        if not module_name or not class_name:
            # Placeholder / not implemented (e.g., SGD)
            return

        try:
            module = importlib.import_module(module_name)
            shape_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            # If the import fails, do not crash; leave drawables empty.
            print(f"[AppModel] failed to load {module_name}.{class_name}: {e}")
            return

        vert_shader, frag_shader = self._shader_paths()
        drawable = shape_cls(vert_shader, frag_shader)
        drawable.setup()
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
        self.selected_idx = 0  # Reset to first shape in new category
        self.load_active_drawable()

    def set_shader(self, shader_idx: int) -> None:
        if shader_idx == self.selected_shader:
            return
        self.selected_shader = shader_idx
        self.load_active_drawable()
