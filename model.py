from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Dict, List, Optional, Tuple


ShaderPaths = Tuple[str, str]


def _default_shader_paths() -> ShaderPaths:
    return ("./shaders/color_interp.vert", "./shaders/color_interp.frag")


class AppModel:

    def __init__(self) -> None:
        self.selected_idx: int = 0
        self.selected_category: int = 0  # 0: 2D, 1: 3D, 2: Mathematical Surface, 3: Model from file, 4: SGD
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
        
        # Hierarchy objects list - REFACTORED: Each object is now a GameObject with proper structure
        self.hierarchy_objects = [
            {
                "id": 0, "name": "Main Camera", "type": "camera", "selected": False,
                "transform": {"position": [0.0, 1.0, -10.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
                "camera_data": {"fov": 60.0, "near": 0.1, "far": 100.0}
            },
            {
                "id": 1, "name": "Directional Light", "type": "light", "selected": False,
                "transform": {"position": [0.0, 3.0, 0.0], "rotation": [50.0, -30.0, 0.0], "scale": [1.0, 1.0, 1.0]},
                "light_data": {"intensity": 1.0, "color": [1.0, 1.0, 0.9]}
            },
            {
                "id": 2, "name": "Math Surface", "type": "math", "selected": False,
                "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
                "math_data": {"function": "(x**2 + y - 11)**2"}
            }
        ]
        self.next_object_id = 4  # For generating unique IDs
        
        # Component data for mesh objects
        self.mesh_components = {
            "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
            "mesh_renderer": {"shader": 0, "texture": "", "color": [1.0, 0.5, 0.0]}
        }

    @property
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
        else:
            return ["Part 2: SGD (Himmelblau)"]

    @property
    def shader_names(self) -> List[str]:
        return ["Color Interpolation", "Gouraud", "Phong"]

    @property
    def category_options(self) -> List[str]:
        return ["2D Shapes", "3D Shapes", "Mathematical Surface", "Model from file", "SGD"]

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
            return

        try:
            module = importlib.import_module(module_name)
            shape_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"[AppModel] failed to load {module_name}.{class_name}: {e}")
            return

        vert_shader, frag_shader = self._shader_paths()
        
        if class_name == "MathematicalSurface":
            try:
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
        self.selected_idx = 0 
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

    def set_object_type(self, obj_type: str) -> None:
        """Set object type: 'mesh', 'light', or 'camera'"""
        if obj_type in ["mesh", "light", "camera"]:
            self.object_type = obj_type

    def add_hierarchy_object(self, name: str, obj_type: str) -> None:
        """Add new object to hierarchy with proper GameObject structure"""
        new_obj = {
            "id": self.next_object_id,
            "name": name,
            "type": obj_type,
            "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}
        }
        
        # Add type-specific components
        if obj_type == "camera":
            new_obj["camera"] = {"fov": 45.0, "near": 0.1, "far": 100.0}
        elif obj_type == "light":
            new_obj["light_data"] = {"intensity": 1.0, "color": [1.0, 1.0, 1.0]}
        elif obj_type in ["2d", "3d", "math", "custom_model"]:
            new_obj["mesh_renderer"] = {"shader_idx": 0, "color": [1.0, 0.5, 0.0]}
            if obj_type == "math":
                new_obj["math_data"] = {"function": "(x**2 + y - 11)**2 + (x + y**2 - 7)**2"}
            elif obj_type == "custom_model":
                new_obj["model_data"] = {"filename": ""}
        
        self.hierarchy_objects.append(new_obj)
        self.next_object_id += 1
    
    def select_hierarchy_object(self, idx: int) -> None:
        """Select hierarchy object by index"""
        # Set new selection (no need to clear old selections with new structure)
        if 0 <= idx < len(self.hierarchy_objects):
            self.selected_hierarchy_idx = idx
        else:
            self.selected_hierarchy_idx = -1

    def select_hierarchy_object(self, idx: int):
        for i, obj in enumerate(self.hierarchy_objects):
            obj["selected"] = (i == idx)
        self.selected_hierarchy_idx = idx

    def get_selected_hierarchy_object(self):
        if 0 <= self.selected_hierarchy_idx < len(self.hierarchy_objects):
            return self.hierarchy_objects[self.selected_hierarchy_idx]
        return None

    def update_object_data(self, obj_id: int, key: str, value):
        # Tìm object theo ID
        obj = next((o for o in self.hierarchy_objects if o["id"] == obj_id), None)
        if not obj: return
        
        # Cập nhật giá trị lồng nhau (VD: "transform.position")
        keys = key.split('.')
        current = obj
        for k in keys[:-1]:
            current = current[k]
        current[keys[-1]] = value
        
        # Kích hoạt vẽ lại nếu sửa phương trình Math
        if key == "math_data.function":
            self.math_function = value # Đồng bộ với biến toàn cục cũ (nếu có dùng)
            self.load_active_drawable()
    
    def get_selected_object_components(self):
        """Get components for currently selected object"""
        if self.selected_hierarchy_idx == -1:
            # Mesh object selected - return global components
            return self.mesh_components
        else:
            # Hierarchy object selected - return the object itself (flat structure)
            selected_obj = self.get_selected_hierarchy_object()
            if selected_obj:
                return selected_obj
            return {}
    
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
            if key == "math_data.function" and selected_obj["type"] == "math":
                self.math_function = value  # Update global math function for compatibility
                self.load_active_drawable()
            elif key == "model_data.filename" and selected_obj["type"] == "custom_model":
                self.model_filename = value  # Update global model filename for compatibility
                self.load_active_drawable()
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

    def reload_current_shape(self) -> None:
        """Reload the current shape"""
        self.load_active_drawable()
