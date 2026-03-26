from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import Scene class
from core.GameObject import GameObject, GameObjectOBJ, GameObjectLight, GameObjectCamera, GameObjectMath


ShaderPaths = Tuple[str, str]


def _default_shader_paths() -> ShaderPaths:
    return ("./shaders/standard.vert", "./shaders/standard.frag")


class AppModel:

    def __init__(self) -> None:
        from components.scene import Scene
        self.selected_idx: int = -1  # -1 means no shape selected
        self.selected_category: int = 1  # 1: 3D (default to 3D instead of 2D)
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
        
        # Initialize Scene
        self.scene = Scene()
        
        # Hierarchy objects list - REFACTORED: Each object is now a GameObject with proper structure
        self.hierarchy_objects = []
        
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
        return ["Solid Color", "Gouraud", "Phong", "Rainbow Interpolation"]

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
        self.drawables: List[Any] = []

        if self.selected_idx == -1:  # No shape selected
            return

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

    def set_object_type(self, obj_type: str) -> None:
        """Set object type: 'mesh', 'light', or 'camera'"""
        if obj_type in ["mesh", "light", "camera"]:
            self.object_type = obj_type

    def add_hierarchy_object(self, name: str, obj_type: str, shape_name: str = "Cube") -> None:
        from core.GameObject import GameObject, GameObjectOBJ, GameObjectLight, GameObjectCamera, GameObjectMath
        import numpy as np
        
        # 1. Khởi tạo đúng class
        if obj_type in ["3d", "custom_model"]:
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
            
            # Map shape names to classes
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
            # Create MathematicalSurface drawable
            vert_shader = "./shaders/color_interp.vert"
            frag_shader = "./shaders/color_interp.frag"
            
            # Import MathematicalSurface
            import sys
            import os
            math_path = os.path.join(os.path.dirname(__file__), 'geometry')
            if math_path not in sys.path:
                sys.path.insert(0, math_path)
            
            from math_surface3d import MathematicalSurface
            new_obj.drawable = MathematicalSurface(vert_shader, frag_shader)
            new_obj.drawable.setup()
            
        elif obj_type == "custom_model":
            new_obj = GameObjectOBJ(name)
            # Create ModelLoader drawable
            vert_shader = "./shaders/standard.vert"
            frag_shader = "./shaders/standard.frag"
            
            # Import ModelLoader
            import sys
            import os
            model_path = os.path.join(os.path.dirname(__file__), 'geometry')
            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            
            from model_loader3d import ModelLoader
            new_obj.drawable = ModelLoader(vert_shader, frag_shader)
            new_obj.drawable.setup()
        elif obj_type == "light":
            new_obj = GameObjectLight(name)
            new_obj.drawable = None # Đèn không cần vẽ lưới (hoặc vẽ icon sau)
        elif obj_type == "camera":
            new_obj = GameObjectCamera(name)
            new_obj.drawable = None
        else:
            new_obj = GameObject(name)
            new_obj.drawable = None
            
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
        self.hierarchy_objects.append(hierarchy_obj)
    
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

    def update_object_data(self, obj_id: int, key_path: str, value: Any) -> None:
        object_idx = next((i for i, o in enumerate(self.scene.objects) if o.id == obj_id), -1)
        
        if object_idx != -1:
            target_obj = self.scene.objects[object_idx]
            self._set_nested_attribute(target_obj, key_path, value)

            if key_path == "mesh_renderer.texture_filename" and value == "":
                self.remove_texture_objects(obj_id)

            # --- THÊM ĐOẠN NÀY ĐỂ XỬ LÝ LỆNH BẤT/TẮT CÁC CHẾ ĐỘ ---
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

            self.update_hierarchy_state(object_idx, target_obj)
    
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
            vert_shader = "./shaders/color_interp.vert"
            frag_shader = "./shaders/color_interp.frag"
            
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
            print(f"[DEBUG] Creating ModelLoader with filename: {self.model_filename}")
            scene_obj.drawable = ModelLoader(vert_shader, frag_shader, filename=self.model_filename)
            scene_obj.drawable.setup()
            print("[DEBUG] ModelLoader recreated")
        else:
            print(f"[DEBUG] Unknown object type: {obj_type}")

    def reload_current_shape(self) -> None:
        """Reload the current shape"""
        self.load_active_drawable()

    def get_hierarchy_drawables(self) -> List[Any]:
        """Create drawable objects for hierarchy objects"""
        hierarchy_drawables = []
        
        for obj in self.hierarchy_objects:
            if obj["type"] in ["3d", "math", "custom_model"]:
                try:
                    # Determine shape type and create appropriate drawable
                    if obj["type"] == "3d":
                        # Use the first 3D shape (Cube) as default
                        module_name, class_name = self._shape_factories()[4]  # Cube is at index 4 when category 1
                        module = importlib.import_module(module_name)
                        shape_cls = getattr(module, class_name)
                        drawable = shape_cls()
                        
                    elif obj["type"] == "math":
                        # Use math surface
                        module_name, class_name = self._shape_factories()[0]  # First math surface
                        module = importlib.import_module(module_name)
                        shape_cls = getattr(module, class_name)
                        drawable = shape_cls(self.math_function)
                        
                    elif obj["type"] == "custom_model":
                        # Use model loader
                        module_name, class_name = self._shape_factories()[0]  # First model
                        module = importlib.import_module(module_name)
                        shape_cls = getattr(module, class_name)
                        drawable = shape_cls(self.model_filename)
                    
                    # Apply transform
                    if hasattr(drawable, 'set_transform'):
                        drawable.set_transform(
                            obj["transform"]["position"],
                            obj["transform"]["rotation"], 
                            obj["transform"]["scale"]
                        )
                    
                    # Apply color if mesh renderer exists
                    if "mesh_renderer" in obj and hasattr(drawable, 'set_color'):
                        drawable.set_color(obj["mesh_renderer"]["color"])
                        
                    hierarchy_drawables.append(drawable)
                    
                except Exception as e:
                    print(f"[AppModel] Failed to create drawable for {obj['name']}: {e}")
                    
        return hierarchy_drawables
