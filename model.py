from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import Scene class
from core.GameObject import GameObject, GameObjectOBJ, GameObjectLight, GameObjectCamera, GameObjectMath


ShaderPaths = Tuple[str, str]


def _default_shader_paths() -> ShaderPaths:
    return ("./shaders/standard.vert", "./shaders/standard.frag")


from libs.transform import Trackball
from libs.loss_functions import LOSS_FUNCTIONS
class AppModel:

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
        self.sgd_wireframe_mode = 0  # 0: fill, 1: wireframe, 2: point
        self.sgd_optimizers_enabled = {
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

    def add_hierarchy_object(self, name: str, obj_type: str, shape_name: str = "Cube") -> None:
        from core.GameObject import GameObject, GameObjectOBJ, GameObjectLight, GameObjectCamera, GameObjectMath
        import numpy as np
        
        # 1. Tạo đúng loại GameObject theo nhu cầu của người dùng.
        # Mesh, light, camera đều được đưa về cùng mô hình scene object để quản lý thống nhất.
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
            print(f"[DEBUG] Creating ModelLoader with filename: {self.model_filename}")
            scene_obj.drawable = ModelLoader(vert_shader, frag_shader, filename=self.model_filename)
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
        
        self.sgd_visualizer = SGDVisualizer(loss_func, x_range=x_range, y_range=y_range, resolution=80)
        
        for opt_name, opt_type in [('GD', 'GD'), ('SGD', 'SGD'), ('MiniBatch', 'MiniBatch'), 
                                    ('Momentum', 'Momentum'), ('Nesterov', 'Nesterov'), ('Adam', 'Adam')]:
            if self.sgd_optimizers_enabled.get(opt_name, True):
                initial_pos = self.sgd_initial_positions.get(opt_name)
                self.sgd_visualizer.add_optimizer(opt_name, opt_type, initial_pos)
        
        self.sgd_visualizer.setup()
        self.sgd_simulation_running = False
        self.sgd_step_count = 0
    
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
    
    def reset_sgd(self):
        """Reset all optimizers to initial positions"""
        if self.sgd_visualizer is None:
            return
        
        for opt_name in self.sgd_visualizer.optimizers:
            initial_pos = self.sgd_initial_positions.get(opt_name)
            self.sgd_visualizer.reset_optimizer(opt_name, initial_pos)
        
        self.sgd_simulation_running = False
        self.sgd_step_count = 0
    
    def get_sgd_stats(self):
        """Get current optimization statistics for all optimizers"""
        if self.sgd_visualizer is None:
            return {}
        
        stats = {}
        for opt_name, opt_data in self.sgd_visualizer.optimizers.items():
            stats[opt_name] = {
                'position': opt_data['position'].tolist(),
                'loss': float(opt_data['loss']),
                'gradient_mag': float(opt_data['gradient_mag']),
                'step': opt_data['step'],
            }
        return stats
    
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
