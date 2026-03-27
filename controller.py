import glfw
import OpenGL.GL as GL
from typing import Optional

from model import AppModel
from viewer import Viewer
from libs.coordinate_system import CoordinateSystem
from libs.shader import Shader
from libs.buffer import VAO, UManager


class AppController:
    def __init__(self, model: Optional[AppModel] = None, view: Optional[Viewer] = None) -> None:
        self.view = view or Viewer()
        self.model = model or AppModel()
        
        # Set model reference in viewer for gizmo interaction
        self.view.set_model_reference(self.model)
        
        self.coord_system = CoordinateSystem(axis_length=20.0, grid_size=1.0, is_3d=False)
        self._setup_coordinate_system()

        self.view.scroll_callback = self.on_scroll
        self.view.mouse_move_callback = self.on_mouse_move
        self.view.mouse_button_callback = self.on_mouse_button
        self.view.key_callback = self.on_key

    def on_scroll(self, window, xoffset, yoffset):
        width, height = glfw.get_window_size(window)
        self.view.trackball.zoom(yoffset, max(width, height))

    def on_mouse_move(self, window, xpos, ypos):
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.view.trackball.drag(self.view.last_mouse_pos, (xpos, ypos), glfw.get_window_size(window))
    
    def on_mouse_button(self, window, button, action, mods):
        """Handle mouse button events for gizmo interaction"""
        # This will be handled by viewer's gizmo system
        pass
    
    def on_key(self, window, key, scancode, action, mods):
        # Dùng PRESS (nhấn) và REPEAT (giữ phím) để camera trượt mượt mà
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_W:
                self.view.cycle_polygon_mode()
            elif key == glfw.KEY_Q:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_S:
                self.model.set_shader((self.model.selected_shader + 1) % 3)
            elif key == glfw.KEY_G:
                self.coord_system.toggle_visibility()
                
            elif key == glfw.KEY_UP:
                if hasattr(self.view.trackball, 'pan'):
                    self.view.trackball.pan(0, -20)  # Lên
            elif key == glfw.KEY_DOWN:
                if hasattr(self.view.trackball, 'pan'):
                    self.view.trackball.pan(0, 20)   # Xuống
            elif key == glfw.KEY_LEFT:
                if hasattr(self.view.trackball, 'pan'):
                    self.view.trackball.pan(20, 0)   # Trái
            elif key == glfw.KEY_RIGHT:
                if hasattr(self.view.trackball, 'pan'):
                    self.view.trackball.pan(-20, 0)  # Phải

            # --- THÊM PHÍM TẮT ĐIỀU KHIỂN ĐÈN ---
            elif key == glfw.KEY_1 and action == glfw.PRESS:
                lights = [obj for obj in self.model.scene.objects if hasattr(obj, 'light_intensity')]
                if len(lights) > 0:
                    lights[0].visible = not lights[0].visible
                    print(f"Đèn 1: {'SÁNG' if lights[0].visible else 'TẮT'}")
            elif key == glfw.KEY_2 and action == glfw.PRESS:
                lights = [obj for obj in self.model.scene.objects if hasattr(obj, 'light_intensity')]
                if len(lights) > 1:
                    lights[1].visible = not lights[1].visible
                    print(f"Đèn 2: {'SÁNG' if lights[1].visible else 'TẮT'}")
            elif key == glfw.KEY_3 and action == glfw.PRESS:
                lights = [obj for obj in self.model.scene.objects if hasattr(obj, 'light_intensity')]
                if len(lights) > 2:
                    lights[2].visible = not lights[2].visible
                    print(f"Đèn 3: {'SÁNG' if lights[2].visible else 'TẮT'}")

    def _setup_coordinate_system(self):
        """Setup coordinate system with simple color shader"""
        vert_shader = "./shaders/color_interp.vert"
        frag_shader = "./shaders/color_interp.frag"
        
        self.coord_vao = VAO()
        self.coord_shader = Shader(vert_shader, frag_shader)
        self.coord_uma = UManager(self.coord_shader)
        
        self.coord_system.setup(self.coord_vao, self.coord_uma)

    def _process_ui_actions(self, actions):
        """Process UI actions and update model accordingly"""
        if 'category_changed' in actions:
            # old_category = self.model.selected_category
            self.model.set_category(actions['category_changed'])
            if self.model.selected_category == 0 or self.model.selected_category == 2:
                self.coord_system.set_mode(is_3d=False)
            else:
                self.coord_system.set_mode(is_3d=True)
            # Reset hierarchy selection when category changes
            self.model.select_hierarchy_object(-1)
        
        if 'shape_changed' in actions:
            self.model.set_selected(actions['shape_changed'])
            # Reset hierarchy selection when mesh shape changes
            self.model.select_hierarchy_object(-1)
        
        if 'shader_changed' in actions:
            self.model.set_shader(actions['shader_changed'])
        
        if 'toggle_coord_system' in actions:
            self.coord_system.toggle_visibility()
        
        if 'math_function_changed' in actions:
            self.model.set_math_function(actions['math_function_changed'])
            if (self.model.selected_category == 2 and self.model.selected_idx == 0):
                self.model.reload_current_shape()
        
        if 'model_filename_changed' in actions:
            self.model.set_model_filename(actions['model_filename_changed'])
            if (self.model.selected_category == 3 and self.model.selected_idx == 0):
                self.model.reload_current_shape()
        
        if 'browse_model_file' in actions:
            self._browse_model_file()
        
        if 'color_changed' in actions:
            self.model.set_color(actions['color_changed'])
            print(f"Color updated to: {actions['color_changed']}")
        
        if 'browse_texture_file' in actions:
            self._browse_texture_file()
        
        if 'clear_texture' in actions:
            self.model.set_texture_filename("")
            print("Texture cleared")
        
        # Handle texture for specific hierarchy objects
        if 'browse_texture_for_object' in actions:
            target_obj = actions['browse_texture_for_object']['obj']
            self._browse_texture_for_specific_object(target_obj)
        
        if 'clear_texture' in actions and 'obj_id' in actions['clear_texture']:
            obj_id = actions['clear_texture']['obj_id']
            self.model.update_object_data(obj_id, "mesh_renderer.texture_filename", "")
            
            target_obj = next((o for o in self.model.scene.objects if o.id == obj_id), None)
            if target_obj and hasattr(target_obj, 'drawable') and target_obj.drawable:
                if hasattr(target_obj.drawable, 'set_texture'):
                    target_obj.drawable.set_texture("")
                
            print(f"Texture cleared for object {obj_id}")
        
        # Add Light and Camera actions
        if 'add_light' in actions:
            light_count = len([obj for obj in self.model.hierarchy_objects if obj["type"] == "light"])
            light_name = f"Light {light_count + 1}"
            self.model.add_hierarchy_object(light_name, "light")
            self.model.select_hierarchy_object(len(self.model.hierarchy_objects) - 1)
            print(f"Added {light_name}")
        
        if 'add_camera' in actions:
            camera_count = len([obj for obj in self.model.hierarchy_objects if obj["type"] == "camera"])
            camera_name = f"Camera {camera_count + 1}"
            self.model.add_hierarchy_object(camera_name, "camera")
            self.model.select_hierarchy_object(len(self.model.hierarchy_objects) - 1)
            print(f"Added {camera_name}")
        
        if 'reset_to_mesh' in actions:
            self.model.set_object_type("mesh")
            print("Reset to Mesh object")
        
        # === NEW: DYNAMIC INSPECTOR ACTIONS ===
        
        # Transform updates
        if 'update_transform_pos' in actions:
            data = actions['update_transform_pos']
            self.model.update_selected_object_data("transform.position", data['value'])
            
        if 'update_transform_rot' in actions:
            data = actions['update_transform_rot']
            self.model.update_selected_object_data("transform.rotation", data['value'])
            
        if 'update_transform_scale' in actions:
            data = actions['update_transform_scale']
            self.model.update_selected_object_data("transform.scale", data['value'])
        
        # Mesh Renderer updates
        if 'update_mesh_shader' in actions:
            data = actions['update_mesh_shader']
            self.model.update_selected_object_data("mesh_renderer.shader_idx", data['value'])
            # Update global shader for compatibility if mesh object selected
            if data['obj_id'] == -1:
                self.model.set_shader(data['value'])
                self.model.reload_current_shape()
                
        if 'update_mesh_color' in actions:
            data = actions['update_mesh_color']
            self.model.update_selected_object_data("mesh_renderer.color", data['value'])
            # Update global color for compatibility if mesh object selected
            if data['obj_id'] == -1:
                self.model.set_color(tuple(data['value']))
        
        # Math Surface updates
        if 'update_math_function' in actions:
            data = actions['update_math_function']
            self.model.update_selected_object_data("math_data.function", data['value'])
            
        # Model Loader updates
        if 'browse_model_for_object' in actions:
            data = actions['browse_model_for_object']
            self._browse_model_for_specific_object(data['obj_id'])
        
        # Light updates
        if 'update_light_intensity' in actions:
            data = actions['update_light_intensity']
            self.model.update_selected_object_data("light_data.intensity", data['value'])
            
        if 'update_light_color' in actions:
            data = actions['update_light_color']
            self.model.update_selected_object_data("light_data.color", data['value'])
        
        # Camera updates
        if 'update_camera_fov' in actions:
            data = actions['update_camera_fov']
            self.model.update_selected_object_data("camera.fov", data['value'])
            
        if 'update_camera_near' in actions:
            data = actions['update_camera_near']
            self.model.update_selected_object_data("camera.near", data['value'])
            
        if 'update_camera_far' in actions:
            data = actions['update_camera_far']
            self.model.update_selected_object_data("camera.far", data['value'])

        # --- Bổ sung vào controller.py ---
        if 'select_object' in actions:
            data = actions['select_object']
            self.model.scene.select_object(data['object'], data['multi_select'])

        if 'clear_selection' in actions:
            self.model.scene.clear_selection()
            
        # Thêm đoạn này vào để cập nhật tự động TẤT CẢ các loại thuộc tính (FOV, Color, Intensity...)
        if 'update_attr' in actions:
            data = actions['update_attr']
            setattr(data['obj'], data['attr'], data['val'])
            
        # --- BẮT SỰ KIỆN NÚT APPLY MATH ---
        if 'apply_math' in actions:
            target_obj = actions['apply_math']['obj']
            print(f"Applying new math function: {target_obj.math_script}")
            
            # Cần chuyển data sang dạng Dictionary để hàm _reload_hierarchy_object cũ của bạn hiểu được
            # (Vì hàm đó của bạn đang viết: obj_id = hierarchy_obj["id"])
            dict_format_obj = {
                "id": target_obj.id,
                "type": "math",
                "name": target_obj.name
            }
            
            # Gọi hàm tạo lại lưới 3D
            self.model._reload_hierarchy_object(dict_format_obj, target_obj.math_script)

        if 'update_obj' in actions:
            data = actions['update_obj']
            self.model.update_object_data(data['id'], data['key'], data['val'])

        if 'set_tool' in actions:
            self.model.active_tool = actions['set_tool']
            print(f"Đã chuyển sang công cụ: {self.model.active_tool}")

        # --- BẬT/TẮT GLOBAL FLAT COLOR ---
        if 'toggle_global_flat_color' in actions:
            current_state = getattr(self.model, 'global_flat_color_enabled', False)
            new_state = not current_state
            self.model.global_flat_color_enabled = new_state
            
            for obj in self.model.scene.objects:
                if hasattr(obj, 'drawable') and obj.drawable:
                    obj.drawable.use_flat_color = new_state
                    
                    if new_state:
                        # --- SỬA Ở ĐÂY: Ép toàn bộ thành MÀU TRẮNG thay vì lấy màu của object ---
                        obj.drawable.set_solid_color([1.0, 1.0, 1.0]) 
            
            status = "BẬT" if new_state else "TẮT"
            print(f"Đã {status} chế độ Flat Color (Trắng) cho toàn bản đồ!")

    def _browse_texture_file(self):
        """Open file browser for texture files using macOS native dialog"""
        import platform
        import subprocess

        if platform.system() == "Darwin":
            try:
                script = '''
                try
                    set chosen_file to choose file with prompt "Select Texture File (.png, .jpg, .tga):"
                    POSIX path of chosen_file
                end try
                '''
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    filename = result.stdout.strip()
                    self.model.set_texture_filename(filename)
                    print(f"Đã chọn file texture: {filename}")
                else:
                    print("Đã hủy chọn file.")
            except Exception as e:
                print(f"Lỗi khi mở hộp thoại Mac: {e}")
        else:
            print("Tính năng chọn texture hiện chỉ hỗ trợ giao diện native trên macOS.")
            print("Vui lòng nhập đường dẫn thủ công.")

    def _browse_model_file(self):
        """Open file browser for model files using macOS native dialog"""
        import platform
        import subprocess

        if platform.system() == "Darwin":
            try:
                script = '''
                try
                    set chosen_file to choose file with prompt "Select 3D Model File (.obj, .ply):"
                    POSIX path of chosen_file
                end try
                '''
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    filename = result.stdout.strip()
                    self.model.set_model_filename(filename)
                    
                    # If there's a selected hierarchy object of type custom_model, reload it
                    if self.model.selected_hierarchy_idx >= 0:
                        selected_obj = self.model.hierarchy_objects[self.model.selected_hierarchy_idx]
                        if selected_obj["type"] == "custom_model":
                            self.model._reload_hierarchy_object(selected_obj)
                        else:
                            # Fallback to loading active drawable
                            self.model.load_active_drawable()
                    else:
                        # No hierarchy object selected, load active drawable
                        self.model.load_active_drawable()
                    
                    print(f"Đã chọn file model: {filename}")
                else:
                    print("Đã hủy chọn file.")
            except Exception as e:
                print(f"Lỗi khi mở hộp thoại Mac: {e}")
        else:
            print("Tính năng chọn model hiện chỉ hỗ trợ giao diện native trên macOS.")
            print("Vui lòng nhập đường dẫn thủ công.")

    # Đổi tham số obj_id thành target_obj
    def _browse_texture_for_specific_object(self, target_obj):
        """Browse texture file for specific hierarchy object"""
        import platform
        import subprocess

        if platform.system() == "Darwin":
            try:
                script = f'''
                try
                    set chosen_file to choose file with prompt "Select Texture File for {target_obj.name} (.png, .jpg, .tga):"
                    POSIX path of chosen_file
                end try
                '''
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    filename = result.stdout.strip()
                    # Gán trực tiếp vào object
                    target_obj.texture_filename = filename
                    
                    if hasattr(target_obj, 'drawable') and target_obj.drawable:
                        if hasattr(target_obj.drawable, 'set_texture'):
                            target_obj.drawable.set_texture(filename)
                        else:
                            print(f"⚠️ Khối {target_obj.name} chưa được nâng cấp để hỗ trợ dán ảnh UV!")
                        
                    print(f"Đã chọn texture cho {target_obj.name}: {filename}")
                else:
                    print("Đã hủy chọn file.")
            except Exception as e:
                print(f"Lỗi khi mở hộp thoại Mac: {e}")
        else:
            print("Tính năng chọn texture hiện chỉ hỗ trợ giao diện native trên macOS.")
            
    def _browse_model_for_specific_object(self, obj_id):
        """Browse model file for specific hierarchy object"""
        import platform
        import subprocess

        if platform.system() == "Darwin":
            try:
                script = '''
                try
                    set chosen_file to choose file with prompt "Select 3D Model File (.obj, .ply):"
                    POSIX path of chosen_file
                end try
                '''
                result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    filename = result.stdout.strip()
                    # Update the specific object's model data
                    self.model.update_selected_object_data("model_data.filename", filename)
                    print(f"Đã chọn file cho object {obj_id}: {filename}")
                else:
                    print("Đã hủy chọn file.")
            except Exception as e:
                print(f"Lỗi khi mở hộp thoại Mac: {e}")
        else:
            print("Tính năng chọn file hiện chỉ hỗ trợ giao diện native trên macOS.")
            print("Vui lòng nhập đường dẫn thủ công.")

    # def _browse_texture_for_specific_object(self, obj_id):
    #     """Browse texture file for specific hierarchy object"""
    #     import platform
    #     import subprocess

    #     if platform.system() == "Darwin":
    #         try:
    #             script = f'''
    #             try
    #                 set chosen_file to choose file with prompt "Select Texture File for Object {obj_id} (.png, .jpg, .tga):"
    #                 POSIX path of chosen_file
    #             end try
    #             '''
    #             result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
                
    #             if result.returncode == 0 and result.stdout.strip():
    #                 filename = result.stdout.strip()
    #                 # Update the specific object's mesh renderer data
    #                 self.model.update_object_data(obj_id, "mesh_renderer.texture_filename", filename)
    #                 print(f"Đã chọn texture cho object {obj_id}: {filename}")
    #             else:
    #                 print("Đã hủy chọn file.")
    #         except Exception as e:
    #             print(f"Lỗi khi mở hộp thoại Mac: {e}")
    #     else:
    #         print("Tính năng chọn texture hiện chỉ hỗ trợ giao diện native trên macOS.")
    #         print("Vui lòng nhập đường dẫn thủ công.")

    def run(self) -> None:
        while not self.view.should_close():
            self.view.poll_events()
            self.view.begin_frame()

            ui_actions = self.view.draw_ui(self.model, self.coord_system)
            
            if 'category_changed' in ui_actions:
                self.model.set_category(ui_actions['category_changed'])
            
            if 'shape_changed' in ui_actions:
                self.model.set_selected(ui_actions['shape_changed'])
            
            if 'math_function_changed' in ui_actions:
                self.model.set_math_function(ui_actions['math_function_changed'])
                self.model.load_active_drawable()
            
            self._process_ui_actions(ui_actions)
            
            view = self.view.trackball.view_matrix()
            projection = self.view.trackball.projection_matrix(glfw.get_window_size(self.view.win))
            
            GL.glUseProgram(self.coord_shader.render_idx)
            self.view.draw_coordinate_system(self.coord_system, projection, view)
            
            self.view.draw_drawables(self.model.drawables, 
                self.model.scene.objects,
                active_tool=self.model.active_tool,
                selected_objects=self.model.scene.selected_objects
            )

            self.view.end_frame()
