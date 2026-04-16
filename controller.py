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
        # Controller là lớp đứng giữa Model và Viewer:
        # - nhận input từ viewer
        # - cập nhật state vào model
        # - quyết định khi nào cần reload shape, đổi shader, đổi camera...
        self.view = view or Viewer()
        self.model = model or AppModel()
        
        # Set model reference in viewer for gizmo interaction
        self.view.set_model_reference(self.model)
        
        # Khởi tạo grid đúng mode ngay frame đầu để tránh cảm giác "lật/nghịch" khi vừa mở app.
        self.coord_system = CoordinateSystem(
            axis_length=20.0,
            grid_size=1.0,
            is_3d=self._is_3d_grid_mode(self.model.selected_category),
        )
        self._setup_coordinate_system()

        self.view.scroll_callback = self.on_scroll
        self.view.mouse_move_callback = self.on_mouse_move
        self.view.mouse_button_callback = self.on_mouse_button
        self.view.key_callback = self.on_key

    @staticmethod
    def _is_3d_grid_mode(category: int) -> bool:
        """Category 0 (2D) uses XY grid; other work modes use XZ grid."""
        return category != 0

    def on_scroll(self, window, xoffset, yoffset):
        # Scroll chuột được quy đổi thành zoom của camera trackball.
        width, height = glfw.get_window_size(window)
        self.view.trackball.zoom(yoffset, max(width, height))

    def on_mouse_move(self, window, xpos, ypos):
        import imgui
        # Nếu chuột đang tương tác với ImGui thì không được kéo camera / object nữa.
        if imgui.get_io().want_capture_mouse:
            self.view.last_mouse_pos = (xpos, ypos)
            return

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            tool = self.model.active_tool
            
            # 1. Nếu chọn HAND (Bàn tay) -> Trượt Camera (Pan)
            if tool == 'hand':
                dx = xpos - self.view.last_mouse_pos[0]
                dy = ypos - self.view.last_mouse_pos[1]
                tb = self.view.trackball
                
                # Gọi đúng hàm pan của Trackball để trượt bằng chuột
                if hasattr(tb, 'pan') and callable(tb.pan):
                    tb.pan(dx, -dy)  # Chú ý -dy để kéo chuột thuận tay
                elif hasattr(tb, 'target') and not callable(tb.target):
                    tb.target[0] -= dx * 0.02
                    tb.target[1] += dy * 0.02
                    
            # 2. Nếu chọn ROTATE (Xoay) -> Xoay góc Camera
            elif tool == 'rotate' or tool == 'select':
                # Ở select/rotate, kéo chuột sẽ quay góc nhìn quanh scene.
                self.view.trackball.drag(self.view.last_mouse_pos, (xpos, ypos), glfw.get_window_size(window))
                
            # 3. Nếu chọn MOVE (Di chuyển) -> Dịch chuyển Camera 3 trục (X,Y,Z)
            elif tool == 'move':
                dx = xpos - self.view.last_mouse_pos[0]
                dy = ypos - self.view.last_mouse_pos[1]
                tb = self.view.trackball
                
                # X,Y: Dịch chuyển ngang (giống Hand)
                if hasattr(tb, 'pan') and not callable(tb.pan):
                    tb.pan[0] -= dx * 0.02
                    tb.pan[1] += dy * 0.02
                elif hasattr(tb, 'target') and not callable(tb.target):
                    tb.target[0] -= dx * 0.02
                    tb.target[1] += dy * 0.02
                
                # Z: Dịch chuyển sâu (zoom) - dùng dy để zoom in/out
                if hasattr(tb, 'distance'):
                    tb.distance += dy * 0.05  # Kéo lên = đi ra, kéo xuống = đi vào
            
            # 4. Scale tool sau này làm Gizmo
            elif tool == 'scale':
                pass 

        self.view.last_mouse_pos = (xpos, ypos)
    
    def on_mouse_button(self, window, button, action, mods):
        """Handle mouse button events for gizmo interaction"""
        # This will be handled by viewer's gizmo system
        pass
    
    def on_key(self, window, key, scancode, action, mods):
        # Dùng PRESS (nhấn) và REPEAT (giữ phím) để camera trượt mượt mà
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_W:
                # W có 2 nghĩa:
                # - nếu đang ở SGD thì đổi render mode của surface loss
                # - nếu không thì đổi kiểu vẽ fill / wireframe / point
                if self.model.selected_category == 4:
                    self.model.sgd_wireframe_mode = (self.model.sgd_wireframe_mode + 1) % 3
                else:
                    self.view.cycle_polygon_mode()
            elif key == glfw.KEY_Q:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_S:
                # S dùng để chuyển qua lại các shader minh họa.
                self.model.set_shader((self.model.selected_shader + 1) % 4)
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

            # --- ĐỔI CAMERA TRONG SCENE BẰNG PHÍM C ---
            elif key == glfw.KEY_C and action == glfw.PRESS:
                cameras = [obj for obj in self.model.scene.objects if hasattr(obj, 'camera_fov')]
                
                # Tổng số góc nhìn = 1 (Scene Camera) + Số lượng Game Camera đang có
                total_views = len(cameras) + 1
                
                self.view.active_camera_idx = (self.view.active_camera_idx + 1) % total_views
                
                if self.view.active_camera_idx == 0:
                    print("🎥 Đã chuyển về: Scene Camera (Góc nhìn tự do)")
                else:
                    active_cam = cameras[self.view.active_camera_idx - 1]
                    print(f"🎥 Đã chuyển góc nhìn sang: {active_cam.name}")

            # --- PHÍM TẮT ĐIỀU KHIỂN SGD ---
            elif key == glfw.KEY_SPACE and action == glfw.PRESS:
                if self.model.selected_category == 4:
                    if self.model.sgd_replay_enabled and not self.model.sgd_simulation_running:
                        self.model.sgd_replay_enabled = False
                        self.model.sgd_replay_step = self.model.sgd_step_count
                    self.model.sgd_simulation_running = not self.model.sgd_simulation_running
                    status = "Running" if self.model.sgd_simulation_running else "Paused"
                    print(f"SGD Simulation: {status}")
            
            elif key == glfw.KEY_R and action == glfw.PRESS:
                if self.model.selected_category == 4:
                    self.model.reset_sgd()
                    print("SGD Reset!")
            
            elif key == glfw.KEY_T and action == glfw.PRESS:
                if self.model.selected_category == 4:
                    self.model.sgd_show_trajectory = not self.model.sgd_show_trajectory
                    print(f"Trajectory: {'On' if self.model.sgd_show_trajectory else 'Off'}")

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
        # Toàn bộ panel UI không sửa state trực tiếp,
        # mà trả về một dictionary actions. Controller đọc actions này
        # và quyết định phải cập nhật model/view như thế nào.
        if 'category_changed' in actions:
            new_cat = actions['category_changed']
            old_cat = self.model.selected_category
            self.model.set_category(new_cat)
            self.view.on_category_changed(old_cat, new_cat)
            
            self.coord_system.set_mode(is_3d=self._is_3d_grid_mode(new_cat))
            
            # Initialize SGD visualizer when switching to category 4
            if new_cat == 4 and self.model.sgd_visualizer is None:
                self.model.init_sgd_visualizer()
            if new_cat == 6:
                self.model.sync_btl2_config()

            self.model.select_hierarchy_object(-1)
            # Giữ nguyên góc camera khi đổi tab/mode để tránh cảm giác "nhảy view".
            # Người dùng vẫn có nút "Center Scene For Demo" nếu muốn canh lại thủ công.
        
        if 'shape_changed' in actions:
            new_shape_idx = actions['shape_changed']
            self.model.set_selected(new_shape_idx)
            # Reset hierarchy selection when mesh shape changes
            self.model.select_hierarchy_object(-1)
            # Chỉ reset camera khi người dùng thực sự chọn một shape preview cụ thể.
            # Trường hợp -1 (clear shape / đổi tab) thì giữ nguyên camera hiện tại.
            if new_shape_idx >= 0 and self.model.selected_category in (0, 1, 2, 3, 5):
                self.view.reset_scene_camera()

        if 'center_scene_for_demo' in actions:
            if self.view.center_scene_view(self.model.scene.objects):
                print("Da canh giua scene demo vao viewport.")
            else:
                self.view.reset_scene_camera()
                print("Scene trong, da reset goc nhin ve mac dinh.")
        
        if 'shader_changed' in actions:
            self.model.set_shader(actions['shader_changed'])
        
        if 'toggle_coord_system' in actions:
            self.coord_system.toggle_visibility()
        
        if 'math_function_changed' in actions:
            # Khi người dùng đổi công thức toán học, nếu đang preview Math Surface
            # thì reload lại mesh để tạo bề mặt mới từ f(x, y).
            self.model.set_math_function(actions['math_function_changed'])
            if (self.model.selected_category == 2 and self.model.selected_idx == 0):
                self.model.reload_current_shape()
        
        if 'model_filename_changed' in actions:
            # Tương tự Math Surface, đổi file model thì phải load lại model preview.
            self.model.set_model_filename(actions['model_filename_changed'])
            if (self.model.selected_category == 3 and self.model.selected_idx == 0):
                self.model.reload_current_shape()
        
        if 'browse_model_file' in actions:
            self._browse_model_file()

        if 'add_default_model' in actions:
            try:
                created = self.model.add_default_model_object(actions['add_default_model'])
                self.view.center_scene_view(self.model.scene.objects)
                print(f"Đã thêm model mặc định: {created.name}")
            except Exception as exc:
                print(f"Lỗi thêm model mặc định: {exc}")
        
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
            
        if 'delete_object' in actions:
            obj_to_delete = actions['delete_object']
            self.model.scene.remove_object(obj_to_delete)
            if obj_to_delete in self.model.scene.selected_objects:
                self.model.scene.selected_objects.remove(obj_to_delete)
            print(f"Deleted object: {obj_to_delete.name}")
            
        # Thêm đoạn này vào để cập nhật tự động TẤT CẢ các loại thuộc tính (FOV, Color, Intensity...)
        if 'update_attr' in actions:
            obj = actions['update_attr']['obj']
            attr = actions['update_attr']['attr']
            val = actions['update_attr']['val']
            
            # --- [ĐỒNG BỘ 2] TỪ INSPECTOR NGƯỢC VỀ TRACKBALL ---
            if hasattr(obj, 'camera_fov') and hasattr(obj, 'trackball'):
                tb = obj.trackball
                if attr == 'position':
                    # TRUYỀN THẲNG SỐ VÀO BIẾN pos2d MÀ KHÔNG QUA HÀM PAN
                    if hasattr(tb, 'pos2d'):
                        tb.pos2d[0] = float(val[0])
                        tb.pos2d[1] = float(val[1])
                        
                    if hasattr(tb, 'distance'): 
                        tb.distance = float(val[2])
                elif attr == 'rotation':
                    if hasattr(tb, 'rot_x'): tb.rot_x = val[0]
                    if hasattr(tb, 'rot_y'): tb.rot_y = val[1]
                    
                # ---> THÊM 3 DÒNG NÀY VÀO ĐÂY <---
                elif attr == 'camera_fov': tb.fov = float(val)
                elif attr == 'camera_near': tb.near = float(val)
                elif attr == 'camera_far': tb.far = float(val)

            # Gán giá trị vào bảng Inspector
            setattr(obj, attr, val)
            
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

        # --- BẬT/TẮT GLOBAL FLAT SHADING ---
        if 'toggle_global_flat_color' in actions:
            current_state = getattr(self.model, 'global_flat_color_enabled', False)
            new_state = not current_state
            self.model.global_flat_color_enabled = new_state
            
            for obj in self.model.scene.objects:
                if hasattr(obj, 'drawable') and obj.drawable:
                    obj.drawable.use_flat_color = new_state
                    
                    if new_state:
                        base_color = getattr(obj, 'color', [1.0, 1.0, 1.0])
                        obj.drawable.set_solid_color(base_color[:3])

            print(f"Đã {'BẬT' if new_state else 'TẮT'} Flat Shading cho toàn scene")
            
        # --- CHUYỂN ĐỔI CHẾ ĐỘ RGB / DEPTH MAP ---
        if 'toggle_display_mode' in actions:
            current_mode = getattr(self.model, 'display_mode', 0)
            self.model.display_mode = 1 if current_mode == 0 else 0
            print(f"Đã chuyển chế độ hiển thị sang: {'Depth Map' if self.model.display_mode == 1 else 'RGB'}") 

        if 'lab_toggle_slerp' in actions:
            self.model.set_lab_slerp_enabled(actions['lab_toggle_slerp'])
            print(f"Lab SLERP: {'On' if self.model.lab_slerp_enabled else 'Off'}")

        if 'lab_rescan_slerp_targets' in actions:
            self.model.refresh_lab_slerp_targets()
            print(f"Lab SLERP targets: {len(self.model.lab_slerp_targets)} sphere(s)")
            
        # SGD actions are now handled by SGDPanel and model methods

        if 'btl2_sync_config' in actions:
            self.model.sync_btl2_config()
            self.model.btl2_last_status = f"Synced config from {self.model.btl2_config_path}"
            print(f"Đã đồng bộ BTL 2 từ config: {self.model.btl2_config_path}")

        if 'btl2_refresh_scene' in actions:
            self.model.refresh_btl2_scene_summary()
            self.model.btl2_last_status = (
                f"Scene summary: cameras={self.model.btl2_scene_camera_count}, "
                f"renderables={self.model.btl2_scene_renderable_count}"
            )
            print(
                "BTL 2 scene summary:",
                f"cameras={self.model.btl2_scene_camera_count},",
                f"renderables={self.model.btl2_scene_renderable_count}"
            )

        if 'btl2_preview_mode' in actions:
            self.model.set_btl2_preview_mode(actions['btl2_preview_mode'])

        if 'btl2_refresh_preview' in actions:
            self.model.refresh_btl2_preview()

        if 'btl2_generate' in actions:
            try:
                self.model.btl2_last_status = "Running: BTL2 generation in progress..."
                if getattr(self.model, 'btl2_source_mode', 'current_scene') == 'current_scene':
                    result = self.model.run_btl2_from_current_scene()
                else:
                    result = self.model.run_btl2_generator()
                print(f"BTL 2 OK: {result['generated_frames']} frame -> {result['output_dir']}")
            except Exception as exc:
                self.model.btl2_last_status = f"Failed: {exc}"
                print(self.model.btl2_last_status)

        if 'btl2_load_preview_scene' in actions:
            try:
                self.model.btl2_last_status = "Running: loading procedural preview into BTL1 scene..."
                result = self.model.load_btl2_procedural_preview_into_scene()
                self.view.center_scene_view(self.model.scene.objects)
                print(
                    "BTL2 preview loaded:",
                    f"mesh={result.get('mesh_objects', 0)},",
                    f"cameras={result.get('cameras', 0)},",
                    f"renderables={result.get('renderables', 0)}",
                )
            except Exception as exc:
                self.model.btl2_last_status = f"Failed: {exc}"
                print(self.model.btl2_last_status)

        if 'btl2_pick_latest_weight' in actions:
            latest_weight = self.model._find_latest_yolo_weight()
            if latest_weight:
                self.model.btl2_detector_weight_path = latest_weight
                self.model.btl2_detector_weight_preset = "fine_tuned"
                self.model.btl2_inference_status = f"Suggested weight: {latest_weight}"
                print(f"Đã gợi ý best.pt mới nhất: {latest_weight}")
            else:
                self.model.btl2_inference_status = "Failed: khong tim thay best.pt trong outputs/training/yolo."
                print(self.model.btl2_inference_status)

        if 'btl2_use_yolov8s_weight' in actions:
            try:
                selected = self.model.set_btl2_detector_weight_preset("yolov8s")
                print(f"Đã chọn preset yolov8s: {selected}")
            except Exception as exc:
                self.model.btl2_inference_status = f"Failed: {exc}"
                print(self.model.btl2_inference_status)

        if 'btl2_use_yolov8m_weight' in actions:
            try:
                selected = self.model.set_btl2_detector_weight_preset("yolov8m")
                print(f"Đã chọn preset yolov8m: {selected}")
            except Exception as exc:
                self.model.btl2_inference_status = f"Failed: {exc}"
                print(self.model.btl2_inference_status)

        if 'btl2_use_yolov8x_weight' in actions:
            try:
                selected = self.model.set_btl2_detector_weight_preset("yolov8x")
                print(f"Đã chọn preset yolov8x: {selected}")
            except Exception as exc:
                self.model.btl2_inference_status = f"Failed: {exc}"
                print(self.model.btl2_inference_status)

        if 'btl2_use_yolo26s_weight' in actions:
            try:
                selected = self.model.set_btl2_detector_weight_preset("yolo26s")
                print(f"Đã chọn preset yolo26s: {selected}")
            except Exception as exc:
                self.model.btl2_inference_status = f"Failed: {exc}"
                print(self.model.btl2_inference_status)

        if 'btl2_use_finetuned_weight' in actions:
            try:
                selected = self.model.set_btl2_detector_weight_preset("fine_tuned")
                print(f"Đã chọn preset fine-tuned: {selected}")
            except Exception as exc:
                self.model.btl2_inference_status = f"Failed: {exc}"
                print(self.model.btl2_inference_status)

        if 'btl2_pick_sample_image' in actions:
            sample_image = self.model._find_current_output_image() or self.model._find_default_inference_image()
            if sample_image:
                self.model.btl2_inference_image_path = sample_image
                self.model.btl2_inference_status = f"Suggested image: {sample_image}"
                print(f"Đã gợi ý ảnh sample: {sample_image}")
            else:
                self.model.btl2_inference_status = "Failed: khong tim thay anh sample trong outputs/btl2."
                print(self.model.btl2_inference_status)

        if 'btl2_browse_weight' in actions:
            self._browse_btl2_weight_file()

        if 'btl2_browse_image' in actions:
            self._browse_btl2_image_file()

        if 'btl2_load_detector' in actions:
            try:
                result = self.model.load_btl2_detector()
                print(f"YOLO loaded: {result['weights']} ({result['device']})")
            except Exception as exc:
                self.model.btl2_inference_status = f"Failed: {exc}"
                print(self.model.btl2_inference_status)

        if 'btl2_run_inference' in actions:
            try:
                self.model.btl2_inference_status = "Running: YOLO inference in progress..."
                result = self.model.run_btl2_inference()
                print(
                    "YOLO inference OK:",
                    f"detections={result.get('detections', 0)},",
                    f"preview={result.get('preview', '')}",
                )
            except Exception as exc:
                self.model.btl2_inference_status = f"Failed: {exc}"
                print(self.model.btl2_inference_status)

    def _browse_file_dialog(self, prompt: str):
        """Open a native macOS file dialog and return the chosen path or None."""
        import platform
        import subprocess

        if platform.system() != "Darwin":
            print("Tính năng chọn file hiện chỉ hỗ trợ giao diện native trên macOS.")
            return None

        try:
            script = f'''
            try
                set chosen_file to choose file with prompt "{prompt}"
                POSIX path of chosen_file
            end try
            '''
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as exc:
            print(f"Lỗi khi mở hộp thoại Mac: {exc}")
        return None

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

    def _browse_btl2_weight_file(self):
        filename = self._browse_file_dialog("Select YOLO Weight File (.pt):")
        if filename:
            self.model.btl2_detector_weight_path = filename
            self.model.btl2_detector_weight_preset = self.model._infer_btl2_weight_preset(filename)
            self.model.btl2_inference_status = f"Selected weight: {filename}"
            print(f"Đã chọn YOLO weight: {filename}")
        else:
            print("Đã hủy chọn weight.")

    def _browse_btl2_image_file(self):
        filename = self._browse_file_dialog("Select Image for YOLO Inference (.png, .jpg):")
        if filename:
            self.model.btl2_inference_image_path = filename
            self.model.btl2_inference_status = f"Selected image: {filename}"
            print(f"Đã chọn ảnh inference: {filename}")
        else:
            print("Đã hủy chọn ảnh.")

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
            
            self._process_ui_actions(ui_actions)
            
            # === SGD Position Change ===
            if 'sgd_pos_changed' in ui_actions and self.model.selected_category == 4:
                if self.model.sgd_visualizer:
                    self.model.reset_sgd()
            
            # === SGD Simulation Step ===
            if self.model.selected_category == 4:
                if self.model.sgd_simulation_running and not self.model.sgd_replay_enabled:
                    for _ in range(self.model.sgd_simulation_speed):
                        if self.model.sgd_step_count < self.model.sgd_max_iterations:
                            self.model.sgd_step()

            # === Lab SLERP Animation Step ===
            self.model.update_lab_slerp_animation()
            
            # --- [ĐỒNG BỘ 1] GIẢI QUYẾT XUNG ĐỘT GIỮA GIZMO VÀ TRACKBALL ---
            cameras = [obj for obj in self.model.scene.objects if hasattr(obj, 'camera_fov')]
            active_cam_idx = self.view.active_camera_idx
            
            for i, cam in enumerate(cameras):
                if not hasattr(cam, 'trackball'): continue
                tb = cam.trackball
                
                # Trạng thái 1: Camera đang ĐƯỢC NHÌN (Lấy Trackball đè ra Inspector)
                if (i + 1) == active_cam_idx:
                    if hasattr(tb, 'pos2d'):
                        cam.position[0] = float(tb.pos2d[0])
                        cam.position[1] = float(tb.pos2d[1])
                    if hasattr(tb, 'distance'):
                        cam.position[2] = float(tb.distance)
                
                # Trạng thái 2: Camera đang TẮT (Lấy vị trí Gizmo/Inspector nạp vào Trackball)
                else:
                    if hasattr(tb, 'pos2d'):
                        tb.pos2d[0] = float(cam.position[0])
                        tb.pos2d[1] = float(cam.position[1])
                    if hasattr(tb, 'distance'):
                        tb.distance = float(cam.position[2])
            
            self.view.draw_drawables(self.model.drawables, 
                self.model.scene.objects,
                coord_system=self.coord_system,
                active_tool=self.model.active_tool,
                selected_objects=self.model.scene.selected_objects
            )

            self.view.end_frame()
