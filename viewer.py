import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import itertools
import numpy as np
from libs.transform import Trackball, lookat, ortho
from PIL import Image

# Import UI components
from components.main_menu import MainMenu
from components.hierarchy_panel import HierarchyPanel
from components.inspector_panel import InspectorPanel

from libs.gizmo import TransformGizmo

class Viewer:
    def __init__(self, width=1280, height=720):
        # Viewer chịu trách nhiệm mở cửa sổ, tạo OpenGL context,
        # dựng ImGui và render mọi thứ lên màn hình.
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.win = glfw.create_window(width, height, 'HCMUT - Unity Engine Clone', None, None)
        glfw.make_context_current(self.win)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)

        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.win)
        # --- SỬA THÀNH: Camera dự phòng khi Scene chưa có Camera nào ---
        self.default_trackball = Trackball()
        self.active_camera_idx = 0
        
        # Store model reference for gizmo interaction
        self.model = None
        
        # Call Unity Style Setup
        self._apply_unity_style()

        self.scroll_callback = None
        self.mouse_move_callback = None
        self.mouse_button_callback = None
        self.key_callback = None
        self.last_mouse_pos = (0.0, 0.0)
        self.fill_modes = itertools.cycle([gl.GL_FILL, gl.GL_LINE, gl.GL_POINT])

        self.gizmo = TransformGizmo()

        self.cube_texture_id, _, _ = self.load_texture("assets/textures/cube-solid.png")
        self.hand_texture_id, _, _ = self.load_texture("assets/textures/hand-solid.png")
        self.move_texture_id, _, _ = self.load_texture("assets/textures/arrows-up-down-left-right-solid.png")
        self.rotate_texture_id, _, _ = self.load_texture("assets/textures/arrows-rotate-solid.png")
        self.scale_texture_id, _, _ = self.load_texture("assets/textures/up-right-from-square-solid.png")

        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_mouse_button_callback(self.win, self.on_mouse_button)
        glfw.set_key_callback(self.win, self._on_key)

    def set_model_reference(self, model):
        """Set reference to AppModel for gizmo interaction"""
        self.model = model

    @property #biến method thành property 
    def trackball(self):
        """0 là Scene Camera (Tự do), >0 là các Game Camera trong cảnh"""
        # Property này quyết định "camera nào đang thật sự được dùng để nhìn scene".
        if not self.model: 
            return self.default_trackball
            
        cameras = [obj for obj in self.model.scene.objects if hasattr(obj, 'camera_fov')]
        
        # Nếu đang ở chế độ 0, HOẶC lỡ xóa mất camera làm index bị lố -> Trả về Scene Camera
        if self.active_camera_idx == 0 or self.active_camera_idx > len(cameras):
            return self.default_trackball
            
        # Nếu đang ở chế độ > 0, trỏ góc nhìn vào Game Camera tương ứng
        target_cam = cameras[self.active_camera_idx - 1]
        if hasattr(target_cam, 'trackball'):
            return target_cam.trackball
            
        return self.default_trackball

    def _apply_unity_style(self):
        style = imgui.get_style()
        style.window_rounding = 0.0
        style.child_rounding = 0.0
        style.frame_rounding = 2.0
        style.grab_rounding = 2.0
        style.window_border_size = 1.0
        style.item_spacing = (8, 4)

        # Unity Dark Palette
        colors = style.colors
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.22, 0.22, 0.22, 1.0) # #383838
        colors[imgui.COLOR_HEADER] = (0.17, 0.17, 0.17, 1.0)           # #2c2c2c
        colors[imgui.COLOR_HEADER_HOVERED] = (0.27, 0.27, 0.27, 1.0)
        colors[imgui.COLOR_HEADER_ACTIVE] = (0.35, 0.35, 0.35, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND] = (0.12, 0.12, 0.12, 1.0)
        colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = (0.12, 0.12, 0.12, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND] = (0.16, 0.16, 0.16, 1.0)
        colors[imgui.COLOR_BUTTON] = (0.34, 0.34, 0.34, 1.0)
        colors[imgui.COLOR_BUTTON_HOVERED] = (0.4, 0.4, 0.4, 1.0)
        colors[imgui.COLOR_NAV_HIGHLIGHT] = (0.1, 0.4, 0.6, 1.0) # Unity Blue

    # Các hàm callback giữ nguyên như code cũ của bạn...
    def _on_scroll(self, window, xoffset, yoffset):
        if self._is_sgd_contour_locked():
            return
        if self.scroll_callback: self.scroll_callback(window, xoffset, yoffset)

    def on_mouse_move(self, window, xpos, ypos):
        if self._is_sgd_contour_locked():
            self.last_mouse_pos = (xpos, ypos)
            return
        # 1. CHUỘT PHẢI: Xoay Camera vòng vòng (Nhưng cũ)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            self.trackball.drag(self.last_mouse_pos, (xpos, ypos), glfw.get_window_size(window))
            
        # 2. CHUỘT TRÁI: Tương tác với Gizmo hoặc Hand Tool
        elif (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS and 
              glfw.get_key(window, glfw.KEY_LEFT_SHIFT) != glfw.PRESS):
            
            current_tool = getattr(self.model, 'active_tool', 'select')
            
            # --- NẾU LÀ HAND TOOL -> Di chuyển Camera (Pan) ---
            if current_tool == 'hand':
                dx = xpos - self.last_mouse_pos[0]
                dy = ypos - self.last_mouse_pos[1]
                
                if hasattr(self.trackball, 'pan'):
                    self.trackball.pan(dx, -dy)
                    
            # --- NẾU LÀ CÁC TOOL CÒN LẠI -> Xử lý kéo Gizmo ---
            else:
                selected_objects = getattr(self.model.scene, 'selected_objects', [])
                if (selected_objects and len(selected_objects) == 1 and 
                    current_tool in ['move', 'rotate', 'scale']):
                    target = selected_objects[0]
                    mouse_pos = (xpos, ypos)
                    
                    view = self.trackball.view_matrix()
                    proj = self.trackball.projection_matrix(glfw.get_window_size(window))
                    win_size = glfw.get_window_size(window)
                    
                    # Nếu đang thao tác object, kéo chuột sẽ chuyển thành transform qua gizmo.
                    self.gizmo.handle_mouse_drag(mouse_pos, target, current_tool, view, proj, win_size)
        
        # Cập nhật vị trí chuột cuối cùng
        self.last_mouse_pos = (xpos, ypos)
    
    def on_mouse_button(self, window, button, action, mods):
        if self._is_sgd_contour_locked():
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                current_tool = getattr(self.model, 'active_tool', 'select')
                selected_objects = getattr(self.model.scene, 'selected_objects', [])
                
                if (selected_objects and len(selected_objects) == 1 and 
                    current_tool in ['move', 'rotate', 'scale']):
                    target = selected_objects[0]
                    mouse_pos = glfw.get_cursor_pos(window)
                    
                    # --- LẤY MA TRẬN CAMERA & MÀN HÌNH ---
                    view = self.trackball.view_matrix()
                    proj = self.trackball.projection_matrix(glfw.get_window_size(window))
                    win_size = glfw.get_window_size(window)
                    
                    # Bắt đầu bấm trúng Gizmo
                    self.gizmo.handle_mouse_press(mouse_pos, target.position, current_tool, view, proj, win_size)
                    
            elif action == glfw.RELEASE:
                self.gizmo.handle_mouse_release()
    def _on_key(self, window, key, scancode, action, mods):
        self.imgui_impl.keyboard_callback(window, key, scancode, action, mods)
        if not imgui.get_io().want_capture_keyboard and self.key_callback:
            self.key_callback(window, key, scancode, action, mods)

    def load_texture(self, image_path):
        # Hàm này dùng cho icon UI của editor, không phải texture của model .obj.
        img = Image.open(image_path).convert("RGBA")
        # Lật ảnh ngược lại để đúng chiều trong OpenGL
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        width, height = img.size
        data = img.tobytes("raw", "RGBA", 0, -1)

        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        
        return texture_id, width, height

    def should_close(self) -> bool:
        return glfw.window_should_close(self.win)

    def poll_events(self) -> None:
        glfw.poll_events()
        self.imgui_impl.process_inputs()

    def begin_frame(self) -> None:
        if self.model and self.model.selected_category == 4:
            gl.glClearColor(0.5, 0.5, 0.5, 1.0)
        else:
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        imgui.new_frame()

    def _is_sgd_contour_locked(self):
        if not self.model or self.model.selected_category != 4:
            return False
        return getattr(self.model, 'sgd_view_mode', '') in ('contour', 'interactive')

    def _get_sgd_contour_camera(self, viewport_size):
        width, height = viewport_size
        aspect = max(width / max(height, 1), 1e-5)
        # Mặt contour thật sự chỉ nằm trong vùng x,y khoảng [-2, 2].
        # Giảm extent xuống một chút để hình ôm khung giữa hơn, đỡ bị lọt thỏm.
        base_extent = 2.18
        if aspect >= 1.0:
            half_w = base_extent * aspect
            half_h = base_extent
        else:
            half_w = base_extent
            half_h = base_extent / aspect

        projection = ortho(-half_w, half_w, -half_h, half_h, -20.0, 20.0)
        view = lookat(np.array([0.0, 0.0, 18.0], dtype=np.float32),
                      np.array([0.0, 0.0, 0.0], dtype=np.float32),
                      np.array([0.0, 0.5, 0.0], dtype=np.float32))
        return view, projection

    def _get_sgd_contour_inset_rect(self, gl_x, gl_y, gl_w, gl_h):
        # Inset mini contour map cố định ở góc phải dưới của viewport scene.
        if gl_w < 220 or gl_h < 180:
            return None

        margin = max(int(round(min(gl_w, gl_h) * 0.03)), 10)
        max_w = gl_w - margin * 2
        max_h = gl_h - margin * 2
        if max_w <= 0 or max_h <= 0:
            return None

        inset_w = max(int(round(gl_w * 0.26)), 150)
        inset_h = max(int(round(gl_h * 0.23)), 110)
        inset_w = min(inset_w, max_w)
        inset_h = min(inset_h, max_h)

        if inset_w < 120 or inset_h < 90:
            return None

        inset_x = gl_x + gl_w - inset_w - margin
        inset_y = gl_y + margin
        return inset_x, inset_y, inset_w, inset_h

    def _draw_sgd_contour_inset(self, sgd_visualizer, gl_x, gl_y, gl_w, gl_h,
                                display_mode, cam_far,
                                show_trajectory, show_projected_trajectory):
        inset_rect = self._get_sgd_contour_inset_rect(gl_x, gl_y, gl_w, gl_h)
        if inset_rect is None:
            return

        inset_x, inset_y, inset_w, inset_h = inset_rect
        gl.glViewport(inset_x, inset_y, inset_w, inset_h)
        gl.glScissor(inset_x, inset_y, inset_w, inset_h)
        gl.glClearColor(0.12, 0.12, 0.12, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        inset_view, inset_projection = self._get_sgd_contour_camera((inset_w, inset_h))
        sgd_visualizer.draw(
            inset_projection,
            inset_view,
            wireframe_mode=0,
            display_mode=display_mode,
            cam_far=cam_far,
            show_trajectory=show_trajectory,
            show_projected_trajectory=show_projected_trajectory,
            show_drop_lines=False,
            show_contours=True,
            view_mode='contour',
        )

    def end_frame(self) -> None:
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.win)

    def draw_drawables(self, drawables, scene_objects, active_tool="select", selected_objects=None):
        # Đây là khối render chính của viewer:
        # lấy camera hiện tại, dựng ma trận view/projection,
        # thu thập đèn trong scene rồi gọi draw cho từng drawable.
        win_w, win_h = glfw.get_window_size(self.win)
        fb_w, fb_h = glfw.get_framebuffer_size(self.win)
        viewport_rect = (315.0, 55.0, max(win_w - 635.0, 10.0), max(win_h - 255.0, 10.0))
        vx, vy, vw, vh = viewport_rect

        scale_x = fb_w / max(win_w, 1)
        scale_y = fb_h / max(win_h, 1)
        gl_x = int(round(vx * scale_x))
        gl_w = max(int(round(vw * scale_x)), 1)
        gl_h = max(int(round(vh * scale_y)), 1)
        gl_y = max(int(round((win_h - (vy + vh)) * scale_y)), 0)

        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glViewport(gl_x, gl_y, gl_w, gl_h)
        gl.glScissor(gl_x, gl_y, gl_w, gl_h)

        if self._is_sgd_contour_locked():
            view, projection = self._get_sgd_contour_camera((gl_w, gl_h))
        else:
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix((gl_w, gl_h))

        scene_lights = [obj for obj in scene_objects if hasattr(obj, 'light_intensity')]
        
        # === LẤY CÁC GIÁ TRỊ GLOBAL ĐỂ TRUYỀN VÀO SHADER ===
        display_mode = getattr(self.model, 'display_mode', 0)
        cam_far      = getattr(self.trackball, 'far', 100.0)
        draw_sgd_inset = False
        inset_show_trajectory = True
        inset_show_projected = True

        # Draw SGD visualization if in SGD category
        if self.model and self.model.selected_category == 4 and self.model.sgd_visualizer:
            wireframe_mode = getattr(self.model, 'sgd_wireframe_mode', 0)
            show_trajectory = getattr(self.model, 'sgd_show_trajectory', True)
            show_projected_trajectory = getattr(self.model, 'sgd_show_projected_trajectory', True)
            show_drop_lines = getattr(self.model, 'sgd_show_drop_lines', True)
            show_contours = getattr(self.model, 'sgd_show_contours', True)
            view_mode = getattr(self.model, 'sgd_view_mode', "combined")
            self.model.sgd_visualizer.draw(
                projection,
                view,
                wireframe_mode,
                display_mode,
                cam_far,
                show_trajectory=show_trajectory,
                show_projected_trajectory=show_projected_trajectory,
                show_drop_lines=show_drop_lines,
                show_contours=show_contours,
                view_mode=view_mode,
            )
            draw_sgd_inset = not self._is_sgd_contour_locked()
            inset_show_trajectory = show_trajectory
            inset_show_projected = show_projected_trajectory

            if getattr(self.model, 'sgd_hover_enabled', True) and view_mode in ('contour', 'interactive', 'combined'):
                mx, my = glfw.get_cursor_pos(self.win)
                vx, vy, vw, vh = viewport_rect
                inside = (vx <= mx <= vx + vw) and (vy <= my <= vy + vh)
                if inside:
                    hover = self.model.sgd_visualizer.pick_hover_info(mx, my, projection, view, viewport_rect)
                    self.model.sgd_hover_info = hover
                    if hover:
                        self._draw_sgd_hover_overlay(hover)
                else:
                    self.model.sgd_hover_info = None
            else:
                self.model.sgd_hover_info = None
        else:
            # Draw regular drawables (mesh objects)
            for drawable in drawables:
                drawable.draw(projection, view, None)
        
        # ==================== VẼ CÁC OBJECT CHÍNH ====================
        for obj in scene_objects:
            if not getattr(obj, 'visible', True):
                continue 

            if hasattr(obj, 'drawable') and obj.drawable is not None:
                if hasattr(obj, 'drawable') and hasattr(obj.drawable, 'set_transform'):
                    obj.drawable.set_transform(obj.position, obj.rotation, obj.scale)
                
                obj.drawable.scene_lights = scene_lights
                
                # ==================== TRUYỀN UNIFORM VÀO SHADER ====================
                if hasattr(obj.drawable, 'shader') and obj.drawable.shader:
                    shader_id = obj.drawable.shader.render_idx
                    gl.glUseProgram(shader_id)
                    
                    # 1. Display Mode (RGB / Depth Map)
                    loc_mode = gl.glGetUniformLocation(shader_id, "u_display_mode")
                    if loc_mode != -1:
                        gl.glUniform1i(loc_mode, display_mode)
                    
                    # 2. Camera Far (cho Depth Map)
                    loc_far = gl.glGetUniformLocation(shader_id, "u_cam_far")
                    if loc_far != -1:
                        gl.glUniform1f(loc_far, cam_far)
                    
                    # 3. Shininess
                    loc_shininess = gl.glGetUniformLocation(shader_id, "u_shininess")
                    if loc_shininess != -1:
                        shininess_val = getattr(obj.drawable, 'shininess', 32.0)
                        gl.glUniform1f(loc_shininess, shininess_val)
                    
                    # 4. Light Range (Attenuation + Cutoff)
                    loc_range = gl.glGetUniformLocation(shader_id, "u_light_range")
                    if loc_range != -1:
                        gl.glUniform1f(loc_range, 50.0)      # ← Bạn có thể chỉnh số này

                obj.drawable.draw(projection, view, None)

            
        if selected_objects and len(selected_objects) == 1:
            target = selected_objects[0]
            # Nếu đang chọn tool Move, Rotate hoặc Scale thì hiện Gizmo lên
            if active_tool in ['move', 'rotate', 'scale']:
                # Truyền vị trí của target vào để nó vẽ trục XYZ ra
                self.gizmo.draw(projection, view, target.position, active_tool)

        if draw_sgd_inset:
            self._draw_sgd_contour_inset(
                self.model.sgd_visualizer,
                gl_x,
                gl_y,
                gl_w,
                gl_h,
                display_mode=display_mode,
                cam_far=cam_far,
                show_trajectory=inset_show_trajectory,
                show_projected_trajectory=inset_show_projected,
            )
            # Khôi phục viewport/scissor scene chính để tránh ảnh hưởng các pass sau.
            gl.glViewport(gl_x, gl_y, gl_w, gl_h)
            gl.glScissor(gl_x, gl_y, gl_w, gl_h)

        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glViewport(0, 0, fb_w, fb_h)

    def _draw_sgd_hover_overlay(self, hover):
        box_w, box_h = 155, 78
        x = hover['screen_x'] + 14
        y = hover['screen_y'] + 14

        imgui.set_next_window_position(x, y)
        imgui.set_next_window_size(box_w, box_h)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.18, 0.18, 0.18, 0.88)
        imgui.begin(
            "SGDHoverOverlay",
            flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
                  imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_INPUTS
        )
        imgui.text(f"x: {hover['x']:.3f}")
        imgui.text(f"y: {hover['y']:.3f}")
        imgui.text(f"z: {hover['z']:.5f}")
        imgui.end()
        imgui.pop_style_color()
                
                

    def cycle_polygon_mode(self) -> None:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, next(self.fill_modes))

    def draw_coordinate_system(self, coord_system, projection, view):
        """Draw coordinate system"""
        if coord_system.visible:
            coord_system.draw(projection, view)

    def draw_ui(self, model, coord_system):
        """Draw UI using components"""
        actions = {}
        win_w, win_h = glfw.get_window_size(self.win)
        
        # Ensure minimum window size
        win_w = max(win_w, 800)
        win_h = max(win_h, 600)
        
        # 1. MAIN MENU BAR
        menu_actions = MainMenu.draw(model)
        actions.update(menu_actions)
        
        # 2. THANH CÔNG CỤ VIEWPORT
        imgui.set_next_window_position(275 + 40, 20)
        imgui.set_next_window_size(max(win_w - 595 - 40, 100), 35)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.15, 0.15, 0.15, 0.9)
        imgui.begin("ViewportToolbar", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        imgui.same_line()
        if imgui.button(" Wireframe", 85, 22):
            self.cycle_polygon_mode()
            
        imgui.same_line()
        grid_status = "Grid: On" if coord_system.visible else "Grid: Off"
        if imgui.button(grid_status, 85, 22):
            actions['toggle_coord_system'] = True
            
        imgui.same_line()
        if imgui.button(" Flat Shading", 100, 22):
            actions['toggle_global_flat_color'] = True
            
        # --- THÊM NÚT NÀY VÀO ĐÂY ---
        imgui.same_line()
        mode_text = "View: RGB" if getattr(model, 'display_mode', 0) == 0 else "View: Depth Map"
        if imgui.button(mode_text, 115, 22):
            actions['toggle_display_mode'] = True
            
        imgui.end()
        imgui.pop_style_color()

        # 3. THANH CÔNG CỤ TRANSFORM TOOLS (Giữ nguyên code gốc)
        imgui.set_next_window_position(275, 20)
        imgui.set_next_window_size(40, max(win_h - 220, 100))
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.15, 0.15, 0.15, 0.9)
        imgui.begin("Tools", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        if imgui.image(self.cube_texture_id, 24, 24):
            print("Đã chọn Object Tool")
            
        def draw_tool_btn(tex_id, tool_name):
            is_active = model.active_tool == tool_name
            if is_active:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 1.0, 1.0) # Màu xanh Unity
                
            if imgui.image_button(tex_id, 16, 16):
                actions['set_tool'] = tool_name
                
            if is_active:
                imgui.pop_style_color(1)

        draw_tool_btn(self.hand_texture_id, 'hand')
        draw_tool_btn(self.move_texture_id, 'move')
        draw_tool_btn(self.rotate_texture_id, 'rotate')
        draw_tool_btn(self.scale_texture_id, 'scale')



        imgui.end()
        imgui.pop_style_color()

        # 4. HIERARCHY PANEL
        hierarchy_actions = HierarchyPanel.draw(model)
        actions.update(hierarchy_actions)
        
        # 5. INSPECTOR PANEL
        inspector_actions = InspectorPanel.draw(model, self.cube_texture_id)
        actions.update(inspector_actions)
        
        # 6. PROJECT & CONSOLE
        imgui.set_next_window_position(0, win_h - 200)
        imgui.set_next_window_size(max(win_w - 320, 100), 200)
        imgui.begin("Project", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        imgui.text("Assets > Models"); imgui.separator()
        if imgui.button("Import Model"): actions['browse_model_file'] = True
        imgui.text(f"Active: {model.model_filename}")
        imgui.end()
        
        # 7. SGD OPTIMIZER PANEL (PHẦN 2)
        if model.selected_category == 4:
            from components.sgd_panel import SGDPanel
            sgd_actions = SGDPanel.draw(model)
            actions.update(sgd_actions)

        # 8. BTL 2 BRIDGE PANEL
        if model.selected_category == 6:
            from components.btl2_panel import BTL2Panel
            btl2_actions = BTL2Panel.draw(model)
            actions.update(btl2_actions)
        
        return actions
