import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import itertools
from libs.transform import Trackball
from PIL import Image

class Viewer:
    def __init__(self, width=1280, height=720):
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
        self.trackball = Trackball()
        
        # Call Unity Style Setup
        self._apply_unity_style()

        self.scroll_callback = None
        self.mouse_move_callback = None
        self.key_callback = None
        self.last_mouse_pos = (0.0, 0.0)
        self.fill_modes = itertools.cycle([gl.GL_FILL, gl.GL_LINE, gl.GL_POINT])

        self.cube_texture_id, _, _ = self.load_texture("assets/textures/cube-solid.png")
        self.hand_texture_id, _, _ = self.load_texture("assets/textures/hand-solid.png")
        self.move_texture_id, _, _ = self.load_texture("assets/textures/arrows-up-down-left-right-solid.png")
        self.rotate_texture_id, _, _ = self.load_texture("assets/textures/arrows-rotate-solid.png")
        self.scale_texture_id, _, _ = self.load_texture("assets/textures/up-right-from-square-solid.png")

        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_cursor_pos_callback(self.win, self._on_mouse_move)
        glfw.set_key_callback(self.win, self._on_key)

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
        if self.scroll_callback: self.scroll_callback(window, xoffset, yoffset)
    def _on_mouse_move(self, window, xpos, ypos):
        if self.mouse_move_callback: self.mouse_move_callback(window, xpos, ypos)
        self.last_mouse_pos = (xpos, ypos)
    def _on_key(self, window, key, scancode, action, mods):
        self.imgui_impl.keyboard_callback(window, key, scancode, action, mods)
        if not imgui.get_io().want_capture_keyboard and self.key_callback:
            self.key_callback(window, key, scancode, action, mods)

    def load_texture(self, image_path):
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
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        imgui.new_frame()

    def end_frame(self) -> None:
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.win)

    def draw_drawables(self, drawables, hierarchy_objects):
        view = self.trackball.view_matrix()
        projection = self.trackball.projection_matrix(glfw.get_window_size(self.win))
        
        for i, obj in enumerate(hierarchy_objects):
            if obj["type"] in ["3d", "math", "custom_model"]:
                pos = obj["transform"]["position"]
                rot = obj["transform"]["rotation"]
                sca = obj["transform"]["scale"]
                
                

    def cycle_polygon_mode(self) -> None:
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, next(self.fill_modes))

    def draw_coordinate_system(self, coord_system, projection, view):
        """Draw coordinate system"""
        if coord_system.visible:
            coord_system.draw(projection, view)

    def draw_ui(self, model, coord_system):
        actions = {}
        win_w, win_h = glfw.get_window_size(self.win)
        
        # 1. MAIN MENU BAR (Lồng nhau y hệt Unity)
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Import Model")[0]: actions['browse_model_file'] = True
                if imgui.menu_item("Exit")[0]: glfw.set_window_should_close(self.win, True)
                imgui.end_menu()

            if imgui.begin_menu("BTL 1"):
                if imgui.begin_menu("2D Shapes"):
                    original_cat = model.selected_category
                    model.selected_category = 0 
                    for idx, name in enumerate(model.menu_options):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 0
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()
                
                if imgui.begin_menu("3D Shapes"):
                    original_cat = model.selected_category
                    model.selected_category = 1
                    for idx, name in enumerate(model.menu_options):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 1
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()
                
                if imgui.begin_menu("Mathematical Surface"):
                    if imgui.menu_item("Z = f(x,y)")[0]:
                        actions['category_changed'] = 2
                        actions['shape_changed'] = 0
                    imgui.end_menu()
                
                if imgui.begin_menu("Model from file"):
                    if imgui.menu_item("Model from .obj/.ply file")[0]:
                        actions['category_changed'] = 3
                        actions['shape_changed'] = 0
                    imgui.end_menu()
                
                if imgui.begin_menu("Optimization (SGD)"):
                    original_cat = model.selected_category
                    model.selected_category = 4
                    for idx, name in enumerate(model.menu_options):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 4
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()
                imgui.end_menu()

            if imgui.begin_menu("BTL 2"):
                if imgui.menu_item("Setup Road Scene")[0]: actions['category_changed'] = 4
                if imgui.begin_menu("Add Traffic Object"):
                    if imgui.menu_item("Main Vehicle")[0]: pass
                    if imgui.menu_item("Pedestrian")[0]: pass
                    if imgui.menu_item("Traffic Light")[0]: pass
                    imgui.end_menu()
                imgui.end_menu()
            imgui.end_main_menu_bar()

        # 2. THANH CÔNG CỤ VIEWPORT (2D/3D, Wireframe, Grid)
        imgui.set_next_window_position(275 + 40, 20)
        imgui.set_next_window_size(win_w - 595 - 40, 35)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.15, 0.15, 0.15, 0.9)
        imgui.begin("ViewportToolbar", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        
        imgui.same_line()
        # Nút bật/tắt Wireframe
        if imgui.button(" Wireframe", 85, 22):
            self.cycle_polygon_mode()
            
        imgui.same_line()
        # Nút bật/tắt Lưới tọa độ (Grid)
        grid_status = "Grid: On" if coord_system.visible else "Grid: Off"
        if imgui.button(grid_status, 85, 22):
            actions['toggle_coord_system'] = True
            
        imgui.end()
        imgui.pop_style_color()

        # 3. BẢNG HIERARCHY (Bên trái)
        imgui.set_next_window_position(275, 20)
        imgui.set_next_window_size(40, win_h - 220)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.15, 0.15, 0.15, 0.9)
        imgui.begin("Tools", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        if imgui.image(self.cube_texture_id, 24, 24):
            print("Đã chọn Object Tool")
        
        if imgui.image_button(self.hand_texture_id, 16, 16):
            print("Đã chọn Hand Tool")
            
        if imgui.image_button(self.move_texture_id, 16, 16):
            print("Đã chọn Move Tool")
            
        if imgui.image_button(self.rotate_texture_id, 16, 16):
            print("Đã chọn Rotate Tool")
            
        if imgui.image_button(self.scale_texture_id, 16, 16):
            print("Đã chọn Scale Tool")

        imgui.end()
        imgui.pop_style_color()

        # 4. BẢNG HIERARCHY (Bên trái)
        imgui.set_next_window_position(0, 20)
        imgui.set_next_window_size(275, win_h - 220)
        imgui.begin("Hierarchy", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        # Menu chuột phải cho toàn bộ Hierarchy window
        if imgui.begin_popup_context_window():
            if imgui.begin_menu("Add 2D Object"):
                original_cat = model.selected_category
                model.selected_category = 0 
                for idx, name in enumerate(model.menu_options):
                    if imgui.menu_item(name)[0]:
                        actions['category_changed'] = 0
                        actions['shape_changed'] = idx
                model.selected_category = original_cat
                imgui.end_menu()
            if imgui.begin_menu("Add 3D Object"):
                original_cat = model.selected_category
                model.selected_category = 1
                for idx, name in enumerate(model.menu_options):
                    if imgui.menu_item(name)[0]:
                        actions['category_changed'] = 1
                        actions['shape_changed'] = idx
                model.selected_category = original_cat
                imgui.end_menu()
            if imgui.begin_menu("Add Mathematical Surface"):
                if imgui.menu_item("Z = f(x,y)")[0]:
                    actions['category_changed'] = 2
                    actions['shape_changed'] = 0
                imgui.end_menu()
            if imgui.begin_menu("Add Model from file"):
                if imgui.menu_item("Model from .obj/.ply file")[0]:
                    actions['category_changed'] = 3
                    actions['shape_changed'] = 0
                imgui.end_menu()
            if imgui.begin_menu("Add Light"):
                if imgui.menu_item("Light")[0]: 
                    actions['add_light'] = True
                imgui.end_menu()
            if imgui.begin_menu("Add Camera"):
                if imgui.menu_item("Camera")[0]: 
                    actions['add_camera'] = True
                imgui.end_menu()
            imgui.separator()
            imgui.menu_item("Delete")
            imgui.end_popup()
        
        if imgui.tree_node("MainScene", imgui.TREE_NODE_DEFAULT_OPEN):
            for i, obj in enumerate(model.hierarchy_objects):
                # BẮT BUỘC PHẢI TÁCH TUPLE Ở ĐÂY để không bị lỗi dính object cuối cùng
                clicked, state = imgui.selectable(f"{obj['name']}##{obj['id']}", obj.get("selected", False))
                if clicked:
                    actions['select_hierarchy_object'] = i
            imgui.tree_pop()

        imgui.end()

        # --- BẢNG INSPECTOR (Bên phải) ---
        imgui.set_next_window_position(win_w - 320, 20)
        imgui.set_next_window_size(320, win_h - 20)
        imgui.begin("Inspector", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        selected_obj = model.get_selected_hierarchy_object()
        
        if not selected_obj:
            imgui.text_disabled("No Object Selected")
        else:
            obj_id = selected_obj.get("id", 0)
            obj_type = selected_obj.get("type", "unknown")
            obj_name = selected_obj.get("name", "Unknown")
            obj_data = selected_obj  # Dùng chung dictionary
            
            # --- HEADER ---
            imgui.checkbox("##active", True); imgui.same_line()
            icon = self.cube_texture_id
            imgui.image(icon, 16, 16)
            imgui.same_line()
            imgui.text_colored(f"{obj_name}", 1.0, 1.0, 1.0, 1.0)
            imgui.separator()
            
            # --- COMPONENT: TRANSFORM (Luôn có) ---
            transform = obj_data.get("transform", {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
            if imgui.collapsing_header("Transform", imgui.TREE_NODE_DEFAULT_OPEN):
                imgui.columns(2, "trans_layout", False)
                imgui.set_column_width(0, 80)
                
                # Position
                imgui.text("Position"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_pos, new_pos = imgui.drag_float3(f"##pos_{obj_id}", transform["position"][0], transform["position"][1], transform["position"][2], 0.1)
                if changed_pos: actions['update_transform_pos'] = {'obj_id': obj_id, 'value': list(new_pos)}
                imgui.pop_item_width(); imgui.next_column()
                
                # Rotation
                imgui.text("Rotation"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_rot, new_rot = imgui.drag_float3(f"##rot_{obj_id}", transform["rotation"][0], transform["rotation"][1], transform["rotation"][2], 1.0)
                if changed_rot: actions['update_transform_rot'] = {'obj_id': obj_id, 'value': list(new_rot)}
                imgui.pop_item_width(); imgui.next_column()
                
                # Scale
                imgui.text("Scale"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_sca, new_sca = imgui.drag_float3(f"##sca_{obj_id}", transform["scale"][0], transform["scale"][1], transform["scale"][2], 0.1)
                if changed_sca: actions['update_transform_scale'] = {'obj_id': obj_id, 'value': list(new_sca)}
                imgui.pop_item_width(); imgui.next_column()
                
                # QUAN TRỌNG: Phải đóng cột trước khi kết thúc khối Transform để các Component sau không bị thụt lề
                imgui.columns(1)

            # --- DYNAMIC COMPONENTS: MESH RENDERER ---
            if obj_type in ["2d", "3d", "math", "custom_model", "mesh"]:
                mesh_renderer = obj_data.get("mesh_renderer", {"shader_idx": 0, "color": [1.0, 0.5, 0.0]})
                if imgui.collapsing_header("Mesh Renderer", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "mesh_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("Shader"); imgui.next_column()
                    imgui.push_item_width(-1)
                    current_shader = mesh_renderer.get("shader_idx", 0)
                    changed_shader, new_shader = imgui.combo(f"##shader_{obj_id}", current_shader, model.shader_names)
                    if changed_shader: 
                        actions['update_mesh_shader'] = {"obj_id": obj_id, "value": new_shader}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Color"); imgui.next_column()
                    imgui.push_item_width(-1)
                    current_color = mesh_renderer.get("color", [1.0, 0.5, 0.0])
                    changed_color, new_color = imgui.color_edit3(f"##color_{obj_id}", *current_color)
                    if changed_color: 
                        actions['update_mesh_color'] = {"obj_id": obj_id, "value": list(new_color)}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.columns(1)

            # --- DYNAMIC COMPONENTS: MATH SCRIPT ---
            if obj_type == "math":
                math_data = obj_data.get("math_data", {"function": "(x**2 + y - 11)**2"})
                if imgui.collapsing_header("Math Script", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.text("z = f(x, y):")
                    imgui.push_item_width(-1)
                    changed_func, new_func = imgui.input_text(f"##fxy_{obj_id}", math_data.get("function", ""), 256)
                    if changed_func: actions['update_math_function'] = {"obj_id": obj_id, "value": new_func}
                    imgui.pop_item_width()
            
            # --- DYNAMIC COMPONENTS: LIGHT SETTINGS ---
            elif obj_type == "light":
                light_data = obj_data.get("light_data", {"intensity": 1.0, "color": [1.0, 1.0, 1.0]})
                if imgui.collapsing_header("Light Settings", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "light_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("Intensity"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_intensity, new_intensity = imgui.drag_float(f"##intensity_{obj_id}", light_data.get("intensity", 1.0), 0.1, 10.0)
                    if changed_intensity: actions['update_light_intensity'] = {"obj_id": obj_id, "value": new_intensity}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Color"); imgui.next_column()
                    imgui.push_item_width(-1)
                    current_light_color = light_data.get("color", [1.0, 1.0, 1.0])
                    changed_light_color, new_light_color = imgui.color_edit3(f"##light_color_{obj_id}", *current_light_color)
                    if changed_light_color: actions['update_light_color'] = {"obj_id": obj_id, "value": list(new_light_color)}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.columns(1)
                    
            # --- DYNAMIC COMPONENTS: CAMERA SETTINGS ---
            elif obj_type == "camera":
                camera_data = obj_data.get("camera_data", {"fov": 60.0, "near": 0.1, "far": 100.0})
                if imgui.collapsing_header("Camera Settings", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "cam_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("FOV"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_fov, new_fov = imgui.slider_float(f"##fov_{obj_id}", camera_data.get("fov", 60.0), 10.0, 120.0)
                    if changed_fov: actions['update_camera_fov'] = {"obj_id": obj_id, "value": new_fov}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Near"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_near, new_near = imgui.drag_float(f"##near_{obj_id}", camera_data.get("near", 0.1), 0.01, 10.0)
                    if changed_near: actions['update_camera_near'] = {"obj_id": obj_id, "value": new_near}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Far"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_far, new_far = imgui.drag_float(f"##far_{obj_id}", camera_data.get("far", 100.0), 1.0, 1000.0)
                    if changed_far: actions['update_camera_far'] = {"obj_id": obj_id, "value": new_far}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.columns(1)

        imgui.end()

        # 4. PROJECT & CONSOLE (Giữ nguyên layout dưới)
        imgui.set_next_window_position(0, win_h - 200)
        imgui.set_next_window_size(win_w - 320, 200)
        imgui.begin("Project", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        imgui.text("Assets > Models"); imgui.separator()
        if imgui.button("Import Model"): actions['browse_model_file'] = True
        imgui.text(f"Active: {model.model_filename}")
        imgui.end()

        return actions