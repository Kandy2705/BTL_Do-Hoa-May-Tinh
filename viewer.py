import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import itertools
from libs.transform import Trackball

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

    def draw_drawables(self, drawables):
        view = self.trackball.view_matrix()
        projection = self.trackball.projection_matrix(glfw.get_window_size(self.win))
        for drawable in drawables:
            drawable.draw(projection, view, None)

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
        imgui.set_next_window_position(275, 20)
        imgui.set_next_window_size(win_w - 595, 35)
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
        imgui.set_next_window_position(0, 20)
        imgui.set_next_window_size(275, win_h - 220)
        imgui.begin("Hierarchy", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        if imgui.tree_node("Objects", imgui.TREE_NODE_DEFAULT_OPEN):
            current_obj_name = model.menu_options[model.selected_idx] if model.selected_idx < len(model.menu_options) else "Unknown"
            imgui.selectable(f"  {current_obj_name}", True)
            
            # Menu chuột phải để tạo nhanh Object
            if imgui.begin_popup_context_window():
                if imgui.begin_menu("Create 2D Object"):
                    if imgui.menu_item("Triangle")[0]: actions['category_changed'], actions['shape_changed'] = 0, 0
                    if imgui.menu_item("Square")[0]: actions['category_changed'], actions['shape_changed'] = 0, 1
                    imgui.end_menu()
                if imgui.begin_menu("Create 3D Object"):
                    if imgui.menu_item("Cube")[0]: actions['category_changed'], actions['shape_changed'] = 1, 0
                    if imgui.menu_item("Sphere")[0]: actions['category_changed'], actions['shape_changed'] = 1, 1
                    imgui.end_menu()
                imgui.separator()
                imgui.menu_item("Delete")
                imgui.end_popup()
                
            imgui.tree_pop()
        imgui.end()

        # 4. BẢNG INSPECTOR (Bên phải - Chỉnh sửa thuộc tính)
        imgui.set_next_window_position(win_w - 320, 20)
        imgui.set_next_window_size(320, win_h - 20)
        imgui.begin("Inspector", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        imgui.checkbox("##active", True); imgui.same_line()
        imgui.text_colored(model.menu_options[model.selected_idx], 1, 1, 1, 1)
        imgui.separator()

        # Component Transform
        if imgui.collapsing_header("Transform", imgui.TREE_NODE_DEFAULT_OPEN):
            imgui.drag_float3("Position", 0.0, 0.0, 0.1)
            imgui.drag_float3("Rotation", 0.0, 0.0, 1.0)
            imgui.drag_float3("Scale", 1.0, 1.0, 0.1)

        if model.selected_category == 3:
            if imgui.collapsing_header("Model Loader", imgui.TREE_NODE_DEFAULT_OPEN):
                imgui.text(f"File: {model.model_filename}")
                if imgui.button("Change Model (.obj/.ply)", width=-1):
                    actions['browse_model_file'] = True

        # Component đặc thù cho BTL 2 [cite: 131, 136]
        if model.selected_category == 4:
            if imgui.collapsing_header("Synthetic Data Gen", imgui.TREE_NODE_DEFAULT_OPEN):
                imgui.button("Export COCO JSON", width=-1)
                imgui.button("Generate Depth Map", width=-1)

        # Chế độ dựng hình & Shader
        if imgui.collapsing_header("Mesh Renderer", imgui.TREE_NODE_DEFAULT_OPEN):
            changed_shader, new_shader = imgui.combo("Shader", model.selected_shader, model.shader_names)
            if changed_shader: actions['shader_changed'] = new_shader

        # Component đặc thù cho SGD [cite: 81]
        if model.selected_category == 2:
            if imgui.collapsing_header("Mathematical Surface Setting", imgui.TREE_NODE_DEFAULT_OPEN):
                imgui.text("Phuong trinh z = f(x, y):")
                imgui.push_item_width(-1) # Kéo dài ô input hết cỡ
                
                # Ô gõ hàm
                changed, new_func = imgui.input_text("##fxy", model.math_function, 256)
                if changed:
                    actions['math_function_changed'] = new_func # Gửi về controller
                    
                imgui.pop_item_width()
                imgui.text_disabled("Vi du: (x**2 + y - 11)**2")

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