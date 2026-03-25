import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import itertools
from libs.transform import Trackball
from PIL import Image

# Import UI components
from components.main_menu import MainMenu
from components.hierarchy_panel import HierarchyPanel
from components.inspector_panel import InspectorPanel

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
        
        # Draw regular drawables (mesh objects)
        for drawable in drawables:
            drawable.draw(projection, view, None)
        
        # Draw hierarchy objects
        hierarchy_drawables = []
        for i, obj in enumerate(hierarchy_objects):
            if obj["type"] in ["3d", "math", "custom_model"]:
                # Create drawable for this hierarchy object
                try:
                    if obj["type"] == "3d":
                        # Import and create Cube as default 3D object
                        from geometry import cube3d
                        drawable = cube3d.Cube("./shaders/basic.vert", "./shaders/basic.frag")
                        
                    elif obj["type"] == "math":
                        # Skip Math Surface for now - focus on texture UI
                        continue
                        
                    elif obj["type"] == "custom_model":
                        # Import and create Model Loader
                        from geometry import model_loader3d
                        drawable = model_loader3d.ModelLoader("")
                    
                    # Apply transform if drawable supports it
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
                    print(f"Failed to create drawable for {obj['name']}: {e}")
        
        # Draw all hierarchy objects
        for drawable in hierarchy_drawables:
            drawable.draw(projection, view, None)
                
                

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
        
        # 1. MAIN MENU BAR
        menu_actions = MainMenu.draw(model)
        actions.update(menu_actions)
        
        # 2. THANH CÔNG CỤ VIEWPORT
        imgui.set_next_window_position(275 + 40, 20)
        imgui.set_next_window_size(win_w - 595 - 40, 35)
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
        if imgui.button(" Shaded", 85, 22):
            print("Đã chọn Shaded Mode")
            
        imgui.end()
        imgui.pop_style_color()

        # 3. THANH CÔNG CỤ TRANSFORM TOOLS (Giữ nguyên code gốc)
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

        # 4. HIERARCHY PANEL
        hierarchy_actions = HierarchyPanel.draw(model)
        actions.update(hierarchy_actions)
        
        # 5. INSPECTOR PANEL
        inspector_actions = InspectorPanel.draw(model, self.cube_texture_id)
        actions.update(inspector_actions)
        
        # 6. PROJECT & CONSOLE
        imgui.set_next_window_position(0, win_h - 200)
        imgui.set_next_window_size(win_w - 320, 200)
        imgui.begin("Project", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        imgui.text("Assets > Models"); imgui.separator()
        if imgui.button("Import Model"): actions['browse_model_file'] = True
        imgui.text(f"Active: {model.model_filename}")
        imgui.end()
        
        return actions