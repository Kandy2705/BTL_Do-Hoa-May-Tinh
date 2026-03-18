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
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.win = glfw.create_window(width, height, 'BTL Do hoa May tinh', None, None)
        glfw.make_context_current(self.win)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)

        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.win)

        self.trackball = Trackball()
        self.last_mouse_pos = (0.0, 0.0)

        self.scroll_callback = None
        self.mouse_move_callback = None
        self.key_callback = None

        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_cursor_pos_callback(self.win, self._on_mouse_move)
        glfw.set_key_callback(self.win, self._on_key)

        self.fill_modes = itertools.cycle([gl.GL_FILL, gl.GL_LINE, gl.GL_POINT])

    def _on_scroll(self, window, xoffset, yoffset):
        if self.scroll_callback:
            self.scroll_callback(window, xoffset, yoffset)

    def _on_mouse_move(self, window, xpos, ypos):
        if self.mouse_move_callback:
            self.mouse_move_callback(window, xpos, ypos)
        self.last_mouse_pos = (xpos, ypos)

    def _on_key(self, window, key, scancode, action, mods):
        if self.key_callback:
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
        """Draw UI and return user interactions"""
        actions = {}
        
        # ===== WINDOW =====
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 1, 0.976, 0.988, 1.0)

        # ===== TITLE =====
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, 0.988, 0.788, 0.855, 1.0)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, 0.988, 0.788, 0.855, 1.0)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, 0.988, 0.788, 0.855, 1.0)

        # ===== HEADER =====
        imgui.push_style_color(imgui.COLOR_HEADER, 0.988, 0.788, 0.855, 1.0)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.988, 0.788, 0.855, 1.0)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.988, 0.788, 0.855, 1.0)

        # ===== SEPARATOR =====
        imgui.push_style_color(imgui.COLOR_SEPARATOR, 1.0, 0.1, 0.3, 1.0)

        # ===== TEXT =====
        imgui.push_style_color(imgui.COLOR_TEXT, 0.651, 0, 0.145, 1.0)

        # ===== FRAME (Combo box, input) =====
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.988, 0.788, 0.855, 1.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, 0.988, 0.522, 0.737, 1.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 0.988, 0.422, 0.647, 1.0)

        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, 1.0, 0.95, 0.97, 1.0)
        imgui.push_style_color(imgui.COLOR_NAV_HIGHLIGHT, 0.988, 0.522, 0.737, 1.0)

        # ===== BUTTON =====
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.988, 0.788, 0.855, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.988, 0.608, 0.737, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.988, 0.475, 0.647, 1.0)

        # ===== BORDER =====
        imgui.push_style_color(imgui.COLOR_BORDER, 0.988, 0.788, 0.855, 1.0)

        # ===== RESIZE GRIP =====
        imgui.push_style_color(imgui.COLOR_RESIZE_GRIP, 0.988, 0.788, 0.855, 1.0)
        imgui.push_style_color(imgui.COLOR_RESIZE_GRIP_HOVERED, 0.988, 0.608, 0.737, 1.0)
        imgui.push_style_color(imgui.COLOR_RESIZE_GRIP_ACTIVE, 0.988, 0.475, 0.647, 1.0)

        # ===== UI =====
        imgui.begin("BTL1: Controls")

        changed_cat, new_cat = imgui.combo(
            "Select Category",
            model.selected_category,
            model.category_options
        )
        if changed_cat:
            actions['category_changed'] = new_cat

        changed_shape, new_shape = imgui.combo(
            "Select Shape",
            model.selected_idx,
            model.menu_options
        )
        if changed_shape:
            actions['shape_changed'] = new_shape

        changed_shader, new_shader = imgui.combo(
            "Select Shader",
            model.selected_shader,
            model.shader_names
        )
        if changed_shader:
            actions['shader_changed'] = new_shader

        coord_status = "On" if coord_system.visible else "Off"
        if imgui.button(f"Coordinate System: {coord_status}"):
            actions['toggle_coord_system'] = True

        imgui.text("Left mouse: rotate | Scroll: zoom")
        imgui.text("Press G to toggle coordinate system")

        imgui.end()

        # ===== POP ALL =====
        imgui.pop_style_color(21)
        
        return actions
