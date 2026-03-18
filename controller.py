import glfw
import imgui
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

        self.model.load_active_drawable()
        
        self.coord_system = CoordinateSystem(axis_length=20.0, grid_size=1.0)
        self._setup_coordinate_system()

        self.view.scroll_callback = self.on_scroll
        self.view.mouse_move_callback = self.on_mouse_move
        self.view.key_callback = self.on_key

    def on_scroll(self, window, xoffset, yoffset):
        width, height = glfw.get_window_size(window)
        self.view.trackball.zoom(yoffset, max(width, height))

    def on_mouse_move(self, window, xpos, ypos):
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.view.trackball.drag(self.view.last_mouse_pos, (xpos, ypos), glfw.get_window_size(window))

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_W:
                self.view.cycle_polygon_mode()
            elif key == glfw.KEY_Q:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_S:
                self.model.set_shader((self.model.selected_shader + 1) % 3)
            elif key == glfw.KEY_G:
                self.coord_system.toggle_visibility()

    def _setup_coordinate_system(self):
        """Setup coordinate system with simple color shader"""
        vert_shader = "./shaders/color_interp.vert"
        frag_shader = "./shaders/color_interp.frag"
        
        self.coord_vao = VAO()
        self.coord_shader = Shader(vert_shader, frag_shader)
        self.coord_uma = UManager(self.coord_shader)
        
        self.coord_system.setup(self.coord_vao, self.coord_uma)

    def _draw_ui(self) -> None:
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
            self.model.selected_category,
            self.model.category_options
        )
        if changed_cat:
            self.model.set_category(new_cat)

        changed_shape, new_shape = imgui.combo(
            "Select Shape",
            self.model.selected_idx,
            self.model.menu_options
        )
        if changed_shape:
            self.model.set_selected(new_shape)

        changed_shader, new_shader = imgui.combo(
            "Select Shader",
            self.model.selected_shader,
            self.model.shader_names
        )
        if changed_shader:
            self.model.set_shader(new_shader)

        coord_status = "On" if self.coord_system.visible else "Off"
        if imgui.button(f"Coordinate System: {coord_status}"):
            self.coord_system.toggle_visibility()

        imgui.text("Left mouse: rotate | Scroll: zoom")
        imgui.text("Press G to toggle coordinate system")

        imgui.end()

        # ===== POP ALL =====
        imgui.pop_style_color(21)

    def run(self) -> None:
        while not self.view.should_close():
            self.view.poll_events()
            self.view.begin_frame()

            self._draw_ui()
            
            view = self.view.trackball.view_matrix()
            projection = self.view.trackball.projection_matrix(glfw.get_window_size(self.view.win))
            
            GL.glUseProgram(self.coord_shader.render_idx)
            self.coord_system.draw(projection, view)
            
            self.view.draw_drawables(self.model.drawables)

            self.view.end_frame()
