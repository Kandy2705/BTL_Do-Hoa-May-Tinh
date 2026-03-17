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
