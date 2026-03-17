import glfw
import imgui
from typing import Optional

from model import AppModel
from viewer import Viewer


class AppController:
    def __init__(self, model: Optional[AppModel] = None, view: Optional[Viewer] = None) -> None:
        self.view = view or Viewer()
        self.model = model or AppModel()

        self.model.load_active_drawable()

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

    def _draw_ui(self) -> None:
        imgui.begin("BTL1: Controls")
        changed, new_idx = imgui.combo("Select Shape", self.model.selected_idx, self.model.menu_options)
        if changed:
            self.model.set_selected(new_idx)

        imgui.text("Dung chuot trai de xoay, cuon de zoom")
        imgui.end()

    def run(self) -> None:
        while not self.view.should_close():
            self.view.poll_events()
            self.view.begin_frame()

            self._draw_ui()
            self.view.draw_drawables(self.model.drawables)

            self.view.end_frame()
