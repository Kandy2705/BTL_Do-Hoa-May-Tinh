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

    def _process_ui_actions(self, actions):
        """Process UI actions and update model accordingly"""
        if 'category_changed' in actions:
            self.model.set_category(actions['category_changed'])
        
        if 'shape_changed' in actions:
            self.model.set_selected(actions['shape_changed'])
        
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
                        print(f"Đã chọn file: {filename}")
                        
                        if (self.model.selected_category == 3 and self.model.selected_idx == 0):
                            self.model.reload_current_shape()
                    else:
                        print("Đã hủy chọn file.")
                except Exception as e:
                    print(f"Lỗi khi mở hộp thoại Mac: {e}")
            else:
                print("Tính năng chọn file hiện chỉ hỗ trợ giao diện native trên macOS.")
                print("Vui lòng nhập đường dẫn thủ công.")

    def run(self) -> None:
        while not self.view.should_close():
            self.view.poll_events()
            self.view.begin_frame()

            ui_actions = self.view.draw_ui(self.model, self.coord_system)
            
            self._process_ui_actions(ui_actions)
            
            view = self.view.trackball.view_matrix()
            projection = self.view.trackball.projection_matrix(glfw.get_window_size(self.view.win))
            
            GL.glUseProgram(self.coord_shader.render_idx)
            self.view.draw_coordinate_system(self.coord_system, projection, view)
            
            self.view.draw_drawables(self.model.drawables)

            self.view.end_frame()
