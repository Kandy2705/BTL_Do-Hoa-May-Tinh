import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
    
from libs.shader import Shader
from libs.transform import Trackball
from libs.buffer import VAO

class Viewer:
    def __init__(self, width=1280, height=720):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, gl.GL_CORE_PROFILE)
        self.win = glfw.create_window(width, height, 'BTL Do hoa May tinh', None, None)
        glfw.make_context_current(self.win)

        # Một số thiết lập OpenGL cơ bản
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)

        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.win)

        self.trackball = Trackball()
        self.last_mouse_pos = (0.0, 0.0)
        glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        
        self.selected_shader = 0
        self.shader_names = ["Color Interpolation", "Gouraud", "Phong"]

        self.menu_options = [
            "2D: Triangle", "2D: Circle", "2D: Star", 
            "3D: Cube", "3D: Sphere (Lat-Long)", 
            "Part 2: SGD (Himmelblau)"
        ]
        self.selected_idx = 0  # Mục đang được chọn
        self.active_drawable = None # Đối tượng đang được hiển thị
        self.drawables = []

    def update_drawable(self):
        """Hàm này sẽ khởi tạo đối tượng mới dựa trên lựa chọn từ menu"""
        # Xóa đối tượng cũ nếu cần để giải phóng bộ nhớ GPU
        self.active_drawable = None
        self.drawables = []

        if self.selected_idx == 0: # Triangle [cite: 32]
            # Giả sử bạn có class Triangle trong geometry/shapes_2d.py
            # model = Triangle("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            pass
        elif self.selected_idx == 3: # Cube 
            try:
                from cube import Cube  # Hoặc từ geometry/shapes_3d.py
            except ImportError:
                # Nếu chưa có module cube, bỏ qua để không crash.
                return

            self.active_drawable = Cube(
                "./shaders/color_interp.vert", 
                "./shaders/color_interp.frag"
            ).setup()
            self.drawables.append(self.active_drawable)
        elif self.selected_idx == 5: # SGD Simulation [cite: 62]
            # Khởi tạo logic cho Phần 2
            pass

    def add(self, drawable):
        """Thêm đối tượng để vẽ mỗi frame."""
        self.drawables.append(drawable)

    def on_scroll(self, window, xoffset, yoffset):
        """Phóng to/thu nhỏ khi cuộn chuột."""
        width, height = glfw.get_window_size(window)
        # yoffset > 0 khi cuộn lên, < 0 khi cuộn xuống
        self.trackball.zoom(yoffset, max(width, height))

    def on_mouse_move(self, window, xpos, ypos):
        """Quay view khi giữ nút trái và kéo."""
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.trackball.drag(self.last_mouse_pos, (xpos, ypos), glfw.get_window_size(window))
        self.last_mouse_pos = (xpos, ypos)

    def run(self):
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            self.imgui_impl.process_inputs()

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # --- VẼ GIAO DIỆN IMGUI ---
            imgui.new_frame()
            imgui.begin("BTL1: Controls")
            
            # Nếu người dùng đổi mục trong Combo Box 
            changed, new_idx = imgui.combo("Select Shape", self.selected_idx, self.menu_options)
            if changed:
                self.selected_idx = new_idx
                self.update_drawable() # Cập nhật đối tượng hiển thị ngay lập tức

            imgui.text("Dung chuot trai de xoay, cuon de zoom")
            imgui.end()

            # --- VẼ OPENGL ---
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(glfw.get_window_size(self.win))

            for drawable in self.drawables:
                drawable.draw(projection, view, None)

            imgui.render()
            self.imgui_impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.win)