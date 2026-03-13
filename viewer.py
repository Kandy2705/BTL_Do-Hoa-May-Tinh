import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import itertools
    
from libs.shader import Shader
from libs.transform import Trackball
from libs.buffer import VAO

class Viewer:
    def __init__(self, width=1280, height=720):
        glfw.init()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
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
        glfw.set_key_callback(self.win, self.key_handler)
        
        self.fill_modes = itertools.cycle([gl.GL_FILL, gl.GL_LINE, gl.GL_POINT])
        
        self.selected_shader = 0
        self.shader_names = ["Color Interpolation", "Gouraud", "Phong"]

        self.menu_options = [
            "2D: Triangle", "2D: Rectangle", "2D: Pentagon", "2D: Hexagon", "2D: Circle", "2D: Star", "2D: Ellipse",
            "3D: Cube", "3D: Sphere (Tetrahedron)", "3D: Sphere (Grid)", "3D: Sphere (Lat-Long)",
            "3D: Cylinder", "3D: Cone", "3D: Tetrahedron",
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

        try:
            if self.selected_idx == 0:  # 2D: Triangle
                from geometry.triangle2d import Triangle
                self.active_drawable = Triangle("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 1:  # 2D: Rectangle
                from geometry.rectangle2d import Rectangle
                self.active_drawable = Rectangle("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 2:  # 2D: Pentagon
                from geometry.pentagon2d import Pentagon
                self.active_drawable = Pentagon("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 3:  # 2D: Hexagon
                from geometry.hexagon2d import Hexagon
                self.active_drawable = Hexagon("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 4:  # 2D: Circle
                from geometry.circle2d import Circle
                self.active_drawable = Circle("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 5:  # 2D: Star
                from geometry.star import Star
                self.active_drawable = Star("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 6:  # 2D: Ellipse
                from geometry.ellipse import Ellipse
                self.active_drawable = Ellipse("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 7:  # 3D: Cube
                from geometry.cube3d import Cube
                self.active_drawable = Cube("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 8:  # 3D: Sphere (Tetrahedron)
                from geometry.sphere_tetrahedron import SphereTetrahedron
                self.active_drawable = SphereTetrahedron("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 9:  # 3D: Sphere (Grid)
                from geometry.sphere_grid import SphereGrid
                self.active_drawable = SphereGrid("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 10:  # 3D: Sphere (Lat-Long)
                from geometry.sphere_latlong import SphereLatLong
                self.active_drawable = SphereLatLong("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 11:  # 3D: Cylinder
                from geometry.cylinder import Cylinder
                self.active_drawable = Cylinder("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 12:  # 3D: Cone
                from geometry.cone import Cone
                self.active_drawable = Cone("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 13:  # 3D: Tetrahedron
                from geometry.tetrahedron import Tetrahedron
                self.active_drawable = Tetrahedron("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
            elif self.selected_idx == 14:  # Part 2: SGD
                pass  # To be implemented
        except ImportError as e:
            print(f"Import error: {e}")
            return

        if self.active_drawable:
            self.drawables.append(self.active_drawable)

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

    def key_handler(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_W:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, next(self.fill_modes))

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