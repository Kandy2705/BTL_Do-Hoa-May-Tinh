import numpy as np
import OpenGL.GL as GL

class CoordinateSystem:
    def __init__(self, axis_length=10.0, grid_size=1.0, is_3d=False):
        self.axis_length = axis_length
        self.grid_size = grid_size
        self.visible = True
        self.is_3d = is_3d  # True: lưới XZ, False: lưới XY
        
        self._generate_axes()
        self._generate_grid()
        
    def _generate_axes(self):
        """Tạo 3 trục với màu sắc khác nhau"""
        vertices = []
        colors = []
        
        vertices.extend([[0, 0, 0], [self.axis_length, 0, 0]])
        colors.extend([[1, 0, 0], [1, 0, 0]])  # Đỏ
        
        vertices.extend([[0, 0, 0], [0, self.axis_length, 0]])
        colors.extend([[0, 1, 0], [0, 1, 0]])  # Xanh lá
        
        vertices.extend([[0, 0, 0], [0, 0, self.axis_length]])
        colors.extend([[0, 0, 1], [0, 0, 1]])  # Xanh dương
        
        self.axis_vertices = np.array(vertices, dtype=np.float32)
        self.axis_colors = np.array(colors, dtype=np.float32)
        
    def _generate_grid(self):
        """Tạo lưới trên mặt phẳng XY"""
        vertices = []
        colors = []
        
        grid_range = int(self.axis_length / self.grid_size)
        
        if self.is_3d:
            for i in range(-grid_range, grid_range + 1):
                z = i * self.grid_size
                vertices.extend([[-self.axis_length, 0, z], [self.axis_length, 0, z]])
                colors.extend([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
            
            for i in range(-grid_range, grid_range + 1):
                x = i * self.grid_size
                vertices.extend([[x, 0, -self.axis_length], [x, 0, self.axis_length]])
                colors.extend([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        else:
            for i in range(-grid_range, grid_range + 1):
                y = i * self.grid_size
                vertices.extend([[-self.axis_length, y, 0], [self.axis_length, y, 0]])
                colors.extend([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
            
            for i in range(-grid_range, grid_range + 1):
                x = i * self.grid_size
                vertices.extend([[x, -self.axis_length, 0], [x, self.axis_length, 0]])
                colors.extend([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        
        self.grid_vertices = np.array(vertices, dtype=np.float32)
        self.grid_colors = np.array(colors, dtype=np.float32)
        
    def setup(self, vao, uma):
        """Thiết lập VAO để render hệ trục tọa độ"""
        vao.add_vbo(0, self.axis_vertices, ncomponents=3, stride=0, offset=None)
        vao.add_vbo(1, self.axis_colors, ncomponents=3, stride=0, offset=None)
        
        self.grid_vao = vao.__class__()
        self.grid_vao.add_vbo(0, self.grid_vertices, ncomponents=3, stride=0, offset=None)
        self.grid_vao.add_vbo(1, self.grid_colors, ncomponents=3, stride=0, offset=None)
        
        self.vao = vao
        self.uma = uma
        
    def draw(self, projection, view, model=None):
        """Vẽ hệ trục tọa độ"""
        if not self.visible:
            return
            
        if model is None:
            model = np.identity(4, dtype=np.float32)
            
        modelview = view @ model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        self.grid_vao.activate()
        GL.glDrawArrays(GL.GL_LINES, 0, self.grid_vertices.shape[0])
        self.grid_vao.deactivate()
        
        self.vao.activate()
        GL.glDrawArrays(GL.GL_LINES, 0, self.axis_vertices.shape[0])
        self.vao.deactivate()
        
    def toggle_visibility(self):
        """Bật/tắt hiển thị hệ trục tọa độ"""
        self.visible = not self.visible
        
    def set_visibility(self, visible):
        """Thiết lập hiển thị hệ trục tọa độ"""
        self.visible = visible
        
    def set_mode(self, is_3d):
        """Chuyển đổi giữa chế độ 2D và 3D"""
        if self.is_3d != is_3d:
            self.is_3d = is_3d
            self._generate_grid()
            if hasattr(self, 'grid_vao'):
                self.grid_vao.add_vbo(0, self.grid_vertices, ncomponents=3, stride=0, offset=None)
                self.grid_vao.add_vbo(1, self.grid_colors, ncomponents=3, stride=0, offset=None)
