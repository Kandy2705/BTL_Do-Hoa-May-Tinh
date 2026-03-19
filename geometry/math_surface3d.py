import sys
import os
import numpy as np
import ctypes
import math
import warnings

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import OpenGL.GL as GL


class MathematicalSurface(object):
    def __init__(self, vert_shader, frag_shader, func=None, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        
        if func is None:
            self.func = lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        else:
            self.func = func
            
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        self._generate_surface()
        
    def _generate_surface(self):
        """Generate vertices, indices, normals and colors for the mathematical surface"""
        import warnings
        
        # 1. Tạo lưới tọa độ THỰC để tính toán hàm f(x,y)
        x_real = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y_real = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        X_real, Y_real = np.meshgrid(x_real, y_real)
        
        # Bắt lỗi toán học (chia 0, căn âm...)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_real = self.func(X_real, Y_real)
            
        Z_real = np.nan_to_num(Z_real, nan=0.0, posinf=100.0, neginf=-100.0)
        Z_real = np.clip(Z_real, -100.0, 100.0) # Gọt bớt các đỉnh quá sắc nhọn
        
        # 2. HÀM CHUẨN HÓA ĐỂ VẼ (Ép X, Y, Z về khối hộp [-2, 2] cho vừa khung hình)
        def normalize_for_draw(arr, target_min=-2.0, target_max=2.0):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min == 0:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min) * (target_max - target_min) + target_min
            
        X_draw = normalize_for_draw(X_real)
        Y_draw = normalize_for_draw(Y_real)
        Z_draw = normalize_for_draw(Z_real)
        
        vertices = []
        normals = []
        colors = []
        
        # Lấy khoảng cách giữa 2 điểm vẽ để tính pháp tuyến (bắt sáng mượt hơn)
        dx_draw = X_draw[0, 1] - X_draw[0, 0]
        dy_draw = Y_draw[1, 0] - Y_draw[0, 0]
        
        # 3. NẠP ĐỈNH, PHÁP TUYẾN VÀ MÀU SẮC
        for i in range(self.resolution):
            for j in range(self.resolution):
                vertices.append([X_draw[i, j], Y_draw[i, j], Z_draw[i, j]])
                
                # Tính Vector pháp tuyến dựa trên hình vẽ đã chuẩn hóa
                if 0 < i < self.resolution-1 and 0 < j < self.resolution-1:
                    dz_dx = (Z_draw[i, j+1] - Z_draw[i, j-1]) / (2 * dx_draw)
                    dz_dy = (Z_draw[i+1, j] - Z_draw[i-1, j]) / (2 * dy_draw)
                    normal = np.array([-dz_dx, -dz_dy, 1.0])
                    normal = normal / np.linalg.norm(normal)
                else:
                    normal = np.array([0.0, 0.0, 1.0])
                normals.append(normal)
                
                # Phối màu quang phổ (Càng cao càng đỏ, càng thấp càng xanh)
                h = (Z_draw[i, j] + 2.0) / 4.0  # Chuyển độ cao về khoảng 0.0 -> 1.0
                colors.append([
                    0.2 + 0.8 * h,          # Đỏ tăng dần theo chiều cao
                    0.3 + 0.4 * (1 - abs(2*h-1)), # Xanh lá ở giữa lưng chừng
                    0.8 - 0.6 * h           # Xanh dương đậm ở vùng trũng
                ])
                
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        
        # 4. NỐI CÁC ĐIỂM THÀNH LƯỚI TAM GIÁC (Giữ nguyên)
        indices = []
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                v0 = i * self.resolution + j
                v1 = i * self.resolution + (j + 1)
                v2 = (i + 1) * self.resolution + j
                v3 = (i + 1) * self.resolution + (j + 1)
                
                indices.extend([v0, v1, v2])
                indices.extend([v2, v1, v3])
        
        self.indices = np.array(indices, dtype=np.int32)
        
    def setup(self):
        """Setup buffers and shader"""
        self.vao = VAO()
        
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_ebo(self.indices)
        
        self.shader = Shader(self.vert_shader, self.frag_shader)
        self.uma = UManager(self.shader)
        
        self.lighting = LightingManager(self.uma)
        
        return self

    def draw(self, projection, view, model):
        """Draw the mathematical surface"""
        if model is None:
            model = T.identity()
            
        GL.glUseProgram(self.shader.render_idx)
        modelview = view @ model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)
        else:
            self.lighting.setup_phong(mode=0)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vao'):
            self.vao.delete()
        if hasattr(self, 'shader'):
            self.shader.delete()
