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

# Import base shape
from base_shape import BaseShape


class Cone(BaseShape):
    # --- THAY ĐỔI: Thêm tham số 'lighting_enabled' để bật/tắt ánh sáng ---
    def __init__(self, vert_shader, frag_shader, func=None, radius=0.15, height=0.4, sectors=16, lighting_enabled=True):
        super().__init__()  # Initialize transform from BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.radius = radius
        self.height = height
        self.sectors = sectors
        self.lighting_enabled = lighting_enabled # Flag to enable/disable lighting
        
        if func is None:
            self.func = lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        else:
            self.func = func
            
        self._generate_geometry()
        
    def _generate_geometry(self):
        """Generate vertices, indices, normals and colors for the cone"""
        verts = []
        indices = []
        normals = []
        colors = []
        
        # 1. Đỉnh đỉnh hình nón (Tip) - Vertex 0
        verts.extend([0.0, self.height, 0.0])
        normals.extend([0.0, 1.0, 0.0]) # Hướng lên trên
        colors.extend([1.0, 1.0, 1.0]) # Mặc định trắng
        
        # 2. Các đỉnh đáy hình nón
        for i in range(self.sectors):
            angle = (2 * math.pi * i) / self.sectors
            x = self.radius * math.sin(angle)
            z = self.radius * math.cos(angle)
            verts.extend([x, 0.0, z])
            
            # Tính Normal bên hông - cho mục đích chiếu sáng
            h_len = math.sqrt(self.radius**2 + self.height**2)
            normal = np.array([self.height/h_len * math.sin(angle), self.radius/h_len, self.height/h_len * math.cos(angle)])
            normals.extend(normal)
            
            colors.extend([1.0, 1.0, 1.0]) # Mặc định trắng

        # 3. Tâm đáy hình nón - Vertex (sectors + 1)
        verts.extend([0.0, 0.0, 0.0])
        normals.extend([0.0, -1.0, 0.0]) # Hướng xuống dưới
        colors.extend([1.0, 1.0, 1.0]) # Mặc định trắng
        
        self.vertices = np.array(verts, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        
        # Tạo các tam giác
        # Tam giác bên hông
        for i in range(1, self.sectors + 1):
            next_v = (i % self.sectors) + 1
            indices.extend([0, next_v, i])
            
        # Tam giác đáy (bít kín đáy)
        center_idx = self.sectors + 1
        for i in range(1, self.sectors + 1):
            next_v = (i % self.sectors) + 1
            indices.extend([center_idx, i, next_v])
            
        self.indices = np.array(indices, dtype=np.int32)
        
    def setup(self):
        """Setup buffers and shader"""
        self.vao = VAO()
        
        # VAO 0: Vertices
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        
        # VAO 1: Colors
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        # VAO 2: Normals
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_ebo(self.indices)
        
        self.shader = Shader(self.vert_shader, self.frag_shader)
        self.uma = UManager(self.shader)
        
        # --- THAY ĐỔI: Chỉ khởi tạo LightingManager nếu được bật ---
        if self.lighting_enabled:
            self.lighting = LightingManager(self.uma)
        
        return self

    def draw(self, projection, view, model):
        """Draw the cone"""
        GL.glUseProgram(self.shader.render_idx)
        
        # Combine model matrices
        if hasattr(self, 'get_transform_matrix'):
            # Gộp transform của BaseShape
            object_transform = self.get_transform_matrix()
            final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        else:
            final_model = (model if model is not None else np.identity(4, dtype=np.float32))
            
        modelview = view @ final_model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # --- THAY ĐỔI: Chỉ setup lighting nếu được bật ---
        if self.lighting_enabled and hasattr(self, 'lighting'):
            if 'gouraud' in self.vert_shader.lower():
                self.lighting.setup_gouraud()
            elif 'phong' in self.vert_shader.lower():
                self.lighting.setup_phong(mode=1)
            else:
                self.lighting.setup_phong(mode=0)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)
    
    # --- THÊM HÀM MỚI: Dùng cho Gizmo để gán màu đơn sắc ---
    def set_solid_color(self, color):
        """Set a single solid color for the entire cone (RGBA list)"""
        if len(color) < 4:
            # Add Alpha=1.0 if not provided
            color = list(color) + [1.0]
            
        # Tạo mảng màu đơn sắc cho tất cả  đỉnh
        colors_rgba = np.array([color] * len(self.vertices), dtype=np.float32)
        
        # Cập nhật dữ liệu màu mới lên Card Đồ Họa (GPU)
        self.vao.activate()
        # VBO Màu sắc nằm ở location 1, 4 thành phần (RGBA)
        self.vao.add_vbo(1, colors_rgba, ncomponents=4, stride=0, offset=None)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vao'):
            self.vao.delete()
        if hasattr(self, 'shader'):
            self.shader.delete()