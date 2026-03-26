# Hình cầu lưới
import sys, os
import numpy as np
import OpenGL.GL as GL
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager
from geometry.base_shape import BaseShape


class SphereLatLong(BaseShape):
    def __init__(self, vert_shader, frag_shader, lat_div=20, long_div=20):
        super().__init__()
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.lat_div = lat_div
        self.long_div = long_div
        
        # --- CÁC BIẾN TRẠNG THÁI CHO SIÊU SHADER ---
        self.use_flat_color = False
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.use_texture = False
        self.texture_id = None
        self.render_mode = 2  # Mặc định là Phong Shading
        
        # TẠO DỮ LIỆU (Vị trí, Pháp tuyến, Màu)
        self.vertices, self.indices = self._generate_sphere()
        self.normals = self.vertices.copy()
        self.colors = np.array([[1.0, 1.0, 1.0]] * len(self.vertices), dtype=np.float32)
        self._generate_texcoords()

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_sphere(self):
        vertices = []
        for i in range(self.lat_div + 1):
            lat = np.pi * (-0.5 + i / self.lat_div)
            for j in range(self.long_div + 1):
                lon = 2 * np.pi * j / self.long_div
                x = np.cos(lat) * np.cos(lon)
                y = np.sin(lat)
                z = np.cos(lat) * np.sin(lon)
                vertices.append([x, y, z])
        vertices = np.array(vertices, dtype=np.float32)

        indices = []
        for i in range(self.lat_div):
            for j in range(self.long_div):
                first = i * (self.long_div + 1) + j
                second = first + self.long_div + 1
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
        indices = np.array(indices, dtype=np.uint32)

        return vertices, indices

    def _generate_texcoords(self):
        """Ma thuật Auto UV Mapping (Spherical Projection)"""
        norms = np.linalg.norm(self.vertices, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Tránh chia cho 0
        norm_v = self.vertices / norms
        
        u = 0.5 + np.arctan2(norm_v[:, 2], norm_v[:, 0]) / (2 * np.pi)
        v = 0.5 - np.arcsin(norm_v[:, 1]) / np.pi
        self.texcoords = np.column_stack((u, v)).astype(np.float32)

    def setup(self):
        # Bắt buộc tuân thủ layout: 0 (Pos), 1 (Color), 2 (Normal), 3 (UV)
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao.add_ebo(self.indices)
        return self

    def set_texture(self, filepath):
        if not filepath:
            self.use_texture = False
            return
        try:
            img = Image.open(filepath).convert("RGBA")
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.tobytes("raw", "RGBA", 0, -1)
            
            if self.texture_id is None:
                self.texture_id = GL.glGenTextures(1)
                
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            
            self.use_texture = True
            print(f"Đã load texture thành công: {filepath}")
        except Exception as e:
            print(f"Lỗi load texture: {e}")
            self.use_texture = False

    def draw(self, projection, view, model=None):
        GL.glUseProgram(self.shader.render_idx)
        
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # --- CÁC CÔNG TẮC CHO SIÊU SHADER ---
        loc_flat = GL.glGetUniformLocation(self.shader.render_idx, "u_use_flat_color")
        if loc_flat != -1: GL.glUniform1i(loc_flat, 1 if self.use_flat_color else 0)
        
        loc_flat_col = GL.glGetUniformLocation(self.shader.render_idx, "u_flat_color")
        if loc_flat_col != -1: 
            GL.glUniform3f(loc_flat_col, self.flat_color[0], self.flat_color[1], self.flat_color[2])
            
        loc_tex = GL.glGetUniformLocation(self.shader.render_idx, "u_use_texture")
        if loc_tex != -1: GL.glUniform1i(loc_tex, 1 if self.use_texture else 0)
        
        loc_mode = GL.glGetUniformLocation(self.shader.render_idx, "u_render_mode")
        if loc_mode != -1: GL.glUniform1i(loc_mode, self.render_mode)
        
        if self.use_texture and self.texture_id is not None:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            loc_sampler = GL.glGetUniformLocation(self.shader.render_idx, "u_texture")
            if loc_sampler != -1: GL.glUniform1i(loc_sampler, 0)
            
        if self.render_mode > 0:
            self.lighting.setup_phong(mode=1)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.size, GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()
        
        if self.use_texture:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    def set_color(self, color):
        self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        
        self.flat_color = np.array(color[:3], dtype=np.float32)

    def set_solid_color(self, color):
        self.use_flat_color = True
        self.flat_color = np.array(color[:3], dtype=np.float32)

    def cleanup(self):
        if hasattr(self, 'vao'): self.vao.delete()
        if hasattr(self, 'shader'): self.shader.delete()
        if hasattr(self, 'texture_id') and self.texture_id is not None:
            GL.glDeleteTextures(1, [self.texture_id])