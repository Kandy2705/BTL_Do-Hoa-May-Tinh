# Hình trụ
import sys, os
import numpy as np
import OpenGL.GL as GL
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager
from geometry.base_shape import BaseShape


class Cylinder(BaseShape):
    def __init__(self, vert_shader, frag_shader, segments=32, radius=1.0, height=2.0):
        super().__init__()
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.segments = segments
        self.radius = radius
        self.height = height
        
        # --- CÁC BIẾN TRẠNG THÁI CHO SIÊU SHADER ---
        self.use_flat_color = False
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.use_texture = False
        self.texture_id = None
        self.render_mode = 2  # Mặc định là Phong Shading
        
        # TẠO DỮ LIỆU (Vị trí, Pháp tuyến, Màu, UV)
        self.vertices, self.normals, self.colors, self.indices = self._generate_cylinder_geometry()
        self.texcoords = self._generate_cylinder_texcoords()

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)
    def _generate_cylinder_geometry(self):
        vertices = []
        normals = []
        colors = []
        indices = []

        for i in range(self.segments + 1):
            theta = 2.0 * np.pi * i / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            
            # Màu mặc định trắng
            color = [1.0, 1.0, 1.0]
            
            vertices.append([x, -self.height/2, z])
            normals.append([np.cos(theta), 0, np.sin(theta)])
            colors.append(color)
            
            vertices.append([x, self.height/2, z])
            normals.append([np.cos(theta), 0, np.sin(theta)])
            colors.append(color)

        bottom_center_idx = len(vertices)
        vertices.append([0, -self.height/2, 0])
        normals.append([0, -1, 0])
        colors.append([1.0, 1.0, 1.0])

        top_center_idx = len(vertices)
        vertices.append([0, self.height/2, 0])
        normals.append([0, 1, 0])
        colors.append([1.0, 1.0, 1.0])

        for i in range(self.segments):
            b1 = i * 2
            t1 = i * 2 + 1
            b2 = (i + 1) * 2
            t2 = (i + 1) * 2 + 1
            indices.extend([b1, t1, b2, t1, t2, b2])

        for i in range(self.segments):
            b1 = i * 2
            b2 = (i + 1) * 2
            indices.extend([bottom_center_idx, b2, b1])

        for i in range(self.segments):
            t1 = i * 2 + 1
            t2 = (i + 1) * 2 + 1
            indices.extend([top_center_idx, t1, t2])

        return (np.array(vertices, dtype=np.float32), 
                np.array(normals, dtype=np.float32),
                np.array(colors, dtype=np.float32),
                np.array(indices, dtype=np.int32))
    
    def _generate_cylinder_texcoords(self):
        """Generate UV coordinates for cylinder texture mapping"""
        texcoords = []
        
        for i in range(self.segments + 1):
            u = i / self.segments
            texcoords.extend([[u, 0.0], [u, 1.0]])
        
        # UV cho tâm đáy và tâm đỉnh
        texcoords.extend([[0.5, 0.0], [0.5, 1.0]])
        
        return np.array(texcoords, dtype=np.float32)

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
            #img = img.transpose(Image.FLIP_TOP_BOTTOM)
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
        
        # Upload view matrix for light transform
        loc_view = GL.glGetUniformLocation(self.shader.render_idx, "view")
        if loc_view != -1: self.uma.upload_uniform_matrix4fv(view, 'view', True)
        
        # --- 1. Truyền công tắc Flat Color ---
        loc_flat = GL.glGetUniformLocation(self.shader.render_idx, "u_use_flat_color")
        if loc_flat != -1: GL.glUniform1i(loc_flat, 1 if self.use_flat_color else 0)
        
        loc_flat_col = GL.glGetUniformLocation(self.shader.render_idx, "u_flat_color")
        if loc_flat_col != -1: 
            GL.glUniform3f(loc_flat_col, self.flat_color[0], self.flat_color[1], self.flat_color[2])
        
        # --- 2. Truyền công tắc Texture ---
        loc_tex = GL.glGetUniformLocation(self.shader.render_idx, "u_use_texture")
        if loc_tex != -1: GL.glUniform1i(loc_tex, 1 if self.use_texture else 0)
        
        # --- 3. Truyền chế độ Render (0: None, 1: Gouraud, 2: Phong) ---
        loc_mode = GL.glGetUniformLocation(self.shader.render_idx, "u_render_mode")
        if loc_mode != -1: GL.glUniform1i(loc_mode, self.render_mode)
        
        if self.use_texture and self.texture_id is not None:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            loc_sampler = GL.glGetUniformLocation(self.shader.render_idx, "u_texture")
            if loc_sampler != -1: GL.glUniform1i(loc_sampler, 0)
        
        # --- HỆ THỐNG ĐA NGUỒN SÁNG (MULTI-LIGHTING) ---
        lights = getattr(self, 'scene_lights', [])
        loc_num_lights = GL.glGetUniformLocation(self.shader.render_idx, "u_num_lights")
        if loc_num_lights != -1: GL.glUniform1i(loc_num_lights, len(lights))
        
        for i, l in enumerate(lights[:4]): # Hỗ trợ tối đa 4 nguồn sáng cùng lúc
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_pos[{i}]"), *l.position)
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_color[{i}]"), *l.light_color)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_intensity[{i}]"), l.light_intensity)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_active[{i}]"), 1 if l.visible else 0)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
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