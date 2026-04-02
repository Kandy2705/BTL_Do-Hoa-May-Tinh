# Hình tròn
import sys, os
import numpy as np
import OpenGL.GL as GL
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager
from geometry.base_shape import BaseShape




class Circle(BaseShape):
    def __init__(self, vert_shader, frag_shader, segments=64):
        super().__init__()
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.segments = segments
        
        # --- CÁC BIẾN TRẠNG THÁI CHO SIÊU SHADER ---
        self.use_flat_color = False
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.use_texture = False
        self.texture_id = None
        self.render_mode = 2  # 2D nên mặc định là 2 (Phong Shading)
        
        # TẠO DỮ LIỆU (Vị trí, Pháp tuyến, Màu)
        self.vertices, self.normals, self.colors, self.texcoords = self._generate_circle_geometry()

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_circle_geometry(self):
        vertices = [[0.0, 0.0, 0.0]]
        normals = [[0.0, 0.0, 1.0]]
        colors = [[1.0, 1.0, 1.0]]
        texcoords = [[0.5, 0.5]]
        
        for i in range(self.segments + 1):
            angle = 2 * np.pi * i / self.segments
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0

            vertices.append([x, y, z])
            normals.append([0, 0, 1])
            colors.append([1.0, 1.0, 1.0])
            texcoords.append([(x + 1.0) * 0.5, (y + 1.0) * 0.5])

        return (np.array(vertices, dtype=np.float32), 
                np.array(normals, dtype=np.float32),
                np.array(colors, dtype=np.float32),
                np.array(texcoords, dtype=np.float32))

    def setup(self):
        # Bắt buộc tuân thủ layout: 0 (Pos), 1 (Color), 2 (Normal), 3 (UV)
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
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
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, self.vertices.shape[0])
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