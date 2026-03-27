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

class SphereGrid(BaseShape):
    def __init__(self, vert_shader, frag_shader, grid_size=20, radius=0.8):
        super().__init__()
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.grid_size = grid_size 
        self.radius = radius
        
        # --- CÁC BIẾN TRẠNG THÁI CHO SIÊU SHADER ---
        self.use_flat_color = False
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.use_texture = False
        self.texture_id = None
        self.render_mode = 2  # Mặc định là Phong Shading
        
        # TẠO DỮ LIỆU LƯỚI (Vị trí, Pháp tuyến, UV, Màu)
        self.vertices, self.normals, self.texcoords, self.colors = self._generate_sphere_from_cube_grid()

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_sphere_from_cube_grid(self):
        all_vertices = []
        all_normals = []
        all_texcoords = []
        all_colors = []

        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ]

        for local_z in directions:
            local_x = np.array([local_z[1], local_z[2], local_z[0]])
            local_y = np.cross(local_z, local_x)

            face_vertices = []
            face_normals = []
            face_texcoords = []
            face_colors = []
            
            for j in range(self.grid_size):
                for i in range(self.grid_size):
                    percent_x = i / (self.grid_size - 1)
                    percent_y = j / (self.grid_size - 1)
                    
                    point_on_cube = local_z + (percent_x * 2 - 1) * local_x + (percent_y * 2 - 1) * local_y
                    point_on_sphere = point_on_cube / np.linalg.norm(point_on_cube)
                    
                    # Vị trí
                    face_vertices.append(point_on_sphere * self.radius)
                    # Pháp tuyến của khối cầu tâm O chính là vector vị trí chuẩn hóa
                    face_normals.append(point_on_sphere)
                    # Tọa độ UV (Map ảnh lặp lại trên 6 mặt cầu cho mượt)
                    face_texcoords.append([percent_x, percent_y])
                    # Màu mặc định (Trắng)
                    face_colors.append([1.0, 1.0, 1.0])
            
            # Khâu các đỉnh thành tam giác (Triangles)
            for j in range(self.grid_size - 1):
                for i in range(self.grid_size - 1):
                    idx = j * self.grid_size + i
                    
                    # 2 tam giác cho mỗi ô vuông
                    indices = [
                        idx, idx + 1, idx + self.grid_size,
                        idx + self.grid_size, idx + 1, idx + self.grid_size + 1
                    ]
                    
                    for k in indices:
                        all_vertices.append(face_vertices[k])
                        all_normals.append(face_normals[k])
                        all_texcoords.append(face_texcoords[k])
                        all_colors.append(face_colors[k])

        return (np.array(all_vertices, dtype=np.float32), 
                np.array(all_normals, dtype=np.float32),
                np.array(all_texcoords, dtype=np.float32),
                np.array(all_colors, dtype=np.float32))

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
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])
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