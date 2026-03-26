# Hình cầu lưới

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import OpenGL.GL as GL
from libs.shader import Shader
from libs.buffer import VAO, UManager

# Import base shape
from base_shape import BaseShape


class SphereGrid(BaseShape):
    def __init__(self, vert_shader, frag_shader, grid_size=20, radius=0.8):
        super().__init__()  # Initialize transform from BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.grid_size = grid_size 
        self.radius = radius
        
        self.vertices, self.colors = self._generate_sphere_from_cube_grid()

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def _generate_sphere_from_cube_grid(self):
        """
        Cách 2: Tạo 6 lưới hình vuông (6 mặt của cube) 
        rồi chuẩn hóa (normalize) từng đỉnh để chiếu lên mặt cầu.
        Vẽ dạng triangles đặc thay vì wireframe.
        """
        all_vertices = []
        all_colors = []

        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ]

        for local_z in directions:
            local_x = np.array([local_z[1], local_z[2], local_z[0]])
            local_y = np.cross(local_z, local_x)

            # Tạo lưới vertices cho mỗi mặt
            face_vertices = []
            face_colors = []
            
            for j in range(self.grid_size):
                for i in range(self.grid_size):
                    percent_x = i / (self.grid_size - 1)
                    percent_y = j / (self.grid_size - 1)
                    
                    point_on_cube = local_z + (percent_x * 2 - 1) * local_x + (percent_y * 2 - 1) * local_y
                    point_on_sphere = point_on_cube / np.linalg.norm(point_on_cube)
                    
                    face_vertices.append(point_on_sphere * self.radius)
                    color = (point_on_sphere + 1.0) * 0.5 
                    face_colors.append(color)
            
            # Tạo triangles từ lưới vertices
            for j in range(self.grid_size - 1):
                for i in range(self.grid_size - 1):
                    # 4 vertices của mỗi square trong lưới
                    idx = j * self.grid_size + i
                    v0 = face_vertices[idx]
                    v1 = face_vertices[idx + 1]
                    v2 = face_vertices[idx + self.grid_size]
                    v3 = face_vertices[idx + self.grid_size + 1]
                    
                    c0 = face_colors[idx]
                    c1 = face_colors[idx + 1]
                    c2 = face_colors[idx + self.grid_size]
                    c3 = face_colors[idx + self.grid_size + 1]
                    
                    # 2 triangles cho mỗi square
                    all_vertices.extend([v0, v1, v2, v2, v1, v3])
                    all_colors.extend([c0, c1, c2, c2, c1, c3])

        return np.array(all_vertices, dtype=np.float32), np.array(all_colors, dtype=np.float32)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors,   ncomponents=3, stride=0, offset=None)
        return self

    def draw(self, projection, view, model=None):
        GL.glUseProgram(self.shader.render_idx)
        
        # Use BaseShape transform
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])
        self.vao.deactivate()
    
    def set_color(self, color):
        """Set color for the sphere grid - override BaseShape method"""
        # Update colors with new color
        self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        # Re-setup the VBO to update colors
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]  # Get the color VBO at location 1
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)