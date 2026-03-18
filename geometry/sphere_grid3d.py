# Hình cầu lưới

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import OpenGL.GL as GL
from libs.shader import Shader
from libs.buffer import VAO, UManager

class SphereGrid:
    def __init__(self, vert_shader, frag_shader, grid_size=20, radius=0.8):
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

            for j in range(self.grid_size):
                for i in range(self.grid_size):
                    for dx, dy in [(0,0), (1,0), (0,1), (0,1), (1,0), (1,1)]:
                        percent_x = (i + dx) / self.grid_size
                        percent_y = (j + dy) / self.grid_size
                        
                        point_on_cube = local_z + (percent_x * 2 - 1) * local_x + (percent_y * 2 - 1) * local_y
                        
                        point_on_sphere = point_on_cube / np.linalg.norm(point_on_cube)
                        
                        all_vertices.append(point_on_sphere * self.radius)
                        
                        color = (point_on_sphere + 1.0) * 0.5 
                        all_colors.append(color)

        return np.array(all_vertices, dtype=np.float32), np.array(all_colors, dtype=np.float32)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors,   ncomponents=3, stride=0, offset=None)
        return self

    def draw(self, projection, view, model=None):
        GL.glUseProgram(self.shader.render_idx) 

        if model is None:
            model = np.identity(4, dtype=np.float32)
            
        modelview = view @ model 

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])
        self.vao.deactivate()