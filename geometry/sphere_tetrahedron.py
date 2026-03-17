import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import OpenGL.GL as GL
import glfw

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs import transform as T

V = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.942809, -0.33333],
    [-0.816497, -0.471405, -0.33333],
    [0.816497, -0.471405, -0.33333],
], dtype=np.float32)

def normalize(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

class SphereTetrahedron:
    def __init__(self, vert_shader, frag_shader, subdiv=6, radius=0.8):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.subdiv = subdiv
        self.radius = radius

        vertex_colors = np.array([
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.out_verts = []
        self.out_colors = []
        
        faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
        
        for i0, i1, i2 in faces:
            a = normalize(V[i0])
            b = normalize(V[i1])
            c = normalize(V[i2]) 
            
            color_a = vertex_colors[i0]
            color_b = vertex_colors[i1]
            color_c = vertex_colors[i2]
            
            self._subdivide_with_colors(a, b, c, color_a, color_b, color_c, self.subdiv)
        
        self.vertices = np.array(self.out_verts, dtype=np.float32) * self.radius
        self.colors = np.array(self.out_colors, dtype=np.float32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

        self.angle_x = 0.0
        self.angle_y = 0.0

    def _subdivide_with_colors(self, a, b, c, color_a, color_b, color_c, n):
        if n <= 0:
            self.out_verts.extend([a, b, c])
            self.out_colors.extend([color_a, color_b, color_c])
            return
            
        m = normalize((a + b) * 0.5)
        p = normalize((a + c) * 0.5)
        o = normalize((b + c) * 0.5)
        
        color_m = (color_a + color_b) * 0.5
        color_p = (color_a + color_c) * 0.5
        color_o = (color_b + color_c) * 0.5
        
        self._subdivide_with_colors(a, m, p, color_a, color_m, color_p, n - 1)
        self._subdivide_with_colors(b, m, o, color_b, color_m, color_o, n - 1)
        self._subdivide_with_colors(c, p, o, color_c, color_p, color_o, n - 1)
        self._subdivide_with_colors(m, p, o, color_m, color_p, color_o, n - 1)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors,   ncomponents=3, stride=0, offset=None)
        return self

    def draw(self, projection, view, _model_unused=None):
        GL.glUseProgram(self.shader.render_idx) 

        modelview = view 

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])
        self.vao.deactivate()