# Hình sao

import numpy as np
import OpenGL.GL as GL
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


class Star:
    def __init__(self, vert_shader, frag_shader, points=5):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        angles = np.linspace(0, 2*np.pi, 2*points, endpoint=False)
        radii = np.tile([1, 0.5], points)

        center = np.array([[0, 0, 0]], dtype=np.float32)

        outer_vertices = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles),
            np.zeros(2*points)
        ])

        outer_vertices = np.vstack([outer_vertices, outer_vertices[0]])

        self.vertices = np.vstack([center, outer_vertices]).astype(np.float32)

        self.normals = np.tile([0, 0, 1], (len(self.vertices), 1)).astype(np.float32)
        self.colors = np.random.rand(len(self.vertices), 3).astype(np.float32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        if 'gouraud' in self.vert_shader.lower() or 'phong' in self.vert_shader.lower():
            self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view @ (model if model is not None else np.identity(4, dtype=np.float32))
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)
        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, len(self.vertices))
        self.vao.deactivate()