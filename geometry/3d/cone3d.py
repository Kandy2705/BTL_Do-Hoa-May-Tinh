# Hình nón 3D

import numpy as np
import OpenGL.GL as GL
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


class Cone:
    def __init__(self, vert_shader, frag_shader, slices=20):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.slices = slices
        self.vertices, self.indices = self._generate_cone()
        self.normals = self._generate_normals()
        self.colors = np.random.rand(len(self.vertices), 3).astype(np.float32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_cone(self):
        vertices = []
        vertices.append([0, 0, 0])
        for i in range(self.slices):
            angle = 2 * np.pi * i / self.slices
            vertices.append([np.cos(angle), 0, np.sin(angle)])
        vertices.append([0, 2, 0])
        vertices = np.array(vertices, dtype=np.float32)

        indices = []
        for i in range(1, self.slices):
            indices.extend([0, i, i+1])
        indices.extend([0, self.slices, 1])
        apex = self.slices + 1
        for i in range(self.slices):
            indices.extend([apex, i+1, (i+1) % self.slices + 1])
        indices = np.array(indices, dtype=np.uint32)

        return vertices, indices

    def _generate_normals(self):
        normals = []
        normals.extend([[0, 0, 0]] * (self.slices + 1))
        normals.append([0, 1, 0])
        return np.array(normals, dtype=np.float32)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        if 'gouraud' in self.vert_shader.lower() or 'phong' in self.vert_shader.lower():
            self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)
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
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.size, GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()