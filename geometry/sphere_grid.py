"""Sphere approximated by grid on plane projected to sphere for rendering in the BTL project."""

import numpy as np
import OpenGL.GL as GL
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


class SphereGrid:
    """Sphere approximated by grid on plane projected to sphere."""

    def __init__(self, vert_shader, frag_shader, stacks=20, slices=20):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.stacks = stacks
        self.slices = slices
        self.vertices, self.indices = self._generate_sphere()
        self.normals = self.vertices.copy()
        self.colors = np.random.rand(len(self.vertices), 3).astype(np.float32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_sphere(self):
        vertices = []
        for i in range(self.stacks + 1):
            phi = np.pi * i / self.stacks
            for j in range(self.slices + 1):
                theta = 2 * np.pi * j / self.slices
                x = np.sin(phi) * np.cos(theta)
                y = np.cos(phi)
                z = np.sin(phi) * np.sin(theta)
                vertices.append([x, y, z])
        vertices = np.array(vertices, dtype=np.float32)

        indices = []
        for i in range(self.stacks):
            for j in range(self.slices):
                first = i * (self.slices + 1) + j
                second = first + self.slices + 1
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
        indices = np.array(indices, dtype=np.uint32)

        return vertices, indices

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